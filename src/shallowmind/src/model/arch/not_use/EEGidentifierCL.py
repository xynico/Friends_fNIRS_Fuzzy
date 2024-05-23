
import torch
import torch.nn as nn
from ..base import BaseArch
from einops.layers.torch import Rearrange
import sys
from functools import partial
from timm.models.vision_transformer import Block
import numpy as np
from ...builder import ARCHS, build_backbone, build_head, build_arch
# L2 Norm
from torch.nn.functional import normalize
from torch import logical_not
from torch.nn import Transformer
from timm.models.vision_transformer import PatchEmbed, Block
from vit_pytorch import ViT

@ARCHS.register_module()
class EEGidentifierCLV0(BaseArch):
    def __init__(self, 
                pretrained = True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),  
                img_size = (60, 1000), # [H, W]
                num_heads = 8,
                depth = 2,
                cl_embed_dim = 128,
                dropout = 0.1,
            ):
        super().__init__()
        # --------------------------------------------------------------------------
        # encoder specifics
        self.encoder = Transformer(
            d_model=img_size[1], 
            nhead=num_heads,
            num_encoder_layers=depth,
            num_decoder_layers=depth,
            dim_feedforward=cl_embed_dim,
            dropout=dropout,
        )
        # if test, dropout = 0
        if not self.training:
            self.encoder.dropout = 0
            
        self.norm = norm_layer(img_size[1])
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
       
        # CL specifics
        self.cl_projector = nn.Sequential(
                nn.Linear(img_size[1], cl_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(cl_embed_dim, cl_embed_dim),
            )
        self.norm_CL = norm_layer(cl_embed_dim)
        self.CL_loss = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)
        # --------------------------------------------------------------------------
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def gen_data(self, data):
        imgs = data['seq']  # [B, 2, CH, TP]
        device = imgs.device
        subjects = data['subject'].detach().cpu().tolist()
        first_occurrence_dict = {subj: None for subj in subjects}
        for idx, subject in enumerate(subjects):
            if first_occurrence_dict[subject] is None:
                first_occurrence_dict[subject] = idx
        first_occurrences_indices = list(first_occurrence_dict.values())
        first_occurrences_indices = torch.tensor(first_occurrences_indices, device=device)
        imgs = imgs[first_occurrences_indices]  # [B', 2, CH, TP]
        imgs = torch.cat([imgs[:, 0, :, :], imgs[:, 1, :, :]], dim=0)  # [2B', CH, TP]
        return imgs

    def forward(self, data):
        imgs = self.gen_data(data) # [2B', CH, TP]

        # Encoder
        latent = self.forward_encoder(imgs) # [2B', D_CL]

        # CL
        latent = self.cl_projector(latent) # [2B', D_CL]
        # latent = self.norm_CL(latent)

        # SupConLoss
        loss = self.forward_loss_cl(latent)

        return loss
    
    def forward_encoder(self, x):
        '''
        x: [B, CH, TP]
        '''
        x = self.encoder(x, x)
        # average pooling
        x = x.mean(dim=1)
        x = self.norm(x)

        return x
    
    def forward_loss_cl(self, cls_token_ouput):
        '''
        cls_token_ouput: concat([N, D], [N, D]) -> [2N, D]
        '''
        N = cls_token_ouput.shape[0] // 2
        cls_token_ouput = torch.cat([cls_token_ouput[:N, :].unsqueeze(1), cls_token_ouput[N:, :].unsqueeze(1)], dim=1)
        loss = self.CL_loss(cls_token_ouput)
        return loss

    def forward_train(self, x, label):
        loss = self.forward(x)
        # pack the output and losses
        return {'loss': loss}

    def forward_test(self, x, label=None):
        loss = self.forward(x)
        return {'loss': loss, 'output': label, 'meta_data': None, 'label': label}
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def gradNorm(net, layer, alpha, dataloader, num_epochs, lr1, lr2, log=False):
    """
    Args:
        net (nn.Module): a multitask network with task loss
        layer (nn.Module): a layers of the full network where appling GradNorm on the weights
        alpha (float): hyperparameter of restoring force
        dataloader (DataLoader): training dataloader
        num_epochs (int): number of epochs
        lr1（float): learning rate of multitask loss
        lr2（float): learning rate of weights
        log (bool): flag of result log
    """
    # init log
    if log:
        log_weights = []
        log_loss = []
    # set optimizer
    optimizer1 = torch.optim.Adam(net.parameters(), lr=lr1)
    # start traning
    iters = 0
    net.train()
    for epoch in range(num_epochs):
        # load data
        for data in dataloader:
            # cuda
            if next(net.parameters()).is_cuda:
                data = [d.cuda() for d in data]
            # forward pass
            loss = net(*data)
            # initialization
            if iters == 0:
                # init weights
                weights = torch.ones_like(loss)
                weights = torch.nn.Parameter(weights)
                T = weights.sum().detach() # sum of weights
                # set optimizer for weights
                optimizer2 = torch.optim.Adam([weights], lr=lr2)
                # set L(0)
                l0 = loss.detach()
            # compute the weighted loss
            weighted_loss = weights @ loss
            # clear gradients of network
            optimizer1.zero_grad()
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True)
            # compute the L2 norm of the gradients for each task
            gw = []
            for i in range(len(loss)):
                dl = torch.autograd.grad(weights[i]*loss[i], layer.parameters(), retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)
            # compute loss ratio per task
            loss_ratio = loss.detach() / l0
            # compute the relative inverse training rate per task
            rt = loss_ratio / loss_ratio.mean()
            # compute the average gradient norm
            gw_avg = gw.mean().detach()
            # compute the GradNorm loss
            constant = (gw_avg * rt ** alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            # clear gradients of weights
            optimizer2.zero_grad()
            # backward pass for GradNorm
            gradnorm_loss.backward()
            # log weights and loss
            if log:
                # weight for each task
                log_weights.append(weights.detach().cpu().numpy().copy())
                # task normalized loss
                log_loss.append(loss_ratio.detach().cpu().numpy().copy())
            # update model weights
            optimizer1.step()
            # update loss weights
            optimizer2.step()
            # renormalize weights
            weights = (weights / weights.sum() * T).detach()
            weights = torch.nn.Parameter(weights)
            optimizer2 = torch.optim.Adam([weights], lr=lr2)
            # update iters
            iters += 1
    # get logs
    if log:
        return np.stack(log_weights), np.stack(log_loss)