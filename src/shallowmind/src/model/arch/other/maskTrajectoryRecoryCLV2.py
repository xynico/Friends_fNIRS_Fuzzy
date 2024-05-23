
import torch
import torch.nn as nn
from .base import BaseArch
from einops.layers.torch import Rearrange
import sys
from functools import partial
from timm.models.vision_transformer import Block
import numpy as np
from util import *
from ..builder import ARCHS, build_backbone, build_head, build_arch
# L2 Norm
from torch.nn.functional import normalize
from wh_models import PMA
from torch import logical_not

@ARCHS.register_module()
class maskTrajectoryRecoryCLV2(BaseArch):
    def __init__(self, pretrained = True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),  
                embed_dim=1024, 
                depth=24, 
                num_heads=16,
                decoder_embed_dim=512, 
                decoder_depth=8, 
                decoder_num_heads=16,
                img_size = [800,2],
                patch_size = [5,2],
                in_chans=1,
                mlp_ratio=4,
                norm_pix_loss=False,
                mask_ratio=0.75,
                special_token_value = -0.01,
                cl_pos = 'after_decoder', 
                cl_embed_dim = 512,
                weight = {
                    'mae_loss': 0.97,
                    'cl_loss': 0.03,
                }, # or 'gradnorm' or 'deltaLoss'
                gradnorm_alpha = 0.12,
            ):
        super().__init__()
        self.weight = weight
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.mask_ratio = mask_ratio
        self.special_token_value = special_token_value
        self.weights = nn.Parameter(torch.ones(2))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.norm_pix_loss = norm_pix_loss
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0]*patch_size[1], bias=True)
        # --------------------------------------------------------------------------


        # --------------------------------------------------------------------------
        # CL specifics
        self.cl_pos = cl_pos
        if self.cl_pos == 'after_encoder':
            self.cl = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, cl_embed_dim)
            )

        elif self.cl_pos == 'after_decoder':
            self.cl = nn.Sequential(
                nn.Linear(int(patch_size[0]*patch_size[1]), int(patch_size[0]*patch_size[1])),
                nn.ReLU(inplace=True),
                nn.Linear(int(patch_size[0]*patch_size[1]), cl_embed_dim)
            )
        elif self.cl_pos == 'after_encoder_avg_pooling':
            self.cl = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, cl_embed_dim)
            )
        elif self.cl_pos == 'after_decoder_avg_pooling':
            self.cl = nn.Sequential(
                nn.Linear(int(patch_size[0]*patch_size[1]), int(patch_size[0]*patch_size[1])),
                nn.ReLU(inplace=True),
                nn.Linear(int(patch_size[0]*patch_size[1]), cl_embed_dim)
            )
        elif self.cl_pos == 'after_encoder_attention_pooling':
            input_size = embed_dim * ( (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) + 1)
            self.cl = nn.Sequential(
                    PMA(dim=input_size, num_heads=4, num_seeds=1),
                    nn.Linear(in_features=embed_dim, out_features=embed_dim),
                )
        self.CL_loss = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)

        # --------------------------------------------------------------------------

        # NormGrad specifics
        if self.weight == 'gradnorm':
            self.gradnorm_alpha = gradnorm_alpha
            self.weights = nn.Parameter(torch.ones(2))  
            self.initial_losses = None
        # --------------------------------------------------------------------------

        # deltaLoss specifics
        if self.weight == 'deltaLoss':
            self.initial_losses = None

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        # print('pos_embed.shape', pos_embed.shape)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, pw*ph)
        """
        assert imgs.shape[2] % self.patch_embed.patch_size[0] == 0

        h = imgs.shape[2] // self.patch_embed.patch_size[0]
        w = imgs.shape[3] // self.patch_embed.patch_size[1]
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, self.patch_embed.patch_size[0], w, self.patch_embed.patch_size[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, pw*ph)
        imgs: (N, 1, H, W)
        """
        pw = self.patch_embed.patch_size[0]
        ph = self.patch_embed.patch_size[1]
        w = 1
        h = int(x.shape[1] / w)
        
        x = x.reshape(shape=(x.shape[0], h, w, pw, ph, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * pw, w * ph))
        return imgs
    
    def random_masking(self, x, num_unpad_patches_h):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        num_unpad_patches_h: [N], number of unpad patches in height, for each sample; others are padding so directly mask them before random shuffling
        """
        mask_ratio = self.mask_ratio
        N, L, D = x.shape  # batch, length, dim

        selected_length = torch.ceil(num_unpad_patches_h *  mask_ratio)
        selected_length = torch.clamp(selected_length, min=1).int()

        # random selection of patches to be unmasked
        selected_ids = [torch.randperm(num_unpad_patches_h[i])[:selected_length[i]] for i in range(N)]
        mask = torch.zeros([N, L], dtype=torch.bool, device=x.device)
        padding = torch.zeros([N, L], dtype=torch.bool, device=x.device)
        for i in range(N):
            mask[i, selected_ids[i]] = True
            padding[i, :num_unpad_patches_h[i]] = True
        return mask, padding
    
    def forward_encoder(self, x, padding_length):

        # x.shape torch.Size([B, 1, 350, 2]) padding_length.shape torch.Size([64]) self.patch_embed(x).shape torch.Size([64, 14, 1024])
    
        # embed patches
        x, num_unpad_patches_h = self.patch_embed(x, padding_length)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        mask, padding = self.random_masking(x, num_unpad_patches_h)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)

        return x, mask, padding

    def forward_decoder(self, x, mask, padding):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # x: [N, L+1, D], mask: [N, L], padding: [N, L]
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if mask[i, j] == 1 and padding[i, j] == 1:
        #             x[i, j, :] = self.mask_token

        mask_indices = (mask == 1) & (padding == 1)
        cl_token = x[:, 0, :]
        x = torch.where(mask_indices.unsqueeze(-1), self.mask_token, x[:, 1:, :])
        x = torch.cat([cl_token.unsqueeze(1), x], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # split cls token
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]
        return x, cls_token

    def forward_loss(self, imgs, pred, mask, padding):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, pw*ph]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask * padding).sum() / (mask * padding).sum()  # mean loss on removed patches
        return loss
    
    def forward_one_sample(self, imgs, padding_length):
        latent, mask, padding = self.forward_encoder(imgs, padding_length) # latent: [N, L+1, D]
        pred, cls_token = self.forward_decoder(latent, mask, padding)

        if self.cl_pos == 'after_encoder':
            cls_token = latent[:, -1, :] # [N, D]

        elif self.cl_pos == 'after_decoder':
            cls_token = cls_token[:, 0, :] # [N, D]

        elif self.cl_pos == 'after_encoder_avg_pooling':
            # mask_padding = logical_not(mask)  * padding # [N, L] if mask_padding[i, j] == 1, then it is a valid patch
            # cls_token = torch.cat([torch.stack([latent[i,j,:] for j in range(mask_padding.shape[1]) if mask_padding[i, j] == 1], dim=1).mean(dim=1).unsqueeze(0)
            #                        for i in range(mask_padding.shape[0])], dim=0) # [N, D]
            cls_token = self.calc_cls_token_avg_enc(mask, padding, latent)
            
        elif self.cl_pos == 'after_decoder_avg_pooling':
            cls_token = torch.cat([cls_token[:, 0, :], pred], dim=1).mean(dim=1)

        elif self.cl_pos == 'after_encoder_attention_pooling':
            # cls_token = latent.view(latent.shape[0], -1) # [N, D]
            cls_token = latent

        return pred, cls_token, mask, padding


    def forward(self, data):
        imgs = torch.cat([data['seq'][:, 0, :, :], data['seq'][:, 1, :, :]], dim=0).unsqueeze(1) # [2*N, 1, H, W]
        padding_length = torch.cat([data['padding_length'][0], data['padding_length'][1]], dim=0) # [2*N]

        pred, cl_token, mask, padding = self.forward_one_sample(imgs, padding_length)

        cls_token_ouput = self.cl(cl_token)
        cls_token_ouput = normalize(cls_token_ouput, dim=-1)

        loss_cl = self.forward_loss_cl(cls_token_ouput, mask, padding)
        loss_mae = self.forward_loss(imgs, pred, mask, padding)
        w1 = self.weight['cl_loss']
        w2 = self.weight['mae_loss']
        loss = w1 * loss_cl + w2 * loss_mae
        return loss, pred, mask, padding_length, loss_cl, loss_mae, cls_token_ouput
    
    def forward_loss_cl(self, cls_token_ouput, mask, padding):
        '''
        cls_token_ouput: concat([N, D], [N, D]) -> [2N, D]
        '''
        N = cls_token_ouput.shape[0] // 2
        # label = torch.arange(N).to(cls_token_ouput.device) # [N]
        cls_token_ouput = torch.cat([cls_token_ouput[:N, :].unsqueeze(1), cls_token_ouput[N:, :].unsqueeze(1)], dim=1)
        loss = self.CL_loss(cls_token_ouput)
        return loss

    def forward_train(self, x, label):
        loss, pred, mask, padding_length, loss_cl, loss_mae, cls_token_ouput = self.forward(x)
        # pack the output and losses
        return {'loss': loss, 'loss_cl': loss_cl, 'loss_mae': loss_mae}

    def forward_test(self, x, label=None):
        loss, pred, mask, padding_length, loss_cl, loss_mae, cls_token_ouput = self.forward(x)
        pred = self.unpatchify(pred)
        label = torch.cat([label[:, 0], label[:, 1]], dim=0) # [2*N]

        meta_data = {'mask': mask, 'padding_length': padding_length, 'label': label, 'cls_token_ouput': cls_token_ouput}
        return {'loss': loss, 'output': pred, 'meta_data': meta_data, 'loss_cl': loss_cl, 'loss_mae': loss_mae}

    def calc_cls_token_avg_enc(self, mask, padding, latent):
        # Assuming mask is a boolean tensor of shape [N, L] and latent is of shape [N, L, D]

        # Invert the mask and use it for masking
        mask_padding = logical_not(mask)  * padding   # Using ~ for logical_not
        mask_padding = torch.cat([torch.zeros(mask_padding.shape[0], 1, dtype=torch.bool, device=mask_padding.device), mask_padding], dim=1)

        # Pre-allocate tensor for cls_token
        N, L, D = latent.size()
        cls_token = torch.zeros(N, D, device=latent.device)

        # Efficient computation using vectorized operations
        for i in range(N):
            valid_patches = latent[i, mask_padding[i]]  # Extract valid patches based on mask
            if valid_patches.nelement() != 0:  # Check if there are any valid patches
                cls_token[i] = valid_patches.mean(dim=0)  # Compute mean of valid patches

        return cls_token

    
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