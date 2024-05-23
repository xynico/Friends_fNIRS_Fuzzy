
import torch
import torch.nn as nn
from .base import BaseArch
from einops.layers.torch import Rearrange
import sys
from functools import partial
import numpy as np
from ..builder import ARCHS, build_backbone, build_head, build_arch
from torch.nn.functional import normalize
from torch.nn import Transformer
from .EEGidentifierCL import EEGidentifierCLV0
import torch.nn.functional as F

@ARCHS.register_module()
class EEGTransClassiferAtt(BaseArch):
    def __init__(self, pretrained = True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),  
                img_size = (60, 1000), # [H, W]
                num_heads = 8,
                depth = 2,
                cl_embed_dim = 128,
                dropout = 0.1,
                model_ckpt = None,
                fixed = True,
                encoder_type = 'Transformer',
            ):
        super().__init__()
        self.__dict__.update(locals())

        # --------------------------------------------------------------------------
        # encoder specifics
        if encoder_type == 'Transformer':
            self.encoder = Transformer(
                d_model=img_size[1], 
                nhead=num_heads,
                num_encoder_layers=depth,
                num_decoder_layers=depth,
                dim_feedforward=cl_embed_dim,
                dropout=dropout,
            )
            self.norm_encoder = norm_layer(img_size[1])
        elif encoder_type == 'LSTM':
            self.encoder = nn.LSTM(
                input_size=img_size[1],
                hidden_size=cl_embed_dim,
                num_layers=depth,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            self.norm_encoder = norm_layer(cl_embed_dim*2)
        else:
            raise NotImplementedError
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # CLS specifics
        if encoder_type == 'Transformer':
            self.cls = nn.Sequential(
                    nn.Linear(2*img_size[1], cl_embed_dim),
                    nn.ReLU(inplace=False),
                    nn.Linear(cl_embed_dim, 2),
                )
        elif encoder_type == 'LSTM':
            self.cls = nn.Sequential(
                    nn.Linear(4*cl_embed_dim, cl_embed_dim),
                    nn.ReLU(inplace=False),
                    nn.Linear(cl_embed_dim, 2),
                )
        self.cls_loss = nn.CrossEntropyLoss()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Attention specifics
        self.att_weight = nn.Parameter(torch.ones(img_size[0], img_size[0]), requires_grad=True)
        self.att_proj = nn.Sequential(
            nn.Linear(img_size[0]*img_size[0], cl_embed_dim),
            nn.ReLU(inplace=False),
            nn.Linear(cl_embed_dim, 2),
        )
        # --------------------------------------------------------------------------
        
        self.initialize_weights()
        self.setup()
    
    def setup(self, stage=None):

        if self.model_ckpt is not None:
            ckpt = torch.load(self.model_ckpt, map_location='cpu')
            ckpt["state_dict"] = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
            ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items() if "encoder" in k}            
            # copy the weights from pre_train_model.encoder to self.encoder
            self.encoder.load_state_dict(ckpt["state_dict"], strict=False)
            if self.fixed == True:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
        

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
        imgs = torch.cat([imgs[:, 0, :, :], imgs[:, 1, :, :]], dim=0)  # [B, CH, TP]
        return imgs

    def calculate_channelwise_cosine_similarity(self, data):
        # data: [B, 2, CH, TP]

        # Extract the two sets of channels
        set1 = data[:, 0, :, :]  # [B, CH, TP]
        set2 = data[:, 1, :, :]  # [B, CH, TP]
        B, CH, TP = set1.shape
        cos_sim = torch.sum(set1.unsqueeze(2) * set2.unsqueeze(1), dim=-1)
        return cos_sim


    def forward(self, data, label):
        imgs = self.gen_data(data) # [2B, CH, TP]

        # encoder
        latent = self.forward_encoder(imgs) # [2B, D_CL]
        latent = torch.cat((latent[:latent.size(0)//2], latent[latent.size(0)//2:]), dim=1)
        latent = self.cls(latent) # [B, 2]

        # attention weight
        imgs = data['seq']  # [B, 2, CH, TP]

        # [B, 2, CH, TP] -> [B, CH, CH]
        cos_sim = self.calculate_channelwise_cosine_similarity(imgs)
        att_weight = torch.softmax(self.att_weight, dim=1)
        cos_sim = cos_sim * att_weight.unsqueeze(0) # [B, CH, CH]
        cos_sim_vector = cos_sim.view(cos_sim.size(0), -1) # [B, CH*CH]
        p_weight = self.att_proj(cos_sim_vector) # [B, 2]

        # [B, 2] * [B, 2] -> [B, 2]
        latent = latent * p_weight
        # 预测和损失
        pred = latent.squeeze(1)
        loss = self.cls_loss(pred, label)
        return loss, pred

    
    def forward_encoder(self, x):
        '''
        x: [B, CH, TP]
        '''
        if self.encoder_type == 'Transformer':
            x = self.encoder(x, x)
            # average pooling
            x = x.mean(dim=1)
            x = self.norm_encoder(x)
        elif self.encoder_type == 'LSTM':
            # x = x.permute(0, 2, 1)
            x, _ = self.encoder(x)
            x = self.norm_encoder(x[:, -1, :])

        return x
    
    def forward_decoder(self, x):
        '''
        x: [B, CH, TP]
        '''
        x = self.decoder(x, x)
        # average pooling
        x = self.norm_decoder(x)

        return x
    

    def forward_train(self, x, label):
        loss, pred = self.forward(x, label)
        # pack the output and losses
        return {'loss': loss}

    def forward_test(self, x, label=None):
        loss, pred = self.forward(x, label)
        pred = pred.argmax(dim=1)
        return {'loss': loss, 'output': pred, 'meta_data': None, 'label': label}
    