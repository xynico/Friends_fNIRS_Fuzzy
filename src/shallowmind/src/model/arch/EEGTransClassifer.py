
import torch
import torch.nn as nn
from .base import BaseArch
from einops.layers.torch import Rearrange
import sys
from functools import partial
import numpy as np
from ..builder import ARCHS, build_backbone, build_head, build_arch
from torch.nn.functional import normalize
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
# from .not_use.EEGidentifierCL import EEGidentifierCLV0
# from .FuzzyTransformerEncoderDecoder import FuzzyTransformer


@ARCHS.register_module()
class EEGTransClassifer(BaseArch):
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

        # encoder specifics
        if encoder_type == 'Transformer':
            self.encoder = Transformer(
                d_model=img_size[1], 
                nhead=num_heads,
                num_encoder_layers=depth,
                num_decoder_layers=depth,
                dim_feedforward=cl_embed_dim,
                dropout=dropout,
                batch_first=True,
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

        elif encoder_type == 'FuzzyTransformer':
            self.encoder = FuzzyTransformer(
                input_shape=(1,img_size[0],img_size[1]),
                projection=True, 
                num_encoder_layers=depth, 
                num_decoder_layers=depth, 
                cl_embed_dim=cl_embed_dim,
                n_rules = num_heads,
            )
            self.norm_encoder = norm_layer(img_size[1])
        elif encoder_type == 'TransformerEncoder':
            encoder_layer = TransformerEncoderLayer(d_model=img_size[1],
                                                     nhead=num_heads, 
                                                     dim_feedforward=cl_embed_dim, 
                                                     dropout=dropout)
            self.encoder = TransformerEncoder(encoder_layer, depth)
            self.norm_encoder = norm_layer(img_size[1])
        elif encoder_type == 'TransformerMergeinTransformer':
            self.encoder = Transformer(
                d_model=img_size[1], 
                nhead=num_heads,
                num_encoder_layers=depth,
                num_decoder_layers=depth,
                dim_feedforward=cl_embed_dim,
                dropout=dropout,
            )
            self.norm_encoder = norm_layer(img_size[1])
        else:
            raise NotImplementedError
            
        
        # --------------------------------------------------------------------------

        # # --------------------------------------------------------------------------
        # # decoder specifics
        # self.decoder = Transformer(
        #     d_model=img_size[1]*2,
        #     nhead=num_heads,
        #     num_encoder_layers=depth,
        #     num_decoder_layers=depth,
        #     dim_feedforward=cl_embed_dim,
        #     dropout=dropout,
        # )
        # self.norm_decoder = norm_layer(img_size[1]*2)
        # # --------------------------------------------------------------------------

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
        elif encoder_type == 'FuzzyTransformer':
            self.cls = nn.Sequential(
                    nn.Linear(2*img_size[1], cl_embed_dim),
                    nn.ReLU(inplace=False),
                    nn.Linear(cl_embed_dim, 2),
                )
        elif encoder_type == 'TransformerMergeinTransformer':
            self.cls = nn.Sequential(
                    nn.Linear(img_size[1], cl_embed_dim),
                    nn.ReLU(inplace=False),
                    nn.Linear(cl_embed_dim, 2),
                )
        else:
            raise NotImplementedError
        self.cls_loss = nn.CrossEntropyLoss()
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

    def forward(self, data, label):
        imgs = self.gen_data(data) # [2B, CH, TP]
        # Encoder
        latent = self.forward_encoder(imgs) # [2B, D_CL]
        if not self.encoder_type == 'TransformerMergeinTransformer':
            latent = torch.cat([latent[:latent.shape[0]//2, :], latent[latent.shape[0]//2:, :]], dim=1)
        # Decoder
        # latent = self.forward_decoder(latent) # [2B, D_CL]

        latent = self.cls(latent) # [B, 2]
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
        elif self.encoder_type == 'FuzzyTransformer':
            x = self.encoder(x)
            x = x.mean(dim=1)
            x = self.norm_encoder(x)
        elif self.encoder_type == 'TransformerMergeinTransformer':
            subj1 = x[:x.shape[0]//2]
            subj2 = x[x.shape[0]//2:]
            x = self.encoder(subj1,subj2)
            # average pooling
            x = x.mean(dim=1)
            x = self.norm_encoder(x)
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
    