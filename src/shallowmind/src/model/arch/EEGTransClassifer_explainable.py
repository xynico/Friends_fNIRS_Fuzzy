
import torch
import torch.nn as nn
from .base import BaseArch
from einops.layers.torch import Rearrange
import sys
from functools import partial
import numpy as np
from ..builder import ARCHS, build_backbone, build_head, build_arch
from torch.nn.functional import normalize
# from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
# from .not_use.EEGidentifierCL import EEGidentifierCLV0
# from .FuzzyTransformerEncoderDecoder import FuzzyTransformer
# import torch
# import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
from torch import Tensor
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention



class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None,
                 return_attention: bool = True) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.return_attention = return_attention
        super(nn.TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(
        self,tgt: Tensor,memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,memory_is_causal: bool = False,
        return_attention: bool = True,
        ) -> Tensor:

        x = tgt if type(tgt) is Tensor else tgt[0]

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        if not self.return_attention:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                x = x + self._ff_block(self.norm3(x))
            else:
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
                x = self.norm3(x + self._ff_block(x))
            return x

        else:
            if self.norm_first:
                
                o1,att = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                x = x + o1
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                x = x + self._ff_block(self.norm3(x))
            else:
                o1,att = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
                x = self.norm1(x + o1)
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
                x = self.norm3(x + self._ff_block(x))
            return x,att


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        if not self.return_attention:
            
            x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
            return self.dropout1(x)
        else:
            x, attn = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True,
                           average_attn_weights=False)

            return self.dropout1(x), attn
    

class TransformerDecoder(nn.TransformerDecoder):
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,return_attention: bool = True
                ) -> Tensor:
        output = tgt

        for mod in self.layers:
            output, att = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         return_attention=return_attention,
                         )

        if self.norm is not None:
            output = self.norm(output)

        return output,att
    
class Transformer(nn.Transformer):
    
    def __init__(self, return_attention: bool = True,
                d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,return_attention,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

data = torch.rand(10, 32, 512)
tgt = torch.rand(10, 32, 512)
transformer_model = Transformer(device='cpu', nhead=8)
output = transformer_model(data, tgt)


@ARCHS.register_module()
class EEGTransClassifer_explainable(BaseArch):
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
        latent, att = self.forward_encoder(imgs) # [2B, D_CL]
        if not self.encoder_type == 'TransformerMergeinTransformer':
            latent = torch.cat([latent[:latent.shape[0]//2, :], latent[latent.shape[0]//2:, :]], dim=1)
        # Decoder
        # latent = self.forward_decoder(latent) # [2B, D_CL]

        latent = self.cls(latent) # [B, 2]
        pred = latent.squeeze(1)
        loss = self.cls_loss(pred, label)
        return loss, pred, att
    
    def forward_encoder(self, x):
        '''
        x: [B, CH, TP]
        '''
        if self.encoder_type == 'Transformer':
            x,att = self.encoder(x, x)
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
        return x,att
    
    def forward_decoder(self, x):
        '''
        x: [B, CH, TP]
        '''
        x = self.decoder(x, x)
        # average pooling
        x = self.norm_decoder(x)

        return x
    

    def forward_train(self, x, label):
        loss, pred,att = self.forward(x, label)
        # pack the output and losses
        return {'loss': loss}

    def forward_test(self, x, label=None):
        loss, pred, att = self.forward(x, label)
        pred = pred.argmax(dim=1)
        return {'loss': loss, 'output': pred, 'meta_data': {'att':att}, 'label': label}
    