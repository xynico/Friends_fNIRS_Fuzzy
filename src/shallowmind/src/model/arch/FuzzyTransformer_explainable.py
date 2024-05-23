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
# from .EEGidentifierCL import EEGidentifierCLV0
# from .FuzzyTransformerEncoderDecoder import FuzzyTransformer
from torch.nn import Linear, Dropout, Softmax
from .EEGTransClassifer_explainable import Transformer


import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F


class FuzzyTransformer(Transformer):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, num_rules, seq_len, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(FuzzyTransformer, self).__init__(
            d_model, nhead, num_encoder_layers, num_decoder_layers, 
            dim_feedforward, dropout, activation, batch_first=True
        )
        self.batch_first = True
        # Replace the standard MultiheadAttention with FuzzyMultiheadAttention in each layer
        self._replace_attention_layers(self.encoder, d_model, nhead, num_rules, seq_len)
        self._replace_attention_layers(self.decoder, d_model, nhead, num_rules, seq_len)

    def _replace_attention_layers(self, transformer_layer, d_model, nhead, num_rules, seq_len):
        for layer in transformer_layer.layers:
            # print("attention class", type(layer.self_attn))
            num_params = sum(p.numel() for p in layer.self_attn.parameters() if p.requires_grad)
            # print(f"original parameters: {num_params}")
            layer.self_attn = FuzzyMultiheadAttention(d_model, nhead, num_rules, seq_len)

            num_params = sum(p.numel() for p in layer.self_attn.parameters() if p.requires_grad)
            # print(f"fuzzy parameters: {num_params}")
            # if isinstance(layer, TransformerEncoderLayer) or isinstance(layer, TransformerDecoderLayer):
            #     # print("layer",layer)
            #     num_params = sum(p.numel() for p in layer.multihead_attn.parameters() if p.requires_grad)
            #     print(f"original parameters: {num_params}")
            #     layer.multihead_attn = FuzzyMultiheadAttention(d_model, nhead, num_rules, seq_len)
            #     num_params = sum(p.numel() for p in layer.multihead_attn.parameters() if p.requires_grad)
            #     print(f"fuzzy parameters: {num_params}")
                

# Example usage
# d_model = 512
# nhead = 8
# num_encoder_layers = 6
# num_decoder_layers = 6
# num_rules = 10
# seq_len = 50

# fuzzy_transformer = FuzzyTransformer(d_model, nhead, num_encoder_layers, num_decoder_layers, num_rules, seq_len)



class FuzzyMultiheadAttention(nn.Module):
    def __init__(
                self, 
                embed_dim, 
                num_heads, 
                num_rules, 
                seq_len, 
                dropout=0.,
                need_weights=True,
                average_attn_weights=False
            ):
        super().__init__()
        self.batch_first = True
        self._qkv_same_embed_dim = True
        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.num_rules = num_rules
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.high_dim = False
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights

        # Initialize rules (keys and values) as parameters
        if self.high_dim:
            self.rules_keys = nn.Parameter(torch.Tensor(num_rules, num_heads, seq_len, self.head_dim))
            self.rules_widths = nn.Parameter(torch.ones(num_rules, num_heads, seq_len,self.head_dim))
            self.value_proj = Linear(embed_dim, embed_dim*num_rules)
        else:

            self.rules_keys = nn.Parameter(torch.Tensor(self.num_heads, self.num_rules, self.head_dim))
            self.rules_widths = nn.Parameter(torch.ones(self.num_heads, self.num_rules, self.head_dim))
            self.value_proj = Linear(embed_dim, embed_dim*num_rules)
            # self.rules_values = nn.Parameter(torch.Tensor(num_rules, embed_dim))
        self.query_proj = Linear(embed_dim, embed_dim)

        # self.value_proj = Linear(embed_dim, embed_dim*num_rules)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(dropout)
        
        # Initialize rules keys and values
        nn.init.normal_(self.rules_keys, mean=0.0, std=0.02)
        # nn.init.normal_(self.rules_values, mean=0.0, std=0.02)
    def custom_distance(self,query):
        if self.high_dim:
            key = self.rules_keys.unsqueeze(0).repeat(query.shape[0], 1, 1, 1,1)
        else:
            key = self.rules_keys.unsqueeze(0).unsqueeze(0)
            key = key.repeat(query.shape[0], query.shape[2],1,1,1).permute(0,2,1,3,4)
            # key=self.rules_keys
        l1_distance = torch.abs(query.unsqueeze(-2).repeat(1,1,1,self.num_rules,1) - key)
        return l1_distance
    def get_z_values(self, query_key_distance):
        # Calculate z values from query-key distance
        if self.high_dim:  
            prot=torch.div(query_key_distance, self.rules_widths)
            root=-torch.square(prot)*0.5
            z_values =root.mean(-1) # HTSK
            return z_values.permute(0, 2, 3, 1)
        else:
            prot=torch.div(query_key_distance.permute(0,2,1,3,4), self.rules_widths.unsqueeze(0).unsqueeze(0))
            root=-torch.square(prot)*0.5
            # print("root", root.shape)
            z_values =root.mean(-1) # HTSK
            return z_values
    def forward(self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False
            ) -> Tuple[Tensor, Optional[Tensor]]:
        # Project query
        batch_size, seq_length, _ = query.size()
        query = self.query_proj(query) * self.scale # [B, seq_len, embed_dim]
        query = query.reshape(batch_size, seq_length, self.num_heads, -1) # [B, seq_len, head, head_dim]
        query = query.transpose(1, 2) # [B, head, seq_len, head_dim]
        # Repeat keys and values for each head
        # key = self.rules_keys.unsqueeze(0).repeat(batch_size, 1, 1, 1,1)

        value = self.value_proj(value) * self.scale # [B, seq_len, embed_dim]
        value = value.reshape(batch_size, seq_length, self.num_heads,  -1, self.num_rules) # [B, seq_len, head, head_dim]
        value = value.transpose(1, 2) # [B, head, seq_len, head_dim]
        # value = value.unsqueeze(0).

        # Calculate L2 distance between query and key
        # Reshape query and key to align sequence dimensions for cdist calculation
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        # _, num_heads, num_rules, seq_len_k, _ = key.shape
        
        # key = key.reshape(batch_size * num_heads, num_rules, seq_len_k, head_dim)

        query_key_distance = self.custom_distance(query)
        # query_key_distance=query_key_distance.permut

        # Calculate attention scores using negative exponentiation
        # attn_scores = torch.exp(-query_key_distance).mean(dim=-1)
        z_values = self.get_z_values(query_key_distance)
        
        # Apply softmax to get normalized attention weights
        attn_weights = self.softmax(z_values)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to fixed values and sum them
        # attn_output = torch.matmul(attn_weights, value)
        if self.high_dim:
            value= value.reshape(batch_size, num_heads, seq_len_q, self.num_rules, head_dim)
        else:
            value= value.permute(0,2,1,4,3)
        # print("attn_weights", attn_weights.shape)
        # print("value", value.shape)
        Fnn_output = (attn_weights.unsqueeze(-1) *value)
        # Average the sum of weighted values
        output = Fnn_output.sum(dim=-2)

        # Apply final linear projection
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, self.num_heads * self.head_dim)
        output = self.out_proj(output)
        return output




@ARCHS.register_module()
class FuzzyTransformer_arch(BaseArch):
    def __init__(self, pretrained = True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),  
                img_size = (60, 1000), # [H, W]
                num_heads = 8,
                depth = 2,
                num_rules=5,
                cl_embed_dim = 128,
                dropout = 0.1,
                model_ckpt = None,
                fixed = True,
                encoder_type = 'Transformer',
            ):
        super().__init__()
        self.__dict__.update(locals())

        # encoder specifics
        self.encoder = FuzzyTransformer(
                d_model=img_size[1], 
                nhead=num_heads,
                num_encoder_layers=depth,
                num_decoder_layers=depth,
                num_rules=num_rules,
                seq_len=img_size[0],
                dim_feedforward=cl_embed_dim,
                dropout=dropout,
            )
        self.norm_encoder = norm_layer(img_size[1])

        self.cls = nn.Sequential(
                    nn.Linear(2*img_size[1], cl_embed_dim),
                    nn.ReLU(inplace=False),
                    nn.Linear(cl_embed_dim, 2),
                )

        self.cls_loss = nn.CrossEntropyLoss()
        # --------------------------------------------------------------------------
        
        self.initialize_weights()
        self.setup()
    def check_parameters(self):
        input_data={'seq':torch.rand(1, 2, self.img_size[0], self.img_size[1]).to(self.device)}
        loss, output = self.forward(input_data,torch.ones(1).long().to(self.device))

        # Create a pseudo target tensor with the same shape as the output
        # For demonstration, let's use random values
        # In a real scenario, this should be your actual target data
        pseudo_target = torch.rand_like(output)

        # Define a loss function, for example, Mean Squared Error
        # loss_function = nn.MSELoss()

        # # Calculate the loss
        # loss = loss_function(output, pseudo_target)

        # # Perform the backward pass
        loss.backward()

        # Check for unused parameters
        for name, param in self.named_parameters():
            if param.grad is None:
                print(f"Parameter '{name}' did not receive gradients.")


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
        self.check_parameters()

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
        x = self.encoder(x, x)
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
