
import torch
import torch.nn.functional as F
import torch.nn as nn
from .base import BaseArch
from einops.layers.torch import Rearrange
import sys
from functools import partial
from ..builder import ARCHS, build_backbone, build_head, build_arch
from einops import rearrange
from functools import partial
from torch.nn import Transformer
       
class Fuzzy_Attention(nn.Module):
    def __init__(self, input_shape, n_rules=None, projection=True, dropout=0.0):
        super(Fuzzy_Attention, self).__init__()
        B, L, D = input_shape  # Adjusted for BLD input shape
        
        self.n_rules = L if n_rules is None else n_rules
        self.to_c = nn.Linear(L, self.n_rules)
        self.projection = projection
        self.centers = nn.Parameter(torch.rand(self.n_rules, D))
        self.widths = nn.Parameter(torch.rand(self.n_rules, D))
        self.dropout = dropout
        
        if self.projection:
            self.to_q = nn.Linear(D, D)

    def get_dist(self, x):
        x = x.unsqueeze(2)  # Shape becomes (B, L, 1, D)
        centers = self.centers.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, R, D)
        l1_distance = torch.abs(x - centers)  # Broadcasting occurs here
        return l1_distance

    def get_z_values(self, x):
        dist = self.get_dist(x)
        aligned_w = self.widths.unsqueeze(0).unsqueeze(0)
        prot = torch.div(dist, aligned_w)
        root = -torch.square(prot) * 0.5
        z_values = root.mean(-1)
        return z_values

    def forward(self, query, key = None, value=None):
        b, l, d = query.shape
        
        # print('x.shape', x.shape)
        if self.projection:
            q = rearrange(self.to_q(rearrange(query, 'b l d -> (b l) d')), '(b l) d -> b l d', b=b, l=l)
        else:
            q = query

        z_outs = self.get_z_values(q)
        Fss = F.softmax(z_outs, dim=-1)  # Shape (B, L, R)
        Fss = F.dropout(Fss, p=self.dropout, training=self.training)


        conq = self.to_c(rearrange(query, 'b r d -> (b d) r')) if value is None else self.to_c(rearrange(value, 'b r d -> (b d) r'))
        conq = rearrange(conq, '(b d) r -> b r d', b=b, r=self.n_rules)  # (B, R, D)

        #(B, L, D) @ (B, D, R) = (B, L, R)
        output = Fss @ conq  # (B, L, R)
        
        return output
    
class FuzzyAttentionEncoderLayer(nn.Module):
    def __init__(self, input_shape, n_rules=None, projection=True, dropout=0.0, dim_feedforward=2048):
        super(FuzzyAttentionEncoderLayer, self).__init__()
        self.atten = Fuzzy_Attention(input_shape, n_rules, projection, dropout)
        B, L, D = input_shape  # Updated input_shape
        self.n_rules = L if n_rules is None else n_rules
        self.dropout = dropout

        self.feed_forward = nn.ModuleList([
            nn.Linear(D, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, D),
            nn.Dropout(dropout),
        ])
        self.norm1 = nn.LayerNorm(normalized_shape=[L, D])  # Updated norm layer
        self.norm2 = nn.LayerNorm(normalized_shape=[L, D])  # Updated norm layer
    
    def _sa_block(self, x):
        attn_output = self.atten(x)
        
        return F.dropout(attn_output, p=self.dropout, training=self.training)
    
    def _ff_block(self, x):
        for layer in self.feed_forward:
            x = layer(x)
        return x
    def forward(self, x):
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(x))
        return x

class FuzzyAttentionDecoderLayer(nn.Module):
    def __init__(self, input_shape, n_rules=None, projection=True, dropout = 0.0, dim_feedforward=2048):
        super(FuzzyAttentionDecoderLayer, self).__init__()
        self.self_attn = Fuzzy_Attention(input_shape, n_rules, projection)
        self.cross_attn = Fuzzy_Attention(input_shape, n_rules, projection)
        B, L, D = input_shape  # Adjusted for BLD input shape
        self.n_rules = L if n_rules is None else n_rules
        self.dropout = dropout


        self.norm1 = nn.LayerNorm(normalized_shape=[L, D])  
        self.norm2 = nn.LayerNorm(normalized_shape=[L, D])  
        self.norm3 = nn.LayerNorm(normalized_shape=[L, D])


        self.feed_forward = nn.ModuleList([
            nn.Linear(D, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, D),
            nn.Dropout(dropout),
        ])
    
    def _sa_block(self, x):
        attn_output = self.self_attn(x)
        return F.dropout(attn_output, p=self.dropout, training=self.training)
    
    def _mha_block(self, x, memory):
        attn_output = self.cross_attn(x, memory, memory)
        return F.dropout(attn_output, p=self.dropout, training=self.training)
    
    def _ff_block(self, x):
        for layer in self.feed_forward:
            x = layer(x)
        return x

    def forward(self, x, memory):
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._mha_block(x, memory))
        x = self.norm3(x + self._ff_block(x))
        return x
    

class Fuzzy_Attention_Encoder(nn.Module):
    def __init__(self, input_shape, n_rules=None, projection=True, n_layers=1, dropout=0.0, dim_feedforward=2048):
        super(Fuzzy_Attention_Encoder, self).__init__()
        self.dropout = dropout

        # Create a series of FuzzyAttentionLayer instances
        self.layers = nn.ModuleList([
            FuzzyAttentionEncoderLayer(input_shape, n_rules, projection, dropout, dim_feedforward) for _ in range(n_layers)
        ])

    def forward(self, x):
        # Sequentially pass the input through all FuzzyAttentionLayer instances
        for layer in self.layers:
            x = layer(x)
        return x
    
class Fuzzy_Attention_Decoder(nn.Module):
    def __init__(self, input_shape, n_rules=None, projection=True, n_layers=1, dropout = 0.0, dim_feedforward=2048):
        super(Fuzzy_Attention_Decoder, self).__init__()
        self.dropout = dropout

        self.layers = nn.ModuleList([
            FuzzyAttentionDecoderLayer(input_shape, n_rules, projection, dropout, dim_feedforward) for _ in range(n_layers)
        ])

    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

class FuzzyTransformer(nn.Module):
    def __init__(self, input_shape, n_rules=None, projection=True, num_encoder_layers=1, num_decoder_layers=1, cl_embed_dim=64, dropout=0.0, dim_feedforward=2048):
        super(FuzzyTransformer, self).__init__()
        B, L, D = input_shape
        # Encoder with multiple layers
        self.encoder = Fuzzy_Attention_Encoder(input_shape, n_rules, projection, n_layers=num_encoder_layers, dropout=dropout, dim_feedforward=cl_embed_dim)

        # Decoder with multiple layers
        self.decoder = Fuzzy_Attention_Decoder(input_shape, n_rules, projection, n_layers=num_decoder_layers, dropout=dropout, dim_feedforward=cl_embed_dim)

    def forward(self, x):
        # Encoder pass
        encoder_output = self.encoder(x)

        # Decoder pass
        decoder_output = self.decoder(x, encoder_output)

        return decoder_output
