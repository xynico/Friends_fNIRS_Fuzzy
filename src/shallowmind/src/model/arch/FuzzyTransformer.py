import torch
import torch.nn as nn
from .base import BaseArch
from einops.layers.torch import Rearrange
from einops import rearrange
import sys
from functools import partial
import numpy as np
from ..builder import ARCHS, build_backbone, build_head, build_arch
from torch.nn.functional import normalize
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
# from .EEGidentifierCL import EEGidentifierCLV0
# from .FuzzyTransformerEncoderDecoder import FuzzyTransformer
from torch.nn import Linear, Dropout, Softmax
from .soft_dtw import SoftDTW
import warnings
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.modules.transformer import MultiheadAttention
from tslearn.metrics import SoftDTWLossPyTorch

class NormalizeLayer(nn.Module):
    def __init__(self, min_val, max_val):
        super(NormalizeLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        # Normalize to [0, 1]
        x_normalized = (x - self.min_val) / (self.max_val - self.min_val)
        # Scale to [-1, 1]
        x_scaled = x_normalized * 2 - 1
        return x_scaled
class FuzzyTransformer_CB(Transformer):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, num_rules, seq_len, dim_feedforward=2048, dropout=0.1, activation="relu",use_projection=True,fuzzy_type='DTW',norm=False,HTSK=True):
        super(FuzzyTransformer_CB, self).__init__(
            d_model, nhead, num_encoder_layers, num_decoder_layers, 
            dim_feedforward, dropout, activation, batch_first=True,
        )
        self.HTSK=HTSK
        self.batch_first = True
        self.use_projection=use_projection
        self.Fuzzy_type=fuzzy_type
        # print("double_FS",double_FS)
        # Replace the standard MultiheadAttention with FuzzyMultiheadAttention in each layer
        self._replace_attention_layers(self.encoder, d_model, nhead, num_rules, seq_len,HTSK=HTSK)
        self._replace_attention_layers(self.decoder, d_model, nhead, num_rules, seq_len,HTSK=HTSK)
    def _replace_attention_layers(self, transformer_layer, d_model, nhead, num_rules, seq_len,HTSK=True):
        # for layer in transformer_layer.layers:o
        for name, child in transformer_layer.named_children():
            # print("attention class", type(layer.self_attn))
            if isinstance(child, MultiheadAttention):
                # Ensure the arguments match your FuzzyMultiheadAttention's __init__ signature
                if self.Fuzzy_type=='double_FS':
                    setattr(transformer_layer, name, Fuzzy_MultiHeadAttention_QK(d_model,nhead,seq_len,num_rules,HTSK))
                elif self.Fuzzy_type=='parallel':
                    setattr(transformer_layer, name, Fuzzy_MultiHeadAttention_PA(d_model,nhead,num_rules,seq_len,HTSK))
                elif self.Fuzzy_type=='DTW':
                    setattr(transformer_layer, name, Fuzzy_MultiHeadAttention_DTW(d_model,nhead,seq_len, num_rules,HTSK)) #embed_size, heads,seq_len,rules
                else:
                    setattr(transformer_layer, name, FuzzyMultiheadAttention(d_model, nhead, num_rules, seq_len,use_projection=self.use_projection,HTSK=HTSK))
               
                    
                # layer.self_attn = FuzzyMultiheadAttention(d_model, nhead, num_rules, seq_len)
            else:
            # Recursively apply this function to submodules
                self._replace_attention_layers(child, d_model, nhead, num_rules, seq_len,HTSK=HTSK)

            # num_params = sum(p.numel() for p in layer.self_attn.parameters() if p.requires_grad)
            # print(f"fuzzy parameters: {num_params}")
            # if isinstance(layer, TransformerEncoderLayer) or isinstance(layer, TransformerDecoderLayer):
            #     # print("layer",layer)
            #     num_params = sum(p.numel() for p in layer.multihead_attn.parameters() if p.requires_grad)
            #     print(f"original parameters: {num_params}")
            #     layer.multihead_attn = FuzzyMultiheadAttention(d_model, nhead, num_rules, seq_len)
            #     num_params = sum(p.numel() for p in layer.multihead_attn.parameters() if p.requires_grad)
            #     print(f"fuzzy parameters: {num_params}")
                


class FuzzyTransformer(Transformer):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, num_rules, seq_len, dim_feedforward=2048, dropout=0.1, activation="relu",use_projection=True,only_cross=False,norm=False,HTSK=True):
        super(FuzzyTransformer, self).__init__(
            d_model, nhead, num_encoder_layers, num_decoder_layers, 
            dim_feedforward, dropout, activation, batch_first=True
        )
        self.batch_first = True
        self.use_projection=use_projection
        # Replace the standard MultiheadAttention with FuzzyMultiheadAttention in each layer
        if only_cross:
            self._replace_attention_layers(self.decoder, d_model, nhead, num_rules, seq_len,only_cross=only_cross,norm=norm,HTSK=HTSK)
        else:
            self._replace_attention_layers(self.encoder, d_model, nhead, num_rules, seq_len,only_cross=only_cross,norm=norm,HTSK=HTSK)
            self._replace_attention_layers(self.decoder, d_model, nhead, num_rules, seq_len,only_cross=only_cross,norm=norm,HTSK=HTSK)
        # print("only_cross",only_cross) 
    def _replace_attention_layers(self, transformer_layer, d_model, nhead, num_rules, seq_len,only_cross=True,norm=False,HTSK=True):
        for layer in transformer_layer.layers:
            # print("attention class", type(layer.self_attn))
            num_params = sum(p.numel() for p in layer.self_attn.parameters() if p.requires_grad)
            # print(f"original parameters: {num_params}")
            if only_cross:
                layer.multihead_attn = FuzzyMultiheadAttention(d_model, nhead, num_rules, seq_len,use_projection=self.use_projection,norm=norm,HTSK=HTSK)
            else:
                layer.self_attn = FuzzyMultiheadAttention(d_model, nhead, num_rules, seq_len,use_projection=self.use_projection,norm=norm,HTSK=HTSK)

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
class fakedata(object):
    is_cuda = False
    device='cca'
    weight=torch.rand(1,1)
    bias=torch.rand(1,1)
class Fuzzy_MultiHeadAttention_PA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_rules, seq_len, dropout=0.,use_projection=True):
        super().__init__()
        self.batch_first = True
        self._qkv_same_embed_dim = True
        print("num_rules",num_rules)
        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        # print("use_projection",use_projection)
        self.num_heads = num_heads
        self.num_rules = num_rules
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Initialize rules (keys and values) as parameters
        self.rules_keys = nn.Parameter(torch.Tensor(self.num_heads, self.num_rules, self.head_dim))
        self.rules_widths = nn.Parameter(torch.ones(self.num_heads, self.num_rules, self.head_dim))
        # self.fuzzy_value_proj = Linear(self.head_dim, self.head_dim*num_rules)
        # for dot product
        # self.dot_value_proj = Linear(self.head_dim, self.head_dim, bias=False)
        self.key_proj= Linear(self.head_dim, self.head_dim, bias=False)
        self.query_proj = Linear(self.head_dim, self.head_dim, bias=False)
    

        self.value_proj = Linear(seq_len,  self.num_rules+seq_len)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(dropout)
        


        self.in_proj_weight=fakedata()
        self.in_proj_bias=fakedata()


        # Initialize rules keys and values
        nn.init.normal_(self.rules_keys, mean=0.0, std=0.02)
        # nn.init.normal_(self.rules_values, mean=0.0, std=0.02)
    def custom_distance(self,query):
        # if self.high_dim:
        #     key = self.rules_keys.unsqueeze(0).repeat(query.shape[0], 1, 1, 1,1)
        
        key = self.rules_keys.unsqueeze(0).unsqueeze(0)
        key = key.repeat(query.shape[0], query.shape[2],1,1,1).permute(0,2,1,3,4)
        # key=self.rules_keys
        l1_distance = torch.abs(query.unsqueeze(-2).repeat(1,1,1,self.num_rules,1) - key)
        return l1_distance
    def get_z_values(self, query_key_distance):
        # Calculate z values from query-key distance
        # if self.high_dim:  
        #     prot=torch.div(query_key_distance, self.rules_widths)
        #     root=-torch.square(prot)*0.5
        #     z_values =root.mean(-1) # HTSK
        #     return z_values.permute(0, 2, 3, 1)
        # else:
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
            is_causal : bool = False,
            return_query: bool = False,
            ) -> Tuple[Tensor, Optional[Tensor]]:
        # Project query
        batch_size, seq_length, _ = query.size()

        query=query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        dot_key=key.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        # dot_value=value.reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        query = self.query_proj(query)  # [B, seq_len, head, head_dim]
        # for dot attention
        dot_query= query
        dot_key=self.key_proj(dot_key)
        # dot_value=self.dot_value_proj(dot_value)
        energy = torch.einsum("nqhd,nkhd->nhqk", [dot_query, dot_key])
        dot_attention = torch.softmax(energy / (self.scale), dim=3)

        # dot_out = torch.einsum("nhql,nlhd->nqhd", [dot_attention, dot_value]).reshape(
        #     batch_size, seq_length, self.num_heads * self.head_dim
        # )

        # print("dot out", dot_out.shape)

        # query = query.reshape(batch_size, seq_length, self.num_heads, -1) # [B, seq_len, head, head_dim]
        query = query* self.scale
        query = query.transpose(1, 2) # [B, head, seq_len, head_dim]
        # Repeat keys and values for each head
        # key = self.rules_keys.unsqueeze(0).repeat(batch_size, 1, 1, 1,1)

        # value = self.dot_value_proj(value) * self.scale # [B, seq_len, embed_dim]
        # value = value.reshape(batch_size, seq_length, self.num_heads,  -1, self.num_rules) # [B, seq_len, head, head_dim]
        # value = value.transpose(1, 2) # [B, head, seq_len, head_dim]
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
        all_weights=torch.cat([attn_weights.permute(0,2,1,3),dot_attention],dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        # print("attn_weights", attn_weights.shape)
        
        # Apply attention weights to fixed values and sum them
        # attn_output = torch.matmul(attn_weights, value)
        value=value.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0,2,3,1)
        value = self.value_proj(value) 
        # fuzzy_value = self.fuzzy_value_proj(value)
        # fuzzy_value=rearrange(fuzzy_value, "b n h (r d) -> b n h r d",r=self.num_rules)
        # print("value", dot_value.shape,fuzzy_value.shape)
        # value= value.permute(0,2,1,4,3)
        # print("attn_weights", attn_weights.shape)
        out=torch.einsum("bhsr, bhdr->bhsd", [all_weights, value])
        output=rearrange(out, "b h s d -> b s (h d)")
        output = self.out_proj(output)
        if not return_query:
            return output
        else:
            return output, query




class Fuzzy_MultiHeadAttention_QK(nn.Module):
    def __init__(self, embed_size, heads,seq_len,rules):
        super(Fuzzy_MultiHeadAttention_QK, self).__init__()
        self.batch_first = True
        self.seq_len=seq_len
        self._qkv_same_embed_dim = True
        self.embed_size = embed_size
        self.num_heads = heads
        self.num_rules=rules
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.num_rules*self.embed_size, bias=False)
        # self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Parameter(torch.rand(self.num_rules, self.seq_len,self.embed_size))
        # self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)
    
    def sequence_l2_distance(self,x, key):
        """
        Calculate the L2 distance between two tensors. The behavior changes based on the shape of `key`.
        - If `key` has a shape (R, N), the function behaves like `calculate_custom_l2_distance`.
        - If `key` has a shape (R, N, D), the function behaves like `calculate_l2_distance`.
        The output tensor has shape (B, R, D).

        Parameters:
        - x: A tensor with shape (B, N, D).
        - key: A tensor with shape (R, N) or (R, N, D).

        Returns:
        - A tensor with shape (B, R, D) representing the L2 distances.
        """
        B, R, D = x.size(0), key.size(0), x.size(2)

        if key.dim() == 2:  # Shape (R, N)
            # Expand key to match x's last dimension
            key_expanded = key.unsqueeze(-1).expand(-1, -1, D)
        elif key.dim() == 3:  # Shape (R, N, D)
            key_expanded = key
        else:
            raise ValueError("Key tensor must have 2 or 3 dimensions.")

        # Expand both tensors for subtraction
        x_expanded = x.unsqueeze(1).expand(B, R, -1, -1)  # Shape (B, R, N, D)
        key_expanded = key_expanded.unsqueeze(0).expand(B, -1, -1, -1)  # Shape (B, R, N, D)

        # Calculate L2 distance
        l2_distance = torch.sqrt(((x_expanded - key_expanded) ** 2).sum(dim=2))  # Resulting shape (B, R, D)

        return l2_distance
    def calculate_l2_distance_bhn(self, tensor1, tensor2):
        # tensor1 and tensor2 shapes are [B, N, heads*head_dim]
        B, N, _ = tensor1.shape

        # Reshape and transpose for L2 distance calculation
        tensor1 = tensor1.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape [B, heads, N, head_dim]
        tensor2 = tensor2.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate squared L2 distance
        distances = torch.sum((tensor1.unsqueeze(3) - tensor2.unsqueeze(2)) ** 2, dim=-1)
        # print("distances n", distances.shape)
        # # Transpose back to [B, N, N, heads]
        # distances = distances.permute(0, 2, 3, 1)
        
        return distances  # Shape [B, H, N, N]

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True,
                is_causal : bool = False,
                return_query: bool = False,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        # Linear transformations, keep shape [B, N, D]
        B, N, _ = query.shape
        values = self.values(value)
        # keys = self.keys(key)
        # queries = self.queries(query)
        # print("queries", queries.shape)
        # Calculate L2 distance for all heads together
        # distance = self.calculate_l2_distance_bhn(queries, keys)
        distance = self.sequence_l2_distance(query, self.keys)
        # print("distance", distance.shape)  
        # Inverse distance as attention, applying softmax over N dimension for each head
        attention_scores = torch.softmax(-distance, dim=-2)  # Shape [B, R, D], x=[B, N, D]->[B, N, R, D], out=[B, N, D]
        # print("attention_scores", attention_scores.shape)
        # Apply attention scores to values
        values = values.reshape(B, N, self.num_rules, self.embed_size)  # Shape [B, heads, N, head_dim]
        # print("values", values.shape)
        out = torch.einsum("brd,bnrd->bnd", [attention_scores, values])
        # print("out", out.shape)
        # Reshape & linear output
        # out = rearrange(out, "b h n d -> b n (h d)")
        out = self.fc_out(out)
        # print("out", out.shape)
        return out
class Fuzzy_MultiHeadAttention_DTW(nn.Module):
    def __init__(self, embed_size, heads,seq_len,rules):
        super(Fuzzy_MultiHeadAttention_DTW, self).__init__()
        self.batch_first = True
        self.seq_len=seq_len
        self._qkv_same_embed_dim = True
        self.embed_size = embed_size
        self.num_heads = heads
        self.num_rules=rules
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.num_rules*self.embed_size, bias=False)
        # self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Parameter(torch.rand(self.num_rules, self.embed_size))
        self.rules_widths = nn.Parameter(torch.ones(self.num_rules))
        # self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)
        self.dtw=SoftDTWLossPyTorch(gamma=1.0)

    def fuzzy_Z_values(self,query_key_distance):

        # Calculate z values from query-key distance
        # print("query_key_distance", query_key_distance.shape,self.rules_widths)
        prot=torch.div(query_key_distance, self.rules_widths)
        root=-torch.square(prot)*0.5
        # z_values =root.mean(-1)
        return root
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True,
                is_causal : bool = False,
                return_query: bool = False,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        # Linear transformations, keep shape [B, N, D]
        B, N, _ = query.shape
        values = self.values(value)
        # keys = self.keys(key)
        # queries = self.queries(query)
        # print("queries", queries.shape)
        # Calculate L2 distance for all heads together
        # distance = self.calculate_l2_distance_bhn(queries, keys)
        # distance = self.sequence_l2_distance(query, self.keys)
        # print("using DTW")
        keys=self.keys.unsqueeze(0).repeat(B,N,1,1)
        Q = query.unsqueeze(2).repeat(1, 1, self.num_rules, 1)
        Q = rearrange(Q, 'b n r d -> (b n r) d').unsqueeze(-1)
        keys=rearrange(keys, 'b n r d -> (b n r) d').unsqueeze(-1)
        distance = self.dtw(Q, keys)
        distance=rearrange(distance, '(b n r) -> b n r',b=B,n=N)
        # distance=self.batch_dtw_distance(query)
        # print("distance", distance.shape)
        z_values=self.fuzzy_Z_values(distance)
        # print("z_values", z_values.shape)
        # print("distance", distance.shape)  
        # Inverse distance as attention, applying softmax over N dimension for each head
        attention_scores = torch.softmax(z_values, dim=-1)  # Shape B,C,R
        # print("attention_scores", attention_scores.shape)
        # Apply attention scores to values
        values = values.reshape(B, N, self.num_rules, self.embed_size)  # Shape [B, seq, R, head_dim]
        # print("values", values.shape)
        out = torch.einsum("bcr,bcrd->bcd", [attention_scores, values])
        # print("out", out.shape)
        # Reshape & linear output
        # out = rearrange(out, "b h n d -> b n (h d)")
        out = self.fc_out(out)
        # print("out", out.shape)
        return out

class Fuzzy_MultiHeadAttention_CB(nn.Module):
    def __init__(self, embed_size, heads,seq_len,rules):
        super(Fuzzy_MultiHeadAttention_CB, self).__init__()
        self.batch_first = True
        self.seq_len=seq_len
        self._qkv_same_embed_dim = True
        self.embed_size = embed_size
        self.num_heads = heads
        self.num_rules=rules
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.num_rules*self.embed_size, bias=False)
        # self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Parameter(torch.rand(self.num_rules, self.seq_len,self.embed_size))
        # self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)
    
    def sequence_l2_distance(self,x, key):
        """
        Calculate the L2 distance between two tensors. The behavior changes based on the shape of `key`.
        - If `key` has a shape (R, N), the function behaves like `calculate_custom_l2_distance`.
        - If `key` has a shape (R, N, D), the function behaves like `calculate_l2_distance`.
        The output tensor has shape (B, R, D).

        Parameters:
        - x: A tensor with shape (B, N, D).
        - key: A tensor with shape (R, N) or (R, N, D).

        Returns:
        - A tensor with shape (B, R, D) representing the L2 distances.
        """
        B, R, D = x.size(0), key.size(0), x.size(2)

        if key.dim() == 2:  # Shape (R, N)
            # Expand key to match x's last dimension
            key_expanded = key.unsqueeze(-1).expand(-1, -1, D)
        elif key.dim() == 3:  # Shape (R, N, D)
            key_expanded = key
        else:
            raise ValueError("Key tensor must have 2 or 3 dimensions.")

        # Expand both tensors for subtraction
        x_expanded = x.unsqueeze(1).expand(B, R, -1, -1)  # Shape (B, R, N, D)
        key_expanded = key_expanded.unsqueeze(0).expand(B, -1, -1, -1)  # Shape (B, R, N, D)

        # Calculate L2 distance
        l2_distance = torch.sqrt(((x_expanded - key_expanded) ** 2).sum(dim=2))  # Resulting shape (B, R, D)

        return l2_distance
    def calculate_l2_distance_bhn(self, tensor1, tensor2):
        # tensor1 and tensor2 shapes are [B, N, heads*head_dim]
        B, N, _ = tensor1.shape

        # Reshape and transpose for L2 distance calculation
        tensor1 = tensor1.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape [B, heads, N, head_dim]
        tensor2 = tensor2.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate squared L2 distance
        distances = torch.sum((tensor1.unsqueeze(3) - tensor2.unsqueeze(2)) ** 2, dim=-1)
        # print("distances n", distances.shape)
        # # Transpose back to [B, N, N, heads]
        # distances = distances.permute(0, 2, 3, 1)
        
        return distances  # Shape [B, H, N, N]

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True,
                is_causal : bool = False,
                return_query: bool = False,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        # Linear transformations, keep shape [B, N, D]
        B, N, _ = query.shape
        values = self.values(value)
        # keys = self.keys(key)
        # queries = self.queries(query)
        # print("queries", queries.shape)
        # Calculate L2 distance for all heads together
        # distance = self.calculate_l2_distance_bhn(queries, keys)
        distance = self.sequence_l2_distance(query, self.keys)
        # print("distance", distance.shape)  
        # Inverse distance as attention, applying softmax over N dimension for each head
        attention_scores = torch.softmax(-distance, dim=-2)  # Shape [B, R, D], x=[B, N, D]->[B, N, R, D], out=[B, N, D]
        # print("attention_scores", attention_scores.shape)
        # Apply attention scores to values
        values = values.reshape(B, N, self.num_rules, self.embed_size)  # Shape [B, heads, N, head_dim]
        # print("values", values.shape)
        out = torch.einsum("brd,bnrd->bnd", [attention_scores, values])
        # print("out", out.shape)
        # Reshape & linear output
        # out = rearrange(out, "b h n d -> b n (h d)")
        out = self.fc_out(out)
        # print("out", out.shape)
        return out



class FuzzyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_rules, seq_len, dropout=0.,use_projection=True,norm=False, HTSK=True):
        super().__init__()
        self.batch_first = True
        self._qkv_same_embed_dim = True
        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        # print("use_projection",use_projection)
        self.num_heads = num_heads
        self.num_rules = num_rules
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.high_dim = False
        self.HTSK=HTSK
        self.norm=None if not norm else NormalizeLayer(0, 1)
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
        if use_projection:
            self.query_proj = Linear(embed_dim, embed_dim)
        else:
            self.query_proj = nn.Sigmoid()
            self.rules_keys = nn.Parameter(torch.rand(num_rules, num_heads, seq_len, self.head_dim))

        # self.value_proj = Linear(embed_dim, embed_dim*num_rules)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(dropout)
        self.in_proj_weight=fakedata()
        self.in_proj_bias=fakedata()

        # Initialize rules keys and values
        nn.init.normal_(self.rules_keys, mean=0.0, std=0.02)
        # nn.init.normal_(self.rules_values, mean=0.0, std=0.02)

    def custom_distance(self,query):
        if self.norm:
            query=self.norm(query)
        if self.high_dim:
            key = self.rules_keys.unsqueeze(0).repeat(query.shape[0], 1, 1, 1,1)
        else:
            key = self.rules_keys.unsqueeze(0).unsqueeze(0)
            key = key.repeat(query.shape[0], query.shape[2],1,1,1).permute(0,2,1,3,4)
            # key=self.rules_keys
        l1_distance = torch.abs(query.unsqueeze(-2).repeat(1,1,1,self.num_rules,1) - key)
        return l1_distance
    def get_menbership(self, query):
        query=torch.tensor(query).unsqueeze(0)
        batch_size, seq_length, _ = query.size()
        query = self.query_proj(query) * self.scale # [B, seq_len, embed_dim]
        query = query.reshape(batch_size, seq_length, self.num_heads, -1) # [B, seq_len, head, head_dim]
        query = query.transpose(1, 2) # [B, head, seq_len, head_dim]
        # Repeat keys and values for each head
        # key = self.rules_keys.unsqueeze(0).repeat(batch_size, 1, 1, 1,1)
        # _, num_heads, num_rules, seq_len_k, _ = key.shape
        
        # key = key.reshape(batch_size * num_heads, num_rules, seq_len_k, head_dim)
        # print("query", query.shape)
        # print("key", self.rules_keys.shape) # [num_heads, num_rules, head_dim]
        query_key_distance = self.custom_distance(query)
        # print("query_key_distance", query_key_distance.shape, self.rules_widths.shape)
        prot=torch.div(query_key_distance.permute(0,2,1,3,4), self.rules_widths.unsqueeze(0).unsqueeze(0))
        root=-torch.square(prot)*0.5
        # prot=torch.div(query_key_distance, self.rules_widths)
        # root=-torch.square(prot)*0.5
        root=torch.exp(root)[0]
        # print("root", root.shape)

        mv=rearrange(root, "t h r d -> t r (h d)")
        # dist=dist[c] # r, (h d)
        # values, indices = torch.topk(dist, 3, dim=1, largest=False)
        # find the minimum distance
        # prot=torch.div(query_key_distance.permute(0,2,1,3,4), self.rules_widths.unsqueeze(0).unsqueeze(0))
        # root=-torch.square(prot)*0.5
        return mv

    def get_z_values(self, query_key_distance):
        # Calculate z values from query-key distance
        if self.high_dim:  
            prot=torch.div(query_key_distance, self.rules_widths)
            root=-torch.square(prot)*0.5
            if self.HTSK:
                z_values =root.mean(-1) # HTSK
            else:
                z_values =root.sum(-1) # TSK
            return z_values.permute(0, 2, 3, 1)
        else:
            prot=torch.div(query_key_distance.permute(0,2,1,3,4), self.rules_widths.unsqueeze(0).unsqueeze(0))
            root=-torch.square(prot)*0.5
            # print("root", root.shape)
            if self.HTSK:
                z_values =root.mean(-1) # HTSK
            else:
                z_values =root.sum(-1) # TSK
            return z_values
    def forward(self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False,
            return_query: bool = False,
            return_atten: bool = False,
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
        # print("query", query.shape)
        # print("key", self.rules_keys.shape) # [num_heads, num_rules, head_dim]
        query_key_distance = self.custom_distance(query)
        # query_key_distance=query_key_distance.permut
        # Calculate attention scores using negative exponentiation
        # attn_scores = torch.exp(-query_key_distance).mean(dim=-1)
        z_values = self.get_z_values(query_key_distance)
        
        # Apply softmax to get normalized attention weights
        attn_weights = self.softmax(z_values)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        # print("attn_weights", attn_weights.shape)
        
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
        if return_atten:
            return output, attn_weights
        if return_query:
            return output, query
        
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
                attention_type = 'only_self',
                use_projection=True,
                norm=False,
                HTSK=True
            ):
        super().__init__()
        self.__dict__.update(locals())

        # encoder specifics
        if attention_type=="only_cross":
            self.encoder = FuzzyTransformer(
                d_model=img_size[1], 
                nhead=num_heads,
                num_encoder_layers=depth,
                num_decoder_layers=depth,
                num_rules=num_rules,
                seq_len=img_size[0],
                dim_feedforward=cl_embed_dim,
                dropout=dropout,
                use_projection=use_projection,
                only_cross=True,
                norm=norm,
                HTSK=HTSK
            )
        elif attention_type=="only_self":
            self.encoder = FuzzyTransformer(
                d_model=img_size[1], 
                nhead=num_heads,
                num_encoder_layers=depth,
                num_decoder_layers=depth,
                num_rules=num_rules,
                seq_len=img_size[0],
                dim_feedforward=cl_embed_dim,
                dropout=dropout,
                use_projection=use_projection,
                only_cross=False,
                norm=norm,
                HTSK=HTSK
            )
        else:
            self.encoder = FuzzyTransformer_CB(
                d_model=img_size[1], 
                nhead=num_heads,
                num_encoder_layers=depth,
                num_decoder_layers=depth,
                num_rules=num_rules,
                seq_len=img_size[0],
                dim_feedforward=cl_embed_dim,
                dropout=dropout,
                use_projection=use_projection,
                fuzzy_type=attention_type,
                norm=norm,
                HTSK=HTSK
            )
        
        self.norm_encoder = norm_layer(img_size[1])
        self.img_size = img_size
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
        # print("self.encoder", self.encoder)
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
