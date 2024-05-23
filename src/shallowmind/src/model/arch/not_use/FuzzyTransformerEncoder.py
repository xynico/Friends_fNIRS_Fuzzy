
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
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
    def forward(self, pair):
        sub1 = pair[:,0]
        sub2 = pair[:, 1]
        batch_size, seq_len, embed_size = sub1.shape
        sub1 = sub1.reshape(-1, embed_size)
        sub2 = sub2.reshape(-1, embed_size)
        keys = self.keys(sub1)
        keys = keys.reshape(batch_size, seq_len,embed_size)
        queries = self.queries(sub2)
        queries = queries.reshape(batch_size, seq_len,embed_size)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other token
        attention = torch.einsum("nqd,nkd->nqk", [queries, keys])

        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=-1)
        return attention

class Weighted_Consequent(nn.Module):
    def __init__(self, input_shape):
        super(Weighted_Consequent, self).__init__()
        sub_dim=input_shape[1]
        channel_dim=input_shape[2]
        time_dim=input_shape[3]
        self.time_linear = nn.Linear(time_dim, 1)   # First linear layer
        self.channel_linear = nn.Linear(channel_dim, 1)     # Second linear layer
        self.sub_linear = nn.Linear(sub_dim, 1)           # Output layer

    def forward(self, x):
        x = F.relu(self.time_linear(x))  # Applying linear transformation and ReLU
        x = x.squeeze(-1)
        x = F.relu(self.channel_linear(x))  # Applying linear transformation and ReLU
        x = x.squeeze(-1)
        x = self.sub_linear(x) # Applying linear transformation and ReLU
        return torch.sigmoid(x)  # Sigmoid for binary classification

class LSTM_consequent(nn.Module):
    def __init__(self, input_shape,dropout=0.0):
        super(LSTM_consequent, self).__init__()
        self.encoder = nn.LSTM(
            input_size=input_shape[-1],
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        # self.norm_decoder=norm_layer(1)
        cl_embed_dim=64
        self.norm_encoder = norm_layer(cl_embed_dim * 2)
        self.cls = nn.Sequential(
            nn.Linear(4 * cl_embed_dim, cl_embed_dim),
            nn.ReLU(inplace=False),
            nn.Linear(cl_embed_dim, 2)
            # nn.Sigmoid()
        )
    def forward(self,x):
        # x=x.view(-1,x.shape[-2],x.shape[-1]) # 128, 2, 40, 33 --> 256, 40, 33
        x=torch.cat([x[:,0],x[:,1]],dim=0)
        latent,_ = self.encoder(x)
        latent = latent[:,-1,:] # 256, 128 -->128, 256

        latent=self.norm_encoder(latent)
        sub1_latent=latent[:int(latent.shape[0]/2)]
        sub2_latent = latent[int(latent.shape[0] / 2):]
        latent = torch.cat([sub1_latent,sub2_latent],dim=-1)
        output=self.cls(latent)
        return output

class CNN_consequent(nn.Module):
    def __init__(self, input_shape):
        super(CNN_consequent, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Calculate the size of the flattened features after conv and pooling layers
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool
        )
        self.convs(torch.randn(input_shape))

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

    def convs(self, x):
        # Dummy forward pass to get the size of the flattened features
        if self._to_linear is None:
            self._to_linear = torch.prod(torch.tensor(x.size()[1:])).item()
        return x
    
class Fuzzy_Attention_Encoder(nn.Module):
    def __init__(self, input_shape, n_rules=None, projection=True):
        super(Fuzzy_Attention_Encoder, self).__init__()
        self.atten=Fuzzy_Attention(input_shape,n_rules,projection)
        B, S, L, D = input_shape
        if n_rules is None:
            self.n_rules=L
        else:
            self.n_rules=n_rules
        self.feed_forward = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(S*D*self.n_rules , int(S*D*self.n_rules/2)),
            nn.ReLU(),
            nn.Linear(int(S*D*self.n_rules/2), 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.norm = nn.LayerNorm(normalized_shape=[S, self.n_rules, D])
    def forward(self,x):
        attn_output = self.atten(x)
        x = self.norm(x + attn_output)
        ff_output = self.feed_forward(x)
        return ff_output
class Fuzzy_Attention(nn.Module):
    def __init__(self,input_shape, n_rules=None, projection=True):
        super(Fuzzy_Attention, self).__init__()
        B, S, L, D = input_shape
        if n_rules is None:
            self.n_rules = L
        else:
            self.n_rules = n_rules
        self.to_c = nn.Linear(D, self.n_rules*D)
        self.projection=projection
        self.centers = nn.Parameter(torch.rand(self.n_rules ,D)) # n_rules=L
        self.widths = nn.Parameter(torch.rand(self.n_rules , D)) # n_rules=L
        if self.projection:
            self.to_q=nn.Linear(D,D)
    def get_dist(self,x):
        x = x.unsqueeze(2)  # Shape becomes (B, C, 1, T)
        cetners = self.centers.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, R, T)
        # Calculate L1 distance
        l1_distance = torch.abs(x - cetners)  # Broadcasting occurs here
        # print(l1_distance.shape)  # Should be (B, C, R, T)
        return l1_distance
    def get_z_values(self, x):
        # print("fuzzy input",x.shape)
        aligned_x = x
        dist=self.get_dist(aligned_x)
        # dist is already sum/prod, not on all dimensions
        aligned_w=self.widths.unsqueeze(0).unsqueeze(0)
        prot = torch.div(dist,aligned_w)
        # HTSK dived D
        root=-torch.square(prot)*0.5
        z_values =root.mean(-1)
        # print("gmv",x.shape,dist.shape,prot.shape,root.shape,membership_values.shape)
        return z_values
    def forward(self,x):
        b, s, c, t = x.shape
        if self.projection:
            x_arranged = rearrange(x, 'b s c t -> (b s c) t')
            q=self.to_q(x_arranged)
            q_rearranged = rearrange(q, '(b s c) t -> (b s) c t', b=b, s=s, c=c)
        else:
            q_rearranged=rearrange(x, 'b s c t -> (b s) c t')
        z_outs=self.get_z_values(q_rearranged)
        Fss = F.softmax(z_outs, dim=-1) # (b s), c, r
        # v_rearranged=rearrange(x, 'b s c t -> (b s t) c')
        v_rearranged=rearrange(x, 'b s c t -> (b s c) t')
        conq=self.to_c(v_rearranged) # (b s c) (t r)
        conq_rearranged=rearrange(conq, '(b s c) (t r)-> (b s c) t r',b=b, s=s, t=t, r=self.n_rules)
        Fss=rearrange(Fss, '(b s) c r -> (b s c) r',b=b, s=s, c=c)
        output=Fss.unsqueeze(-1) @ conq_rearranged # (b s c) t 1
        output=rearrange(output, '(b s c) t x-> b s c t x',b=b, s=s, c=c, t=t).squeeze(-1)
        return output



class SOFIN(nn.Module):
    def __init__(self, n_rules, input_shape,  output_dim=1, distance_metric="L1", order=1, rule_type="attention"):
        super(SOFIN, self).__init__()
        self.input_dim=input_shape[-3]*input_shape[-2]*input_shape[-1]
        self.output_dim = output_dim
        self.distance_metric = distance_metric
        self.order=order
        self._commitment_cost = 0.25
        self.rule_dropout=nn.Dropout(0.0)
        self.n_rules=n_rules
        # at begining we don't have data, just random create one rule
        self.softmax = nn.Softmax(dim=-1)
        self.rule_type = rule_type
        if self.rule_type == "attention":
            self.rules = nn.ModuleList([HTSK_Attention(input_shape,output_dim,distance_metric) for i in range(n_rules)])
            self.atten = LSTM_consequent(input_shape=input_shape)
        else:
            self.rules = nn.ModuleList([HTSK_Fuzzy_rule(self.input_dim,output_dim,distance_metric,order) for i in range(n_rules)])
    def get_all_fs(self,x):
        z_outs=[]
        with torch.no_grad():
            for rule in self.rules:
                z = rule.get_FS(x)
                z_outs.append(z)
            z_outs = torch.stack(z_outs,dim=-1)
            all_fs=F.softmax(z_outs,dim=-1)
        return all_fs
    def update_rules(self, x):
        # print("x",x.shape)
        # print("rule number before drop", len(self.rules))
        self.drop_rules(x)
        # print("rule number after drop",len(self.rules))
        self.add_rules(x)
        self.n_rules = len(self.rules)
        # print("rule number after add",len(self.rules))
    def drop_rules(self,x,threshold=0.8):
        while len(self.rules)>=2:
            all_fs=self.get_all_fs(x)
            mask = torch.all(all_fs > threshold, dim=0)
            # Get the indices where the condition is not met
            # print("mask",mask.shape,mask)
            mask=[not elem for elem in mask]
            if any(mask):
                if False in mask:
                    self.rules = nn.ModuleList([module for module, include in zip(self.rules, mask) if include])
                else:
                    break
            else:
                break
    def add_rules(self, x):
        ood_samples=x
        while len(ood_samples)>2 and len(self.rules) <=20:
            threshold=1.1/len(self.rules)
            if len(self.rules)<=1:
                self.rules.append(HTSK_Fuzzy_rule(self.input_dim,self.output_dim,self.distance_metric,self.order))
                ood_samples = ood_samples[1:]
                pass
            all_fs=self.get_all_fs(ood_samples)
            max_probabilities, predicted_classes = torch.max(all_fs, dim=1)
            confused_mask = max_probabilities < threshold
            # print("max_probabilities",max_probabilities)
            ood_indices = torch.nonzero(confused_mask).squeeze()
            if ood_indices.dim() == 0:
                break
            ood_samples=ood_samples[ood_indices]
            if len(ood_samples)>2:
                self.rules.append(HTSK_Fuzzy_rule(self.input_dim,self.output_dim,self.distance_metric,self.order, center=ood_samples[0]).to(x.device))
                ood_samples=ood_samples[1:]
    def forward(self, x):
        x=x.float()
        if self.rule_type=="attention":
            attention=self.atten(x).view(-1,self.input_dim**2)
            cq_outs = []
            z_outs = []
            for rule in self.rules:
                z, cq = rule(x,attention)
                cq_outs.append(cq)
                z_outs.append(z)
        else:
            if len(x.shape)>2:
                x=x.view(x.shape[0],-1)
            cq_outs = []
            z_outs = []
            for rule in self.rules:
                z, cq = rule(x)
                cq_outs.append(cq)
                z_outs.append(z)
        z_outs = torch.stack(z_outs,dim=-1)
        fs_outs=F.softmax(z_outs,dim=-1)
        fs_outs=self.rule_dropout(fs_outs)
        cq_outs = torch.stack(cq_outs,dim=-2)
        FNN_outs=cq_outs * fs_outs.unsqueeze(-1)
        FNN_outs = FNN_outs.sum(-2)
        return FNN_outs

class HTSK_Fuzzy_rule(nn.Module):
    """ one rule"""
    def __init__(self,input_dim,output_dim=1,distance_metric="L1", order=0, center=None,cq_width=64):
        super().__init__()
        if center is None:
            self.center=nn.Parameter(torch.rand(input_dim))
        else:
            self.center = nn.Parameter(center)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.widths = nn.Parameter(torch.ones(input_dim)*0.25, requires_grad=True)
        self.distance_metric=distance_metric
        self.order=order
        if self.order==0:
            self.consequent = nn.Parameter(torch.randn(output_dim))
        elif self.order==1:
            self.consequent = nn.Linear(self.input_dim, output_dim, bias=True)
            self.consequent = nn.Sequential(nn.Linear(self.input_dim,output_dim,bias=True))
        elif self.order>=2:
            layers = [nn.Linear(self.input_dim, cq_width)]
            for l in range (self.order-1):
                layers.append(nn.Linear(cq_width, cq_width))
            layers.append(nn.Linear(cq_width, output_dim))
            # layers.append(nn.Sigmoid)
            self.consequent = nn.Sequential(*layers)
    def get_dist(self,x):
        if len(x.shape)==1:
            x=x.unsqueeze(0)
        # first normalize the input
        x=F.sigmoid(x)
        if self.distance_metric=="L1":
            torch_dist = torch.abs(x - self.center)
        elif self.distance_metric=="L2":
            torch_dist = torch.square(x - self.center)
        return torch_dist
    def get_z_values(self, x):
        # print("fuzzy input",x.shape)
        aligned_x = x
        dist=self.get_dist(aligned_x)
        # dist is already sum/prod, not on all dimensions
        aligned_w=self.widths
        prot = torch.div(dist,aligned_w)
        # HTSK dived D
        root=-torch.square(prot)*0.5
        z_values =root
        # print("gmv",x.shape,dist.shape,prot.shape,root.shape,membership_values.shape)
        return z_values
    def get_FS(self, x):
        # mvs=self.get_Membership_values(x)
        #
        # fs=mvs.prod(-1)
        # HTSK dived D
        mvs = self.get_z_values(x)
        fs = mvs.mean(-1)
        # print("gfs", x.shape, mvs.shape)
        return fs
    def forward(self,x):
        fs=self.get_FS(x)
        # conq = self.consequent
        # conq=torch.Tensor(conq).to('cuda')
        # out=fs @ conq
        if self.order==0:
            return fs, self.consequent
        else:
            return fs, self.consequent(x)

class HTSK_Attention(nn.Module):
    """ one rule"""
    def __init__(self,input_shape,output_dim=1,distance_metric="L1", center=None,cq_width=64):
        super().__init__()
        if center is None:
            self.center=nn.Parameter(torch.rand(input_shape[-2]**2))
        else:
            self.center = nn.Parameter(center)
        self.output_dim = output_dim
        self.input_shape = input_shape
        self.widths = nn.Parameter(torch.rand(input_shape[-2]**2)*0.25, requires_grad=True)
        self.distance_metric = distance_metric

        self.consequent = CNN_consequent(input_shape)
    def get_dist(self,x):
        if len(x.shape)==1:
            x=x.unsqueeze(0)
        # first normalize the input
        x=F.sigmoid(x)
        if self.distance_metric=="L1":
            torch_dist = torch.abs(x - self.center)
        elif self.distance_metric=="L2":
            torch_dist = torch.square(x - self.center)
        return torch_dist
    def get_z_values(self, x):
        # print("fuzzy input",x.shape)
        aligned_x = x
        dist=self.get_dist(aligned_x)
        # dist is already sum/prod, not on all dimensions
        aligned_w=self.widths
        prot = torch.div(dist,aligned_w)
        # HTSK dived D
        root=-torch.square(prot)*0.5
        z_values =root
        # print("gmv",x.shape,dist.shape,prot.shape,root.shape,membership_values.shape)
        return z_values
    def get_FS(self, x):
        # mvs=self.get_Membership_values(x)
        #
        # fs=mvs.prod(-1)
        # HTSK dived D
        mvs = self.get_z_values(x)
        fs = mvs.mean(-1)
        # print("gfs", x.shape, mvs.shape)
        return fs
    def forward(self, x,attention):
        fs = self.get_FS(attention)
        # conq = self.consequent
        # conq=torch.Tensor(conq).to('cuda')
        # out=fs @ conq
        return fs, self.consequent(x)

@ARCHS.register_module()
class FuzzyTransformer_arch(Fuzzy_Attention_Encoder):

    def __init__(self, input_shape, n_rules=None, projection=True, pretrained=False):
        super(FuzzyTransformer_arch, self).__init__(input_shape, n_rules, projection)
        self.criterion = nn.CrossEntropyLoss()

    def forward_train(self, x, label):
        pred = self.forward(x['seq'])
        loss = self.criterion(pred, label)
        return {'loss': loss}
    
    def forward_test(self, x, label=None):
        pred = self.forward(x['seq'])
        
        loss = self.criterion(pred, label)
        return {'loss': loss, 'output': pred.argmax(dim=1), 'meta_data': None, 'label': label}
    
