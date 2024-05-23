
import torch
import torch.nn.functional as F
import torch.nn as nn
from .base import BaseArch
from einops.layers.torch import Rearrange
import sys
from functools import partial
from ..builder import ARCHS, build_backbone, build_head, build_arch
# from .EEGidentifierCL import EEGTransClassifer
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
        # --------------------------------------------------------------------------
        # encoder specifics
        if encoder_type == 'Transformer':
            # self.encoder = Transformer(
            #     d_model=img_size[1], 
            #     nhead=num_heads,
            #     num_encoder_layers=depth,
            #     num_decoder_layers=depth,
            #     dim_feedforward=cl_embed_dim,
            #     dropout=dropout,
            # )
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
        # self.cls_loss = nn.CrossEntropyLoss()
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
        # imgs = data

        imgs = data['seq']  # [B, 2, CH, TP]
        imgs = torch.cat([imgs[:, 0, :, :], imgs[:, 1, :, :]], dim=0)  # [2B, CH, TP]
        return imgs

    def forward(self, data):
        imgs = self.gen_data(data) # [2B, CH, TP]

        # Encoder
        latent = self.forward_encoder(imgs) # [2B, D_CL]
        latent = torch.cat([latent[:latent.shape[0]//2, :], latent[latent.shape[0]//2:, :]], dim=1)
        # Decoder
        # latent = self.forward_decoder(latent) # [2B, D_CL]

        latent = self.cls(latent) # [B, 2]
        pred = latent.squeeze(1)
        return pred
    
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
            nn.Linear(cl_embed_dim, 2),
            # nn.Sigmoid()
        )
    def forward(self,x):
        x=x.view(-1,x.shape[-2],x.shape[-1]) # 128, 2, 40, 33 --> 256, 40, 33
        latent,_=self.encoder(x)
        latent=latent[:,-1,:] # 256, 128 -->128, 256

        latent=self.norm_encoder(latent)
        sub1_latent=latent[:int(latent.shape[0]/2)]
        sub2_latent = latent[int(latent.shape[0] / 2):]
        latent = torch.cat([sub1_latent,sub2_latent],dim=-1)
        output=self.cls(latent)
        return output

class SOFIN(BaseArch):
    def __init__(self, n_rules, input_shape,  output_dim=1, distance_metric="L1", order=1, rule_type="attention"):
        super(SOFIN, self).__init__()
        self.input_dim=input_shape[-2]
        self.output_dim = output_dim
        self.distance_metric = distance_metric
        self.order=order
        self._commitment_cost = 0.25
        self.rule_dropout=nn.Dropout(0.0)
        self.n_rules=n_rules
        # at begining we don't have data, just random create one rule
        self.rule_type = rule_type
        self.rules = HTSK_Attention(input_shape,output_dim,distance_metric)
        # model_para = dict(
        #     img_size = (40, 33), # [H, W]
        #     num_heads = 11,
        #     depth = 2,
        #     cl_embed_dim = 64,
        #     dropout = 0.3,
        #     encoder_type = 'LSTM',
        #     model_ckpt = None,
        #     fixed = False,
        # )
        # # from .EEGidentifierCL import EEGTransClassifer
        # self.rules = EEGTransClassifer(**model_para)

    def forward(self, x):
        cq = self.rules(x)
        return cq
    

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
            self.consequent = nn.Parameter(torch.rand(output_dim))
        elif self.order==1:
            self.consequent = nn.Linear(self.input_dim, output_dim, bias=True)
            self.consequent = nn.Sequential(nn.Linear(self.input_dim,output_dim,bias=True),nn.Sigmoid())
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
            self.consequent = nn.Parameter(torch.rand(output_dim))
        elif self.order==1:
            self.consequent = nn.Linear(self.input_dim, output_dim, bias=True)
            self.consequent = nn.Sequential(nn.Linear(self.input_dim,output_dim,bias=True),nn.Sigmoid())
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

        # self.consequent = CNN_consequent(input_shape)
        # self.consequent = LSTM_consequent(input_shape)
        model_para = dict(
            img_size = (40, 33), # [H, W]
            num_heads = 11,
            depth = 2,
            cl_embed_dim = 64,
            dropout = 0.3,
            encoder_type = 'LSTM',
            model_ckpt = None,
            fixed = False,
        )
        # from .EEGidentifierCL import EEGTransClassifer
        self.consequent = EEGTransClassifer(**model_para)
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
    def forward(self, x):
        return self.consequent(x)
    

@ARCHS.register_module()
class SOFIN_arch(SOFIN):

    def __init__(self, n_rules, input_shape,  output_dim=1, distance_metric="L1", order=1, rule_type="attention", pretrained=False):
        super(SOFIN_arch, self).__init__(n_rules, input_shape,  output_dim, distance_metric, order, rule_type)
        self.criterion = nn.CrossEntropyLoss()

    def forward_train(self, x, label):
        pred = self.forward(x)
        loss = self.criterion(pred, label)
        return {'loss': loss}
    
    def forward_test(self, x, label=None):
        pred = self.forward(x)
        loss = self.criterion(pred, label)
        return {'loss': loss, 'output': pred.argmax(dim=1), 'meta_data': None, 'label': label}
    
