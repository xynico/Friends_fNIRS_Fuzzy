
import torch
import torch.nn as nn
from .base import BaseArch
from einops.layers.torch import Rearrange
from functools import partial
import pytorch_lightning as pl
from timm.models.vision_transformer import Block
import numpy as np
from util import *
from .maskTrajectoryRecoryCLV2 import maskTrajectoryRecoryCLV2
from ..builder import ARCHS, build_backbone, build_head, build_arch
from torch.nn.functional import normalize
from torch.nn import functional as F

@ARCHS.register_module()
class maskTrajectoryRecoryCLV2S2(pl.LightningModule):
    def __init__(self, step_1_para, step_2_para, **kwargs):
        super().__init__()
        self.step_1_para = step_1_para
        self.step_2_para = step_2_para
        self.save_hyperparameters()
        self.pre_train_model = None

        # In step 2, we will use the cls token output of step 1 as the representation of the trajectory
        # So we need to set the cls token output of step 1 as the input of step 2
        # We will classify the cls token output of step 1 into 2 classes, which represent if the two trajectories are from the same person or not
        # The classification is done by a Transformer decoder
        # The cls token output of step 1 is a B*L+1*D tensor, 
        #   where B is the batch size, 
        #   L is the number of patches, 
        #   D is the dimension of the cls token output(step 1 encoder embedding dimension)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
                Block(
                    dim = step_2_para['cls_decoder_dim'],
                    num_heads = step_2_para['cls_decoder_num_heads'],
                    mlp_ratio = step_2_para['cls_mlp_ratio'],
                    qkv_bias = True,
                    norm_layer=norm_layer,
                )
            for i in range(step_2_para['cls_decoder_depth'])
        ])
        self.cls_decoder_norm = norm_layer(step_2_para['cls_decoder_dim'])
        self.cls_decoder_pred = nn.Linear(step_2_para['cls_decoder_dim'], 2)
        # Loss function
        self.loss_cls = nn.CrossEntropyLoss()
        self.initialize_weights()
        self.setup()

    def setup(self, stage=None):
        # 在setup阶段加载预训练模型
        self.pre_train_model = maskTrajectoryRecoryCLV2(**self.step_1_para).to(self.device)
        ckpt = torch.load(self.step_2_para['model_ckpt'], map_location='cpu')
        # keys in ckpt["state_dict"]: ['model', 'model.head.weight', 'model.head.bias',...]
        # change to ['head.weight', 'head.bias',...] to load the model
        ckpt["state_dict"] = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
        self.pre_train_model.load_state_dict(ckpt["state_dict"])

        # 固定预训练模型的参数
        for param in self.pre_train_model.parameters():
            param.requires_grad = False

        # 将模型设置为评估模式
        self.pre_train_model.eval()
    
    def initialize_weights(self):
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

    def get_repr(self, imgs, padding_length):
        '''
        cls_token_ouput: concat([N, D], [N, D]) -> [2N, D]
        '''
        pred, cl_token, mask, padding = self.pre_train_model.forward_one_sample(imgs, padding_length)
        cls_token_ouput = self.pre_train_model.cl(cl_token)
        cls_token_ouput = normalize(cls_token_ouput, dim=-1)
        return cls_token_ouput
    
    def forward_data_gen(self, data):
        subj_list = data['subj'] # [N]
        # if subj have the same value, then they are from the same person
        imgs = torch.cat([data['seq'][:, 0, :, :], data['seq'][:, 1, :, :]], dim=0).unsqueeze(1) # [2*N, 1, H, W]
        padding_length = torch.cat([data['padding_length'][0], data['padding_length'][1]], dim=0) # [2*N]
        cls_token_ouput = self.get_repr(imgs, padding_length) # [2*N, D]
        
        # Positive sample
        cls_token_output_pos = torch.cat([cls_token_ouput[:cls_token_ouput.shape[0]//2, :], cls_token_ouput[cls_token_ouput.shape[0]//2:, :]], dim=-1)  # [N, 2*D]
        label_pos = torch.ones(cls_token_output_pos.shape[0]).long()

        # Negative sample
        shuffled_indices = torch.randperm(cls_token_ouput.shape[0]//2).long()
        neg_subj_list = subj_list[shuffled_indices]
        cls_token_output_neg = torch.cat([cls_token_ouput[:cls_token_ouput.shape[0]//2, :], cls_token_ouput[cls_token_ouput.shape[0]//2:, :][shuffled_indices]], dim=-1)
        neg_label = torch.tensor(subj_list == neg_subj_list, dtype=torch.long).long()
        cls_token_output_combined = torch.cat([cls_token_output_pos, cls_token_output_neg], dim=0)  # [2N, 2*D]
        label = torch.cat([label_pos.to(self.device), neg_label.to(self.device)], dim=0)
        
        cls_token_output_combined = cls_token_output_combined.unsqueeze(1)
        return cls_token_output_combined, label
    
    def forward_one_sample(self, cls_token_output_combined):
        for block in self.blocks:
            cls_token_output_combined = block(cls_token_output_combined)
        cls_token_output_combined = self.cls_decoder_norm(cls_token_output_combined)
        predictions = self.cls_decoder_pred(cls_token_output_combined)  # [2N, 2]
        predictions = predictions.squeeze(1)
        return predictions

    def forward(self, x):
        cls_token_output_combined, label = self.forward_data_gen(x)
        predictions = self.forward_one_sample(cls_token_output_combined)
        loss = self.loss_cls(predictions, label)
        return loss, predictions, label
    
    def forward_train(self, x, label):
        loss, predictions, label = self.forward(x)
        # pack the output and losses
        return {'loss': loss}

    def forward_test(self, x, label=None):
        loss, predictions, label = self.forward(x)
        label = label.to(torch.int64)
        pred_label = torch.argmax(predictions, dim=-1)
        return {'loss': loss, 'output': pred_label,'label': label}

    



        