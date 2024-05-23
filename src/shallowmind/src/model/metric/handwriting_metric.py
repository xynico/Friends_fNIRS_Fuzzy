from copy import deepcopy
import torchvision
import pytorch_lightning as pl
from ..builder import METRICS
from ..utils import pascal_case_to_snake_case
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import PIL
import torch
from copy import deepcopy
import torchvision
import pytorch_lightning as pl
from ..builder import METRICS
from ..utils import pascal_case_to_snake_case
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
import torch
from matplotlib.font_manager import FontProperties
font_path = "simsun.ttc"  # Ensure this path is correct
font_prop = FontProperties(fname=font_path)
from sklearn.metrics import pairwise_distances

@METRICS.register_module()
class handwritingCosSimAcc(pl.LightningModule):
    def __init__(self, 
                  metric_name='handwritingCosSimAcc', 
                  by = ['mask', 'padding_length', 'label', 'cls_token_ouput'],
                  label2idx_path = None,
                  **kwargs):
        super(handwritingCosSimAcc, self).__init__()
        self.__dict__.update(locals())
        self.metric_name = pascal_case_to_snake_case(metric_name)

    def forward(self, pred, target, meta_data=None):
        pred, target, meta_data = self.merge_data(pred, target, meta_data)
        acc = self.cal_acc(meta_data, index=self.trainer.global_step)
        return acc
    
    def merge_data(self, pred, target, meta_data):
        target = torch.cat([target[:,0,:,:], target[:,1,:,:]], dim=0).unsqueeze(1)
        meta_data = {k: torch.cat([meta[k] for meta in meta_data], dim=0) for k in meta_data[0].keys()}
        return pred, target, meta_data
            
    def cal_acc(self, meta_data, index=0):
        cls_token_ouput = meta_data['cls_token_ouput'].detach().cpu().numpy() # [2*B, E]
        B = cls_token_ouput.shape[0] // 2
        cos_sim = 1 - pairwise_distances(cls_token_ouput[:B], cls_token_ouput[B:], metric='cosine')
        pred = np.argmax(cos_sim, axis=1)
        target = np.arange(B)
        acc = np.sum(pred == target) / B
        return acc


    

@METRICS.register_module()
class handwritingMaskVisualization(pl.LightningModule):
    def __init__(self, samples_idx=[12,24,54,64],
                  metric_name='MaskedCorr', 
                  image_name='generated_signals', 
                  save_image_path = None,
                  special_token = dict(
                    penoff_symbol= (-1,-1),
                    penon_symbol = (-1,0),
                    padding_token = (0, -1),
                  ),
                  width_char = 1000,
                  height_char = 1000,
                  special_token_value = -1,
                  by = ['mask', 'padding_length', 'label'],
                  label2idx_path = None,
                  patch_size_w = 5,
                  **kwargs):
        super(handwritingMaskVisualization, self).__init__()
        self.__dict__.update(locals())

        if metric_name is None:
            raise ValueError('metric_name is required')
        
        # if label2idx_path is not None and os.path.exists(label2idx_path):
        #     self.label2idx = np.load(label2idx_path, allow_pickle=True).item()
        #     self.idx2label = {v: k for k, v in self.label2idx.items()} # {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'...}
        # else:
        #     self.label2idx = None
        #     self.idx2label = None
        

        self.skip_tokens = [v for k, v in self.special_token.items()]
        if self.save_image_path is not None: os.makedirs(self.save_image_path, exist_ok=True)
        self.metric_name = pascal_case_to_snake_case(metric_name)

    def forward(self, pred, target, meta_data=None):
        pred, target, meta_data = self.merge_data(pred, target, meta_data)
        self.log_image(pred, target, meta_data, name=self.image_name, index=self.trainer.global_step)
        return self.log_corr_gpu(pred, target, meta_data, name=self.metric_name, index=self.trainer.global_step)
    
    def merge_data(self, pred, target, meta_data):
        target = torch.cat([target[:,0,:,:], target[:,1,:,:]], dim=0).unsqueeze(1)
        meta_data = {k: torch.cat([meta[k] for meta in meta_data], dim=0) for k in meta_data[0].keys()}
        return pred, target, meta_data

    def log_image(self, image, target, meta_data, name="gimg", index=0):
        if self.trainer.global_step == 0: return None
        
        sample_idx = []
        for sidx in self.samples_idx:
            if sidx < len(image):
                sample_idx.append(sidx)
        self.samples_idx = sample_idx
        
        image = image[self.samples_idx][:,0,:,:].detach().cpu().numpy() # [B, T, 2 (x,y)]
        target = target[self.samples_idx][:,0,:,:].detach().cpu().numpy() # [B, T, 2 (x,y)]
        mask = meta_data['mask'][self.samples_idx].detach().cpu().numpy() # [B, T]
        padding_length = meta_data['padding_length'][self.samples_idx].detach().cpu().numpy() # [B]
        # label = meta_data['label'][self.samples_idx].detach().cpu().numpy() # [B]
        patch_size_w = np.array([self.patch_size_w])

        # log image to the logger
        for i in range(len(self.samples_idx)):
            mask_patch2seq = mask[i].repeat(patch_size_w, axis=0)
            fig = plt.figure()
            canvas = FigureCanvas(fig)
            signal_data = image[i]
            target_data = target[i]
            signal_data = [(signal_data[j, 0], signal_data[j, 1]) if mask_patch2seq[j] else (target_data[j, 0], target_data[j, 1]) for j in range(signal_data.shape[0])]
            target_data = [(target_data[j, 0], target_data[j, 1]) for j in range(target_data.shape[0])]
            signal_data = signal_data[:-padding_length[i]]
            target_data = target_data[:-padding_length[i]]

            signal_data = [self.check_if_speical_token(stroke) for stroke in signal_data]
            target_strokes = np.array([stroke for stroke in target_data if stroke not in self.skip_tokens])
            signal_strokes = np.array([stroke for stroke in signal_data if stroke not in self.skip_tokens])

            if len(signal_strokes) == 0: continue
            if len(target_strokes) == 0: continue
            plt.plot(signal_strokes[:, 0]*self.width_char, -signal_strokes[:, 1]*self.height_char, 'r', label='generated', alpha=0.5)
            plt.plot(target_strokes[:, 0]*self.width_char, -target_strokes[:, 1]*self.height_char, 'k', label='target', alpha=0.5)
            # plt.title(f'Label: {self.idx2label[label[i]]}', fontproperties = font_prop)
            plt.legend()
            canvas.draw()
            img_str = canvas.tostring_rgb()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image_data = np.frombuffer(img_str, dtype='uint8').reshape(int(height), int(width), 3)
            plt.close(fig)
            self.trainer.logger.experiment.log_image(image_data=image_data, name=f'{name}-step_idx{index}-img_idx{self.samples_idx[i]}')
            
    def log_corr_gpu(self, image, target, meta_data, name="gimg", index=0):
        mask = meta_data['mask'].cuda()
        corr = []
        for idx in range(len(image)):
            patch_size_w = torch.tensor([self.patch_size_w], device=image.device)
            mask_patch2seq = mask[idx].repeat_interleave(patch_size_w, dim=0)
            sig = image[idx][0,mask_patch2seq, :].view(-1)
            tar = target[idx][0,mask_patch2seq, :].view(-1)
            # Compute correlation using PyTorch
            if sig.numel() > 1:  # Ensure there are at least two elements
                sig_mean = torch.mean(sig)
                tar_mean = torch.mean(tar)
                sig -= sig_mean
                tar -= tar_mean
                corr_coeff = torch.sum(sig * tar) / torch.sqrt(torch.sum(sig ** 2) * torch.sum(tar ** 2))
                if not torch.isnan(corr_coeff):
                    corr.append(corr_coeff)
        # Compute the mean of correlations
        if corr:
            corr_mean = torch.mean(torch.stack(corr)).cpu().item()
        else:
            corr_mean = float('nan')  # or some default value in case of no valid correlation
        
        return corr_mean
        
    def check_if_speical_token(self, point):
        x,y = point
        if x < 0:
            x = self.special_token_value
            if y < 0:
                y = self.special_token_value
            else:
                y = 0
        else:
            if y < 0:
                y = self.special_token_value
                x = 0
        return (x,y)

    def save_image(self, image, name="generated_images", index=0):
        raise NotImplementedError

@METRICS.register_module()
class HandwritingCLCorrelation(pl.LightningModule):
    def __init__(self, metric_name='HandwritingCLcorrelation', eps=1e-8, detach_target=True):
        """
        Compute correlation between the output and the target

        Args:
            eps (float, optional): Used to offset the computed variance to provide numerical stability.
                Defaults to 1e-12.
            detach_target (bool, optional): If True, `target` tensor is detached prior to computation. Appropriate when
                using this as a loss to train on. Defaults to True.
        """
        super().__init__()
        self.metric_name = metric_name
        self.eps = eps
        self.detach_target = detach_target

    def forward(self, output, target):
        target = torch.cat([target[:,0,:,:], target[:,1,:,:]], dim=0).unsqueeze(1)
        if self.detach_target:
            target = target.detach()
        delta_out = output - output.mean(0, keepdim=True)
        delta_target = target - target.mean(0, keepdim=True)

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()
        return corrs.mean()