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

@METRICS.register_module()
class ImageVisualization(pl.LightningModule):
    def __init__(self, n_sample=3, metric_name='ImageVisualization', image_name='generated_images', **kwargs):
        super(ImageVisualization, self).__init__()
        if metric_name is None:
            raise ValueError('metric_name is required')
        self.n_sample = n_sample
        self.image_name = image_name
        self.metric_name = pascal_case_to_snake_case(metric_name)

    def forward(self, pred, target=None):
        self.log_image(pred, name=self.image_name, index=self.trainer.global_step)
        return None

    def log_image(self, image, name="generated_images", index=0):
        image = image[:self.n_sample].detach().cpu()
        # log image to the logger
        for i in range(self.n_sample):
            self.trainer.logger.experiment.log_image(image_data=image[i], name=f'name={name}'+'-'+
                                                                               f'step={index}'+'-'+
                                                                               f'index={i}')

    def save_image(self, image, name="generated_images", index=0):
        raise NotImplementedError

@METRICS.register_module()
class SignalVisualization(pl.LightningModule):
    def __init__(self, samples_idx=[12,24,54,64],
                  metric_name='SignalVisualization', 
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
                  patch_size_w = 5,
                  **kwargs):
        super(SignalVisualization, self).__init__()
        self.__dict__.update(locals())

        if metric_name is None:
            raise ValueError('metric_name is required')
        

        self.skip_tokens = [v for k, v in self.special_token.items()]
        if self.save_image_path is not None: os.makedirs(self.save_image_path, exist_ok=True)
        self.metric_name = pascal_case_to_snake_case(metric_name)

    def forward(self, pred, target, meta_data=None):
        corr = self.log_corr_gpu(pred, target, meta_data, name=self.image_name, index=self.trainer.global_step)
        self.log_image(pred, target, meta_data, name=self.image_name, index=self.trainer.global_step)
        return corr
    
    def log_image(self, image, target, meta_data, name="gimg", index=0):
        if self.trainer.global_step == 0: return None
        sample_idx = []
        for sidx in self.samples_idx:
            if sidx < len(image):
                sample_idx.append(sidx)
        self.samples_idx = sample_idx

        _meta_data = []
        for batch_meta in meta_data:
            sub_meta_data = []
            for i in range(len(batch_meta['mask'])):
                sub_meta_data.append({k: batch_meta[k][i] for k in batch_meta.keys()})
            _meta_data.extend(sub_meta_data)
        meta_data = _meta_data

        image = image[self.samples_idx][:,0,:,:].detach().cpu().numpy() # [B, T, 2 (x,y)]
        target = target[self.samples_idx][:,0,:,:].detach().cpu().numpy() # [B, T, 2 (x,y)]
        
        # log image to the logger
        for i in range(len(self.samples_idx)):
            mask = meta_data[self.samples_idx[i]]['mask'].detach().cpu().numpy()
            padding_length = meta_data[self.samples_idx[i]]['padding_length'].detach().cpu().numpy()
            label = meta_data[self.samples_idx[i]]['label'].detach().cpu().numpy()
            patch_size_w = np.array([self.patch_size_w])
            mask_patch2seq = mask.repeat(patch_size_w, axis=0)
            
            fig = plt.figure()
            signal_data = image[i]
            target_data = target[i]

            signal_data = [(signal_data[j, 0], signal_data[j, 1]) if mask_patch2seq[j] else (target_data[j, 0], target_data[j, 1]) for j in range(signal_data.shape[0])]
            target_data = [(target_data[j, 0], target_data[j, 1]) for j in range(target_data.shape[0])]

            signal_data = signal_data[:-padding_length]
            target_data = target_data[:-padding_length]
            mask = mask[:-padding_length]

            signal_data = [self.check_if_speical_token(stroke) for stroke in signal_data]
            target_strokes = np.array([stroke for stroke in target_data if stroke not in self.skip_tokens])
            signal_strokes = np.array([stroke for stroke in signal_data if stroke not in self.skip_tokens])

            if len(signal_strokes) == 0: continue
            if len(target_strokes) == 0: continue

            plt.plot(signal_strokes[:, 0]*self.width_char, -signal_strokes[:, 1]*self.height_char, 'r', label='generated', alpha=0.5)
            plt.plot(target_strokes[:, 0]*self.width_char, -target_strokes[:, 1]*self.height_char, 'k', label='target', alpha=0.5)
            plt.legend()

            if self.save_image_path is not None:
                plt.savefig(os.path.join(self.save_image_path, f'{name}-step_idx{index}-img_idx{self.samples_idx[i]}.png'))
            plt.close()

            if os.path.exists(os.path.join(self.save_image_path, f'{name}-step_idx{index}-img_idx{self.samples_idx[i]}.png')):
                image_data = PIL.Image.open(os.path.join(self.save_image_path, f'{name}-step_idx{index}-img_idx{self.samples_idx[i]}.png'))
                self.trainer.logger.experiment.log_image(image_data=image_data, name=f'{name}-step_idx{index}-img_idx{self.samples_idx[i]}')
            
    def log_corr_gpu(self, image, target, meta_data, name="gimg", index=0):
        mask = []
        for batch_meta in meta_data:
            mask += [batch_meta['mask'][i] for i in range(len(batch_meta['mask']))]            
        print(len(mask))

        corr = []
        for idx in range(len(image)):
            patch_size_w = torch.tensor([self.patch_size_w], device=image.device)
            mask_patch2seq = mask[idx].repeat_interleave(patch_size_w, dim=0)
            print(mask_patch2seq.shape)
            print(image[idx].shape)
            print(target[idx].shape)
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
