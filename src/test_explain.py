# from tqdm import tqdm
from shallowmind.src.model import ModelInterface
from shallowmind.src.data import DataInterface
from shallowmind.api.infer import prepare_inference
import pandas as pd
import torch
from tqdm import tqdm
import pickle
import numpy as np
import pytorch_lightning as pl
pl.seed_everything(42)
from einops import rearrange
import os
import numpy as np
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from torch import nn

def convert(linear_layer,y):
    W = linear_layer.weight.data
    b = linear_layer.bias.data
    print("W",W.min(),W.max(),W.shape)
    print("b",b.min(),b.max(),b.shape)
    W_pinv = torch.pinverse(W)
    print("W_pinv",W_pinv.min(),W_pinv.max(),W_pinv.shape)
    # Adjust the bias shape for the operation
    b = b.unsqueeze(0)  # Make b the same shape as y for subtraction
    # Attempt to recover x using the pseudoinverse
    recovered_x = torch.matmul(W_pinv, (y - b).T).T  # Adjusted operation
    return recovered_x

def find_highest_scored_file_and_config(base_dir,condition):

    # Function to extract F1 score from file name
    def extract_f1_score_v2(file_name):
        # Splitting by '-' and finding the part with 'val_f1_score'
        parts = file_name.split('-')
        for part in parts:
            if part.startswith('val_f1_score='):
                try:
                    return float(part.split('=')[1])
                except ValueError:
                    return 0.0
        return 0.0

    # Initialize variables to store the results
    best_score = 0.0
    best_file = ""
    best_dir = ""
    config_file = ""

    for root, dirs, files in os.walk(base_dir):
        if "ckpts" in root.split(os.sep):
            for file in files:
                match=True
                for con in condition:
                    if con not in file:
                        match=False
                if file.endswith(".ckpt") and match:
                    current_score = extract_f1_score_v2(file)
                    if current_score > best_score:
                        best_score = current_score
                        best_file = file
                        best_dir = root
                        # Reset config_file for a new best score
                        config_file = ""

                        # Attempt to find a config file in the directory containing the best ckpt so far
                        config_candidates = [f for f in os.listdir(os.path.dirname(root)) if f.endswith('.py')]
                        if config_candidates:
                            config_file = os.path.join(os.path.dirname(root), config_candidates[0])

    return best_dir, best_file, best_score, config_file
# base_dir = "/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/work_dir"
# condition=['hand']

# best_dir, best_file, best_score, config_file=find_highest_scored_file_and_config(base_dir,condition=condition)
# print("best_score",best_score)
# save_dir='/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/'
# os.makedirs(save_dir, exist_ok=True)
# ckpt = os.path.join(best_dir, best_file)
# config = config_file

# no reversed:
# ckpt = '/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/work_dir/AFTF-ds=PictureClass_l-d=1-nh=19-r=10-bs=128-ls=0.015-label=hand/ckpts/exp_name=AFTF-ds=PictureClass_l-d=1-nh=19-r=10-bs=128-ls=0.015-label=hand-cfg=FuzzyTramsformer_ALL_num_rules10_num_heads19_dataset_name=PictureClass_label=hand_base_lr=1.5e-2_depth=1_batch_size=128-bs=128-seed=42-val_f1_score=0.7414.ckpt'
# config = '/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/work_dir/AFTF-ds=PictureClass_l-d=1-nh=19-r=10-bs=128-ls=0.015-label=hand/FuzzyTramsformer_ALL_num_rules10_num_heads19_dataset_name=PictureClass_label=hand_base_lr=1.5e-2_depth=1_batch_size=128.py'

# reversed:
ckpt = '/home/xiaowjia/data/Friends_fNIRS/work_dir/FTF-ds=PictureClass_d-d=3-nh=5-r=10-bs=256-ls=0.0015-label=hand/ckpts/exp_name=FTF-ds=PictureClass_d-d=3-nh=5-r=10-bs=256-ls=0.0015-label=hand-cfg=FuzzyTramsformer_num_rules10_num_heads5_dataset_name=PictureClass_label=hand_base_lr=1.5e-3_depth=3_batch_size=256-bs=256-seed=42-val_f1_score=0.8364.ckpt'
config = '/home/xiaowjia/data/Friends_fNIRS/work_dir/FTF-ds=PictureClass_d-d=3-nh=5-r=10-bs=256-ls=0.0015-label=hand/FuzzyTramsformer_num_rules10_num_heads5_dataset_name=PictureClass_label=hand_base_lr=1.5e-3_depth=3_batch_size=256.py'

# transformer-reversed
# ckpt = '/home/xiaowjia/data/Friends_fNIRS/work_dir/baseline_TFe-ds=PictureClass_d-d=2-nh=5-bs=128-ls=0.0015-label=hand/ckpts/exp_name=baseline_TFe-ds=PictureClass_d-d=2-nh=5-bs=128-ls=0.0015-label=hand-cfg=tramsformer_num_heads5_dataset_name=PictureClass_label=hand_base_lr=1.5e-3_depth=2_batch_size=128-bs=128-seed=42-val_f1_score=0.8206.ckpt'
# config = '/home/xiaowjia/data/Friends_fNIRS/work_dir/baseline_TFe-ds=PictureClass_d-d=2-nh=5-bs=128-ls=0.0015-label=hand/tramsformer_num_heads5_dataset_name=PictureClass_label=hand_base_lr=1.5e-3_depth=2_batch_size=128.py'

output_path = '/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/output/output_all.pkl'

data_module, model = prepare_inference(config, ckpt)
data_module.setup()

# 1. draw centers
# 2. Friend Attention statistics


centers=model.model.encoder.encoder.layers[0].self_attn.rules_keys
# print("centers",centers.shape)
centers=rearrange(centers, 'h r d -> r (h d)')
test_loader = data_module.test_dataloader()
data_table = test_loader.dataset.data_table
model = model.eval()


device = torch.device('cpu')
model = model.to(device)
val_loader = data_module.val_dataloader()
data_table = val_loader.dataset.data_table
res = pd.DataFrame(columns=['pred', 'label'])

embs = []
all_data = []
data_info = test_loader.dataset.data_index_table


def plot_center(center, i):
    """
    Plot the raw data alongside its reconstruction for a specific batch and channel index.
    
    Parameters:
    - batch_idx: Index for the batch.
    - channel_idx: Index for the channel.
    - original_data: The original dataset.
    - reconstructed_data: The reconstructed dataset.
    """
    
    # Plotting
    plt.figure(figsize=(10, 4))
    print("center",center.shape)
    plt.plot(center.detach().numpy(), label='Rule center', marker='o')
    # plt.plot(predicted_data.detach().numpy(), label='Reconstructed Data', linestyle='--', marker='x')
    plt.title('Center [{}]'.format(i+1))
    plt.xlabel('Data Point Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('../output_liang/center[{}].png'.format(i+1))
    plt.show()
def convert(linear_layer,y):
    # linear_layer = nn.Linear(in_features=3, out_features=3)
    # print("y",y.shape)
    y=y.unsqueeze(0)
    # y=rearrange(y, 'b c d -> (b c) d')
    # Get weight W and bias b
    W = linear_layer.weight.data
    b = linear_layer.bias.data
    # Define input x: shape (1, 3)
    # x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=False, dtype=torch.float32)
    # Compute output y
    # y = linear_layer(x)
    # Calculate the pseudoinverse of W
    W_pinv = torch.pinverse(W)
    # Adjust the bias shape for the operation
    b = b.unsqueeze(0)  # Make b the same shape as y for subtraction
    # Attempt to recover x using the pseudoinverse
    recovered_x = torch.matmul(W_pinv, (y - b).T).T  # Adjusted operation
    # print("recovered_x",recovered_x.shape)
    # recovered_x=rearrange(recovered_x, '(b c) d -> b c d',c=40)
    # print("shapes",W.shape,b.shape,y.shape,recovered_x.shape)
    return recovered_x[0]
    # print("Recovered x:", recovered_x)

# draw centers
# print(cneter.s)

# scale=model.model.encoder.encoder.layers[0].self_attn.scale
# # print("centers",centers.shape)
# for i in range(len(centers)):
#     # print("center",centers[i].shape)/
#     converted_center=convert(model.model.encoder.encoder.layers[0].self_attn.query_proj,centers[i])
#     # print("converted_center",converted_center.shape)
#     plot_center(converted_center[0],i)
all_atten=[]
for batch_idx, d in tqdm(enumerate(test_loader), total=len(test_loader)):
    torch.cuda.empty_cache()
    data = {'seq': d[0]['seq'].to(device),}
    # all_data.append(data['seq'])
    
    label = d[1].to(device)
    # find critical time points
    with torch.no_grad():
        imgs = model.model.gen_data(data)
        # print("model",model)
        # Need FS and 
        # print(model.model.encoder.encoder.layers[0].self_attn)
        # print(imgs)
        output,attention= model.model.encoder.encoder.layers[0].self_attn(imgs,imgs,imgs,return_atten=True)
        # scale = model.model.encoder.encoder.layers[0].self_attn.scale
        # print("attention",attention.shape)
        # Modified by Jiang
        attention = torch.cat([attention[:attention.shape[0]//2, :], attention[attention.shape[0]//2:, :]], dim=1)
        all_atten.append(attention)
        # all_atten.append(attention)
        # scale=model.model.encoder.encoder.layers[0].self_attn.scale
        # used_query=rearrange(query, 'b h c d -> b c (h d)')
        # used_query=used_query[0,0]
        # print("used_query",used_query.shape)
        # pred_img=convert(model.model.encoder.encoder.layers[0].self_attn.query_proj,used_query)
        # print("pred_img",pred_img.shape,imgs.shape)
        # # pred_img=pred_img(y, 'b h c d -> b c (h d)')
        # plot_raw_vs_reconstructed(imgs[0,0],pred_img[0])
        all_data.append(imgs)
        # print(imgs.shape, query.shape)
        latent = model.model.forward_encoder(imgs)
        
        latent = torch.cat([latent[:latent.shape[0]//2, :], latent[latent.shape[0]//2:, :]], dim=1)
        embs.append(latent.cpu().numpy())
        latent = model.model.cls(latent)
        pred = latent.squeeze(1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        res = pd.concat([res, pd.DataFrame({'pred': pred, 'label': label})], axis=0)



data_info = data_info.reset_index(drop=True)
res = res.reset_index(drop=True)
res = pd.concat([data_info, res], axis=1)

res["check"] = res["pred"] == res["label"]
correct_ind=res["check"].index[res["check"]==True].tolist()
incorrect_ind=res["check"].index[res["check"]==False].tolist()

data_iter = iter(test_loader)
batch = next(data_iter)

# `batch` now contains the first batch of your test data
# Typically, batch will be a tuple or list containing your data and labels/targets
# You can unpack it like this:
inputs, targets = batch
all_data = torch.cat(all_data, dim=0)
all_atten=torch.cat(all_atten, dim=0)
embs = np.concatenate(embs, axis=0)

# print('model',model)
# labels=res['label'].values
# print("labels",len(labels))
# all_data=rearrange(all_data, 'b h c d -> b c (h d)')
# all_data=all_data.mean(axis=1)
print("all_data",all_data.shape, all_data.min(), all_data.max())
print("centers",centers.shape, centers.min(), centers.max())
print('num_correct',len(correct_ind)/len(res))
print("correct min max",all_atten[correct_ind].min(),all_atten[correct_ind].max())
print("incorrect min max",all_atten[incorrect_ind].min(),all_atten[incorrect_ind].max())
# Modified by Jiang
scale=model.model.encoder.encoder.layers[0].self_attn.scale
print("model",model)
query_test=model.model.encoder.encoder.layers[0].self_attn.query_proj(all_data)*scale

print("query_test",query_test.shape,query_test.min(),query_test.max())
print("centers",centers.shape,centers.min(),centers.max())
converted_centers = []
for i in range(len(centers)):
    converted_center=convert(model.model.encoder.encoder.layers[0].self_attn.query_proj,centers[i]/scale)
    converted_centers.append(converted_center)
    print("converted_center",converted_center.shape,converted_center.min(),converted_center.max())
    # plot_center(converted_center[0],i)

output_all = {
    'all_atten': all_atten,
    'embs': embs,
    'res': res,
    'correct_ind': correct_ind,
    'converted_centers': converted_centers,
    'centers': centers,
    'scale': scale

}

import pickle
with open(output_path, 'wb') as f:
    pickle.dump(output_all, f)

# torch.save(all_atten[correct_ind], '../output_liang/all_atten.pkl')
