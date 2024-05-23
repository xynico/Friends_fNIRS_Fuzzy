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
def denormalize(x_scaled, min_val, max_val):
    # Scale from [-1, 1] to [0, 1]
    x_normalized = (x_scaled + 1) / 2
    # Denormalize to [min_val, max_val]
    x = x_normalized * (max_val - min_val) + min_val
    return x

def convert(linear_layer,y):
    W = linear_layer.weight.data
    b = linear_layer.bias.data
    W_pinv = torch.pinverse(W)
    # Adjust the bias shape for the operation
    b = b.unsqueeze(0)  # Make b the same shape as y for subtraction
    # Attempt to recover x using the pseudoinverse
    recovered_x = torch.matmul(W_pinv, (y - b).T).T  # Adjusted operation
    return recovered_x
def plot_umap(all_data,correct_ind, special_data, save_dir):
    # Assuming 'all_data' is your dataset and has the shape [17028, 2, 40, 33]
    # And 'special_data' is a subset of 'all_data' with a special characteristic

    # Flatten the data for UMAP
    subject_data=rearrange(all_data, 'b h c d -> b c (h d)')
    subject_data=rearrange(subject_data, 'b h d-> (b h) d')
    # subject_data=subject_data[correct_ind[0]]
    data_flattened=subject_data
    # data_flattened=rearrange(subject_data, 'c t -> (c t) w')
    # data_flattened = rearrange(all_data, 'b c h w -> (b c h) w')
    special_data_flattened = special_data
    if not os.path.exists(save_dir+'umap_data.pkl'):
        # Apply UMAP
        umap_reducer = UMAP(n_components=2,low_memory=False)  # Use 2 for 2D visualization
        data_umap = umap_reducer.fit_transform(data_flattened)
        special_data_umap = umap_reducer.transform(special_data_flattened)  # Transform special data
        # save
        with open(save_dir + 'umap_data.pkl', 'wb') as f:
            pickle.dump(data_umap, f)
        with open(save_dir + 'umap_special_data.pkl', 'wb') as f:
            pickle.dump(special_data_umap, f)
    else:
        with open(save_dir + 'umap_data.pkl', 'rb') as f:
            data_umap = pickle.load(f)
        with open(save_dir + 'umap_special_data.pkl', 'rb') as f:
            special_data_umap = pickle.load(f)

    # Visualize
    plt.scatter(data_umap[:, 0], data_umap[:, 1], s=1, label='All Data', alpha=0.3, c='blue')

    plt.scatter(special_data_umap[:, 0], special_data_umap[:, 1], s=10, marker='*', label='Special Data', alpha=0.7, c='red')

    plt.title('UMAP Visualization of the Data')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    figure_name = 'umap_norm.png'
    
    # Save the visualization
    plt.savefig(save_dir + figure_name)
    plt.show()

save_dir='/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/output_liang/'
os.makedirs(save_dir, exist_ok=True)
ckpt = '/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/work_dir_rule/FTF-ds=PictureClass_d-d=1-nh=5-r=10-bs=128-ls=0.00015-label=hand-drs=42/ckpts/exp_name=FTF-ds=PictureClass_d-d=1-nh=5-r=10-bs=128-ls=0.00015-label=hand-drs=42-cfg=Fuzzy_norm-bs=128-seed=42-val_f1_score=0.7670.ckpt'
config = '/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/Fuzzy_norm.py'
data_module, model = prepare_inference(config, ckpt)
data_module.setup()

centers=model.model.encoder.encoder.layers[0].self_attn.rules_keys

centers=rearrange(centers, 'h r d -> r (h d)')
print("centers",centers.shape)
test_loader = data_module.test_dataloader()
data_table = test_loader.dataset.data_table
model = model.eval()
# print(model)

device = torch.device('cpu')
model = model.to(device)
val_loader = data_module.val_dataloader()
data_table = val_loader.dataset.data_table
res = pd.DataFrame(columns=['pred', 'label'])

embs = []
data_info = test_loader.dataset.data_index_table
all_data=[]
for batch_idx, d in tqdm(enumerate(test_loader), total=len(test_loader)):
    torch.cuda.empty_cache()
    data = {'seq': d[0]['seq'].to(device),}
    # all_data.append(data['seq'])
    label = d[1].to(device)
    with torch.no_grad():
        imgs = model.model.gen_data(data)
        # print(model.model.encoder.encoder.layers[0].self_attn)
        output,query= model.model.encoder.encoder.layers[0].self_attn(imgs,imgs,imgs,return_query=True)
        all_data.append(query)
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
embs = np.concatenate(embs, axis=0)
all_data = torch.cat(all_data, dim=0).cpu().numpy()
# labels=res['label'].values
# print("labels",len(labels))
print("all_data",all_data.shape, 'centers', centers.shape)
min_val=model.model.encoder.encoder.layers[0].self_attn.norm.min_val
max_val=model.model.encoder.encoder.layers[0].self_attn.norm.max_val
scale=model.model.encoder.encoder.layers[0].self_attn.scale
# reverse all_data
de_data=denormalize(all_data, min_val, max_val)/scale
de_center=denormalize(centers, min_val, max_val)/scale

plot_umap(de_data, correct_ind,de_center.detach().numpy(), save_dir)
