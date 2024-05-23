import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from shallowmind.src.model import ModelInterface
from shallowmind.src.data import DataInterface
from shallowmind.api.infer import prepare_inference
from shallowmind.src.utils import load_config
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

pl.seed_everything(42)

def config_function(config):
    config.data['train']['data_root'] = config.data['train']['data_root'].replace('/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/dataset', '/data/xiaowjia/Friends_fNIRS/data')
    config.data['val']['data_root'] = config.data['val']['data_root'].replace('/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/dataset', '/data/xiaowjia/Friends_fNIRS/data')
    config.data['test']['data_root'] = config.data['test']['data_root'].replace('/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/dataset', '/data/xiaowjia/Friends_fNIRS/data')

    config.data['train']['temp_save_folder'] = config.data['train']['temp_save_folder'].replace('/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/dataset', '/data/xiaowjia/Friends_fNIRS/data')
    config.data['val']['temp_save_folder'] = config.data['val']['temp_save_folder'].replace('/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/dataset', '/data/xiaowjia/Friends_fNIRS/data')
    config.data['test']['temp_save_folder'] = config.data['test']['temp_save_folder'].replace('/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/dataset', '/data/xiaowjia/Friends_fNIRS/data')
    return config

def load_model_data(config_path, ckpt_path,confing_function=None):
    data_module, model = prepare_inference(config_path, ckpt_path, confing_function=confing_function)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    return model, test_loader

def model_inference(model, test_loader, device='cuda', metrics=['accuracy', 'recall', 'precision', 'f1', 'roc_auc', 'pr_auc']):
    # inference
    model = model.to(device)
    model = model.eval()
    data_table = test_loader.dataset.data_table
    predictions = []
    labels = []
    for batch_id,d in enumerate(test_loader):
        with torch.no_grad():
            data = {'seq': d[0]['seq'].to(device),}
            label = d[1].to(device)
            out = model.model.forward(data, label) 
            if len(out) == 2:
                loss, pred = out[0], out[1]
            elif len(out) == 3:
                loss, pred, att = out[0], out[1], out[2]
            pred = pred.argmax(dim=1)
            predictions.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    res = pd.DataFrame(columns=metrics, index=[0])
    for metric in metrics:
        if metric == 'accuracy':
            res[metric] = accuracy_score(labels, predictions)
        elif metric == 'recall':
            res[metric] = recall_score(labels, predictions)
        elif metric == 'precision':
            res[metric] = precision_score(labels, predictions)
        elif metric == 'f1':
            res[metric] = f1_score(labels, predictions)
        elif metric == 'roc_auc':
            res[metric] = roc_auc_score(labels, predictions)
        elif metric == 'pr_auc':
            res[metric] = average_precision_score(labels, predictions)
    return res


work_dir = '/data/xiaowjia/Friends_fNIRS/work_dir_rule'
save_path = '/data/xiaowjia/Friends_fNIRS/output/results_rule.csv'
if os.path.exists(save_path):
    res_table = pd.read_csv(save_path)
else:
    res_table = pd.DataFrame(columns=['experiment_path', 'dataset_type', 'dataset_label', 'dataset_name', 'base_lr', 'batch_size', 'model_name', 'model_type', 'num_heads', 'depth', 'num_rules']+['accuracy', 'recall', 'precision', 'f1', 'roc_auc', 'pr_auc'])

experiments_paths = [f for f in os.listdir(work_dir)]
for experiment_path in tqdm(experiments_paths, total=len(experiments_paths)):
    try:
        if not os.path.exists(os.path.join(work_dir, experiment_path, 'ckpts','last.ckpt')): 
            print(f'no ckpt: {experiment_path}')
            continue
        if experiment_path in res_table['experiment_path'].values: 
            print(f'skip: {experiment_path}')
            continue
        print(f'processing: {experiment_path}')
        config_poth = [f for f in os.listdir(os.path.join(work_dir, experiment_path)) if f.endswith('.py')][0]
        config_path = os.path.join(work_dir, experiment_path, config_poth)
        ckpt_path = [f for f in os.listdir(os.path.join(work_dir, experiment_path, 'ckpts')) if not f.startswith('last')]
        ckpt_path = sorted(ckpt_path, key=lambda x: float(x.split('=')[-1].split('.')[0]), reverse=True)[0]
        ckpt_path = os.path.join(work_dir, experiment_path, 'ckpts', ckpt_path)
        # read config details

        # load config
        model, test_loader = load_model_data(config_path, ckpt_path, confing_function=config_function)
        config = load_config(config_path)
        res = model_inference(model, test_loader)

        # add para to res
        res['experiment_path'] = experiment_path

        dataset_type = config['dataset_para']['type']
        res['dataset_type'] = dataset_type

        dataset_label = config['dataset_para']['label']
        res['dataset_label'] = dataset_label

        dataset_name = config['dataset_name']
        res['dataset_name'] = dataset_name

        base_lr = config['base_lr']
        res['base_lr'] = base_lr

        batch_size = config['batch_size']
        res['batch_size'] = batch_size

        model_name = config['model']['type']
        res['model_name'] = model_name

        model_type = config['model_para']['encoder_type']
        res['model_type'] = model_type

        num_heads = config['model_para']['num_heads']
        res['num_heads'] = num_heads

        depth = config['model_para']['depth']
        res['depth'] = depth

        try:
            num_rules = config['model_para']['num_rules']
            res['num_rules'] = num_rules
        except:
            res['num_rules'] = 0
        
        res_table = pd.concat([res_table, res], axis=0)
        res_table.to_csv(save_path, index=False)
    except Exception as e:
        print(f'failed: {experiment_path}: {e}')
        continue