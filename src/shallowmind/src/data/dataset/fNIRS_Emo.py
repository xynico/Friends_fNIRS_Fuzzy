import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
warnings.filterwarnings("ignore")
from ..builder import DATASETS
import torch
import pandas as pd
from ..pipeline import Compose
import numpy as np
from sklearn.model_selection import train_test_split

# Function to load data from a file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Function to save data
def save_data(data, data_path):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)

@DATASETS.register_module()
class fNIRS_Emo(torch.utils.data.Dataset):
    def __init__(self,
                data_root = None,
                test_size=0.3,
                dataset_random_state = 42,
                temp_save_folder = None,
                tier = 'train',
                max_items = None,
                sampler=None,
                pipeline=None,
                **kwargs):
        self.__dict__.update(locals())
        self.setup()

        self.pipeline = Compose(self.pipeline)
        if self.sampler is not None:
            if "torchsampler" in self.sampler:
                import torchsampler
                self.data_sampler = getattr(torchsampler, self.sampler.replace("torchsampler.",""))(self)
            elif "imblearn.under_sampling" in self.sampler:
                import imblearn.under_sampling
                self.data_sampler = getattr(imblearn, self.sampler.replace("imblearn.under_sampling.",""))(self)
            else:
                self.data_sampler = getattr(torch.utils.data, self.sampler)(self)
        else:
            self.data_sampler = torch.utils.data.RandomSampler(self)

    def setup(self):

        if not os.path.exists(self.temp_save_folder):
            os.makedirs(self.temp_save_folder)

        self.epoch_data = load_data(self.data_root)
        for hand in self.epoch_data.keys():
            for relationship_label in self.epoch_data[hand].keys():
                for emotion_label in self.epoch_data[hand][relationship_label].keys():
                    for pair_id in self.epoch_data[hand][relationship_label][emotion_label].keys():
                        for isubj, subj in enumerate(self.epoch_data[hand][relationship_label][emotion_label][pair_id]):
                            self.epoch_data[hand][relationship_label][emotion_label][pair_id][isubj] = self.normalize(subj)
        self.gen_data_table()
        self.split_data()
    
    def normalize(self, data):
        '''
        data: [n_channels, data_length]
        '''
        data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
        return data

    def gen_data_table(self):
        # get the data table
        columns = ['hand','relationship_label', 'emotion_label', 'pair_id', 'isubj', 'sample_number']
        self.data_table = pd.DataFrame(columns=columns)
        data = [ (hand, relationship_label, emotion_label, pair_id, isubj, len(subj))
                for hand in self.epoch_data.keys()
                for relationship_label in self.epoch_data[hand].keys()
                for emotion_label in self.epoch_data[hand][relationship_label].keys()
                for pair_id in self.epoch_data[hand][relationship_label][emotion_label].keys()
                for isubj, subj in enumerate(self.epoch_data[hand][relationship_label][emotion_label][pair_id])
            ]
        self.data_table = pd.DataFrame(data, columns=columns)
        
    
    def split_data(self):

        # split the data table into train and test by balancing subject, state and date, 3 keys
        need_delete_pair_id = []
        for irow, row in self.data_table.iterrows():
            if row['sample_number'] < 5:
                pair_idx = row['pair_id']
                need_delete_pair_id.append(pair_idx)
        need_delete_pair_id = list(set(need_delete_pair_id))
        self.data_table = self.data_table[~self.data_table['pair_id'].isin(need_delete_pair_id)]
        self.data_table = self.data_table.reset_index(drop=True)
        # randomly split the data into train and test in sample level
        rows = []
        for irow, row in self.data_table.iterrows():
            train_idx = np.random.choice(range(row['sample_number']), size=int(row['sample_number']*(1-self.test_size)), replace=False)
            test_idx = list(set(range(row['sample_number']))-set(train_idx))
            row['train_idx'] = train_idx
            row['test_idx'] = test_idx
            rows.append(row)
        self.data_table = pd.DataFrame(rows)

        # save the data table
        if self.temp_save_folder is not None:
            self.data_table.to_csv(os.path.join(self.temp_save_folder, 'data_table.csv'), index=False)
        
        self.data_table['trial_idx'] = self.data_table['train_idx'] if self.tier=='train' else self.data_table['test_idx']


    def __len__(self):
        assert self.max_items is not None, "max_items must be set"
        return int(self.max_items * 0.5)
    
    def __getitem__(self, idx):

        # randomly select a pair_id
        pair_idx = np.random.choice(self.data_table['pair_id'].unique(), size=1, replace=False)[0] # pair_idx = f"{hand}-{relationship_label}-{key[7:]}"
        hand, relationship_label, key = pair_idx.split('-')

        # randomly select one emotion label
        emotion_label = np.random.choice([2,3], size=1, replace=False)[0]

        # randomly select one trial per subject
        trial_idx1 = np.random.choice(self.data_table[(self.data_table['pair_id']==pair_idx)  & (self.data_table['isubj']==0) & (self.data_table['emotion_label']==emotion_label)]['trial_idx'].values[0], size=1, replace=False)[0]
        trial_idx2 = np.random.choice(self.data_table[(self.data_table['pair_id']==pair_idx)  & (self.data_table['isubj']==1) & (self.data_table['emotion_label']==emotion_label)]['trial_idx'].values[0], size=1, replace=False)[0]
        
        # get the data
        data1 = torch.tensor(self.epoch_data[int(hand)][int(relationship_label)][emotion_label][pair_idx][0][trial_idx1], dtype=torch.float32)
        data2 = torch.tensor(self.epoch_data[int(hand)][int(relationship_label)][emotion_label][pair_idx][1][trial_idx2], dtype=torch.float32)

        # concat the data from 2 x [n_channels, data_length] to [2, n_channels, data_length]
        data = torch.cat([data1.unsqueeze(0), data2.unsqueeze(0)], dim=0)
        
        return {
                'seq': data,
            }, torch.tensor(int(relationship_label), dtype=torch.long)
    