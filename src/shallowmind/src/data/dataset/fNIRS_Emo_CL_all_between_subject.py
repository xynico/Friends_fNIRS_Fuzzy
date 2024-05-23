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
class fNIRS_Emo_CL_all_between_subject(torch.utils.data.Dataset):
    def __init__(self,
                data_root = None,
                test_size=0.3,
                dataset_random_state = 42,
                temp_save_folder = None,
                tier = 'train',
                sampler=None,
                pipeline=None,
                label = 'hand',
                normalize_mode = 'normal',
                minmax_value = None,
                subject_split = 'within',
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
                            self.epoch_data[hand][relationship_label][emotion_label][pair_id][isubj] = self.normalize(subj, self.normalize_mode)
        self.gen_data_table()
        self.split_data()
        self.gen_data_index_table()
    
    def normalize(self, data, mode = 'normal'):
        '''
        data: [n_channels, data_length]
        '''
        if mode == 'normal':
            data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
        elif mode == 'minmax':
            data = (data - self.minmax_value[0]) / (self.minmax_value[1] - self.minmax_value[0])
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
        if self.subject_split == 'between':
            self.split_data_between_subject()
        elif self.subject_split == 'within':
            self.split_data_within_subject()

    def split_data_between_subject(self):
        np.random.seed(self.dataset_random_state)
        # split the data table into train and test.

        need_delete_pair_id = []
        for irow, row in self.data_table.iterrows():
            if row['sample_number'] < 5:
                pair_idx = row['pair_id']
                need_delete_pair_id.append(pair_idx)
        need_delete_pair_id = list(set(need_delete_pair_id))
        self.data_table = self.data_table[~self.data_table['pair_id'].isin(need_delete_pair_id)]
        self.data_table = self.data_table.reset_index(drop=True)
        self.data_table['uni_pair_id'] = self.data_table['pair_id'].apply(lambda x: '-'.join([x.split('-')[1], x.split('-')[2]]))

        ## Split the subjects into train and test first
        train_subjects, test_subjects = train_test_split(self.data_table['uni_pair_id'].unique()
                        , test_size=self.test_size, random_state=self.dataset_random_state)
        rows = []
        for irow, row in self.data_table.iterrows():
            if row['uni_pair_id'] in train_subjects:
                train_idx = list(range(row['sample_number']))
                test_idx = []
            else:
                train_idx = []
                test_idx = list(range(row['sample_number']))
            row['train_idx'] = train_idx
            row['test_idx'] = test_idx
            rows.append(row)
        self.data_table = pd.DataFrame(rows)
        
        self.data_table['trial_idx'] = self.data_table['train_idx'] if self.tier=='train' else self.data_table['test_idx']
        # save the data table
        if os.path.join(self.temp_save_folder, f'data_table_{self.subject_split}_{self.tier}.csv'):
            self.data_table.to_csv(os.path.join(self.temp_save_folder, f'data_table_{self.subject_split}_{self.tier}.csv'), index=False)


        
        


    
    def split_data_within_subject(self):
        np.random.seed(self.dataset_random_state)

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
        
        self.data_table['trial_idx'] = self.data_table['train_idx'] if self.tier=='train' else self.data_table['test_idx']
        # save the data table
        if os.path.join(self.temp_save_folder, f'data_table_{self.tier}.csv'):
            self.data_table.to_csv(os.path.join(self.temp_save_folder, f'data_table_{self.tier}.csv'), index=False)

    def gen_data_index_table(self):

        if os.path.exists(os.path.join(self.temp_save_folder, f'data_index_table_{self.subject_split}_{self.tier}.csv')):
            self.data_index_table = pd.read_csv(os.path.join(self.temp_save_folder, f'data_index_table_{self.subject_split}_{self.tier}.csv'))
        else:
            self.data_index_table = []
            for hand in [1,2]:
                for relationship_label in [0,1]:
                    for emotion_label in [2,3]:
                        rows = self.data_table[(self.data_table['hand']==hand) & (self.data_table['relationship_label']==relationship_label) & (self.data_table['emotion_label']==emotion_label)]
                        for pair_idx in tqdm(rows['pair_id'].unique(), desc=f'hand={hand}, relationship_label={relationship_label}, emotion_label={emotion_label}'):
                            sub_rows = rows[(rows['pair_id']==pair_idx) & (rows['emotion_label']==emotion_label)]
                            if len(sub_rows[sub_rows['isubj']==0]) == 0 or len(sub_rows[sub_rows['isubj']==1]) == 0:
                                continue
                            trial_idxes1 = sub_rows[sub_rows['isubj']==0]['trial_idx'].values[0]
                            trial_idxes2 = sub_rows[sub_rows['isubj']==1]['trial_idx'].values[0]
                            for itrial1 in trial_idxes1:
                                for itrial2 in trial_idxes2:
                                    self.data_index_table.append({
                                        'hand': hand,
                                        'relationship_label': relationship_label,
                                        'emotion_label': emotion_label,
                                        'pair_idx': pair_idx,
                                        'trial_idxes1': itrial1,
                                        'trial_idxes2': itrial2,
                                    })
            self.data_index_table = pd.DataFrame(self.data_index_table)
            self.data_index_table.to_csv(os.path.join(self.temp_save_folder, f'data_index_table_{self.subject_split}_{self.tier}.csv'), index=False)

    def __len__(self):
        return len(self.data_index_table)
    
    def __getitem__(self, idx):
        row = self.data_index_table.iloc[idx]
        hand = int(row['hand'])
        relationship_label = int(row['relationship_label'])
        emotion_label = int(row['emotion_label'])
        pair_idx = row['pair_idx']
        trial_idxes1 = int(row['trial_idxes1'])
        trial_idxes2 = int(row['trial_idxes2'])
        data1 = torch.tensor(self.epoch_data[hand][relationship_label][emotion_label][pair_idx][0][trial_idxes1], dtype=torch.float32)
        data2 = torch.tensor(self.epoch_data[hand][relationship_label][emotion_label][pair_idx][1][trial_idxes2], dtype=torch.float32)
        data = torch.cat([data1.unsqueeze(0), data2.unsqueeze(0)], dim=0)
        subj_id = f"{emotion_label}_{pair_idx}"           

        if self.label == 'hand':
            hand = 0 if hand == 2 else 1
            y = torch.tensor(int(hand), dtype=torch.long)
        elif self.label == 'relationship':
            y = torch.tensor(int(relationship_label), dtype=torch.long)
        elif self.label == 'emotion':
            emotion_label = 0 if emotion_label == 2 else 1
            y = torch.tensor(int(emotion_label), dtype=torch.long)
        return {
                'seq': data,
                # 'subject': int(subj_id),
            }, y