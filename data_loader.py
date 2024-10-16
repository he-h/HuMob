import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, DistributedSampler
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from predict import generate_sequence
from utils import *

class MobTimeSeriesDatasetWImputation(Dataset):
    def __init__(self, dataset, input_seq_length, predict_seq_length, subsample=False, subsample_number=100):
        self.input_seq_length = input_seq_length
        self.predict_seq_length = predict_seq_length

        # Precompute the labels and convert dataset to a DataFrame for efficient processing
        dataset = pd.DataFrame(dataset)
        dataset['label'] = 200 * (dataset['x'] - 1) + (dataset['y'] - 1)

        # Sort the dataset once to avoid multiple sorts later
        dataset.sort_values(by=['uid', 'd', 't'], inplace=True)
        self.dataset = dataset

        # Store unique uids
        self.uids = dataset['uid'].unique()
        if subsample:
            self.uids = self.uids[:subsample_number]

        # Prepare sequences
        self.input_seq_feature, self.input_seq_label, self.predict_seq_feature, self.predict_seq_label = self.prepare_sequences()

        print(f'Loaded {len(self.input_seq_feature)} sequences')

    def prepare_sequences(self):
        input_seq_feature = []
        input_seq_label = []
        predict_seq_feature = []
        predict_seq_label = []

        for uid in self.uids:
            uid_df = self.dataset[self.dataset['uid'] == uid]
            max_days = uid_df['d'].max() + 1 # 75 but 60 for test users

            full_seq_x, full_seq_y = self.generate_sequence(uid_df, range(max_days), uid)

            num_seq = max_days - self.input_seq_length - self.predict_seq_length + 1

            if num_seq > 0:
                for i in range(num_seq):
                    input_seq_feature_ = full_seq_x[i * 48:(i + self.input_seq_length) * 48]
                    input_seq_label_ = full_seq_y[i * 48:(i + self.input_seq_length) * 48]
                    predict_seq_feature_ = full_seq_x[(i + self.input_seq_length) * 48:(i + self.input_seq_length + self.predict_seq_length) * 48]
                    predict_seq_label_ = full_seq_y[(i + self.input_seq_length) * 48:(i + self.input_seq_length + self.predict_seq_length) * 48]
                    
                    input_seq_feature.append(input_seq_feature_)
                    input_seq_label.append(input_seq_label_)
                    predict_seq_feature.append(predict_seq_feature_)
                    predict_seq_label.append(predict_seq_label_)

        return (torch.tensor(np.array(input_seq_feature), dtype=torch.long),
                torch.tensor(np.array(input_seq_label), dtype=torch.long),
                torch.tensor(np.array(predict_seq_feature), dtype=torch.long),
                torch.tensor(np.array(predict_seq_label), dtype=torch.long))

    def generate_sequence(self, data_by_day, days, uid):
        # for each uid, generate a sequence of 75 days
        # Vectorized approach
        seq_x = []
        seq_y = []

        for d in days:
            day_data = data_by_day[data_by_day['d'] == d].set_index('t')
            full_day_x = np.array([[d, t, uid, d % 7, t % 24] for t in range(48)]) # TODO: all the input and forward should be changed, 8am to 6pm
            full_day_y = np.full(48, 40000)  # default value [40000, 40000, ... 40000]

            matching_times = day_data.index.values
            full_day_y[matching_times] = day_data['label'].values

            seq_x.extend(full_day_x)
            seq_y.extend(full_day_y)
            
        # seq_x shape is (75*48, 5), seq_y shape is (75*48,)

        return np.array(seq_x), np.array(seq_y)

    def __len__(self):
        return len(self.input_seq_feature)

    def __getitem__(self, idx):
        '''
        The format of seq_feature is (uid, day, time, day_of_week)
        The format of seq_label is the label of the location
        '''
        return (self.input_seq_feature[idx], self.input_seq_label[idx],
                self.predict_seq_feature[idx], self.predict_seq_label[idx])
        

class MobTimeSeriesDataset(Dataset):
    def __init__(self, dataset, input_seq_length, predict_seq_length, subsample=False, subsample_number=100, look_back_len=24, multiple=2):
        self.input_seq_length = input_seq_length
        self.predict_seq_length = predict_seq_length

        # Precompute the labels and convert dataset to a DataFrame for efficient processing
        dataset = pd.DataFrame(dataset)
        dataset['label'] = 200 * (dataset['x'] - 1) + (dataset['y'] - 1)

        # Sort the dataset once to avoid multiple sorts later
        dataset.sort_values(by=['uid', 'd', 't'], inplace=True)
        self.dataset = dataset

        # Store unique uids
        self.uids = dataset['uid'].unique()
        if subsample:
            self.uids = self.uids[:subsample_number]

        # Prepare sequences
        self.input_seq_feature, self.input_seq_label, self.predict_seq_feature, self.predict_seq_label = self.prepare_sequences(look_back_len=look_back_len, multiple=multiple)

        print(f'Loaded {len(self.input_seq_feature)} sequences')

    def prepare_sequences(self, look_back_len=24, multiple=5):
        input_seq_feature = []
        input_seq_label = []
        predict_seq_feature = []
        predict_seq_label = []

        for idx, uid in enumerate(self.uids):
            uid_df = self.dataset[self.dataset['uid'] == uid]
            
            predict_user = False
            if idx >= len(self.uids) - 3000:
                predict_user = True

            full_seq_x, full_seq_y = generate_sequence(uid_df)

            num_seq = (len(uid_df) - self.input_seq_length - self.predict_seq_length + 1)//look_back_len
            
            if num_seq > 0:
                for i in range(num_seq):
                    input_seq_feature_ = full_seq_x[i * look_back_len : i * look_back_len + self.input_seq_length]
                    input_seq_label_ = full_seq_y[i * look_back_len : i * look_back_len + self.input_seq_length]
                    predict_seq_feature_ = full_seq_x[ i * look_back_len + self.input_seq_length : i * look_back_len + self.input_seq_length + self.predict_seq_length]
                    predict_seq_label_ = full_seq_y[i * look_back_len + self.input_seq_length : i * look_back_len + self.input_seq_length + self.predict_seq_length]
                    
                    input_seq_feature.append(input_seq_feature_)
                    input_seq_label.append(input_seq_label_)
                    predict_seq_feature.append(predict_seq_feature_)
                    predict_seq_label.append(predict_seq_label_)
                    
                    if predict_user:
                        for i in range(1, multiple):
                            input_seq_feature.append(input_seq_feature_)
                            input_seq_label.append(input_seq_label_)
                            predict_seq_feature.append(predict_seq_feature_)
                            predict_seq_label.append(predict_seq_label_)
                        

        return (torch.tensor(np.array(input_seq_feature), dtype=torch.long),
                torch.tensor(np.array(input_seq_label), dtype=torch.long),
                torch.tensor(np.array(predict_seq_feature), dtype=torch.long),
                torch.tensor(np.array(predict_seq_label), dtype=torch.long))

    def generate_sequence(self, data):
        uid = data['uid'].values[0]
        # for each uid, generate a sequence of 75 days
        # Vectorized approach
        seq_x = []
        seq_y = []

        previous_d = data['d'].values[0]
        previous_t = data['t'].values[0]
        
        for _, row in data.iterrows():
            d = row['d']
            t = row['t']
            label = row['label']
            
            delta_t = (t - previous_t) + 48 * (d - previous_d)
            
            seq_x.append([d, t, uid, d % 7, t % 24, delta_t])
            seq_y.append(label)
            
            previous_d = d
            previous_t = t

        return np.array(seq_x), np.array(seq_y)

    def __len__(self):
        return len(self.input_seq_feature)

    def __getitem__(self, idx):
        return (self.input_seq_feature[idx], self.input_seq_label[idx],
                self.predict_seq_feature[idx], self.predict_seq_label[idx])

def train_test_mob_time_series_dataloader(rank, world_size, city, input_seq_length, predict_seq_length, subsample=False, subsample_number=100, 
                                          test_size=0.1, batch_size=64, random_seed=42, look_back_len=24):
    # Load the dataset using Pandas directly for faster processing
    data_files = f'data/humob/city{city}_groundtruthdata.csv.gz'
    dataset = pd.read_csv(data_files)

    # Initialize the custom dataset
    custom_dataset = MobTimeSeriesDataset(dataset, input_seq_length, predict_seq_length, subsample=subsample, subsample_number=subsample_number, look_back_len=look_back_len)

    # Split the dataset into training and testing sets
    dataset_size = len(custom_dataset)
    train_size = int(dataset_size * (1 - test_size))
    test_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)  # No shuffle for testing

    # Create DataLoaders with prefetching enabled
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=4, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False, num_workers=4, prefetch_factor=2)

    return train_loader, test_loader

def split_df_by_uid(city, test_size=0.1, random_seed=None, subsample=False, subsample_number=100):
    '''
    Split DataFrame by uid
    '''
    if city == 'A':
        df = pd.read_csv('your_path_here/cityA_groundtruthdata.csv.gz', compression='gzip')
    elif city in ['B', 'C', 'D']:
        df = pd.read_csv(f'your_path_here/city{city}_challengedata.csv.gz', compression='gzip')
    else:
        raise ValueError('City not found')
    
    if subsample:
        uids = df['uid'].unique()[:subsample_number]
        df = df[df['uid'].isin(uids)]
    
    uids = df['uid'].unique()
    generate_uid_list = uids[-3000:]
    generate_df = df[df['uid'].isin(generate_uid_list)]
    
    remain_df = df[df['x'] != 999]
    print(f'Original data size: {len(df)}, Remain data size: {len(remain_df)}, Generate data size: {len(generate_df)}')
    selected_uids = uids[:-3000]
    # random select 10% of the users
    _, test_uids = train_test_split(selected_uids, test_size=test_size, random_state=random_seed)
    
    test_df = remain_df[remain_df['uid'].isin(test_uids)]
    
    # train_df contains the rest of the users and uid in test_df with d<60
    train_df = remain_df[~remain_df['uid'].isin(test_uids)]
    partial_generate_df = generate_df[generate_df['d'] < 60]
    partial_test_df = remain_df[remain_df['d'] < 60]
    train_df = pd.concat([train_df, partial_test_df, partial_generate_df])
    
    return train_df, test_df, generate_df

def train_test_generate_mob_time_series_dataloader(city, input_seq_length, predict_seq_length, 
                                                   subsample=False, subsample_number=100,
                                                   test_size=0.1, batch_size=64, random_seed=42, 
                                                   look_back_len=24, world_size=None, rank=None, multiple=2):
    train_df, test_df, generate_df = split_df_by_uid(city, test_size=test_size, random_seed=random_seed, 
                                                     subsample=subsample, subsample_number=subsample_number)
    dataset = MobTimeSeriesDataset(train_df, input_seq_length, predict_seq_length, look_back_len=look_back_len, multiple=multiple)
    
    if world_size is not None and rank is not None:
        # Distributed training setup
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                                  drop_last=True, num_workers=4, prefetch_factor=2)
    else:
        # Non-distributed training setup
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                  drop_last=True, num_workers=4, prefetch_factor=2)
    
    return train_loader, test_df, generate_df
