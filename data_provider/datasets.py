import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from preprocess import z_score, sub_mean


def preprocess(data):
    '''
    data: DataFrame
    '''
    data = data[data['Time (Seconds)'] < 3600]
    # data['zscore'] = z_score(data['ECG Raw Data'])
    return data


def rm_power_freq(data, freq=60):
    '''
    remove the powerline frequency
    '''
    fft_results = np.fft.rfft(data)
    fft_results[freq] = 0.0

    filtered_signal = np.fft.irfft(fft_results)
    return filtered_signal


class EMGDataset(Dataset):
    def __init__(self, root, subject_list, activities, transform=None):
        self.root = root
        self.df_list = []
        self.mean = 2.5
        self.std = 0.123
        self.sample_freq = 500
        for subject_id in subject_list:
            for act_id, act in enumerate(activities):
                print(f'Loading Subject {subject_id} {act} EMG data...')
                file_path = os.path.join(root, f'Subject_{subject_id}_cleaned', f'{act_id+1}_{act} EMG.xlsx')
                df = pd.read_excel(file_path, sheet_name=None)
                df = pd.concat(df.values(), ignore_index=True)
                df = preprocess(df)
                self.df_list.append(df)

    def __len__(self):
        return len(self.df_list) * 3600

    def __getitem__(self, idx):
        df_id = idx // 3600
        seg_id = idx % 3600
        df = self.df_list[df_id]
        seg = df.loc[df['Time (Seconds)'].between(seg_id, seg_id + 1)]
        clean_seg = rm_power_freq(seg['ECG Raw Data'].values)
        if clean_seg.shape[0] < self.sample_freq:
            out = np.pad(clean_seg, (0, self.sample_freq - clean_seg.shape[0]), 'edge')
        else:
            out = clean_seg[:self.sample_freq]
        out = (out - self.mean) / self.std
        return torch.from_numpy(out).float().reshape(1, -1)
    

class PulseData(Dataset):
    def __init__(self, root, subject_list, activities, transform=None):
        self.root = root
        self.df_list = []
        self.mean = 8.38e-23
        self.std = 7.51e-24
        self.sample_freq = 140
        for subject_id in subject_list:
            for act_id, act in enumerate(activities):
                print(f'Loading Subject {subject_id} {act} ECG data...')
                file_path = os.path.join(root, f'Subject_{subject_id}_cleaned', f'{act_id+1}_{act} Pulse data.xlsx')
                df = pd.read_excel(file_path, sheet_name=None)
                df = pd.concat(df.values(), ignore_index=True)
                df = preprocess(df)
                self.df_list.append(df)

    def __len__(self):
        return len(self.df_list) * 3600

    def __getitem__(self, idx):
        df_id = idx // 3600
        seg_id = idx % 3600
        df = self.df_list[df_id]
        seg = df.loc[df['Time (Seconds)'].between(seg_id, seg_id + 1)]
        seg['zscore'] = z_score(seg['Data'], self.mean, self.std)
        clean_seg = seg[seg['zscore'].abs() < 3]['zscore'].values
        
        if clean_seg.shape[0] < self.sample_freq:
            out = np.pad(clean_seg, (0, self.sample_freq - clean_seg.shape[0]), 'edge')
        else:
            out = clean_seg[:self.sample_freq]

        return torch.from_numpy(out).float().reshape(1, -1)