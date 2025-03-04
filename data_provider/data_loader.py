import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from joblib import load

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def find_cloest_int(array, value):
    '''
    Find the index of the closest integer in a sorrted array.
    '''
    idx = np.searchsorted(array, value, side="left")



class Dataset_Custom(Dataset):
    def __init__(self, root_path,
                 data_path=None, 
                 flag='train', size=None,
                 features='S',
                 target='OT', scale=True, 
                 timeenc=0, freq='h', train_only=False,
                 scaler_path=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.subject_list = [1, 2, 3, 4]
        self.activities = ['Biking', 'VR', 'Hand grip', 'Stroop']
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        assert self.seq_len == self.pred_len
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        # if scaler_path:
            # self.scaler = load(scaler_path)
            # print('Loaded scaler from', scaler_path)
        self.build_data()

    def build_data(self):
        self.feat_arr = []
        self.target_arr = []
        self.stamp_arr = []
        self.borders = [-0.5]

        for subject_id in self.subject_list:
            for act in self.activities:
                datapath = os.path.join(self.root_path, f'Subject_{subject_id}-cleaned-{act}.csv')
                feat, target, stamp = self.__read_data__(datapath, set_type=self.set_type)
                self.feat_arr.append(feat)
                self.target_arr.append(target)
                self.stamp_arr.append(stamp)
                self.borders.append(self.borders[-1] + len(feat) - self.seq_len - self.label_len + 1)

    def __read_data__(self, datapath, set_type=0):
        df_raw = pd.read_csv(datapath)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        num_train = int(len(df_raw) * 0.9)
        num_vali = len(df_raw) - num_train
        border1s = [0, num_train - self.seq_len]
        border2s = [num_train, num_train + num_vali]
        border1 = border1s[set_type]
        border2 = border2s[set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:-1]    # remove date and target
            df_data = df_raw[cols_data]

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data_x = data[border1:border2]
        data_y = df_raw[[self.target]].values[border1:border2]
        return data_x, data_y, data_stamp

    def __getitem__(self, index):
        series_idx = np.searchsorted(self.borders, index, side="right") - 1
        idx = index - int(self.borders[series_idx] + 0.5)

        data_x = self.feat_arr[series_idx]
        data_y = self.target_arr[series_idx]
        data_stamp = self.stamp_arr[series_idx]

        s_begin = idx + self.label_len
        s_end = s_begin + self.seq_len

        r_begin = idx
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data_x[s_begin:s_end]
        # seq_y = self.data_x[r_begin:r_end]
        target_y = data_y[r_begin:r_end]

        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        return seq_x, target_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return int(self.borders[-1]+0.5)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, 
                 timeenc=0, freq='15min', cols=None, train_only=False,
                 scaler_path=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        if scaler_path:
            self.scaler = load(scaler_path)
            print('Loaded scaler from', scaler_path)
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')

        cols.remove(self.target)
        border1 = 0              # len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols + [self.target]]
            cols_data = df_raw.columns[1:-1]
            df_data = df_raw[cols_data]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = df_raw[[self.target]].values[border1:border2]
        # if self.inverse:
        #     self.data_y = df_data.values[border1:border2]
        # else:
        #     self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = self.label_len + index * self.pred_len
        s_end = s_begin + self.seq_len

        r_begin = s_begin - self.label_len
        r_end = s_end

        seq_x = self.data_x[s_begin:s_end]

        target_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, target_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x - self.label_len) // self.pred_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Classification(Dataset):
    def __init__(self, root_path,
                 subjects=[1, 2, 3, 4], 
                 flag='train', size=None,
                 features='S',
                 target='OT', scale=True, 
                 timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.subject_list = subjects
        self.activities = ['Biking', 'VR', 'Hand grip', 'Stroop']
        self.act2label = {key: value for value, key in enumerate(self.activities)}
        self.num_classes = len(self.activities)
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        flag = flag.lower()
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        # if scaler_path:
            # self.scaler = load(scaler_path)
            # print('Loaded scaler from', scaler_path)
        self.build_data()

    def build_data(self):
        self.feat_arr = []
        self.target_arr = []
        self.stamp_arr = []
        self.borders = [-0.5]

        for subject_id in self.subject_list:
            for act in self.activities:
                datapath = os.path.join(self.root_path, f'Subject_{subject_id}-cleaned-{act}.csv')
                feat, stamp = self.__read_data__(datapath, set_type=self.set_type)
                self.feat_arr.append(feat)
                self.target_arr.append(self.act2label[act])
                self.stamp_arr.append(stamp)
                self.borders.append(self.borders[-1] + len(feat) - self.seq_len - self.label_len + 1)

    def __read_data__(self, datapath, set_type=0):
        df_raw = pd.read_csv(datapath)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if set_type == 2:
            border1 = 0
            border2 = len(df_raw)
        else:
            num_train = int(len(df_raw) * 0.9)
            num_vali = len(df_raw) - num_train
            border1s = [0, num_train - self.seq_len]
            border2s = [num_train, num_train + num_vali]
            border1 = border1s[set_type]
            border2 = border2s[set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:-1]    # remove date and target
            df_data = df_raw[cols_data]

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data_x = data[border1:border2]
        return data_x, data_stamp

    def __getitem__(self, index):
        series_idx = np.searchsorted(self.borders, index, side="right") - 1
        idx = index - int(self.borders[series_idx] + 0.5)

        data_x = self.feat_arr[series_idx]
        label = self.target_arr[series_idx]
        data_stamp = self.stamp_arr[series_idx]

        s_begin = idx + self.label_len
        s_end = s_begin + self.seq_len

        seq_x = data_x[s_begin:s_end]

        seq_x_mark = data_stamp[s_begin:s_end]

        return seq_x, label, seq_x_mark

    def __len__(self):
        return int(self.borders[-1]+0.5)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
