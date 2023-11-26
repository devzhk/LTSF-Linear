import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

subject_list = [1, 2, 3, 4]
activities = ['Biking', 'VR', 'Hand grip', 'Stroop']
data_root = '../fatigue/data'

def read_data(data_path):
    df_raw = pd.read_csv(data_path)
    num_train = int(len(df_raw) * 0.9)

    cols_data = df_raw.columns[1:-1]    # remove date and target
    df_data = df_raw[cols_data]

    return df_data[0:num_train].values


def train_scaler(outdir):
    scaler = StandardScaler()
    feat_list = []
    for subject_id in subject_list:
        for act in activities:
            data_path = os.path.join(data_root, f'Subject_{subject_id}-cleaned-{act}.csv')
            data = read_data(data_path)
            feat_list.append(data)
    feats = np.concatenate(feat_list, axis=0)
    scaler.fit(feats)
    scaler_path = os.path.join(outdir, f'train_scaler.joblib')    
    dump(scaler, scaler_path)
    print(scaler.mean_, scaler.scale_)


if __name__ == '__main__':
    outdir = 'assets'
    os.makedirs(outdir, exist_ok=True)
    train_scaler(outdir)
    scaler = load(os.path.join(outdir, 'train_scaler.joblib'))
    print(scaler.mean_, scaler.scale_)

