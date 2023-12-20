#%%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
root = '../data'
subjects = [1, 2, 3, 4, 5]
actitivties = ['Stroop','VR', 'Hand grip', 'Biking']
features = ['Chem', 'EMG', 'GSR', 'Pulse data', 'Temp']
#%%
file_path = os.path.join(root, 'Subject_2_cleaned','2_VR EMG feature list.xlsx')
raw_file_path = os.path.join(root, 'Subject_2_cleaned','2_VR EMG.xlsx')
df = pd.read_excel(file_path)

# %%
print(df.head(len(df) // 3600 + 1))
print(len(df))
# %%
df.tail(10)
# %%
file_path = os.path.join(root, 'Subject_2_cleaned','2_VR Pulse feature list.xlsx')
df = pd.read_excel(file_path)
print(df.tail(10))

# %%

# act_id = 1
# subject_id = subjects[2]
# act = actitivties[act_id]
# feature = features[3]

for act_id in range(4):
    for subject_id in subjects[1:]:
        for feature in ['EMG', 'Pulse data']:
            act = actitivties[act_id]
            file_path = os.path.join(root, f'Subject_{subject_id}_cleaned', f'{act_id + 1}_{act} {feature}.xlsx')
            print(file_path)
            df = pd.read_excel(file_path)
            print(len(df))
            print(df.tail(5))
# %%
