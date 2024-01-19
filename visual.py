#%%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import z_score, sub_mean, bandpass_filter
#%%
root = '../data'
subjects = [1, 2, 3, 4, 5]
actitivties = ['Stroop','VR', 'Hand grip', 'Biking']
features = ['Chem', 'EMG', 'GSR', 'Pulse data', 'Temp']
#%%

feat = 'Pulse data'
act_id = 2
act = actitivties[act_id]
subject_id = subjects[0]

file_path = os.path.join(root, f'Subject_{subject_id}_cleaned', f'{act_id + 1}_{act} {feat}.xlsx')
dodf = pd.read_excel(file_path, sheet_name=None)
df = pd.concat(dodf.values(), ignore_index=True)
#%%
df = df[df['Time'] < 3600]
df['zscore'] = z_score(df['Data'])
new_df = df[df['zscore'].abs() < 3]
#%%
print(df['Data'].mean())
print(df['Data'].std())
#%%
for k in range(200, 227):
    signal = new_df.loc[new_df['Time'].between(k, k+1)]['zscore'].values
    print(signal.shape)
    fft_results = np.fft.rfft(signal)
    # fft_results[60] = 0.0
    plt.plot(np.abs(fft_results), label=f'{k}-{k+1}')
    print(f'{k}-{k+1} segment: max freq at {np.argmax(np.abs(fft_results))} Hz')
    print(np.argmax(np.abs(fft_results)))
# %%
feat = 'EMG'
act_id = 2
act = actitivties[act_id]
subject_id = subjects[0]

file_path = os.path.join(root, f'Subject_{subject_id}_cleaned', f'{act_id + 1}_{act} {feat}.xlsx')
dodf = pd.read_excel(file_path, sheet_name=None)
df = pd.concat(dodf.values(), ignore_index=True)
df = df[df['Time (Seconds)'] < 3600]
df['sub_mean'] = sub_mean(df['ECG Raw Data' if feat == 'EMG' else 'Data'])
# %%
df['zscore'] = z_score(df['ECG Raw Data' if feat == 'EMG' else 'Data'])
# %%
df[df['Time (Seconds)'] < 1].plot(x='Time (Seconds)', y='sub_mean')
# %%
for k in range(225, 300):
    signal = df.loc[df['Time (Seconds)'].between(k, k+1)]['sub_mean'].values
    fft_results = np.fft.rfft(signal)
    # fft_results[60] = 0.0
    plt.plot(np.abs(fft_results), label=f'{k}-{k+1}')
    print(f'{k}-{k+1} segment: max freq at {np.argmax(np.abs(fft_results))} Hz')
    print(np.argmax(np.abs(fft_results)))
# plt.legend()
plt.title('Frequency domain')
plt.ylabel('Amplitude')
plt.show()
#%%
np.abs(fft_results)[59:62]

#%%
filtered_signal = bandpass_filter(signal, 10, 20, 500)
filtered_fft_results = np.fft.fft(filtered_signal)
filtered_fft_freq = np.fft.fftfreq(len(filtered_signal), 1 / len(filtered_signal))
plt.plot(np.abs(filtered_fft_results))
plt.title('Frequency domain')
plt.ylabel('Amplitude')
plt.show()
# %%
df['ECG Raw Data' if feat == 'EMG' else 'Data'].std()

# %%
# %%
def plot_dict(d, title):
    plt.figure(figsize=(20, 10))
    for k, v in d.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.savefig(f'figs/{title}.png')
    plt.close()

# act_id = 1
# subject_id = subjects[2]
# act = actitivties[act_id]
# feature = features[3]

df_list = []
for feature in ['EMG']:
    for act_id in range(4):
        zscore_dict = {}
        raw_dict = {}
        for subject_id in subjects:
            act = actitivties[act_id]
            file_path = os.path.join(root, f'Subject_{subject_id}_cleaned', f'{act_id + 1}_{act} {feature}.xlsx')
            print(file_path)
            dodf = pd.read_excel(file_path, sheet_name=None)
            df = pd.concat(dodf.values(), ignore_index=True)
            df = df[df['Time (Seconds)'] < 3600]
            print(df['Time (Seconds)'].max())
            # df['zscore'] = z_score(df['ECG Raw Data' if feature == 'EMG' else 'Data'])
            # zscore_dict[f'Subject-{subject_id}'] = df['zscore']
            # raw_dict[f'Subject-{subject_id}'] = sub_mean(df['ECG Raw Data' if feature == 'EMG' else 'Data'])
        # plot_dict(zscore_dict, f'{act} {feature} zscore')
        # plot_dict(raw_dict, f'{act} {feature} raw - mean calibrated')