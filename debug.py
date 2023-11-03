#%%
import os
import numpy as np
import pandas as pd

#%%
pred_dir = os.path.join('results', 'Sub1-Biking_Transformer_custom_ftMS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_results_0')
pred_path = os.path.join(pred_dir, 'real_prediction.npy')
data = np.load(pred_path)
raw = pd.read_csv('../fatigue/data/Subject_1-clean-Biking.csv')
label = raw['Fatigue level']
# %%
import matplotlib.pyplot as plt
plt.plot(np.reshape(data, (-1,)), label='prediction')
plt.plot(label, label='Truth')
plt.legend()
plt.savefig('test.png')
# %%
