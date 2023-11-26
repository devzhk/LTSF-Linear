#%%
import os
import numpy as np
import pandas as pd


#%%
subject_id = 1
activity = 'HandGrip'
pred_dir = os.path.join('results', f'sub{subject_id}-{activity.lower()}_Transformer_custom_ftMS_sl64_ll4_pl64_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_results_0')
pred_path = os.path.join(pred_dir, 'real_pred.npz')

data = np.load(pred_path)
preds = data['preds']
ground_truths = data['ground_truths']
# %%
import matplotlib.pyplot as plt
plt.plot(np.reshape(preds, (-1,)), label='Prediction')
plt.plot(np.reshape(ground_truths, (-1,)), label='Ground truth')
plt.legend()
plt.title(f'Subject{subject_id}-{activity}')
plt.savefig(f'prediction-sub{subject_id}-{activity}.png')
# %%

# %%
print(preds.shape, ground_truths.shape)
# %%
