#%%
import os
import numpy as np
import pandas as pd

#%%
pred_dir = os.path.join('results', 'sub1-biking_Transformer_custom_ftMS_sl64_ll4_pl64_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_results_0')
pred_path = os.path.join(pred_dir, 'real_pred.npz')

data = np.load(pred_path)
preds = data['preds']
ground_truths = data['ground_truths']
# %%
import matplotlib.pyplot as plt
plt.plot(np.reshape(preds, (-1,)), label='prediction')
plt.plot(np.reshape(ground_truths, (-1,)), label='Truth')
plt.legend()
plt.savefig('prediction.png')
# %%

# %%
print(preds.shape, ground_truths.shape)
# %%
