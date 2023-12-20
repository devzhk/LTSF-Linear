#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
img_dir = 'figs'
os.makedirs(img_dir, exist_ok=True)
for activity in ['Biking', 'VR', 'Hand grip', 'Stroop']:
    for train_size in [1, 2, 4]:
        dir_path = f'Train{train_size}-selfnorm_Transformer_custom_ftMS_sl64_ll4_pl64_dm512_nh8_el2_dl2_df2048_fc1_ebtimeF_dtTrue_results_0'
        pred_path = os.path.join('results', dir_path, 'predicts',f'Subject_5-{activity}.npz')
        preds = np.load(pred_path)['preds']
        plt.plot(np.reshape(preds, (-1,)), label=f'Prediction-with{train_size}')
    ground_truth = np.load(pred_path)['ground_truths']
    plt.plot(np.reshape(ground_truth, (-1,)), label='Ground truth')
    plt.legend()
    plt.title(f'Subject5-{activity}')
    plt.savefig(os.path.join(img_dir, f'Test-Subject_5-{activity}.png'))
    plt.close()


# %%
