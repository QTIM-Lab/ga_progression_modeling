import os, sys
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

def deviation(aff):
    return np.linalg.norm(aff[:2, :2] - np.eye(2), 'fro')

def is_good_registration(grid_path):

    # Load the sampling grid
    params, _ = torch.load(grid_path, weights_only=False)

    # get the affine component
    affine = params[0, -3:, :].T.numpy() # (2, 3)
    affine = np.concatenate([affine[:, 1:], affine[:, :1]], axis=1)
    affine = np.concatenate([affine, np.array([[0, 0, 1]])], axis=0)
    affine = np.linalg.inv(affine)

    if deviation(affine) < 0.1:
        return True
    else:
        return False

file = 'results/11142024_coris/registration_results_af/results.csv'
save = 'results/11142024_coris/registration_results_af/results_good_regs.csv'
df = pd.read_csv(file)
# remove all paths where registration failed
df = df[df.status != 'Fail']

bad_regs = []
for i, row in df.iterrows():

    # skip first visits
    if isinstance(row.status, float) and isinstance(row.params, float):
        continue

    if not is_good_registration(row.params):
        bad_regs.append(i)

# drop rows with bad regs
df = df.drop(bad_regs)

# save csv
df.to_csv(save, index=False)