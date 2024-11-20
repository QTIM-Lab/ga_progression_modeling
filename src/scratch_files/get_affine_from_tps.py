import os, sys
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Resize, Grayscale, ToTensor
import cv2
from matplotlib import pyplot as plt

def load_image(path, size=(256, 256), mode='rgb'):
    x = Image.open(path)
    x = Resize(size)(x)
    x = Grayscale()(x) if mode == 'gray' else x
    x = ToTensor()(x)
    return x.unsqueeze(0)

df = pd.read_csv('results/09172024_coris/registration_results_af.csv')
df = df[(df.PID == 1480306) & (df.Laterality == 'OD')].reset_index(drop=True)
# df = df[(df.PID == 264825) & (df.Laterality == 'OS')].reset_index(drop=True)

T = 17
for t in range(T):
    baseline = load_image(df.iloc[0].file_path_coris).squeeze(0)
    visit1 = load_image(df.iloc[t].file_path_coris).squeeze(0)
    if isinstance(df.iloc[t].params, float):
        continue
    theta = torch.load(df.iloc[t].params, weights_only=False)
    affine = theta[0][0, -3:, :].T.numpy() # (2, 3)
    affine = np.concatenate([affine[:, 1:], affine[:, :1]], axis=1)
    affine = np.concatenate([affine, np.array([[0, 0, 1]])], axis=0)
    affine = np.linalg.inv(affine)
    deviation = np.linalg.norm(affine[:2, :2] - np.eye(2), 'fro')
    print(t)
    print(affine)
    print(deviation)

    visit1_image = torch.permute(visit1, (1, 2, 0)).numpy() # (c, h, w) -> (h, w, c)
    warped_image = cv2.warpAffine(visit1_image, affine[:2, :], (visit1_image.shape[0], visit1_image.shape[1]))
    if warped_image.ndim == 2: # adding extra dim for grayscale warp
        warped_image = warped_image[:, :, None]

    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(baseline.permute((1, 2, 0)))
    plt.subplot(132)
    plt.imshow(visit1_image)
    plt.subplot(133)
    plt.imshow(baseline.permute((1, 2, 0)), alpha=0.5)
    plt.imshow(warped_image, alpha=0.5)
    plt.savefig(f'test_{t}.png')