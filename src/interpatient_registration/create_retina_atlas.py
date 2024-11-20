import os, sys
sys.path.append('/sddata/projects/LightGlue')
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import cv2
import os, sys, argparse
import random
import torch
from PIL import Image
import kornia as K
import kornia.geometry as KG
import matplotlib.pyplot as plt
from monai.losses import GlobalMutualInformationLoss
from torchvision.transforms import ToPILImage
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.eyeliner import EyeLinerP

def gmi_loss(pred, target, reduction=None):
    return GlobalMutualInformationLoss()(pred, target)

def load_timg(file_name, device='cpu'):
    """Loads the image with OpenCV and converts it to a torch.Tensor."""
    assert os.path.isfile(file_name), f"Invalid file {file_name}"  # nosec
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    tensor = K.image_to_tensor(img, None).float() / 255.0
    return K.color.bgr_to_rgb(tensor).to(device)

def register_image(moving_image, reference_image, resize_shape, large_shape, method='eyeliner', device='cpu'):
    """Register moving image to the reference image and return the warped image in large_shape."""
    
    # use eyeliner API
    if method == 'eyeliner':
        # Load EyeLiner API for registration
        eyeliner = EyeLinerP(
            reg='affine', # registration technique to use (tps or affine)
            lambda_tps=1.0, # set lambda value for tps
            image_size=(3, 256, 256), # image dimensions
            device=device
            )

        # Resize and move images to GPU if available
        img1 = KG.transform.resize(load_timg(moving_image, device=device), resize_shape)
        img2 = KG.transform.resize(reference_image.to(device), resize_shape)

        # store inputs
        data = {
        'fixed_input': img2,
        'moving_input': img1,
        }

        # compute registration
        theta, _ = eyeliner(data)
        
        # apply registration to image and mask
        try:
            img1_warped = eyeliner.apply_transform(theta[1].squeeze(0), img1.cpu().squeeze(0))
        except:
            img1_warped = eyeliner.apply_transform(theta.squeeze(0), img1.cpu().squeeze(0))
        img1_warped =  KG.transform.resize(img1_warped.to(device), large_shape).unsqueeze(0)

    # use kornia's registration API to register images using intensity based metric
    else:
        # Initialize the image registrator on the selected device
        registrator = KG.ImageRegistrator(
            model_type="similarity", 
            optimizer=torch.optim.Adam, 
            loss_fn=gmi_loss, 
            pyramid_levels=5, 
            lr=1e-5, 
            num_iterations=100, 
            tolerance=1e-4, 
            warper=None
        ).to(device)

        # Resize and move images to GPU if available
        img1 = KG.transform.resize(load_timg(moving_image, device=device), resize_shape)
        img2 = KG.transform.resize(reference_image.to(device), resize_shape)

        # Perform registration on GPU
        model, intermediate = registrator.register(img1, img2, output_intermediate_models=True)

        with torch.no_grad():
            # Warp image using the computed transformation model
            img1_warped = KG.homography_warp(img1, model, img2.shape[-2:])
            img1_warped = KG.transform.resize(img1_warped, large_shape)
        
    return img1_warped

def main(df_lat, method='eyeliner', n_iterations=5, save_folder='.', use_cuda=False):
    # Set device based on whether CUDA is enabled
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    resize_shape = (256, 256)
    large_shape = (1536, 1536)

    # add weightage factor to weight the average image
    df_lat['pat_weight'] = df_lat.groupby('PID')['file_path_coris'].transform('count')
    df_lat['pat_weight'] = 1 / (df_lat['PID'].nunique() * df_lat['pat_weight'])

    for i in range(n_iterations):

        # just use the average image for the i=0 reference
        if i == 0:
            avg_image = 0
            for j, row in df_lat.iterrows():
                avg_image += KG.transform.resize(load_timg(row.file_path_coris, device), large_shape) * row.pat_weight
            reference_image = avg_image.clone()

        # register all images to i-1th reference
        else:
            avg_image = 0
            for j, row in df_lat.iterrows():
                # Perform registration on GPU if available
                try:
                    img1_warped_large = register_image(row.file_path_coris, reference_image, resize_shape, large_shape, method=method, device=device)
                except Exception as e:
                    img1_warped_large = reference_image
                    print(f'Error found: {e}')
                avg_image += img1_warped_large

            avg_image = avg_image / len(df_lat)
            reference_image = avg_image.clone()

        # Save the reference image for each iteration (convert to CPU for saving)
        ToPILImage()(reference_image.squeeze(0).cpu()).save(os.path.join(save_folder, f'reference_{i}.png'))

    return

if __name__ == '__main__':
    import argparse
    import pandas as pd
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/GA_progression_modelling_data_redone/clean_data_talisa_08302024.csv')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--lat', default='OD')
    parser.add_argument('--method', default='eyeliner')
    parser.add_argument('--save-to', default='atlas')
    parser.add_argument('--use-cuda', action='store_true')
    args = parser.parse_args()

    # get AF images
    df = pd.read_csv(args.csv)
    df_lat = df[(df.Procedure == 'Af') & (df.Laterality == args.lat)] # only working with af images now

    # run iterative atlas generation
    main(df_lat, method=args.method, n_iterations=args.n, save_folder=args.save_to, use_cuda=args.use_cuda)
