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

def load_timg(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor."""
    assert os.path.isfile(file_name), f"Invalid file {file_name}"  # nosec
    # load image with OpenCV
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    # convert image to torch tensor
    tensor = K.image_to_tensor(img, None).float() / 255.0
    return K.color.bgr_to_rgb(tensor)

def register_image(moving_image, reference_image, resize_shape, large_shape, model=None, method='eyeliner', device='cpu'):
    """Register moving image to the reference image and return the warped image in large_shape."""
    
    # use eyeliner API
    if method == 'eyeliner':

        # Resize and move images to GPU if available
        img1 = KG.transform.resize(load_timg(moving_image).to(device), resize_shape)
        img2 = KG.transform.resize(load_timg(reference_image).to(device), resize_shape)

        if model is None:
            # Load EyeLiner API for registration
            eyeliner = EyeLinerP(
                reg='affine', # registration technique to use (tps or affine)
                lambda_tps=1.0, # set lambda value for tps
                image_size=(3, 256, 256), # image dimensions
                device=device
                )

            # store inputs
            data = {
            'fixed_input': img2, #.unsqueeze(0),
            'moving_input': img1 #.unsqueeze(0)
            }
            print(data['fixed_input'].shape, data['moving_input'].shape)

            # compute registration
            theta, _ = eyeliner(data)

        # apply registration to image and mask
        try:
            img1_warped = eyeliner.apply_transform(theta[1].squeeze(0), img1.squeeze(0))
        except:
            img1_warped = eyeliner.apply_transform(theta.squeeze(0), img1.squeeze(0))
        img1_warped =  KG.transform.resize(img1_warped.to(device), large_shape)

    # use kornia's registration API to register images using intensity based metric
    else:
        # Resize and move images to GPU if available
        img1 = KG.transform.resize(load_timg(moving_image).to(device), resize_shape)
        img2 = KG.transform.resize(load_timg(reference_image).to(device), resize_shape)

        # Perform registration on GPU
        if model is None:
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

            model, intermediate = registrator.register(img1, img2, output_intermediate_models=True)

            with torch.no_grad():
                # Warp image using the computed transformation model
                img1_warped = KG.homography_warp(img1, model, img2.shape[-2:])
                img1_warped = KG.transform.resize(img1_warped, large_shape)
            return img1_warped, model
        else:
            with torch.no_grad():
                # Warp image using the computed transformation model
                img1_warped = KG.homography_warp(img1, model, img2.shape[-2:])
                img1_warped = KG.transform.resize(img1_warped, large_shape)
            return img1_warped

def main(df, atlas, method, save_folder, use_cuda, patients=None):
    # Set device based on whether CUDA is enabled
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    resize_shape = (256, 256)
    large_shape = (1536, 1536)

    # Register all OD images to the reference
    area_threshold = 0.0
    for mrn, mrn_df in df.groupby('PID'):
        for lat, lat_df in mrn_df.groupby('Laterality'):
            # only keep the latest exam date
            # TODO: uncomment this
            lat_df = lat_df[lat_df.mm_area > area_threshold]
            if len(lat_df) == 0:
                continue
            # lat_df = lat_df[lat_df.ExamDate == lat_df.ExamDate.min()]
            lat_df = lat_df[lat_df.ExamDate == lat_df.ExamDate.max()]
            for date, date_df in lat_df.groupby('ExamDate'):
                area = date_df.mm_area.unique()[0]
                if area > area_threshold:
                    for i, row in date_df.iterrows():
                        img_warped_large, model = register_image(row.file_path_coris, atlas, resize_shape, large_shape, method=method, device=device)
                        seg_warped_large = register_image(row.file_path_ga_seg, atlas, resize_shape, large_shape, model=model, method=method, device=device)
                        vessel_warped_large = register_image(row.file_path_vessel_seg, atlas, resize_shape, large_shape, model=model, method=method, device=device)

                        # save images
                        save_image = os.path.join(save_folder, 'images_registered/', os.path.basename(row.file_path_coris).replace('.j2k', '.png'))
                        save_ga = os.path.join(save_folder, 'ga_registered/', os.path.basename(row.file_path_coris).replace('.j2k', '.png'))
                        save_vessel = os.path.join(save_folder, 'vessels_registered/', os.path.basename(row.file_path_coris).replace('.j2k', '.png'))

                        # Save the reference image for each iteration (convert to CPU for saving)
                        ToPILImage()(img_warped_large.squeeze(0).cpu()).save(save_image)
                        ToPILImage()(seg_warped_large.squeeze(0).cpu()).save(save_ga)
                        ToPILImage()(vessel_warped_large.squeeze(0).cpu()).save(save_vessel)
                        break          
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/GA_progression_modelling_data_redone/clean_data_talisa_08302024.csv')
    parser.add_argument('--lat', default='OD')
    parser.add_argument('--atlas', default='average_OD_image.png')
    parser.add_argument('--method', default=None)
    parser.add_argument('--save-to', default='reg2atlas')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--specific-pat', default='specific_patients.txt')
    args = parser.parse_args()

    # get 46 patients
    with open(args.specific_pat, 'r') as f:
        select_patients = f.read().splitlines()
        select_patients = [(int(pat.split('_')[0]), pat.split('_')[1]) for pat in select_patients]

    # get AF images
    df = pd.read_csv(args.csv)
    df = df[(df.Procedure == 'Af') & (df.Laterality == args.lat)]
    df_subset = df[df[['PID', 'Laterality']].apply(tuple, axis=1).isin(select_patients)]

    # run iterative atlas generation
    main(df_subset, atlas=args.atlas, method=args.method, save_folder=args.save_to, use_cuda=args.use_cuda, patients=select_patients)