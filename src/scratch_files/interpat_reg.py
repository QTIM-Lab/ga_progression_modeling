import pandas as pd
import numpy as np
import cv2
import os
import imageio
import torch
from PIL import Image
import kornia as K
import kornia.geometry as KG
import matplotlib.pyplot as plt

# # get AF images
# df = pd.read_csv('data/GA_progression_modelling_data_redone/clean_data_talisa_08302024.csv')
# df_af_od = df[(df.Procedure == 'Af') & (df.Laterality == 'OD')]
# df_af_os = df[(df.Procedure == 'Af') & (df.Laterality == 'OS')]

# ==================
# Get Reference Eyes
# ==================

if not os.path.exists('average_OD_image.png'):
    # get average OD image
    avg_od_img = np.zeros((1536, 1536))
    for i, row in df_af_od.iterrows():
        img = cv2.imread(row.file_path_coris, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (1536, 1536))
        avg_od_img += img
    avg_od_img = avg_od_img / len(df_af_od)
    cv2.imwrite('average_OD_image.png', avg_od_img)
else:
    print('average_OD_image.png already exists')

if not os.path.exists('average_OS_image.png'):
    # get average OS image
    avg_os_img = np.zeros((1536, 1536))
    for i, row in df_af_os.iterrows():
        img = cv2.imread(row.file_path_coris, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (1536, 1536))
        avg_os_img += img
    avg_os_img = avg_os_img / len(df_af_os)
    cv2.imwrite('average_OS_image.png', avg_os_img)
else:
    print('average_OS_image.png already exists')

# ======================
# Register to References
# ======================

def load_timg(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor."""
    assert os.path.isfile(file_name), f"Invalid file {file_name}"  # nosec
    # load image with OpenCV
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    # convert image to torch tensor
    tensor = K.image_to_tensor(img, None).float() / 255.0
    return K.color.bgr_to_rgb(tensor)

# Register all OD images to the reference
for i, row in df_af_od.iterrows():

    save_image = os.path.join('OD_images/images_registered/', os.path.basename(row.file_path_coris).replace('.j2k', '.png'))
    save_ga = os.path.join('OD_images/ga_registered/', os.path.basename(row.file_path_coris).replace('.j2k', '.png'))
    save_vessel = os.path.join('OD_images/vessels_registered/', os.path.basename(row.file_path_coris).replace('.j2k', '.png'))

    if os.path.exists(save_image) and os.path.exists(save_ga) and os.path.exists(save_vessel):
        continue

    registrator = KG.ImageRegistrator("similarity")

    img1 = KG.transform.resize(load_timg(row.file_path_coris), (1536, 1536))
    msk1 = KG.transform.resize(load_timg(row.file_path_ga_seg), (1536, 1536))
    vessel1 = KG.transform.resize(load_timg(row.file_path_vessel_seg), (1536, 1536))
    img2 = KG.transform.resize(load_timg('average_OD_image.png'), (1536, 1536))
    model, intermediate = registrator.register(img1, img2, output_intermediate_models=True)

    with torch.no_grad():
        # save image
        img1_warped = KG.homography_warp(img1, model, img2.shape[-2:])
        img1_warped = K.tensor_to_image((img1_warped * 255.0).byte())
        # save mask
        msk1_warped = KG.homography_warp(msk1, model, img2.shape[-2:])
        msk1_warped = K.tensor_to_image((msk1_warped * 255.0).byte())
        # save vessel
        vessel1_warped = KG.homography_warp(vessel1, model, img2.shape[-2:])
        vessel1_warped = K.tensor_to_image((vessel1_warped * 255.0).byte())

        Image.fromarray(img1_warped).save(os.path.join('OD_images/images_registered', os.path.basename(row.file_path_coris).replace('.j2k', '.png')))
        Image.fromarray(msk1_warped).save(os.path.join('OD_images/ga_registered', os.path.basename(row.file_path_coris).replace('.j2k', '.png')))
        Image.fromarray(vessel1_warped).save(os.path.join('OD_images/vessels_registered', os.path.basename(row.file_path_coris).replace('.j2k', '.png')))

# Register all OS images to the reference
for i, row in df_af_os.iterrows():

    save_image = os.path.join('OS_images/images_registered/', os.path.basename(row.file_path_coris).replace('.j2k', '.png'))
    save_mask = os.path.join('OS_images/ga_registered/', os.path.basename(row.file_path_coris).replace('.j2k', '.png'))
    save_vessel = os.path.join('OS_images/vessels_registered/', os.path.basename(row.file_path_coris).replace('.j2k', '.png'))

    if os.path.exists(save_image) and os.path.exists(save_ga) and os.path.exists(save_vessel):
        continue

    registrator = KG.ImageRegistrator("similarity")

    img1 = KG.transform.resize(load_timg(row.file_path_coris), (1536, 1536))
    msk1 = KG.transform.resize(load_timg(row.file_path_ga_seg), (1536, 1536))
    vessel1 = KG.transform.resize(load_timg(row.file_path_vessel_seg), (1536, 1536))
    img2 = KG.transform.resize(load_timg('average_OS_image.png'), (1536, 1536))
    model, intermediate = registrator.register(img1, img2, output_intermediate_models=True)

    with torch.no_grad():
        # save image
        img1_warped = KG.homography_warp(img1, model, img2.shape[-2:])
        img1_warped = K.tensor_to_image((img1_warped * 255.0).byte())
        # save mask
        msk1_warped = KG.homography_warp(msk1, model, img2.shape[-2:])
        msk1_warped = K.tensor_to_image((msk1_warped * 255.0).byte())
        # save vessel
        vessel1_warped = KG.homography_warp(vessel1, model, img2.shape[-2:])
        vessel1_warped = K.tensor_to_image((vessel1_warped * 255.0).byte())
        
        Image.fromarray(img1_warped).save(os.path.join('OS_images/images_registered', os.path.basename(row.file_path_coris).replace('.j2k', '.png')))
        Image.fromarray(msk1_warped).save(os.path.join('OS_images/ga_masks_registered', os.path.basename(row.file_path_coris).replace('.j2k', '.png')))
        Image.fromarray(vessel1_warped).save(os.path.join('OS_images/vessels_registered', os.path.basename(row.file_path_coris).replace('.j2k', '.png')))

def main(images, atlas, method, save_folder, use_cuda):
    
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/GA_progression_modelling_data_redone/clean_data_talisa_08302024.csv')
    parser.add_argument('--lat', default='OD')
    parser.add_argument('--atlas', default='average_OD_image.png')
    parser.add_argument('--method', default='eyeliner')
    parser.add_argument('--save-to', default='reg2atlas')
    parser.add_argument('--use-cuda', action='store_true')
    args = parser.parse_args()

    # get AF images
    df = pd.read_csv(args.csv)
    images = df[(df.Procedure == 'Af') & (df.Laterality == args.lat)].file_path_coris.tolist()

    # run iterative atlas generation
    main(images, atlas=args.atlas, method=args.method, save_folder=args.save_to, use_cuda=args.use_cuda)