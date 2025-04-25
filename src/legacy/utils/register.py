import os, sys
sys.path.append('/sddata/projects/LightGlue/src/')

import warnings
warnings.filterwarnings("ignore")
import random
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image

from eyeliner import EyeLinerP
from eyeliner.utils import load_image

device = 'cpu'

# Load EyeLiner API for registration
eyelinerp = EyeLinerP(
    reg='affine', # registration technique to use (tps or affine)
    lambda_tps=1.0, # set lambda value for tps
    image_size=(3, 256, 256), # image dimensions
    device=device
    )

def tensor2numpy(tensor):
    return torch.permute(tensor, (1, 2, 0)).numpy()

def draw_lines_on_image(image, lines, color=(255, 0, 0), thickness=2):
    """
    Draws lines on a NumPy image using OpenCV.

    Parameters:
        image (np.ndarray): Input image of shape (H, W, C) where C is the number of channels.
        lines (np.ndarray): Array of lines with each row as [start_x, start_y, end_x, end_y].
        color (tuple): Line color as (B, G, R). Defaults to red (255, 0, 0).
        thickness (int): Thickness of the lines. Defaults to 2.

    Returns:
        np.ndarray: The image with the lines drawn.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("image must be a NumPy array of shape (H, W, C).")
    
    if not isinstance(lines, np.ndarray) or lines.shape[1] != 4:
        raise ValueError("lines must be a NumPy array with shape (N, 4).")
    
    # Create a copy of the image to draw on
    output_image = image.copy()
    
    # Draw each line on the image
    for start_x, start_y, end_x, end_y in lines:
        start_point = (int(start_x), int(start_y))
        end_point = (int(end_x), int(end_y))
        cv2.line(output_image, start_point, end_point, color, thickness)
    
    return output_image

def register(af_df, slo_df, oct_df):

    w, h = Image.open(af_df.file_path_coris).size
    w_slo, h_slo = Image.open(slo_df.file_path_coris.item()).size

    # load each image
    slo_image = load_image(slo_df.file_path_coris.item(), size=(256, 256), mode='rgb').to(device) # (256, 256), [0, 1]
    # slo_ga_seg = load_image(slo_df.file_path_ga_seg.item(), size=(256, 256), mode='rgb').to(device) # (256, 256), [0, 1]

    faf_image = load_image(af_df.file_path_coris, size=(256, 256), mode='rgb').to(device) # (256, 256), [0, 1]
    faf_ga_seg = load_image(af_df.file_path_ga_seg, size=(256, 256), mode='rgb').to(device) # (256, 256), [0, 1]

    # load oct scanlines
    scanlines = torch.from_numpy(oct_df[['Start_X', 'Start_Y', 'End_X', 'End_Y']].values / slo_df.Scale_X.item()).float() # (1536, 1536)
    scanlines = 256. * scanlines / w_slo # (256, 256) 

    # store inputs
    data = {
    'fixed_input': faf_image,
    'moving_input': slo_image
    }

    # compute registration
    A, _ = eyelinerp(data)

    # apply registration to image and mask
    reg_slo_image = eyelinerp.apply_transform(A.squeeze(0), slo_image.squeeze(0)) # (256, 256), [0, 1] 
    # reg_slo_ga_seg = eyelinerp.apply_transform(A.squeeze(0), slo_ga_seg.squeeze(0)) # (256, 256), [0, 1]

    # apply registration to the scanlines
    reg_scanlines_start = eyelinerp.apply_transform_points(A.squeeze(0), scanlines[:, :2]) 
    reg_scanlines_end = eyelinerp.apply_transform_points(A.squeeze(0), scanlines[:, 2:])
    reg_scanlines = torch.cat([reg_scanlines_start, reg_scanlines_end], dim=-1) # (256, 256)
    reg_scanlines = w_slo * reg_scanlines / 256. # (1536, 1536)
    scanlines = w_slo * scanlines / 256. # (1536, 1536)
 
    # convert to numpy 
    faf_image = tensor2numpy(faf_image.cpu().squeeze(0) * 255).astype(np.uint8)
    faf_ga_seg = tensor2numpy(faf_ga_seg.cpu().squeeze(0) * 255).astype(np.uint8)
    slo_image = tensor2numpy(slo_image.cpu().squeeze(0) * 255).astype(np.uint8)
    # slo_ga_seg = tensor2numpy(slo_ga_seg.cpu().squeeze(0) * 255).astype(np.uint8)
    reg_slo_image = tensor2numpy(reg_slo_image.cpu() * 255).astype(np.uint8)
    # reg_slo_ga_seg = tensor2numpy(reg_slo_ga_seg.cpu() * 255).astype(np.uint8)

    # resize registered image to the original size of fixed image
    faf_image = cv2.resize(faf_image, (h, w), interpolation=cv2.INTER_CUBIC) # (1536, 1536), [0, 1]
    faf_ga_seg = cv2.resize(faf_ga_seg, (h, w), interpolation=cv2.INTER_CUBIC) # (1536, 1536), [0, 1]
    slo_image = cv2.resize(slo_image, (h, w), interpolation=cv2.INTER_CUBIC) # (1536, 1536), [0, 1]
    # slo_ga_seg = cv2.resize(slo_ga_seg, (h, w), interpolation=cv2.INTER_CUBIC) # (1536, 1536), [0, 1]
    reg_slo_image = cv2.resize(reg_slo_image, (h, w), interpolation=cv2.INTER_CUBIC) # (1536, 1536), [0, 1]
    # reg_slo_ga_seg = cv2.resize(reg_slo_ga_seg, (h, w), interpolation=cv2.INTER_CUBIC) # (1536, 1536), [0, 1]

    # cache - images for display
    slo_image_w_scanlines = draw_lines_on_image(slo_image, scanlines.numpy())
    faf_image_w_scanlines = draw_lines_on_image(faf_image, reg_scanlines.numpy())
    reg_slo_image_w_scanlines = draw_lines_on_image(reg_slo_image, reg_scanlines.numpy())

    cache = {
        'faf_image': faf_image,
        'faf_ga_seg': faf_ga_seg,
        'faf_image_w_scanlines': faf_image_w_scanlines,
        'slo_image_w_scanlines': slo_image_w_scanlines,
        'reg_slo_image_w_scanlines': reg_slo_image_w_scanlines,
    }
    # return reg_slo_image, reg_slo_ga_seg, reg_scanlines, cache
    return reg_slo_image, reg_scanlines, cache

def copyimage(src, dst):
    # Split the destination path into directory, base filename, and extension
    directory, filename = os.path.split(dst)
    basename, ext = os.path.splitext(filename)
    
    # Initialize counter
    counter = 1
    new_dst = dst
    
    # Keep incrementing counter until we find a filename that doesn't exist
    while os.path.exists(new_dst):
        new_dst = os.path.join(directory, f"{basename}_{counter}{ext}")
        counter += 1
    
    # Save the image to the unique destination path
    Image.open(src).save(new_dst)
    return new_dst

def save_image(img, dst):
    # Split the destination path into directory, base filename, and extension
    directory, filename = os.path.split(dst)
    basename, ext = os.path.splitext(filename)
    
    # Initialize counter
    counter = 1
    new_dst = dst
    
    # Keep incrementing counter until we find a filename that doesn't exist
    while os.path.exists(new_dst):
        new_dst = os.path.join(directory, f"{basename}_{counter}{ext}")
        counter += 1
    
    # Save the image array to the unique destination path
    Image.fromarray(img).save(new_dst)
    return new_dst

def main():
    df = pd.read_csv('data/GA_progression_modelling_data_redone/data_csvs/clean_data_talisa_ga_10312024.csv')

    df_results = {
        'image_path_orig': [],
        'seg_path': [],
        'pid': [],
        'lat': [],
        'examdate': [],
        'group': [],
        'image_type': [],
        'start_x': [],
        'start_y': [],
        'end_x': [],
        'end_y': [],
        'order': []
    }

    os.makedirs('oct_2_af_registeration/', exist_ok=True)
    os.makedirs('oct_2_af_registeration/faf_images/', exist_ok=True)
    os.makedirs('oct_2_af_registeration/faf_segs/', exist_ok=True)
    os.makedirs('oct_2_af_registeration/slo_images/', exist_ok=True)
    # os.makedirs('oct_2_af_registeration/slo_segs/', exist_ok=True)
    os.makedirs('oct_2_af_registeration/oct_images/', exist_ok=True)
    os.makedirs('oct_2_af_registeration/summary/', exist_ok=True)

    specific_ids = [(6013723, 'OS', '03/02/2022'), (88395, 'OD', '10/16/2019'), (1264366, 'OS', '11/10/2022'), (169896, 'OD', '10/12/2020'), (1966666, 'OD', '01/23/2017')]

    # A. MRN: 6013723 | Laterality: OS | Date: 03/02/2022 | AF: 1 | IR: 0 | OCT: 0 | Has duplicates: False
    # A. MRN: 88395 | Laterality: OD | Date: 10/16/2019 | AF: 2 | IR: 1 | OCT: 19 | Has duplicates: False
    # A. MRN: 88395 | Laterality: OD | Date: 10/16/2019 | AF: 2 | IR: 1 | OCT: 49 | Has duplicates: False
    # A. MRN: 1264366 | Laterality: OS | Date: 11/10/2022 | AF: 0 | IR: 1 | OCT: 49 | Has duplicates: False
    # A. MRN: 169896 | Laterality: OD | Date: 10/12/2020 | AF: 1 | IR: 1 | OCT: 19 | Has duplicates: False
    # A. MRN: 1966666 | Laterality: OD | Date: 01/23/2017 | AF: 3 | IR: 1 | OCT: 19 | Has duplicates: False

    for mrn, mrn_df in df.groupby('PID'):
        for lat, lat_df in mrn_df.groupby('Laterality'):
            for date, date_df in lat_df.groupby('ExamDate'):

                ### Comment: Take specific patients (for demo purposes)
                # if (mrn, lat, date) in specific_ids:
                #     pass
                # else:
                #     continue

                print(f'Processing MRN: {mrn}, Lat: {lat}, Date: {date}')

                df_af = date_df[date_df.Procedure == 'Af']
                # skip patients without AF
                if len(df_af) == 0:
                    continue

                df_ir_oct = date_df[date_df.Procedure == 'Ir_oct']

                if len(df_ir_oct) > 0:
                    # if patient has multiple OCTs, pick one that is 49 raster if available, other pick whatever is available
                    df_ir_oct_ = df_ir_oct[df_ir_oct.type == 'BScan']
                    group_sizes = {group: len(group_df) for group, group_df in df_ir_oct_.groupby('ImageGroup')}
                    selected_group = random.choice([key for key, value in group_sizes.items() if value == 49]) if any(value == 49 for value in group_sizes.values()) else random.choice(list(group_sizes))
                    df_ir_oct = df_ir_oct[df_ir_oct['ImageGroup'] == selected_group]

                for i, af_group_df in df_af.iterrows():

                    # get ir and oct image
                    df_ir = df_ir_oct[df_ir_oct.type == 'SLOImage']
                    df_oct = df_ir_oct[df_ir_oct.type == 'BScan'].sort_values('ImageNumber')
                    oct_group = df_ir_oct.ImageGroup.iloc[0] if len(df_ir_oct) > 0 else None

                    # save faf image, seg, and oct bscans
                    faf_img_save_as = 'oct_2_af_registeration/faf_images/' + os.path.basename(af_group_df.file_path_coris).replace('.j2k', '.png')
                    faf_img_save_as = copyimage(af_group_df.file_path_coris, faf_img_save_as)

                    faf_seg_save_as = 'oct_2_af_registeration/faf_segs/' + os.path.basename(af_group_df.file_path_ga_seg).replace('.j2k', '.png')
                    faf_seg_save_as = copyimage(af_group_df.file_path_ga_seg, faf_seg_save_as)

                    if len(df_oct) > 0:
                        oct_img_save_as = []
                        for i, row in df_oct.iterrows():
                            oct_img_save_as_ = 'oct_2_af_registeration/oct_images/' + os.path.basename(row.file_path_coris).replace('.j2k', '.png')
                            oct_img_save_as_ = copyimage(row.file_path_coris, oct_img_save_as_)
                            oct_img_save_as.append(oct_img_save_as_)                  

                    # register Af and its segmentation to Ir
                    try:
                        # reg_slo_image, reg_slo_ga_seg, reg_scanlines, cache = register(af_group_df, df_ir, df_oct)
                        reg_slo_image, reg_scanlines, cache = register(af_group_df, df_ir, df_oct)
                    
                        # visualize registrations
                        plt.figure(figsize=(20, 6))
                        plt.subplot(1,5,1)
                        plt.imshow(cache['faf_image'], plt.cm.gray)
                        plt.subplot(1,5,2)
                        plt.imshow(cache['slo_image_w_scanlines'], plt.cm.gray)
                        plt.subplot(1,5,3)
                        plt.imshow(cache['reg_slo_image_w_scanlines'], plt.cm.gray)
                        plt.subplot(1,5,4)
                        plt.imshow(cache['faf_image'], plt.cm.gray, alpha=0.5)
                        plt.imshow(reg_slo_image, plt.cm.gray, alpha=0.5)
                        plt.subplot(1,5,5)
                        plt.imshow(cache['faf_image_w_scanlines'], plt.cm.gray)
                        plt.savefig(f'oct_2_af_registeration/summary/af{i}_{oct_group}.png')

                        # save image to dataframe and folder
                        reg_slo_img_save_as = 'oct_2_af_registeration/slo_images/reg_' + os.path.basename(df_ir.file_path_coris.item()).replace('.j2k', '.png')
                        reg_slo_img_save_as = save_image(reg_slo_image, reg_slo_img_save_as)
                        # Image.fromarray(reg_slo_image).save(reg_slo_img_save_as)

                        # reg_slo_seg_save_as = 'oct_2_af_registeration/slo_segs/reg_' + os.path.basename(df_ir.file_path_coris.item()).replace('.j2k', '.png')
                        # reg_slo_seg_save_as = save_image(reg_slo_ga_seg, reg_slo_seg_save_as)
                        # Image.fromarray(reg_slo_ga_seg).save(reg_slo_seg_save_as)

                        # Add FAF image data
                        df_results['pid'].append(mrn)
                        df_results['lat'].append(lat)
                        df_results['examdate'].append(date)
                        df_results['image_path_orig'].append(faf_img_save_as)
                        df_results['seg_path'].append(faf_seg_save_as)
                        df_results['group'].append(oct_group)
                        df_results['image_type'].append('FAF')
                        df_results['start_x'].append(None)
                        df_results['start_y'].append(None)
                        df_results['end_x'].append(None)
                        df_results['end_y'].append(None)
                        df_results['order'].append(0)

                        # Add IR image data
                        df_results['pid'].append(mrn)
                        df_results['lat'].append(lat)
                        df_results['examdate'].append(date)
                        df_results['image_path_orig'].append(reg_slo_img_save_as)
                        # df_results['seg_path'].append(reg_slo_seg_save_as)
                        df_results['seg_path'].append(None)
                        df_results['group'].append(oct_group)
                        df_results['image_type'].append('SLOImage')
                        df_results['start_x'].append(None)
                        df_results['start_y'].append(None)
                        df_results['end_x'].append(None)
                        df_results['end_y'].append(None)
                        df_results['order'].append(0)

                        # Add OCT bscan data
                        df_results['pid'] += [mrn]*len(df_oct)
                        df_results['lat'] += [lat]*len(df_oct)
                        df_results['examdate'] += [date]*len(df_oct)
                        df_results['image_path_orig'] += oct_img_save_as
                        df_results['seg_path'] += [None]*len(df_oct)
                        df_results['group'] += [oct_group]*len(df_oct)
                        df_results['image_type'] += ['BScan']*len(df_oct)
                        df_results['start_x'] += reg_scanlines[:, 0].tolist()
                        df_results['start_y'] += reg_scanlines[:, 1].tolist()
                        df_results['end_x'] += reg_scanlines[:, 2].tolist()
                        df_results['end_y'] += reg_scanlines[:, 3].tolist()
                        df_results['order'] += list(range(1, len(df_oct)+1))
                    
                    except Exception as error:
                        print(error)

                        # skip if registration doesn't work
                        print('Cannot register')
                        # Add FAF image data
                        df_results['pid'].append(mrn)
                        df_results['lat'].append(lat)
                        df_results['examdate'].append(date)
                        df_results['image_path_orig'].append(faf_img_save_as)
                        df_results['seg_path'].append(faf_seg_save_as)
                        df_results['group'].append(oct_group)
                        df_results['image_type'].append('FAF')
                        df_results['start_x'].append(None)
                        df_results['start_y'].append(None)
                        df_results['end_x'].append(None)
                        df_results['end_y'].append(None)
                        df_results['order'].append(0)

                        if len(df_ir_oct) > 0:
                            # Add IR image data
                            df_results['pid'].append(mrn)
                            df_results['lat'].append(lat)
                            df_results['examdate'].append(date)
                            
                            reg_slo_img_save_as = 'oct_2_af_registeration/slo_images/' + os.path.basename(df_ir.file_path_coris.item()).replace('.j2k', '.png')
                            reg_slo_img_save_as = copyimage(df_ir.file_path_coris.item(), reg_slo_img_save_as)
                            df_results['image_path_orig'].append(reg_slo_img_save_as)

                            # reg_slo_seg_save_as = 'oct_2_af_registeration/slo_segs/' + os.path.basename(df_ir.file_path_ga_seg.item()).replace('.j2k', '.png')
                            # reg_slo_seg_save_as = copyimage(df_ir.file_path_ga_seg.item(), reg_slo_seg_save_as)
                            # df_results['seg_path'].append(reg_slo_seg_save_as)
                            df_results['seg_path'].append(None)

                            df_results['group'].append(oct_group)
                            df_results['image_type'].append('SLOImage')
                            df_results['start_x'].append(None)
                            df_results['start_y'].append(None)
                            df_results['end_x'].append(None)
                            df_results['end_y'].append(None)
                            df_results['order'].append(0)

                            # Add OCT bscan data
                            df_results['pid'] += [mrn]*len(df_oct)
                            df_results['lat'] += [lat]*len(df_oct)
                            df_results['examdate'] += [date]*len(df_oct)
                            df_results['image_path_orig'] += oct_img_save_as
                            df_results['seg_path'] += [None]*len(df_oct)
                            df_results['group'] += [oct_group]*len(df_oct)
                            df_results['image_type'] += ['BScan']*len(df_oct)
                            df_results['start_x'] += (df_oct.Start_X.values / df_ir.Scale_X.item()).tolist()
                            df_results['start_y'] += (df_oct.Start_Y.values / df_ir.Scale_X.item()).tolist()
                            df_results['end_x'] += (df_oct.End_X.values / df_ir.Scale_X.item()).tolist()
                            df_results['end_y'] += (df_oct.End_Y.values / df_ir.Scale_X.item()).tolist()
                            df_results['order'] += list(range(1, len(df_oct)+1))  

    df_results = pd.DataFrame(df_results)
    df_results.to_csv('oct_2_af_registeration/image_key.csv', index=False)
    return

if __name__ == '__main__':
    main()