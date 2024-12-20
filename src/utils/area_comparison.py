# import libraries
import os, sys
import argparse 

import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from math import isnan
import torch
from torchvision import transforms
from datetime import datetime
import seaborn as sns
import scipy as sp
import cv2

# utility functions
def compute_pixel_area(seg, size):
    img = np.array(Image.open(seg).convert('L').resize(size))
    img = np.where(img > 128, 1, 0)
    area = np.sum(img)
    return area

def compute_pixel_perimeter(seg, size):
    img = np.array(Image.open(seg).convert('L').resize(size))
    img = np.where(img > 128, 255, 0).astype(np.uint8)

    # Find contours using OpenCV
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the perimeter of the contours
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    return perimeter

def compute_ncc(seg, size, area_threshold=100):
    img = np.array(Image.open(seg).convert('L').resize(size))
    img = np.where(img > 128, 255, 0).astype(np.uint8)

    # Find the number of connected components using OpenCV
    # num_labels, labels = cv2.connectedComponents(img, connectivity=8)
    # labeled_array, num_features = sp.ndimage.label(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on the area threshold
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

    # Create a new blank image to draw the filtered contours
    filtered_img = np.zeros_like(img)
    cv2.drawContours(filtered_img, filtered_contours, -1, 255, thickness=cv2.FILLED)

    # The number of connected components is the number of labels minus 1
    # (since label 0 is the background)
    # ncc = num_labels - 1
    # ncc = num_features
    ncc = len(filtered_contours)

    return ncc, filtered_img

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_data')
    parser.add_argument('--img_col')
    parser.add_argument('--ga_col')
    parser.add_argument('--vessel_col')
    parser.add_argument('--pid_col')
    parser.add_argument('--laterality_col')
    parser.add_argument('--date_col')
    parser.add_argument('--modality')
    parser.add_argument('--manual_area_data')
    parser.add_argument('--save_as')
    args = parser.parse_args()

    # get inputs
    imaging_data = args.image_data
    save_to = args.save_as
    image_column = args.img_col
    ga_seg_column = args.ga_col
    vessel_seg_column = args.vessel_col
    patient_id_column = args.pid_col
    laterality_column = args.laterality_col
    date_column = args.date_col
    manual_area_data = args.manual_area_data
    seg_mode = args.modality

    # load imaging data
    df_imaging_data = pd.read_csv(imaging_data)

    # ===============
    # compute AI area
    # ===============
    for mrn, mrn_df in df_imaging_data.groupby(patient_id_column):
        for lat, lat_df in mrn_df.groupby(laterality_column):
            for date, date_df in lat_df.groupby(date_column):

                # get slo and af rows
                slo = date_df[(date_df.Procedure == 'Ir_oct') & (date_df.type == 'SLOImage')]
                if seg_mode == 'Af':
                    af = date_df[date_df.Procedure == 'Af']
                    idx = af.index
                    print(f'Af: {len(af)}, Ir: {len(slo)}')
                elif seg_mode == 'Ir':
                    af = date_df[(date_df.Procedure == 'Ir_oct') & (date_df.type == 'SLOImage')]
                    idx = af.index
                    print(f'Ir: {len(af)}')
                else:
                    raise ValueError('GA size calculations only supported for Af and Ir types!')

                # handle more than one AF case
                number_of_foci = []
                px_perimeter = []
                mm_perimeter = []
                px_areas = []
                mm_areas = []
                for i, af_row in af.iterrows():
                    for j, slo_row in slo.iterrows():
                        xslo, yslo, scale_x, scale_y = int(slo_row.XSlo), int(slo_row.YSlo), slo_row.Scale_X, slo_row.Scale_Y
                        
                        # compute area of lesion
                        px_area = compute_pixel_area(af_row[ga_seg_column], (xslo, yslo))
                        mm_area = px_area * scale_x * scale_y
                        px_areas.append(px_area)
                        mm_areas.append(mm_area)

                        # compute perimeter of lesion
                        px_peri = compute_pixel_perimeter(af_row[ga_seg_column], (xslo, yslo))
                        mm_peri = px_peri * scale_x
                        px_perimeter.append(px_peri)
                        mm_perimeter.append(mm_peri)

                        # compute number of connected components
                        ncc, _ = compute_ncc(af_row[ga_seg_column], (xslo, yslo))
                        assert isinstance(ncc, int), ncc
                        number_of_foci.append(ncc)

                df_imaging_data.loc[idx, 'px_area'] = np.mean(px_areas)
                df_imaging_data.loc[idx, 'mm_area'] = np.mean(mm_areas)
                df_imaging_data.loc[idx, 'px_perimeter'] = np.mean(px_perimeter)
                df_imaging_data.loc[idx, 'mm_perimeter'] = np.mean(mm_perimeter)
                df_imaging_data.loc[idx, 'n_foci'] = np.mean(number_of_foci)

    # if seg_mode == 'Af':
    #     df_imaging_data_af = df_imaging_data[df_imaging_data.Procedure == 'Af'].reset_index(drop=True)

    #     for idx, row in df_imaging_data_af.iterrows():
    #         mrn, lat, date = row[patient_id_column], row[laterality_column], row[date_column]

    #         for j, slo_row in slo.iterrows():
    #             xslo, yslo, scale_x, scale_y = int(slo_row.XSlo), int(slo_row.YSlo), slo_row.Scale_X, slo_row.Scale_Y

    #             # get slo and af rows
    #             slo = df_imaging_data[(df_imaging_data[patient_id_column] == mrn) & (df_imaging_data[laterality_column] == lat) & (df_imaging_data[date_column] == date) & (df_imaging_data.Procedure == 'Ir_oct') & (df_imaging_data.type == 'SLOImage')]

    #             # compute area of lesion
    #             px_area = compute_pixel_area(row[ga_seg_column], (xslo, yslo))
    #             mm_area = px_area * scale_x * scale_y
    #             px_areas.append(px_area)
    #             mm_areas.append(mm_area)

    #             # compute perimeter of lesion
    #             px_peri = compute_pixel_perimeter(row[ga_seg_column], (xslo, yslo))
    #             mm_peri = px_peri * scale_x
    #             px_perimeter.append(px_peri)
    #             mm_perimeter.append(mm_peri)

    #             # compute number of connected components
    #             ncc, _ = compute_ncc(row[ga_seg_column], (xslo, yslo))
    #             assert isinstance(ncc, int), ncc
    #             number_of_foci.append(ncc)

    #             df_imaging_data_af.loc[idx, 'px_area'] = np.mean(px_areas)
    #             df_imaging_data_af.loc[idx, 'mm_area'] = np.mean(mm_areas)
    #             df_imaging_data_af.loc[idx, 'px_perimeter'] = np.mean(px_perimeter)
    #             df_imaging_data_af.loc[idx, 'mm_perimeter'] = np.mean(mm_perimeter)
    #             df_imaging_data_af.loc[idx, 'n_foci'] = np.mean(number_of_foci)

    # ================
    # get manual areas
    # ================
    df_ga_manual = pd.read_excel(manual_area_data)
    df_ga_manual = df_ga_manual[['SID', 'MRN', 'Eye', 'faf_date', 'GA Size 1 (Final)', 'Latest FAF date (before 6/2023)', 'GA Size 2 (Final)']]

    # chunk dataset into two parts
    df_ga_manual_t1 = df_ga_manual[['SID', 'MRN', 'Eye', 'faf_date', 'GA Size 1 (Final)']]
    df_ga_manual_t1 = df_ga_manual_t1.rename(columns={
        'GA Size 1 (Final)': 'GA Size'
    })
    df_ga_manual_t2 = df_ga_manual[['SID', 'MRN', 'Eye', 'Latest FAF date (before 6/2023)', 'GA Size 2 (Final)']]
    df_ga_manual_t2 = df_ga_manual_t2.rename(columns={
        'Date LAST FAF (Up to 6/2023)': 'faf_date',
        'GA Size 2 (Final)': 'GA Size'
    })

    # concatenate the chunks
    df_ga_manual = pd.concat([df_ga_manual_t1, df_ga_manual_t2], axis=0, ignore_index=True)
    df_ga_manual = df_ga_manual[~df_ga_manual['GA Size'].isnull()]
    df_ga_manual['faf_date'] = pd.to_datetime(df_ga_manual['faf_date'])

    # ====================================
    # Merge manual area df with AI area df
    # ====================================

    if seg_mode == 'Af':
        df_af = df_imaging_data[df_imaging_data.Procedure == 'Af'].copy()
        df_af[date_column] = pd.to_datetime(df_af[date_column])
    elif seg_mode == 'Ir':
        df_af = df_imaging_data[(df_imaging_data.Procedure == 'Ir_oct') & (df_imaging_data.type == 'SLOImage')].copy()
        df_af[date_column] = pd.to_datetime(df_af[date_column])

    # merge df_af with df_ga_manual
    merged_df = pd.merge(df_af, df_ga_manual.rename({'MRN': patient_id_column, 'Eye': laterality_column, 'faf_date': date_column}, axis=1), how='left')
    merged_df.to_csv(save_to, index=False)
