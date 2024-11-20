import os, sys
sys.path.append('/sddata/projects/LightGlue')
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
import pickle
from utils import get_contour_points, find_intersections, get_bscans
import torch
from src.utils import load_image
from src.eyeliner import EyeLinerP
import warnings
warnings.filterwarnings("ignore")

# Load EyeLiner API for registration
eyeliner = EyeLinerP(
    reg='tps', # registration technique to use (tps or affine)
    lambda_tps=1.0, # set lambda value for tps
    image_size=(3, 256, 256) # image dimensions
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--save')
    return parser.parse_args()

def tensor2numpy(tensor):
    return torch.permute(tensor, (1, 2, 0)).numpy()

def register(af_df, slo_df, device='cpu'):

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    slo_image = load_image(slo_df.file_path_coris.item(), size=(256, 256), mode='rgb').to(device)
    faf_image = load_image(af_df.file_path_coris.item(), size=(256, 256), mode='rgb').to(device)
    faf_ga_seg = load_image(af_df.file_path_ga_seg.item(), size=(256, 256), mode='rgb').to(device)

    # store inputs
    data = {
    'fixed_input': slo_image,
    'moving_input': faf_image
    }

    # compute registration
    theta, _ = eyeliner(data)

    # apply registration to mask as well
    reg_faf_ga_seg = eyeliner.apply_transform(theta[1].squeeze(0), faf_ga_seg.squeeze(0))

    return tensor2numpy(reg_faf_ga_seg.cpu())

def get_scanlines(oct_df, slo_df):
    # get scanning lines from oct
    scale_x, scale_y = slo_df.Scale_X.item(), slo_df.Scale_Y.item()
    scanlines = oct_df[['Start_X', 'Start_Y', 'End_X', 'End_Y']].values / np.array([scale_x, scale_y, scale_x, scale_y]).T
    return scanlines.astype(int)

def compute_intersection(reg_faf_ga_seg, scanlines):

    # get contour points
    contour, _ = get_contour_points(np.uint8(reg_faf_ga_seg * 255), largest_contour=True)

    # get intersection points
    points = find_intersections(contour, scanlines)

    return points, contour

class IntersectionPointSet():
    ''' Intersection Point Set class '''
    def __init__(self, t, phi, omega, beta, rc):
        self.t = t
        self.phi = phi
        self.omega = omega
        self.beta = beta
        self.rc = rc

    def save(self, filename: str):
        # Serialize the object and save it to a file
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str):
        # Load the object from a file
        with open(filename, 'rb') as file:
            return pickle.load(file)

def main(args):

    # save folder
    os.makedirs(args.save, exist_ok=True)

    # load base dataframe
    df = pd.read_csv(args.csv)

    # ================================================================
    # ASSUMPTION: Every case has 1 AF, 1 en-face OCT and 1 OCT volume!
    # ================================================================

    intersection_point_set = []

    for mrn, mrn_df in df.groupby('PID'):
        for lat, lat_df in mrn_df.groupby('Laterality'):
            T = len(lat_df[lat_df.Procedure == 'Af'].ExamDate.unique())
            print(f'Processing MRN: {mrn}, Laterality: {lat}, Timepoints: {T}')

            # only take patients with > 1 visits
            if T > 1:
                for date, date_df in lat_df.groupby('ExamDate'):

                    # get paths to af 
                    af_df = date_df[date_df.Procedure == 'Af']
                    slo_df = date_df[date_df.type == 'SLOImage']
                    oct_df = date_df[date_df.type == 'BScan'].sort_values(by='Start_Y', ascending=False)

                    print(f'Found {len(af_df)} AF, {len(slo_df)} SLO, and {len(oct_df)} OCT on {date}.')

                    # register af to slo image
                    reg_faf_ga_seg = register(af_df, slo_df)

                    # if no signal in the image, then skip it
                    if np.sum(reg_faf_ga_seg) == 0:
                        print('Skipping - No segmentation found!')
                        continue

                    # get scanlines from oct
                    scanlines = get_scanlines(oct_df, slo_df)

                    # compute intersection points
                    point_set, contour = compute_intersection(reg_faf_ga_seg, scanlines)

                    print(point_set)
                    sys.exit(0)

                    # get a-scans for those intersection points
                    ascan_set = get_bscans(point_set, scanlines, oct_df, window_size=50)

                    # save to intersection point set
                    P = IntersectionPointSet(t=date, phi=point_set, omega=ascan_set, beta=contour, rc=len(oct_df))
                    P.save(f'{mrn}_{lat}_{date}.pkl')

            else:
                print('Skipping - only 1 visit found!')

    return

if __name__ == '__main__':
    main(parse_args())