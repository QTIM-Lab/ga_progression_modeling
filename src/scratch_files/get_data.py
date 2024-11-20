import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision.transforms import Resize, Grayscale, ToTensor
from math import isnan

# get the specific patients with symmetric and non symmetric patterns of growth
patients = [
    (37923, 'OD'), #
    (88395, 'OS'), #
    (169896, 'OS'),
    (2065388, 'OD'),
    (2065388, 'OS'),
    (196031, 'OS'), #
    (1264366, 'OS'),
    (1629706, 'OS'), #
    (2065388, 'OS'), #
    (4930414, 'OD'),
    (4930414, 'OS') 
]

def get_contour(seg):
    # find contour
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image for the contour
    contour_image = np.zeros(seg.shape, dtype=np.uint8)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    return contours, contour_image

def load_image(path, size=(256, 256), mode='rgb'):
    x = Image.open(path)
    x = Resize(size)(x)
    x = Grayscale()(x) if mode == 'gray' else x
    x = ToTensor()(x)
    return x.unsqueeze(0)

def transform(img_file, theta):

    # load image
    image_tensor = load_image(img_file)

    # handle unregistered images
    if isinstance(theta, float) and isnan(theta):
        image_tensor = np.uint8(np.where(image_tensor.numpy() > 0.5, 1, 0) * 255)
        return image_tensor.squeeze()

    # Load the sampling grid
    grid = torch.load(theta, weights_only=True)[1]
    
    # Apply the sampling grid
    registered_image = F.grid_sample(image_tensor, grid, align_corners=True).squeeze(0)

    # Post-process
    registered_image = np.uint8(np.where(registered_image.numpy() > 0.5, 1, 0) * 255)
    
    return registered_image.squeeze()

df = pd.read_csv('results/09172024_coris/registration_results_af.csv')
df = df[df.Procedure == 'Af']
df = df[df[['PID', 'Laterality']].apply(tuple, axis=1).isin(patients)]

for mrn, mrn_df in df.groupby('PID'):
    for lat, lat_df in mrn_df.groupby('Laterality'):
        print(f'Processing {mrn}, {lat}')

        # register ga_segs
        data = {'date': [], 'segs': [], 'contour': [], 'contour_image': []}
        for date, date_df in lat_df.sort_values(by='ExamDate').groupby('ExamDate'):

            date_df = date_df.drop_duplicates(subset=['Procedure'])

            # apply registration to the segmentations
            ga_seg_registered = transform(date_df.file_path_ga_seg.item(), date_df.params.item())

            # get contour 
            contour, contour_image = get_contour(ga_seg_registered)

            if ga_seg_registered.sum() > 0:
                data['date'].append(date)
                data['segs'].append(ga_seg_registered)
                data['contour'].append(contour)
                data['contour_image'].append(contour_image)
