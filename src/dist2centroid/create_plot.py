import sys
import cv2
import numpy as np
import pandas as pd
import os
import math
from matplotlib import pyplot as plt
from PIL import Image
import torch
from torch.nn import functional as F
from math import isnan
from torchvision.transforms import Resize, Grayscale, ToTensor

angle_to_hour = {
    90: 12,
    60: 1,
    30: 2,
    0: 3,
    330: 4,
    300: 5,
    270: 6,
    240: 7,
    210: 8,
    180: 9,
    150: 10,
    120: 11
}

def find_center(contours):
    # Assume 'contours' is a list of contours produced by cv2.findContours
    # Find the center of the first contour
    if len(contours) > 0:
        first_contour = contours[0][0]
        M = cv2.moments(first_contour)
        
        if M['m00'] != 0:
            # Calculate centroid
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
            print(f"Center of the first contour: ({center_x}, {center_y})")
            print(f'Radius={int(math.sqrt(center_x**2 + center_y**2))}')
        else:
            print("The contour has zero area, can't calculate center.")

    return (center_x, center_y)

def get_contour(seg):
    # find contour
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image for the contour
    contour_image = np.zeros(seg.shape, dtype=np.uint8)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
    return contours, contour_image

def load_image(path, size=(256, 256), mode='rgb'):
    x = Image.open(path)
    w, h = x.size
    x = Resize(size)(x)
    x = Grayscale()(x) if mode == 'gray' else x
    x = ToTensor()(x)
    return x.unsqueeze(0), (h, w)

def transform(img_file, theta):

    # load image
    image_tensor, _ = load_image(img_file)

    # handle unregistered images
    if isinstance(theta, float) and isnan(theta):
        image_tensor = cv2.resize(image_tensor.numpy().squeeze(), (1536, 1536))
        image_tensor = np.uint8(np.where(image_tensor > 0.5, 1, 0) * 255)
        return image_tensor.squeeze()

    # Load the sampling grid
    grid = torch.load(theta, weights_only=True)[1]
    
    # Apply the sampling grid
    registered_image = F.grid_sample(image_tensor, grid, align_corners=True).squeeze(0)

    # Post-process
    registered_image = cv2.resize(registered_image.numpy().squeeze(), (1536, 1536))
    registered_image = np.where(registered_image > 0.5, 1, 0)
    registered_image = np.uint8(registered_image * 255)
    
    return registered_image.squeeze()

def process_segmentation_image(data, center, num_angles=12, mrn='', lat=''):

    # Calculate the maximum possible radius
    max_radius = int(math.sqrt(center[0]**2 + center[1]**2))
    
    # Calculate angles
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    
    results = {'dates': [], 'angles': [], 'distances': []}
    
    for i, (date, _, _, img) in enumerate(zip(data['date'], data['segs'], data['contour'], data['contour_image'])):
        cimg = np.repeat(img[:, :, None], repeats=3, axis=-1)
        for angle in angles:
            # Convert angle to radians
            theta = np.radians(angle)
            
            # Calculate end point of the radial line
            end_x = int(center[0] + max_radius * math.cos(-1 * theta))
            end_y = int(center[1] + max_radius * math.sin(-1 * theta))
            
            # Create a blank image for the radial line
            line_image = np.zeros(img.shape, dtype=np.uint8)
            cv2.line(line_image, center, (end_x, end_y), 255, 1)
            
            # Find intersection
            # cv2.imwrite(f'clock_{angle_to_hour[angle]}.png', line_image)
            intersection = cv2.bitwise_and(img, line_image)
            
            # Find the non-zero pixels (intersection points)
            y, x = np.nonzero(intersection)

            #####
            cv2.line(cimg, center, (end_x, end_y), 255, 1)
            #####
            
            if len(x) > 0 and len(y) > 0:
                # Find the nearest intersection point to the center
                distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                nearest_index = np.argmin(distances)
                distance = distances[nearest_index]
                point = (x[nearest_index], y[nearest_index])
                cv2.circle(cimg, point, 5, (0, 0, 255), -1)

                # save results
                results['dates'].append(date)
                results['angles'].append(angle_to_hour[angle])
                results['distances'].append(distance)

        #####
        cv2.imwrite(f'dist2centroid_results/{mrn}_{lat}/contour_plot_wradial_{i}.png', cimg)
        #####

    # sys.exit(0)
    
    return pd.DataFrame(results)

if __name__ == '__main__':

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
        (4930414, 'OS'),
        (1173138, 'OS'),
        (1179649, 'OS'),
        (1577868, 'OD'),
        (1577868, 'OS'),
        (1629706, 'OS'),
        (1765980, 'OS'),
        (2074936, 'OS'),
        (2173985, 'OD'),
        (4391754, 'OS'),
        (5434794, 'OS'),
        (5443557, 'OS')
    ]

    # get AF images for specific patients
    df = pd.read_csv('results/09172024_coris/registration_results_af.csv')
    df = df[df.Procedure == 'Af']
    df = df[df[['PID', 'Laterality']].apply(tuple, axis=1).isin(patients)]

    for mrn, mrn_df in df.groupby('PID'):
        for lat, lat_df in mrn_df.groupby('Laterality'):
            print(f'Processing {mrn}, {lat}')

            ##### METADATA #####
            os.makedirs(f'dist2centroid_results/{mrn}_{lat}/', exist_ok=True)
            ####################

            # register ga_segs
            data = {'date': [], 'segs': [], 'contour': [], 'contour_image': []}

            # convert to datetime format
            lat_df.ExamDate = pd.to_datetime(lat_df.ExamDate)

            # apply registration to the segmentation maps for each date
            for date, date_df in lat_df.sort_values(by='ExamDate', ascending=True).groupby('ExamDate'):

                # drop repeat AF scans from a date
                date_df = date_df.drop_duplicates(subset=['Procedure'])

                # apply registration to the segmentations
                ga_seg_registered = transform(date_df.file_path_ga_seg.item(), date_df.params.item())

                # get contour 
                contour, contour_image = get_contour(ga_seg_registered)

                # exclude visits where the GA area is 0
                if ga_seg_registered.sum() > 0:
                    data['date'].append(date)
                    data['segs'].append(ga_seg_registered)
                    data['contour'].append(contour)
                    data['contour_image'].append(contour_image)

            ###### METADATA #####
            contour_plot = np.where(np.sum(np.stack(data['contour_image'], axis=0), axis=0) > 0, 255, 0)
            cv2.imwrite(f'dist2centroid_results/{mrn}_{lat}/contour_plot_initial.png', contour_plot)
            #####################

            # get center and max radius
            center = find_center(data['contour'])

            ##### METADATA ######
            contour_plot = cv2.circle(np.repeat(np.uint8(contour_plot)[:, :, None], repeats=3, axis=-1), center, 5, (0, 0, 255), -1)
            cv2.imwrite(f'dist2centroid_results/{mrn}_{lat}/contour_plot_wcenter.png', contour_plot)
            #####################

            # get radial distances
            res = process_segmentation_image(data, center, mrn=mrn, lat=lat)

            # plot results
            res = res.sort_values(by=['angles', 'dates']) 

            ##### METADATA ######
            res.to_csv(f'dist2centroid_results/{mrn}_{lat}/raw_data.csv', index=False)
            #####################

            # Create figure
            plt.figure(figsize=(12, 6))

            # Subplot for the image with clock numbers
            ax1 = plt.subplot(121)
            img = np.array(Image.open(f'results/09172024_coris/powerpoint_af/metadata/baseline_wcontours/{mrn}_{lat}_wcontours.png'))
            ax1.imshow(img)

            # Set the center of the clock and the radius for placing the numbers
            center = (0.4, 0.5)  # Move the center slightly to the left (x coordinate < 0.5)
            radius = 0.3

            # Loop through the angle-to-hour mapping and place the numbers
            for angle, hour in angle_to_hour.items():
                # Convert angle to radians for use in trigonometric functions
                angle_rad = np.deg2rad(angle)
                
                # Calculate X, Y positions relative to the center and radius
                x = center[0] + radius * np.cos(angle_rad)  # X position
                y = center[1] + radius * np.sin(angle_rad)  # Y position

                # Place the numbers with red color
                ax1.text(x, y, str(hour), transform=ax1.transAxes,
                        fontsize=12, ha='center', va='center', color='red')  # Change color to red

            # Turn off axis for a cleaner look
            ax1.axis('off')

            # Subplot for the distances from centroid plot
            ax2 = plt.subplot(122)

            # Get the colormap with 12 distinct colors
            colormap = plt.cm.get_cmap('tab20', 12)

            # Assuming `res` is a DataFrame with 'angles', 'contour_idx', and 'distances' columns
            for i, (angle, angle_df) in enumerate(res.groupby('angles')):
                angle_df['dates'] = pd.to_datetime(angle_df['dates'])
                angle_df = angle_df.sort_values(by='dates', ascending=True)
                color = colormap(i)
                ax2.plot(angle_df['dates'], angle_df['distances'], '-o', color=color, label=angle)

            ax2.set_xlabel('Time')
            ax2.set_ylabel('Distance from GA Centroid (pixels)')
            ax2.legend(title="Clock Hour")

            # Save the figure
            plt.savefig(f'dist2centroid_results/{mrn}_{lat}/{mrn}_{lat}.png', bbox_inches='tight')

'''
Old code:

# # Get the bounding box for all contours
# x_min, y_min, x_max, y_max = float('inf'), float('inf'), -float('inf'), -float('inf')

# for contour in contours[0]:
#     x, y, w, h = cv2.boundingRect(contour)
#     x_min = min(x_min, x)
#     y_min = min(y_min, y)
#     x_max = max(x_max, x + w)
#     y_max = max(y_max, y + h)

# # Calculate the center of the bounding box
# center_x = (x_min + x_max) // 2
# center_y = (y_min + y_max) // 2
# print(f"Center of the first contour: ({center_x}, {center_y})")
'''