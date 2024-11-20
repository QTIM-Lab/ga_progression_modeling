import sys
import cv2
import pandas as pd
import os
import numpy as np
from shutil import copy

# Function to visualize segmentation mask as a contour on the image
def visualize_contour(image_path, mask_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    # Read the segmentation mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, image.shape[:2])
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Green color for the contours
    
    # Save the resulting image
    cv2.imwrite(output_path, image)

# Function to process the CSV file
def process_csv(csv_path, output_folder):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    df = df[df.Procedure == 'Af']
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        image_path = row['file_path_coris']
        mask_path = row['file_path_ga_seg']
        output_path = os.path.join(output_folder, f"output_{index}.png")
        visualize_contour(image_path, mask_path, output_path)
        # cv2.imwrite(output_folder + os.path.basename(image_path).replace('.j2k', '.png'), img)
        # copy(f'/sddata/projects/GA_progression_modeling/more_seg_data/af_images_w_contours/output_{index}.png', '/sddata/projects/GA_progression_modeling/more_seg_data/af_images_reannotate_preds/' + os.path.basename(image_path).replace('.j2k', '.png'))

# Main function
if __name__ == "__main__":
    csv_path = 'data/GA_progression_modelling_data_redone/clean_data_talisa_ga_10312024.csv'
    output_folder = 'results/11082024_talisa_ga/af_images_w_contours/'
    process_csv(csv_path, output_folder)
    
