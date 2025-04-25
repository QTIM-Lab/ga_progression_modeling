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
from time import sleep

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

# utility functions
def compute_pixel_area(row, args):
    ''' Compute pixel area'''
    img = np.array(Image.open(row[args.ga_col]).convert('L').resize((int(row[args.size_x_col]), int(row[args.size_y_col]))))
    img = np.where(img > 128, 1, 0).astype(np.uint8)
    area = np.sum(img) * row[args.scale_x_col] * row[args.scale_y_col]
    return area

def compute_pixel_perimeter(row, args):
    ''' Compute pixel perimeter '''
    img = np.array(Image.open(row[args.ga_col]).convert('L').resize((int(row[args.size_x_col]), int(row[args.size_y_col]))))
    img = np.where(img > 128, 255, 0).astype(np.uint8)

    # Find contours using OpenCV
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the perimeter of the contours
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours) * row[args.scale_x_col]
    return perimeter

def compute_ncc(row, args, area_threshold=0.5):
    ''' Compute number of connected components '''
    img = np.array(Image.open(row[args.ga_col]).convert('L').resize((int(row[args.size_x_col]), int(row[args.size_y_col]))))
    img = np.where(img > 128, 255, 0).astype(np.uint8)

    # Find the number of connected components using OpenCV
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on the area threshold
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) * row[args.scale_x_col] * row[args.scale_y_col] > area_threshold]

    # Create a new blank image to draw the filtered contours
    filtered_img = np.zeros_like(img)
    cv2.drawContours(filtered_img, filtered_contours, -1, 255, thickness=cv2.FILLED)
    ncc = len(filtered_contours)
    return ncc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Computes GA area from segmentations")
    parser.add_argument('--csv')
    parser.add_argument('--ga_col', default='file_path_ga_seg', type=str)
    parser.add_argument('--size_x_col', default='XSlo', type=str)
    parser.add_argument('--size_y_col', default='YSlo', type=str)
    parser.add_argument('--scale_x_col', default='Scale_X', type=str)
    parser.add_argument('--scale_y_col', default='Scale_Y', type=str)
    parser.add_argument('--save_as', default='', type=str)
    args = parser.parse_args()

    # load imaging data
    df = pd.read_csv(args.csv)

    # Startup console
    console = Console()

    progress = Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )

    # Create a progress bar using Rich
    with Live(console=console, refresh_per_second=10) as live:
        # Add a task to the progress bar
        task_desc = "Computing GA Area"
        task_id = progress.add_task(task_desc, total=len(df))
        panel = Panel(Group(progress), title=task_desc)
        live.update(panel)

        # compute area
        results = {args.ga_col: [], 'mm_area': [], 'mm_perimeter': [], 'ncc': []}
        for i, row in df.iterrows():
            progress.update(task_id, advance=1)

            # compute area
            mm_area = compute_pixel_area(row, args)

            # compute perimeter
            mm_perimeter = compute_pixel_perimeter(row, args)

            # compute number of connected components
            ncc = compute_ncc(row, args)

            # save results
            results[args.ga_col].append(row[args.ga_col])
            results['mm_area'].append(mm_area)
            results['mm_perimeter'].append(mm_perimeter)
            results['ncc'].append(ncc)
        
        results = pd.DataFrame(results)
        results = pd.merge(df, results, how='left', on=args.ga_col)
        results.to_csv(args.save_as, index=False)

    