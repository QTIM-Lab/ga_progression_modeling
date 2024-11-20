import numpy as np
import cv2

import os, sys
import argparse
from PIL import Image
from math import isnan
import torch
from torchvision import transforms
from datetime import datetime
import seaborn as sns
import scipy as sp
from itertools import combinations

# ===============
# General Utils
# ===============

def load_seg(filepath, shape):
    img = Image.open(filepath).convert('L').resize(shape)
    return np.where(np.array(img) > 0, 255., 0.)

def draw_scanlines(slo_image, lines_array):
    # load image
    if isinstance(slo_image, str):
        image = cv2.imread(slo_image)
    else:
        image = np.copy(slo_image)
    
    # Define line color and thickness
    line_color = (255, 0, 0)  # Blue color in BGR
    line_thickness = 2
    
    # Draw lines on the image
    for line in lines_array:
        start_point = (line[0], line[1])
        end_point = (line[2], line[3])
        cv2.line(image, start_point, end_point, line_color, line_thickness)

    return image

# ==================== 
# Get en-face OCT data
# ====================

def get_scanlines(slo_1, slo_2, oct_1, oct_2):
    scale_x_1, scale_y_1 = slo_1.scale_x, slo_1.scale_y
    scan_lines_1 = oct_1[['start_x', 'start_y', 'end_x', 'end_y']].values / np.array([scale_x_1, scale_y_1, scale_x_1, scale_y_1]).T
    scan_lines_1 = np.round(scan_lines_1)

    scale_x_2, scale_y_2 = slo_2.scale_x, slo_2.scale_y
    scan_lines_2 = oct_2[['start_x', 'start_y', 'end_x', 'end_y']].values / np.array([scale_x_2, scale_y_2, scale_x_2, scale_y_2]).T
    scan_lines_2 = np.round(scan_lines_2)
    
    return scan_lines_1, scan_lines_2

def remove_close_points(points_array, min_distance):

    ''' Suppresses nearby points in an array given a distance threshold '''
    
    # Create a list to keep track of points to retain
    retained_points = [points_array[0]]
    
    for point in points_array[1:]:
        distances = np.linalg.norm(np.array(retained_points) - point, axis=1)
        if np.all(distances >= min_distance):
            retained_points.append(point)
            
    return np.stack(retained_points, axis=0)
    
def get_contour_points(mask, largest_contour=True):
    ''' Function to get contour coordinates from segmentation mask '''

    # Detect the contour of the circle
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the largest contour by area
    if largest_contour:
        contour = max(contours, key=cv2.contourArea)
    else:
        contour = contours[0]

    # draw contours on mask
    contour_map = np.zeros_like(mask)
    cv2.drawContours(contour_map, contour, -1, (255, 255, 255), 1)

    # return points, contour_map
    return contour.squeeze(), contour_map

def find_intersections(contour, lines):

    ''' Function to get the intersection of a mask and scan lines '''
    
    # find intersection of contour and scanning lines
    intersections = []
    for line in lines:
        start_x, start_y, end_x, end_y = line
        for point in contour:
            contour_x, contour_y = point
            if start_y == contour_y:
                intersections.append([contour_x, contour_y])
                
    return np.array(intersections)

def intersection_with_contour_shapely(line_point, line_dir, contour):
    """
    Find the intersection of a line with a contour using Shapely.
    """
    from shapely.geometry import LineString, Point
    
    # Create the infinite line
    line_end = line_point + 1000 * line_dir  # Extend the line sufficiently
    line = LineString([line_point, line_end])
    
    # Create the contour as a series of line segments
    contour_line = LineString(contour)
    
    # Calculate intersection
    intersection = line.intersection(contour_line)
    
    if not intersection.is_empty:
        if intersection.geom_type == 'Point':
            return intersection
        elif intersection.geom_type == 'MultiPoint':
            # Return the closest intersection point
            points = list(intersection)
            dists = [line_point.distance(point) for point in points]
            return points[np.argmin(dists)]
    return None

def find_normal_vector(contour, point_index, h=1):
    """
    Calculate the normal vector at a point on a given contour.
    """
    n = len(contour)
    p_prev = contour[(point_index - h) % n]
    p_next = contour[(point_index + h) % n]
    tangent = p_next - p_prev
    normal = np.array([-tangent[1], tangent[0]])
    normal = normal / np.linalg.norm(normal)
    return normal

def get_growth_rates(inner_contour, outer_contour, inner_intersection_points):
    
    # get orthogonal vectors to the points
    inner_intersection_points_new = []
    inner_intersection_points_normals = []
    outer_intersection_points = []
    
    for point in inner_intersection_points:
        # compute normal vector
        point_index = np.where((inner_contour == np.array(point)).all(axis=1))[0] 
        point_normal = find_normal_vector(inner_contour, point_index.item(), h=5)
        outer_point = intersection_with_contour_shapely(point, point_normal, outer_contour)
        if outer_point is None:
            continue
        outer_point = np.array([outer_point.x, outer_point.y])
        
        # store normals, inner and outer intersection points
        inner_intersection_points_new.append(point)
        inner_intersection_points_normals.append(point_normal)
        outer_intersection_points.append(outer_point)
    
    inner_intersection_points_new = np.stack(inner_intersection_points_new, axis=0)
    inner_intersection_points_normals = np.stack(inner_intersection_points_normals, axis=0)
    outer_intersection_points = np.stack(outer_intersection_points, axis=0)

    return inner_intersection_points_new, inner_intersection_points_normals, outer_intersection_points

# =================== 
# Get OCT Volume data
# ===================
def create_oct_volume(df):
    volume = []
    for f in df.file_path_coris:
        bscan = np.array(Image.open(f))
        volume.append(bscan)
    return np.stack(volume, axis=0)

def get_a_scans(volume, points, scan_lines, window_size=None):
    levels = scan_lines[:, 1]
    ascans = []
    for (x_slo, y_slo) in points:

        # grab the bscan
        slice_index = np.where(levels == y_slo)[0].item()
        oct_bscan = volume[slice_index]

        # scale the x value as per oct dimension
        x_oct = int(1024 * (x_slo - scan_lines[0, 0]) / (scan_lines[0, 2] - scan_lines[0, 0]))

        # grab a window around x location
        if window_size == None:
            ascan_window = oct_bscan
        else:
            ascan_window = oct_bscan[:, x_oct-window_size:x_oct+window_size]

        ascans.append(ascan_window)
        
    return np.stack(ascans, axis=0) # (496, 100)

def get_bscans(inner_intersection_points, inner_scan_lines, inner_oct, window_size=50):
    # create oct volume
    oct_volume_array = create_oct_volume(inner_oct)
    
    # get window region of B-scan
    intersection_points_ascans = get_a_scans(oct_volume_array, inner_intersection_points, inner_scan_lines, window_size=window_size)
    
    return intersection_points_ascans

# =========================== 
# Get BScans and Growth Rates
# ===========================
def get_bscans_and_growth_rates(af_paths, slo_paths, oct_paths):

    # get dates
    date1, date2 = list(sorted(af_paths.ExamDate.unique()))

    # extract filepaths
    inner_seg_mask = af_paths[af_paths.ExamDate == date1].file_path_ga_seg.item()
    outer_seg_mask = af_paths[af_paths.ExamDate == date2].file_path_ga_seg.item()

    # load seg mask
    xslo, yslo = int(slo_paths.iloc[0].xslo.item()), int(slo_paths.iloc[0].yslo.item())
    inner_seg_mask = load_seg(inner_seg_mask, shape=(xslo, yslo))
    outer_seg_mask = load_seg(outer_seg_mask, shape=(xslo, yslo))
    
    # convert mask to contours
    inner_contour, _ = get_contour_points(np.uint8(inner_seg_mask))
    outer_contour, _ = get_contour_points(np.uint8(outer_seg_mask))

    # get the scanlines from oct
    inner_slo = slo_paths[slo_paths.ExamDate == date1]
    outer_slo = slo_paths[slo_paths.ExamDate == date2]
    inner_oct = oct_paths[oct_paths.ExamDate == date1].sort_values(by='start_y', ascending=False)
    outer_oct = oct_paths[oct_paths.ExamDate == date2].sort_values(by='start_y', ascending=False)
    inner_scan_lines, _ = get_scanlines(inner_slo, outer_slo, inner_oct, outer_oct)

    # find intersection points of inner contour with the scan lines
    inner_intersection_points = find_intersections(inner_contour, inner_scan_lines.astype(int))

    # get b-scans
    b_scans = get_bscans(inner_intersection_points, inner_scan_lines, inner_oct)

    # get intersection points
    inner_intersection_points_new, inner_intersection_normals, outer_intersection_points = get_growth_rates(inner_contour, outer_contour, inner_intersection_points)

    return b_scans, inner_intersection_points_new, inner_intersection_normals, outer_intersection_points
