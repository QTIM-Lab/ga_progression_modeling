import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from matplotlib import cm
from scipy.ndimage import map_coordinates

def load_masks(files):
    masks = []
    for filepath in files:
        mask = np.array(Image.open(filepath).convert('L')) / 255.  # Convert to grayscale
        masks.append(mask)
    return np.stack(masks, axis=0)

def generate_heatmap(masks):
    # Assuming masks contain binary values (0 or 1)
    # Sum across all masks and normalize by the number of masks
    mask_sum = np.sum(masks, axis=0)
    heatmap = mask_sum / masks.shape[0]  # Proportion of masks with 1 at each pixel
    return heatmap

def draw_contours_on_image(image, masks):
    """Draw contours from the masks onto the given image using OpenCV."""
    # Make a copy of the image to avoid modifying the original
    contour_image = image.copy()

    for mask in masks:
        # Find contours in the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the contours on the image
        # Use a distinct color (e.g., blue) and a thickness of 1-2 for visibility
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)  # Draw in yellow for visibility (BGR)

    return contour_image

def plot_2d_heatmap_on_reference(title, heatmap, reference_image_path, masks=None, output_path=None):
    # Load the reference image
    reference_image = np.array(Image.open(reference_image_path).convert('L'))
    # reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2RGB)  # Convert to RGB for color drawing

    # Draw GA contours on the reference image
    # contour_image = draw_contours_on_image(reference_image_rgb, masks)

    # Create a normalization object to ensure that the heatmap values are scaled between 0 and 1
    norm = Normalize(vmin=0, vmax=1)

    # Create the figure and axes
    plt.figure(figsize=(8, 8))
    
    # Display the reference image in the background
    plt.imshow(reference_image, cmap='gray')  # Show the reference image in grayscale
    
    # Overlay the heatmap on top of the reference image
    plt.imshow(heatmap, cmap='hot', alpha=0.5, interpolation='nearest', norm=norm)  # Adjust alpha for transparency
    # Add a colorbar without the label
    plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])  # Set ticks between 0 and 1

    # Title and display
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')

def plot_3d_heatmap_on_reference(title, heatmap, reference_image_path, masks=None, output_path=None):
    """Plot a 3D heatmap with the reference image overlaid on the X-Y plane."""
    # Load the reference image
    reference_image = np.array(Image.open(reference_image_path).convert('L'))
    h, w = reference_image.shape

    # Create a mesh grid for x, y coordinates
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    X, Y = np.meshgrid(x, y)
    Z = heatmap

    # Create a figure with 3D axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create coordinate meshgrid for the image
    x_img = np.linspace(0, w-1, w)
    y_img = np.linspace(0, h-1, h)
    X_img, Y_img = np.meshgrid(x_img, y_img)

    # Plot the image on the x-y plane
    # Using gray colormap for grayscale representation
    img_plot = ax.plot_surface(X_img, Y_img, np.zeros_like(X_img), facecolors=plt.cm.gray(reference_image / 255.), shade=False)

    # Plot the heatmap as a 3D surface with transparency
    surf = ax.plot_surface(X, Y, Z, cmap='hot', norm=Normalize(vmin=0, vmax=1), edgecolor='none', alpha=0.5)

    # Customize axis labels and plot settings
    ax.set_xlabel('X (Pixels)')
    ax.set_ylabel('Y (Pixels)')
    ax.set_zlabel('Heatmap Intensity')
    ax.set_title(title)

    # Customize view angle for better visualization
    ax.view_init(elev=60, azim=-60)

    # Add a color bar for the heatmap
    mappable = cm.ScalarMappable(cmap='hot', norm=Normalize(vmin=0, vmax=1))
    mappable.set_array(Z)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cbar.ax.tick_params(labelsize=10)

    # set limits
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_zlim(0, 1)

    # Save the output if needed 
    plt.savefig(output_path, bbox_inches='tight')

def get_radial_intensity_profiles(heatmap, num_lines=12):
    """
    Generate intensity profiles along radial lines from the center of the heatmap.
    
    Parameters:
    - heatmap: 2D numpy array representing the heatmap.
    - num_lines: Number of radial lines to sample (default is 12).
    
    Returns:
    - A dictionary containing angles and intensity profiles.
    """
    h, w = heatmap.shape
    center_y, center_x = h // 2, w // 2  # Determine the center of the heatmap
    
    # Create an array to store intensity profiles
    radial_profiles = {}
    
    # Generate radial lines at evenly spaced angles (in radians)
    angles = np.linspace(0, 2 * np.pi, num_lines, endpoint=False)
    
    # Determine the maximum radius (distance from the center to the image edge)
    max_radius = int(np.hypot(center_y, center_x))
    
    # For each angle, compute the intensity along the radial line
    for angle in angles:
        # Generate x, y coordinates along the radial line
        x_coords = np.linspace(center_x, center_x + max_radius * np.cos(angle), max_radius)
        y_coords = np.linspace(center_y, center_y + max_radius * np.sin(angle), max_radius)
        
        # Extract intensities using interpolation
        intensities = map_coordinates(heatmap, [y_coords, x_coords], order=1, mode='constant', cval=0)
        
        # Store the intensity profile along this line
        radial_profiles[np.degrees(angle)] = intensities
        
    return radial_profiles

def plot_radial_profiles(title, radial_profiles, output_path=None):
    """
    Plot radial intensity profiles as 1D plots.
    
    Parameters:
    - radial_profiles: Dictionary containing angles and intensity profiles.
    """
    # Create a figure for the radial intensity profiles
    plt.figure(figsize=(12, 6))
    
    for angle, intensities in radial_profiles.items():
        plt.plot(intensities, label=f'Angle: {angle:.1f}°')
        
    # Customize plot
    plt.xlabel('Distance from Center (pixels)')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)

def draw_radial_lines_on_heatmap(title, reference_image_path, heatmap, num_lines=12, line_length=None, output_path=None):
    """
    Draw radial lines overlaid on the heatmap, colored according to heatmap intensities.
    
    Parameters:
    - image: 2D array of the reference image.
    - heatmap: 2D array of the heatmap values.
    - num_lines: Number of radial lines to draw.
    - line_length: Optional length of each radial line (default is the minimum dimension).
    
    Returns:
    - An image with radial lines colored by heatmap intensity.
    """

    # Load the reference image
    reference_image = np.array(Image.open(reference_image_path).convert('L'))

    # Define the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the reference image
    ax.imshow(reference_image, cmap='gray')
    
    # Overlay the heatmap (using alpha blending)
    # ax.imshow(heatmap, cmap='hot', alpha=0.6)

    # Calculate the center of the image
    h, w = heatmap.shape
    center_y, center_x = h // 2, w // 2
    
    # Default line length is half of the smallest dimension
    if line_length is None:
        line_length = min(h, w) // 2
    
    # Generate angles evenly spaced around a circle
    angles = np.linspace(0, 2 * np.pi, num_lines, endpoint=False)

    # Plot each radial line
    for angle in angles:
        # Calculate the end point of the line
        end_x = int(center_x + line_length * np.cos(angle))
        end_y = int(center_y + line_length * np.sin(angle))

        # Get the line points using OpenCV's line iterator
        line_points = list(zip(*cv2.line(np.zeros_like(heatmap, dtype=np.uint8), 
                                         (center_x, center_y), (end_x, end_y), 1).nonzero()))

        # Extract the heatmap intensity values along the line
        intensities = [heatmap[pt[0], pt[1]] for pt in line_points]
        
        # Draw the line segments with varying colors based on heatmap intensities
        for i in range(1, len(line_points)):
            # Get the segment points
            start_pt = line_points[i - 1]
            end_pt = line_points[i]
            
            # Normalize intensity value for colormap
            intensity_value = intensities[i]
            color = plt.cm.hot(intensity_value)  # Using the 'hot' colormap for heatmap intensity
            
            # Draw the line with the colormap value
            ax.plot([start_pt[1], end_pt[1]], [start_pt[0], end_pt[0]], color=color, linewidth=2)
    
    # Customize the plot
    ax.set_title(title)
    ax.axis('off')
    
    # Show the final plot
    plt.savefig(output_path)

def get_diameter_intensity_profiles(heatmap, num_diameters=4):
    """
    Extract intensity profiles along diameters through the center of the heatmap.
    
    Parameters:
    - heatmap: 2D array representing the heatmap intensities.
    - num_diameters: Number of diameters to consider (default is 4: horizontal, vertical, and two diagonals).
    
    Returns:
    - A dictionary containing intensity profiles for each diameter.
    """
    h, w = heatmap.shape
    center_y, center_x = h // 2, w // 2

    # Calculate the theoretical diameter of the image
    max_diameter = int(np.sqrt((w / 2) ** 2 + (h / 2) ** 2)) * 2
    r = max_diameter / 2

    # Store the profiles
    profiles = {}
    
    # Define angles for horizontal, vertical, and diagonal diameters
    angles = np.linspace(0, np.pi, num_diameters, endpoint=False)

    # Extract profile along each diameter
    for idx, angle in enumerate(angles):
        # Calculate start and end points for the line
        x1 = int(center_x - r * np.cos(angle))
        y1 = int(center_y - r * np.sin(angle))
        x2 = int(center_x + r * np.cos(angle))
        y2 = int(center_y + r * np.sin(angle))
        
        # Use OpenCV to get points along the line
        line_mask = np.zeros_like(heatmap, dtype=np.uint8)
        cv2.line(line_mask, (x1, y1), (x2, y2), 1, 1)
        points = list(zip(*line_mask.nonzero()))

        # Get intensity values along the diameter
        intensity_profile = [heatmap[pt[0], pt[1]] for pt in points]

        # Save the profile with a label
        profiles[f"Angle {np.degrees(angle):.1f}°"] = intensity_profile
    
    return profiles

def plot_diameter_profiles(title, profiles, output_path):
    """
    Plot intensity profiles along diameters.

    Parameters:
    - profiles: Dictionary of profiles returned by extract_diameter_profiles.
    """
    # Plot each profile
    plt.figure(figsize=(10, 6))
    for label, profile in profiles.items():
        plt.plot(profile, label=label)
    
    # Customize plot
    plt.title(title)
    plt.xlabel("Distance along Diameter (pixels)")
    plt.ylabel("Intensity Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas', default='average_OD_image.png')
    parser.add_argument('--lat', default='OD')
    parser.add_argument('--masks-folder', default='OD_masks/')
    parser.add_argument('--save-to', default='OD_heatmaps/')
    parser.add_argument('--use-cuda', action='store_true')
    args = parser.parse_args()

    masks_files = [os.path.join(args.masks_folder, f) for f in os.listdir(args.masks_folder)]

    # Load masks
    masks = load_masks(masks_files)

    # generate heatmaps
    heatmap = generate_heatmap(masks)

    # Generate plots from heatmap
    # 2d heatmaps
    plot_2d_heatmap_on_reference(f'2D Heatmap of GA on Reference {args.lat} Eye Image', heatmap, args.atlas, masks, output_path=os.path.join(args.save_to, 'heatmap', f'{args.lat}_2d_heatmap_output.png'))
    # 3d heatmaps
    plot_3d_heatmap_on_reference(f'3D Heatmap of GA on Reference {args.lat} Eye Image', heatmap, args.atlas, masks, output_path=os.path.join(args.save_to, 'heatmap', f'{args.lat}_3d_heatmap_output.png'))
    # radial profiles
    plot_diameter_profiles(f'Radial Profiles - {args.lat}', get_diameter_intensity_profiles(heatmap, num_diameters=4), output_path=os.path.join(args.save_to, 'heatmap', f'{args.lat}_radial_profiles.png'))