import os
import pandas as pd
from PIL import Image
import concurrent.futures
import time

# Function to convert and save a j2k image as png
def convert_j2k_to_png(j2k_path, output_folder):
    try:
        # Load the j2k image
        with Image.open(j2k_path) as img:
            # Construct the output path
            file_name = os.path.splitext(os.path.basename(j2k_path))[0] + '.png'
            output_path = os.path.join(output_folder, file_name)
            
            # Save as png
            img.save(output_path, 'PNG')
            print(f"Saved {output_path}")
    except Exception as e:
        print(f"Error converting {j2k_path}: {e}")

# Function to process the j2k files using threading
def process_j2k_files(csv_path, output_folder, num_workers=8):
    # Load CSV
    df = pd.read_csv(csv_path)
    j2k_files = df['file_path_coris'].tolist()  # Assuming the column name is 'file_path'
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Use ThreadPoolExecutor to parallelize the process
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit the tasks to the thread pool
        futures = [executor.submit(convert_j2k_to_png, j2k_file, output_folder) for j2k_file in j2k_files]
        
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    start_time = time.time()

    # Paths
    csv_path = 'data/GA_progression_modelling_data_redone/clean_data_ohsu_09202024.csv'  # Update with your CSV path
    output_folder = '/sddata/projects/Retina-Seg/test_images_2/'  # Update with your desired output folder

    # Start processing with threading
    process_j2k_files(csv_path, output_folder, num_workers=8)

    print(f"Finished in {time.time() - start_time} seconds")
