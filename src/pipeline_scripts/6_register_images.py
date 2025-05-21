''' Executes EyeLiner Pairwise Registration Pipeline on an Image Dataset '''

# =================
# Install libraries
# =================

import argparse
import logging
import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
from eyeliner import EyeLinerP
from eyeliner.utils import none_or_str, load_image
from eyeliner.lightglue import viz2d
from matplotlib import pyplot as plt
import json
from dateutil.parser import parse

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

# Startup console
console = Console()

progress = Progress(
    TextColumn("[bold green]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
)

# def is_date_format(date_string):
#     formats = ['%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%Y/%m/%d']
    
#     for date_format in formats:
#         try:
#             # Attempt to parse the date string using the current format
#             datetime.strptime(date_string, date_format)
#             return True
#         except ValueError:
#             continue  # If it fails, try the next format
    
#     return False

# def is_date_format(val):
#     import re
#     return bool(re.match(r"\d{2}-\d{2}-\d{4}", str(val))) or bool(re.match(r"\d{2}/\d{2}/\d{4}", str(val)))

def is_date(string):
    try:
        parse(string, fuzzy=False)
        return True
    except (ValueError, TypeError):
        return False

def log_progress(task_name, progress, total):
    message = json.dumps({
        "task": task_name,
        "progress": progress,
        "total": total
    })
    print(message)
    sys.stdout.flush()

def visualize_kp_matches(fixed, moving, keypoints_fixed, keypoints_moving):
    # visualize keypoint correspondences
    viz2d.plot_images([fixed.squeeze(0), moving.squeeze(0)])
    viz2d.plot_matches(keypoints_fixed.squeeze(0), keypoints_moving.squeeze(0), color="lime", lw=0.2)
    return

def create_logger(log_file_name):
    """Creates a logger object that writes to a specified log file."""
    # Create a logger
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

    # Create a file handler that logs debug and higher level messages
    handler = logging.FileHandler(log_file_name)
    handler.setLevel(logging.DEBUG)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)

    return logger, handler

def parse_args():
    parser = argparse.ArgumentParser('Registers a sequence of images')
    # data args
    parser.add_argument('--csv', default='results/test/pipeline_files/images_3.csv', type=str, help='Dataset csv path')
    parser.add_argument('--img_col', default='file_path_coris', type=str, help='Image column')
    parser.add_argument('--vessel_col', default='file_path_vessel_seg', type=none_or_str, help='Vessel column')
    parser.add_argument('--pid_col', default='PID', type=str, help='MRN column')
    parser.add_argument('--lat_col', default='Laterality', type=str, help='Laterality column')
    parser.add_argument('--sequence_col', default='ExamDate', type=str, help='Sequence ordering column')
    parser.add_argument('--size', type=int, default=256, help='Size of images')
    parser.add_argument('--input', default='vessel', help='Input image to keypoint detector', choices=['image', 'vessel'])

    # keypoint detector args
    parser.add_argument('--reg2start', default=True, help='Register all timepoints to the start of the sequence')
    parser.add_argument('--reg_method', help='Registration method', type=str, default='tps')
    parser.add_argument('--lambda_tps', help='TPS lambda parameter', type=float, default=1.)

    # misc
    parser.add_argument('--device', default='cuda:1', help='Device to run program on')
    parser.add_argument('--output_folder', default='', help='Location to save registration output')
    parser.add_argument('--save_as', default='', help='Location to save results')
    args = parser.parse_args()
    return args

class SequentialDataset(Dataset):
    def __init__(self, path, mrn_col, lat_col, sequence_col, image_col, vessel_col=None, input_dim=(512, 512), cmode='rgb', input='img'):
        super(SequentialDataset, self).__init__()
        self.path = path
        self.data = pd.read_csv(self.path)
        self.mrn_col = mrn_col
        self.lat_col = lat_col
        self.sequence_col = sequence_col
        unique_combos = self.data[[mrn_col, lat_col]].drop_duplicates(subset=[mrn_col, lat_col])
        self.unique_combos = list(unique_combos.itertuples(index=False, name=None))
        self.image_col = image_col
        self.vessel_col = vessel_col
        self.cmode = cmode
        self.input_dim = input_dim if isinstance(input_dim, tuple) else (input_dim, input_dim)
        self.input = input

    def __len__(self):
        return len(self.unique_combos)

    def __getitem__(self, index):

        data = dict()
        data['images'] = []
        data['inputs'] = []

        # get the patient
        pat, lat = self.unique_combos[index]

        # get the patient data
        mrn_data = self.data[(self.data[self.mrn_col] == pat) & (self.data[self.lat_col] == lat)]
        # print(mrn_data[self.sequence_col].dropna().iloc[0], is_date_format(mrn_data[self.sequence_col].dropna().iloc[0]))
        # sys.exit(0)
        if is_date(mrn_data[self.sequence_col].dropna().iloc[0]):
            mrn_data.loc[:, self.sequence_col] = pd.to_datetime(mrn_data.loc[:, self.sequence_col]).dt.date
        else:
            mrn_data.loc[:, self.sequence_col] = mrn_data[self.sequence_col].astype(int)
        mrn_data = mrn_data.sort_values(by=self.sequence_col)
        data['df'] = mrn_data

        for i, row in mrn_data.iterrows():

            image = load_image(row[self.image_col])
            data['images'].append(image)

            if self.input == 'image':
                data['inputs'].append(image)

            elif self.input == 'vessel':
                assert self.vessel_col is not None
                vessel = load_image(row[self.vessel_col])
                data['inputs'].append(vessel)

            else:
                raise ValueError('Only img and vessel mode supported.')

        return data

def main(args):

    device = torch.device(args.device)

    # load dataset
    dataset = SequentialDataset(
        path=args.csv,
        image_col=args.img_col,
        mrn_col=args.pid_col,
        lat_col=args.lat_col,
        sequence_col=args.sequence_col,
        vessel_col=args.vessel_col,
        input_dim=(args.size, args.size),
        cmode='rgb',
        input=args.input
    )

    # load pipeline
    eyeliner = EyeLinerP(
        reg=args.reg_method,
        lambda_tps=args.lambda_tps,
        image_size=(3, args.size, args.size),
        device=device
    )

    # make directory and csv to store registration results
    results = []
    reg_matches_save_folder = os.path.join(args.output_folder, 'registration_keypoint_matches')
    reg_params_save_folder = os.path.join(args.output_folder, 'registration_params')
    reg_logs_save_folder = os.path.join(args.output_folder, 'registration_logs') 
    os.makedirs(reg_logs_save_folder, exist_ok=True)

    with Live(console=console, refresh_per_second=10) as live:
        for i in range(len(dataset)):

            # get patient sequential data
            batch_data = dataset[i]

            # progress bar
            mrn = batch_data['df'][args.pid_col].unique()[0] 
            lat = batch_data['df'][args.lat_col].unique()[0] 
            task_desc = f"Registering {mrn}_{lat}"
            task_id = progress.add_task(task_desc, total=len(batch_data['images']) - 1)
            panel = Panel(Group(progress), title='Registration')
            live.update(panel)

            # make dir
            os.makedirs(os.path.join(reg_params_save_folder, f'{mrn}_{lat}'), exist_ok=True)
            os.makedirs(os.path.join(reg_matches_save_folder, f'{mrn}_{lat}'), exist_ok=True)

            # create logs file
            logger, handler = create_logger(os.path.join(reg_logs_save_folder, f'{mrn}_{lat}.log'))

            # registration intermediate tensors saved here 
            sequence_registered_images = [batch_data['images'][0]]
            sequence_registered_inputs = [batch_data['inputs'][0]]

            # registration filepaths are saved here
            registration_matches_filenames = [None]
            registration_params_filepaths = [None]
            statuses = ['None']
            
            for j in range(1, len(batch_data['images'])):
                progress.update(task_id, advance=1)

                is_registered = False

                # register to the starting point
                if args.reg2start:

                    try:
                        logger.info(f"Registering timepoint {j} to 0.")

                        # prepare input dictionary
                        data = {
                            'fixed_input': sequence_registered_inputs[0],
                            'moving_input': batch_data['inputs'][j],
                            'fixed_image': sequence_registered_images[0],
                            'moving_image': batch_data['images'][j],
                        }

                        # compute the registration and save it
                        theta, cache = eyeliner(data)
                        filename = os.path.join(reg_params_save_folder, f'{mrn}_{lat}', f'reg_{i}_{j}_0.pth')
                        registration_params_filepaths.append(filename)
                        torch.save(theta, filename)

                        # visualize keypoint matches and save
                        filename = os.path.join(reg_matches_save_folder, f'{mrn}_{lat}', f'kp_match_{i}_{j}_0.png')
                        visualize_kp_matches(
                            data['fixed_image'], 
                            data['moving_image'], 
                            cache['kp_fixed'], 
                            cache['kp_moving']
                            )
                        plt.savefig(filename)
                        plt.close()
                        registration_matches_filenames.append(filename)

                        is_registered = True
                        logger.info(f"Successfully registered timepoint {j} to 0.")

                    except Exception as e:

                        # get the previous timepoint
                        k = j - 1
                        
                        # log the error
                        logger.error(f"Could not register timepoint {j} to 0. Function failed with error: {e}.")

                        # could not register
                        while True:
                            # don't register if you're back to timepoint 0!
                            if k == 0:
                                is_registered = False
                                registration_params_filepaths.append(None)
                                registration_matches_filenames.append(None)
                                logger.info(f"Saving unregistered image.")
                                break

                            # try to re-register
                            try:
                                logger.info(f"Registering timepoint {j} to {k}.")

                                # prepare input dictionary
                                data = {
                                    'fixed_input': sequence_registered_inputs[k],
                                    'moving_input': batch_data['inputs'][j],
                                    'fixed_image': sequence_registered_images[k],
                                    'moving_image': batch_data['images'][j],
                                }

                                # compute registration and save
                                theta, cache = eyeliner(data)
                                filename = os.path.join(reg_params_save_folder, f'{mrn}_{lat}', f'reg_{i}_{j}_{k}.pth')
                                registration_params_filepaths.append(filename)
                                torch.save(theta, filename)

                                # visualize keypoint matches and save
                                filename = os.path.join(reg_matches_save_folder, f'{mrn}_{lat}', f'kp_match_{i}_{j}_{k}.png')
                                visualize_kp_matches(
                                    data['fixed_image'], 
                                    data['moving_image'], 
                                    cache['kp_fixed'], 
                                    cache['kp_moving']
                                    )
                                plt.savefig(filename)
                                plt.close()
                                registration_matches_filenames.append(filename)

                                is_registered = True
                                logger.info(f"Successfully registered timepoint {j} to {k}.")
                                break
                            except:
                                is_registered = False
                                logger.error(f"Could not register timepoint {j} to {k}. Function failed with error: {e}.")
                                k = k - 1

                # register to the previous timepoint
                else:
                    k = j - 1
                    
                    while True:
                        try:
                            logging.info(f"Registering timepoint {j} to {k}.")

                            # prepare input dictionary
                            data = {
                                'fixed_input': sequence_registered_inputs[k],
                                'moving_input': batch_data['inputs'][j],
                                'fixed_image': sequence_registered_images[k],
                                'moving_image': batch_data['images'][j],
                            }

                            # compute the registration and save it
                            theta, cache = eyeliner(data)
                            filename = os.path.join(reg_params_save_folder, f'{mrn}_{lat}', f'reg_{i}_{j}_{k}.pth')
                            registration_params_filepaths.append(filename)
                            torch.save(theta, filename)

                            # visualize keypoint matches and save
                            filename = os.path.join(reg_matches_save_folder, f'{mrn}_{lat}', f'kp_match_{i}_{j}_{k}.png')
                            visualize_kp_matches(
                                data['fixed_image'], 
                                data['moving_image'], 
                                cache['kp_fixed'], 
                                cache['kp_moving']
                                )
                            plt.savefig(filename)
                            plt.close()
                            registration_matches_filenames.append(filename)

                            is_registered = True
                            logging.info(f"Successfully registered timepoint {j} to {k}.")
                            break

                        except Exception as e:
                            is_registered = False
                            logging.error(f"Could not register timepoint {j} to {k}. Function failed with error: {e}.")
                            if k == 0:
                                registration_params_filepaths.append(None)
                                registration_matches_filenames.append(None)
                                logging.info(f"Saving unregistered image.")
                                break
                            else:
                                k = k - 1

                # create registered image and store for next registration
                if is_registered:

                    # apply paramters to image
                    try:
                        reg_image = eyeliner.apply_transform(theta[1], data['moving_image'].squeeze(0))
                    except:
                        reg_image = eyeliner.apply_transform(theta, data['moving_image'].squeeze(0))

                    try:
                        reg_input = eyeliner.apply_transform(theta[1], data['moving_input'].squeeze(0))
                    except:
                        reg_input = eyeliner.apply_transform(theta, data['moving_input'].squeeze(0))

                    sequence_registered_images.append(reg_image)
                    sequence_registered_inputs.append(reg_input)
                    statuses.append('Pass')
                else:
                    sequence_registered_images.append(data['moving_image'])
                    sequence_registered_inputs.append(data['moving_input'])
                    statuses.append('Fail') 

            # save registered sequence
            df = batch_data['df']
            df['params'] = registration_params_filepaths
            df['matches'] = registration_matches_filenames
            df['logs'] = [os.path.join(reg_logs_save_folder, f'patient_{i}.log')]*len(df)
            df['status'] = statuses
            results.append(df)

            # Remove handler to prevent duplicate logs in subsequent iterations
            logger.removeHandler(handler)

            # Close the file handler to release the file
            handler.close()

            # Reset task
            progress.remove_task(task_id)
        
    # save results file
    results = pd.concat(results, axis=0, ignore_index=False)
    results.to_csv(args.save_as, index=False)
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)