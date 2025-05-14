''' An inference script to get GA segmentation masks '''

import os, sys, argparse
from transformers import SamProcessor, SamModel
import yaml
import torch
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import lightning.pytorch as pl
from typing import Optional
from tqdm import tqdm
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    Identityd,
    LoadImaged,
    ResizeWithPadOrCropd,
    Resized,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    MapTransform,
    Transform,
    Lambdad,
    ToTensord,
    Activations,
    AsDiscrete,
    SaveImage
)
from monai.losses.dice import DiceLoss
import json

def log_progress(task_name, progress, total):
    message = json.dumps({
        "task": task_name,
        "progress": progress,
        "total": total
    })
    print(message)
    sys.stdout.flush()

def get_label_bbox(gt, bbox_shift=20):
    y_indices, x_indices = np.where(gt > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    h, w = gt.shape
    zero_a, zero_b = np.array([0, 0])
    x_min = max(zero_a, x_min - random.randint(0, bbox_shift))
    x_max = min(h, x_max + random.randint(0, bbox_shift))
    y_min = max(zero_b, y_min - random.randint(0, bbox_shift))
    y_max = min(w, y_max + random.randint(0, bbox_shift))
    bboxes = [float(x_min), float(y_min), float(x_max), float(y_max)]
    return bboxes

class LoadImagePILd(MapTransform):
    def __init__(self, keys, mode="RGB"):
        """
        Load image from file path using PIL and convert to RGB (or specified mode).
        
        Args:
            keys (list[str]): Keys in the input dictionary to apply the transform to.
            mode (str): Color mode to convert the image to (default: "RGB").
        """
        super().__init__(keys)
        self.mode = mode

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            path = d[key]
            image = Image.open(path).convert(self.mode)
            d[key] = np.array(image)
        return d

class LoadImagedCV2(MapTransform):
    def __init__(self, keys, convert_to_rgb=True):
        """
        Load image using OpenCV from a path in the data dictionary.

        Args:
            keys (list[str]): Keys in the input dictionary to transform.
            convert_to_rgb (bool): If True, convert all images to RGB format (3 channels).
        """
        super().__init__(keys)
        self.convert_to_rgb = convert_to_rgb

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            path = d[key]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image path does not exist: {path}")

            # load image
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                raise ValueError(f"Failed to load image from {path} using OpenCV")

            # Convert grayscale to RGB if requested
            if self.convert_to_rgb:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3 and img.shape[2] == 4:
                    # RGBA to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # If no conversion is requested, make sure it's float32 for consistency
                img = img.astype(np.float32)

            d[key] = img
        return d

class PreprocessMedSAMInputsd(MapTransform):
    def __init__(self, preprocessor, label_bbox_option="label", image_key="image", label_key="label"):
        super().__init__(keys=[image_key])
        self.preprocessor = preprocessor
        self.label_bbox_option = label_bbox_option
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)

        image = d[self.image_key]  # should be tensor (3, H, W)
        mask = d.get(self.label_key, None)

        # Get bounding box
        if mask is not None:
            if self.label_bbox_option == "label":
                input_boxes = [get_label_bbox(mask, bbox_shift=0)]
            elif self.label_bbox_option == "padded_label":
                input_boxes = [get_label_bbox(mask)]
            elif self.label_bbox_option == "image":
                input_boxes = [0., 0., float(image.shape[2]), float(image.shape[1])]
            else:
                raise ValueError(f"Unsupported bbox option: {self.label_bbox_option}")
        else:
            input_boxes = [0., 0., float(image.shape[2]), float(image.shape[1])]

        # Run MedSAM preprocessor
        output = self.preprocessor(
            [image.numpy()],
            input_boxes=[[input_boxes]],
            return_tensors="pt",
            do_rescale=False
        )

        # Inject outputs into dict
        d["pixel_values"] = output["pixel_values"][0]
        d["original_sizes"] = output["original_sizes"][0]
        d["reshaped_input_sizes"] = output["reshaped_input_sizes"][0]
        d["input_boxes"] = output["input_boxes"][0]

        # remove image from d
        del d["image"]

        return d

class ApplyCLAHEd(MapTransform):
    def __init__(self, keys, clip_limit, tile_grid_size, channels="first"):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on a 2D or 3D image.
        
        Args:
            keys (list): Keys in the data dictionary to apply the transform to.
            clip_limit (float): Threshold for contrast limiting.
            tile_grid_size (tuple): Size of the grid for histogram equalization.
            channels (str): "first" for (C, H, W), "last" for (H, W, C).
        """
        super().__init__(keys)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if channels not in {"first", "last"}:
            raise ValueError("channels must be 'first' or 'last'")
        self.channels = channels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            # Convert to uint8 for CLAHE
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

            if img.ndim == 2:
                # Grayscale
                img = self.clahe.apply(img)

            elif img.ndim == 3:
                if self.channels == "first":  # (C, H, W)
                    img = np.stack([self.clahe.apply(img[c]) for c in range(img.shape[0])])
                elif self.channels == "last":  # (H, W, C)
                    img = np.stack([self.clahe.apply(img[..., c]) for c in range(img.shape[2])], axis=-1)

            else:
                raise ValueError(f"Unsupported image shape for CLAHE: {img.shape}")

            d[key] = img.astype(np.float32)
        return d

class GenericModel(pl.LightningModule):
    '''
    Credit to Weights and Biases
    '''
    def __init__(self, configs):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = configs['lr']
        self.optimizer_name = configs['optimizer_name']
        self.scheduler_name = configs['scheduler_name']
        self.adamw_weight_decay = configs['adamw_weight_decay']
        self.sgd_momentum = configs['sgd_momentum']

    # will be used during inference
    def forward(self, x):
        return x

    def common_step(self, batch, batch_idx):
        loss = 0
        metric_value = 0

        return loss, metric_value

    def training_step(self, batch, batch_idx):
        loss, metric_value = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', metric_value, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric_value = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', metric_value, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metric_value = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', metric_value, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Optimizer
        optimizer: torch.optim.Optimizer  # Type hint for optimizer
        if self.optimizer_name == 'adam':
            # Default adam settings, only experiment with AdamW decay
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)
        elif self.optimizer_name == 'adamw':
            # AdamW uses weight decay
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.adamw_weight_decay)
        elif self.optimizer_name == 'sgd':
            # Define an SGD optimizer with momentum
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.sgd_momentum)

        # Scheduler
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        if self.scheduler_name == 'exponential_decay':
            # Exponential decay scheduler
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95,  # Decay rate
            )
        elif self.scheduler_name == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=25,  # Maximum number of iterations
                eta_min=self.learning_rate/50,  # Minimum learning rate
            )
        elif self.scheduler_name == 'cyclic_lr':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.learning_rate/25,
                max_lr=self.learning_rate*25,
                step_size_up=100
            )
        else:
            return optimizer

        return [optimizer], [scheduler]

class SegmentationMedSAM(GenericModel):
    '''
    Credit to Weights and Biases
    '''
    def __init__(self, configs, preprocessor):
        super().__init__(configs=configs)

        self.model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
        self.preprocessor = preprocessor
        self.metric = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    # will be used during inference
    def forward(self, x):
        pixel_values, mask_labels, original_sizes, reshaped_input_sizes, input_boxes = x
        low_res_masks = self.model(
            pixel_values=pixel_values,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
            input_boxes=input_boxes,
            multimask_output=False
        )
        return low_res_masks

    def common_step(self, batch, batch_idx):
        pixel_values, mask_labels, original_sizes, reshaped_input_sizes, input_boxes = batch

        # Forward pass
        outputs = self(batch)

        # resize to original resolution
        probs = self.preprocessor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            original_sizes.cpu(),
            reshaped_input_sizes.cpu(),
            binarize=False
        )

        probs = torch.cat(probs, dim=0)
        probs = probs.squeeze(1)  # squeeze off class dim for metric calculation

        loss = self.metric(probs, mask_labels.cpu())
        metric_value = 1 - loss.item()

        return loss, metric_value

def get_dataloader(csv_path, img_col, label_col, preprocessor, batch_size, num_workers, label_bbox_option, use_histeq):
    
    # load dataframe
    data_df = pd.read_csv(csv_path)

    # Build dicts
    data_dicts = []
    for i, row in data_df.iterrows():
        log_progress('Loading data', i+1, len(data_df))
        item = {
            "image": row[img_col],
            "image_path": row[img_col]
        }
        data_dicts.append(item)

    full_transform = Compose([
        # LoadImagedCV2(keys=["image"]),
        # LoadImagePILd(keys=["image"]),
        Lambdad(keys=["image"], func=lambda x: np.array(Image.open(x).convert("RGB"))),
        ApplyCLAHEd(keys=["image"], clip_limit=4.0, tile_grid_size=(8, 8), channels='last') if use_histeq else Identityd(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=2),
        Resized(keys=["image"], spatial_size=(1024, 1024)),
        NormalizeIntensityd(keys=["image"], subtrahend=[0., 0., 0.], divisor=[255., 255., 255.], nonzero=False, channel_wise=True),
        ToTensord(keys=["image"], dtype=np.float32),
        PreprocessMedSAMInputsd(preprocessor, label_bbox_option=label_bbox_option)
    ])
    
    # Build dataset and loader
    dataset = Dataset(data=data_dicts, transform=full_transform) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) 
    return dataloader

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Initialize Preprocessor
    preprocess = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

    # Get the test dataset
    dataloader = get_dataloader(
        csv_path=args.csv,
        img_col=args.image_col,
        label_col=None,
        preprocessor=preprocess,
        batch_size=1,
        num_workers=4,
        label_bbox_option='image',
        use_histeq=args.hist_eq
    )

    # load model configs
    configs = {
        'lr': 0,
        'optimizer_name': 'adam',
        'scheduler_name': 'none',
        'adamw_weight_decay': 0.0001,
        'sgd_momentum': 0.1
    }

    # load model
    model = SegmentationMedSAM.load_from_checkpoint(
        checkpoint_path=args.weights_path,
        map_location=device,
        configs=dict(configs),
        preprocessor=preprocess
    )
    model.eval()

    results = {args.image_col: [], args.ga_col: []}
    for idx, batch in enumerate(dataloader):
        log_progress('Run Inference', idx+1, len(dataloader))

        # for saving
        file_path = os.path.join(args.output_folder, "seg_" + os.path.basename(batch['image_path'][0]).replace('.j2k', '.png'))
        if os.path.exists(file_path):
            continue

        # get inputs
        pixel_values = batch["pixel_values"].to(device)
        original_sizes = batch["original_sizes"].to(device)
        reshaped_input_sizes = batch["reshaped_input_sizes"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        batch_device = (pixel_values, None, original_sizes, reshaped_input_sizes, input_boxes)

        # run forward pass
        with torch.no_grad():
            preds = model(batch_device)

        # resize to high resolution
        preds = model.preprocessor.image_processor.post_process_masks(
            preds.pred_masks.cpu(),
            original_sizes.cpu(),
            reshaped_input_sizes.cpu(),
            binarize=False
        )
        # get probs
        pred_probs = torch.cat([pred for pred in preds], dim=0)

        # generate segmentation maps
        pred_seg_map = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])(pred_probs).numpy()
        
        # Generate the file path for saving
        cv2.imwrite(file_path, (pred_seg_map.squeeze() * 255).astype(np.uint8))
        results[args.image_col].append(batch["image_path"][0])
        results[args.ga_col].append(file_path)

    # merge with original data
    df_segmentations = pd.DataFrame(results)
    df_segmentations.to_csv(args.save_as, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segments GA from imaging")
    parser.add_argument('--csv', default='results/test/pipeline_files/images_1.csv', type=str)
    parser.add_argument('--image_col', default='file_path_coris', type=str)
    parser.add_argument('--ga_col', default='file_path_ga_seg', type=str)
    parser.add_argument('--hist_eq', default=True, type=bool)
    parser.add_argument('--weights_path', default='weights/ga_seg/epoch=32-step=1023.ckpt', type=str)
    parser.add_argument('--output_folder', default='', type=str)
    parser.add_argument('--save_as', default='', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()
    main(args)

