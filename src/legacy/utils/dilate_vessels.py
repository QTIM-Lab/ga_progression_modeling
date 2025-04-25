import os, sys
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import pytorch_lightning as pl
from monai.losses.dice import DiceLoss

def load_image(image_path, preprocess):
    image = np.array(Image.open(image_path).convert('RGB').resize((1024, 1024)))
    image = image / 255.
    image = preprocess(
            [image.transpose(2, 0, 1)],
            input_boxes = [[[0., 0., 1024., 1024.]]],
            return_tensors="pt",
            do_rescale=False
        )
    return image

def run_medsam(model, image_path, processor, device):
    image = load_image(image_path, processor)
    with torch.no_grad():
        low_res_masks = model.model(
            pixel_values=image['pixel_values'].to(device),
            input_boxes=image['input_boxes'].to(device),
            multimask_output=False
        ) 
    high_res_masks = processor.image_processor.post_process_masks(
            low_res_masks.pred_masks.cpu(),
            image['original_sizes'].cpu(),
            image['reshaped_input_sizes'].cpu(),
            binarize=False
        )
    high_res_masks = torch.cat(high_res_masks, dim=0)
    high_res_masks = high_res_masks.sigmoid()
    return high_res_masks.squeeze(0), image['pixel_values'].squeeze(0)

# Load data for 180 patients 
path = 'results/11182024_coris/area_comparisons_af_12202024_model_5_wmetadata_affine.csv'
df = pd.read_csv(path)
with open('results/11182024_coris/specific_pats_af_12202024_model_5.txt') as f:
    pats = f.read().splitlines()
    pats = [int(pat.split('_')[0]) for pat in pats]
df = df[df.PID.isin(pats)]
df.file_path_coris = df.file_path_coris.apply(lambda x: x.replace('~/mnt/SLCE-1', '/persist/PACS'))

# Load the image
image = df.file_path_coris.tolist()[0]
# vessels = df.file_path_vessel_seg.tolist()[0]

# load thick vessel segmentor
from transformers import SamModel, SamProcessor
checkpoint_path = '/scratch90/veturiy/projects/GA_segmentation_Advaith/segmentation_generic/lightning_logs/vessel_seg/version_0/checkpoints/epoch=99-step=500.ckpt'
class MedSAM(pl.LightningModule):
    def __init__(self, preprocessor, pretrained=None):
        super().__init__()
        
        # load model with pretrained weights
        self.model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
        if pretrained is not None:
            self.model.load_state_dict(torch.load(pretrained)['state_dict'], strict=False)

        # load preprocessing function and metric
        self.preprocessor = preprocessor
        self.metric = DiceLoss(sigmoid=True, include_background=True, squared_pred=True, reduction="mean")

    def forward(self, batch):
        ''' Predicts the raw low resolution mask from MedSAM '''
        x_cur, y_cur, cache = batch

        # if past timepoint not provided, do not condition
        low_res_masks = self.model(
            pixel_values=x_cur,
            input_boxes=cache['input_boxes'].to(self.device),
            multimask_output=False
        ) 

        return low_res_masks

    def common_step(self, batch, batch_idx):
        x_cur, y_cur, cache = batch

        # Forward pass
        outputs = self(batch)

        # resize to original resolution
        probs = self.preprocessor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            cache['x_cur_original_sizes'].cpu(),
            cache['x_cur_reshaped_input_sizes'].cpu(),
            binarize=False
        )

        probs = torch.cat(probs, dim=0)
        probs = probs.squeeze(1)  # squeeze off class dim for metric calculation
        loss = self.metric(probs, y_cur.cpu())
        metric_value = 1 - loss.item()
        return loss, metric_value

    def training_step(self, batch, batch_idx):
        loss, metric_value = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', metric_value, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric_value = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc', metric_value, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metric_value = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_acc', metric_value, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.00005)

processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
model = MedSAM.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    map_location='cpu',
    preprocessor=processor
    )
model.to('cuda:0')
model.eval()  # Set to evaluation mode

thick_vessels, image = run_medsam(model, image, processor, 'cuda:0')
from torchvision.transforms import ToPILImage
ToPILImage()(image).save('image.png')
ToPILImage()(thick_vessels).save('vessels.png')