''' An inference script to get vessel segmentation masks '''

import os, sys, argparse
import yaml
from PIL import Image
import numpy as np
import pandas as pd
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    Identityd,
    LoadImaged,
    Resized,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    Lambdad,
    ToTensord,
    AsDiscrete
)
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,broadcastRGB=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.broad = broadcastRGB

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        
        # self.inc = DoubleConv(n_channels, 224)
        # self.down1 = Down(224, 448)
        # factor = 2 if bilinear else 1
        # self.down2 = Down(448, 896 // factor)
        # self.up1 = Up(896, 448 // factor, bilinear)
        # self.up2 = Up(512, 224,bilinear)
        # self.outc = OutConv(224,n_classes)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x = self.up1(x3,x2)
        # x = self.up2(x,x1)
        # logits = self.outc(x)
        out = torch.div(torch.exp(logits),1+torch.exp(logits))
        if(self.broad):
            out = out.repeat(1,3,1,1)
        return out

def log_progress(task_name, progress, total):
    message = json.dumps({
        "task": task_name,
        "progress": progress,
        "total": total
    })
    print(message)
    sys.stdout.flush()

def get_dataloader(csv_path, img_col, batch_size, num_workers, use_histeq):
    
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
        Lambdad(keys=["image"], func=lambda x: np.array(Image.open(x).convert("RGB"))),
        EnsureChannelFirstd(keys=["image"], channel_dim=2),
        Resized(keys=["image"], spatial_size=(320, 320)),
        NormalizeIntensityd(keys=["image"], subtrahend=[0., 0., 0.], divisor=[255., 255., 255.], nonzero=False, channel_wise=True),
        ToTensord(keys=["image"], dtype=np.float32)
    ])
    
    # Build dataset and loader
    dataset = Dataset(data=data_dicts, transform=full_transform) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) 
    return dataloader

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get the test dataset
    dataloader = get_dataloader(
        csv_path=args.csv,
        img_col=args.image_col,
        batch_size=1,
        num_workers=4,
        use_histeq=args.hist_eq
    )

    # load ensemble model
    models = []
    for i, w in enumerate(os.listdir(args.weights_path)):
        log_progress('Load Ensemble Model', i+1, len(os.listdir(args.weights_path)))
        weight = os.path.join(args.weights_path, w)
        model = UNet(3,1,bilinear=False)
        model.load_state_dict(torch.load(weight, weights_only=False)['model'])
        model.eval()
        models.append(model.to(device))

    results = {'file_path_coris': [], args.vessel_col: []}
    for idx, batch in enumerate(dataloader):
        log_progress('Run Inference', idx+1, len(dataloader))

        # for saving
        file_path = os.path.join(args.output_folder, "seg_" + os.path.basename(batch['image_path'][0]).replace('.j2k', '.png'))
        if os.path.exists(file_path):
            continue

        # get inputs
        batch_device = batch["image"].to(device)

        # run forward pass
        preds = []
        with torch.no_grad():
            for model in models:
                pred = model(batch_device)
                preds.append(pred)
        preds = torch.cat(preds, dim=0).mean(dim=0)

        # generate segmentation maps
        pred_seg_map = Compose([AsDiscrete(threshold=0.5)])(preds).cpu().numpy()
        pred_seg_map = preds.cpu().numpy()

        # Generate the file path for saving
        cv2.imwrite(file_path, (pred_seg_map.squeeze() * 255).astype(np.uint8))
        results['file_path_coris'].append(batch["image_path"][0])
        results[args.vessel_col].append(file_path)

    # merge with original data
    df_segmentations = pd.DataFrame(results)
    df_segmentations.to_csv(args.save_as, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segments retinal vessels from imaging")
    parser.add_argument('--csv', default='results/test/pipeline_files/images_1.csv', type=str)
    parser.add_argument('--image_col', default='file_path_coris', type=str)
    parser.add_argument('--vessel_col', default='file_path_vessel_seg', type=str)
    parser.add_argument('--hist_eq', default=False, type=bool)
    parser.add_argument('--weights_path', default='weights/vessel_seg', type=str)
    parser.add_argument('--output_folder', default='', type=str)
    parser.add_argument('--save_as', default='', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()
    main(args)

