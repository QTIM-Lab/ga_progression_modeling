import os
from functools import partial
import argparse
import pandas as pd

def change_modality_rows(row, img_path_col, modality_col, modality_val, seg_folder, prefix=''):
    if row[modality_col] == modality_val:
        return os.path.join(seg_folder, prefix + os.path.basename(row[img_path_col]).replace('.j2k', '.png'))
    else:
        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--img_path_col')
    parser.add_argument('--seg_path_col')
    parser.add_argument('--modality_col')
    parser.add_argument('--modality_val')
    parser.add_argument('--prefix')
    parser.add_argument('--seg_folder')
    parser.add_argument('--save_as')
    args = parser.parse_args()

    func = partial(change_modality_rows, 
                   img_path_col=args.img_path_col, 
                   modality_col=args.modality_col, 
                   modality_val=args.modality_val, 
                   seg_folder=args.seg_folder,
                   prefix=args.prefix)

    df = pd.read_csv(args.csv)
    df[args.seg_path_col] = df.apply(func, axis=1)
    for i, row in df[df[args.modality_col] == args.modality_val].iterrows():
        assert os.path.exists(row[args.seg_path_col]), row[args.seg_path_col]
    df.to_csv(args.save_as, index=False)