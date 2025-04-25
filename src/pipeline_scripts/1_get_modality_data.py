import argparse
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extracts images of specific modality")
    parser.add_argument('--csv')
    parser.add_argument('--modality_col')
    parser.add_argument('--modality_val')
    parser.add_argument('--n_images', default=None, type=int)
    parser.add_argument('--save_as')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df[args.modality_col] == args.modality_val].reset_index(drop=True)
    df['file_path_coris'] = df['file_path_coris'].apply(lambda x: x.replace('~/mnt/SLCE-1', '/persist/PACS'))
    if args.n_images:
        df = df.iloc[:args.n_images]
    print(f'Found {len(df)} images of {args.modality_val} modality')
    df.to_csv(args.save_as, index=False)