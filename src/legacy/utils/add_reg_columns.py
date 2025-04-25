import argparse
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--reg')
    parser.add_argument('--save_as')
    args = parser.parse_args()

    # update dataframe
    src_df = pd.read_csv(args.csv)
    reg_df = pd.read_csv(args.reg)
    src_df['params'] = reg_df['params']
    src_df['matches'] = reg_df['matches']
    src_df['video'] = reg_df['video']
    src_df['logs'] = reg_df['logs']
    src_df.to_csv(args.save_as, index=False)