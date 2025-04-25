''' (Optional) Imputes scale factors and sizes from OCT imaging into FAF imaging '''

import os, argparse
import pandas as pd

def main(args):

    # load dataframe
    df = pd.read_csv(args.csv)

    # impute the missing values for XSlo, YSlo, Scale_X, Scale_Y with the mode
    for col in ['XSlo', 'YSlo', 'Scale_X', 'Scale_Y']:
        mode = df[col].mode()
        if not mode.empty:  # check in case the entire column is NaN
            df[col] = df[col].fillna(mode.iloc[0])

    # df = df[df.Procedure == 'Af']
    df.to_csv(args.save_as, index=False)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--save_as')
    args = parser.parse_args()
    main(args)