import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges pandas dataframes")
    parser.add_argument('--csvs', nargs='+', required=True, help='List of CSV files to join. First is base, rest are joined to it.')
    parser.add_argument('--join_on', required=True, help='Column name to join on.')
    parser.add_argument('--save_as', required=True, help='Path to save the resulting CSV.')
    args = parser.parse_args()

    # Load the base dataframe
    base_df = pd.read_csv(args.csvs[0])

    # Iteratively left join all other CSVs
    for csv_path in args.csvs[1:]:
        df_to_join = pd.read_csv(csv_path)
        base_df = base_df.merge(df_to_join, on=args.join_on, how='left')

    # Save the resulting dataframe
    base_df.to_csv(args.save_as, index=False)
