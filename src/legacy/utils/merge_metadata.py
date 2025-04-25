import os, sys
import argparse
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv')
    parser.add_argument('--pid_col', default="PID")
    parser.add_argument('--patinfo', default=None)
    parser.add_argument('--registryinfo', default=None)
    parser.add_argument('--prsinfo', default=None)
    parser.add_argument('--gompertz', default=None)
    parser.add_argument('--save_as')
    args = parser.parse_args()

    # load df
    merged_df = pd.read_csv(args.data_csv)

    if args.patinfo:
        df_patient_data = pd.read_csv(args.patinfo, encoding='unicode_escape')
        df_patient_data = df_patient_data[['primarymrn', 'birthdate']].drop_duplicates()
        merged_df = pd.merge(merged_df, df_patient_data.rename({'primarymrn': args.pid_col, 'birthdate': 'dob'}, axis=1), how='left')
        merged_df['dob'] = pd.to_datetime(merged_df['dob'])

    if args.registryinfo:
        df_registry_data = pd.read_csv(args.registryinfo, encoding='unicode_escape')
        registry_patients = df_registry_data.mrn.unique().tolist()
        merged_df['in_registry'] = merged_df[args.pid_col].apply(lambda x: x in registry_patients)

    if args.prsinfo:
        df_prs_data = pd.read_csv(args.prsinfo, sep='\t')
        df_prs_data = df_prs_data[['MRN', 'PGS004606']].drop_duplicates()
        merged_df = pd.merge(merged_df, df_prs_data.rename({'MRN': args.pid_col}, axis=1), how='left')

    if args.gompertz:
        df_gompertz_data = pd.read_csv(args.gompertz)
        df_gompertz_data = df_gompertz_data.drop_duplicates()
        df_gompertz_data['PID'] = df_gompertz_data.ident.apply(lambda x: int(x.split('_')[0]))
        df_gompertz_data['Laterality'] = df_gompertz_data.ident.apply(lambda x: x.split('_')[1])
        merged_df = pd.merge(merged_df, df_gompertz_data, how='left')

    merged_df.to_csv(args.save_as, index=False)