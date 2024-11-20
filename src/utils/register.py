import os, sys
import pandas as pd

df = pd.read_csv('data/GA_progression_modelling_data_redone/clean_data_talisa_ga_10312024.csv')

# n_afs = 0
for pat, pat_df in df.groupby('PID'):
    for lat, lat_df in pat_df.groupby('Laterality'):
        for date, date_df in lat_df.groupby('ExamDate'):
            af = len(date_df[date_df.Procedure == 'Af'])
            oct = len(date_df[(date_df.Procedure == 'Ir_oct') & (date_df.type == 'SLOImage')])
            

            
