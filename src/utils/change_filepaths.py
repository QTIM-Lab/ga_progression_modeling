import os, sys
import pandas as pd

filename = 'data/GA_progression_modelling_data_redone/data_csvs/clean_data_talisa_ga_10312024.csv'
save_as = filename
df = pd.read_csv(filename)
df_new = df.copy(deep=True)

########### CHANGE FILENAME IN A COLUMN ##############

df_new['file_path_ga_seg'] = df_new.apply(lambda row: row.file_path_ga_seg.replace('segmentations_talisa_ga', 'segmentations/talisa_ga') if row['Procedure'] == 'Af' else None, axis=1)
df_new['file_path_vessel_seg'] = df_new.apply(lambda row: row.file_path_vessel_seg.replace('segmentations_talisa_ga', 'segmentations/talisa_ga') if row['Procedure'] == 'Af' else None, axis=1)
print(df_new[df_new['Procedure'] == 'Af'].file_path_ga_seg.iloc[0])
print(df_new[df_new['Procedure'] == 'Af'].file_path_ga_seg.apply(os.path.exists))
print(df_new[df_new['Procedure'] == 'Af'].file_path_vessel_seg.apply(os.path.exists))

df_new.to_csv(filename, index=False)
########### CHANGE FILENAME IN A COLUMN ##############