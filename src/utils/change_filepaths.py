import os, sys
import pandas as pd

filename = 'data/GA_progression_modelling_data_redone/clean_data_talisa_ga_10312024.csv'
save_as = filename
df = pd.read_csv(filename)
df_new = df.copy(deep=True)

########### CHANGE FILENAME IN A COLUMN ##############
old_val = ''
new_val = ''

df_new['file_path_ga_seg'] = df_new.apply(lambda row: '/sddata/data/GA_progression_modelling_data_redone/segmentations_talisa_ga/ga_segs_masks_ir/seg_' + os.path.basename(row.file_path_coris).replace('.j2k', '.png') if row['type'] == 'SLOImage' else row.file_path_ga_seg, axis=1)
# print(df_new[df_new['type'] == 'SLOImage'].file_path_ga_seg.apply(os.path.exists))
# print(df_new.columns)

df_new.to_csv(filename, index=False)
########### CHANGE FILENAME IN A COLUMN ##############