import os
import pandas as pd

mistakes = os.listdir('more_seg_data/af_images_reannotate')
df2 = pd.read_csv('data/GA_progression_modelling_data_redone/clean_data_coris_09162024.csv')
df2 = df2[df2.Procedure == 'Af']
df2 = df2[df2.file_path_coris.apply(os.path.basename).isin([m.replace('png', 'j2k') for m in mistakes])]
df2[['file_path_coris', 'file_path_ga_seg']].to_csv('mistakes.csv', index=False)