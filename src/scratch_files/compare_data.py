import os, sys
import pandas as pd

df_talisa_ga = pd.read_csv('data/GA_progression_modelling_data_redone/clean_data_talisa_ga_10312024.csv')
df_coris = pd.read_csv('data/GA_progression_modelling_data_redone/clean_data_coris_09162024.csv')

talisa_ga_pats = list(df_talisa_ga.PID.unique())
coris_pats = list(df_coris.PID.unique())

print(f'Number of patients in CORIS: {len(coris_pats)}')
print(f'Number of patients in Talisa GA set: {len(talisa_ga_pats)}')

res = tuple(str(pat) for pat in talisa_ga_pats if pat not in coris_pats)
print(res, len(res))