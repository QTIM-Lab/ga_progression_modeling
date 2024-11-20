import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/08212024_talisa/area_comparisons.csv')

res = {'MRN': [], 'Lat': [], 'n_visits': []}
for mrn, mrn_df in df.groupby('PID'):
    for lat, lat_df in mrn_df.groupby('Laterality'):
        res['MRN'].append(mrn)
        res['Lat'].append(lat)
        res['n_visits'].append(len(lat_df.ExamDate.unique()))

res = pd.DataFrame(res)

for n in range(20):
   visits = res[res.n_visits >= n]
   print(f'Number of patients with >= {n} visits: {len(visits.MRN.unique())}')