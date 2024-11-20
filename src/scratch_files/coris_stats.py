import pandas as pd
import sys

df = pd.read_csv('data/GA_progression_modelling_data_redone/clean_data_coris_09162024.csv')
# get visits with AF imaging (optional)
df = df[df.Procedure == 'Af']

print(f'Number of unique patients: {df.PID.nunique()}')
print(f'Number of images: {df.file_path_coris.nunique()}')

res = {'MRN': [], 'Lat': [], 'Earliest': [], 'Latest': [], 'n_visits': [], 'n_images': []}
for mrn, mrn_df in df.groupby('PID'):
    for lat, lat_df in mrn_df.groupby('Laterality'):
        lat_df.ExamDate = pd.to_datetime(lat_df.ExamDate)
        if len(lat_df.ExamDate.unique()) > 3:
            res['MRN'].append(mrn)
            res['Lat'].append(lat)
            res['Earliest'].append(lat_df.ExamDate.min())
            res['Latest'].append(lat_df.ExamDate.max())
            res['n_visits'].append(len(lat_df.ExamDate.unique()))
            res['n_images'].append(len(lat_df.file_path_coris.unique()))
res = pd.DataFrame(res)

# print(f'Earliest date: {res.Earliest.min()}, Latest date: {res.Latest.max()}')

for n in range(30):
   visits = res[res.n_visits > n]
#    visits.Earliest = pd.to_datetime(visits.Earliest)
#    visits.Latest = pd.to_datetime(visits.Latest)
   print(f'Number of patients with > {n} visits: {len(visits.MRN.unique())}, # images: {visits.n_images.sum()}, Earliest: {visits.Earliest.min()}, Latest: {visits.Latest.max()}')