import pandas as pd
import sys
import numpy as np
from matplotlib import pyplot as plt
import itertools
import seaborn as sns
from sklearn.cluster import KMeans

def get_acceleration(traj):
    growth_rates = []
    for i in range(1, len(traj)):
        prev = traj[i-1]
        cur = traj[i]
        rate = (cur[1] - prev[1]) / ((cur[0] - prev[0]).days / 365.25)
        growth_rates.append(rate)
    return np.mean(growth_rates)

# load the ga comparisons
df = pd.read_csv('results/07312024/area_comparisons.csv')

# plot trajectory of growth area and get growth rates
trajectories = []
growth_rates = []
start_area = []
for mrn, mrn_df in df.groupby('PID'):
    for lat, lat_df in mrn_df.groupby('Laterality'):
        lat_df = lat_df.drop_duplicates(subset='ExamDate')
        lat_df.loc[:, 'ExamDate'] = pd.to_datetime(lat_df.loc[:, 'ExamDate'])
        lat_df = lat_df.sort_values(by='ExamDate')

        # get monotonically increasing GA growth area
        traj = []
        step = 0
        for date, date_df in lat_df.groupby('ExamDate'):
            if step == 0:
                traj.append((date, date_df.mm_area.unique()[0].item()))
                step += 1
            else:
                if date_df.mm_area.unique()[0].item() >= traj[step-1][1]:
                    traj.append((date, date_df.mm_area.unique()[0].item()))
                    step += 1
            
        # only keep patients with atleast 3 timepoints
        if len(traj) >= 3:
            growth_rates.append(get_acceleration(traj))
            trajectories.append(traj)
            start_area.append(traj[0][1])

growth_rates = pd.Series(growth_rates)
start_area = pd.Series(start_area)

colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])

# Plot growth trajectories over time
plt.figure(figsize=(6, 6))
for traj in trajectories:
    x_coords, y_coords = zip(*traj)
    plt.plot(x_coords, y_coords, linestyle='--', marker='o', color=next(colors))
plt.xlabel('ExamDate')
plt.ylabel('GA Area (mm^2)')
plt.xticks(rotation=90)
plt.title('GA Area vs Time')
plt.savefig('growth_vs_time.png')

# count    107.000000
# mean       4.072091
# std        7.786181
# min        0.000000
# 25%        0.000000
# 50%        0.432062
# 75%        3.803764
# max       39.006132
# dtype: float64

# Plot growth rates, clustering patients as slow and fast progressors 
np.random.seed(0)
# growth_rates = growth_rates[start_area > 3.8]
# print(growth_rates.describe())
data = np.stack([growth_rates.values, start_area.values], axis=1) #np.array(growth_rates, start_area) #.reshape(-1, 1)  # Reshape for KMeans
# print(data.shape)
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
print(centers)

# Prepare data for stripplot
# data = pd.DataFrame({'Value': data.flatten(), 'Cluster': labels})
data = pd.DataFrame({'growth_rate': data[:, 0], 'start_area': data[:, 1], 'Cluster': labels})
plt.figure(figsize=(10, 5))
# sns.stripplot(x='Value', data=data, hue='Cluster', jitter=True, palette='viridis', size=8)
sns.scatterplot(data=data, y='growth_rate', x='start_area', hue='Cluster', palette='viridis', size=8)
# plt.scatter(centers, [0]*len(centers), c='red', marker='x', s=100, label='Centroids')
plt.scatter(centers[:, 1], centers[:, 0], c='red', marker='x', s=100, label='Centroids')
# plt.xlabel('Growth rate (mm^2/year)')
plt.legend()
plt.title('Distribution of growth rates')
plt.savefig('growth_rates.png')
sys.exit(0)

# calculate the margin between clusters
slow_progressors_threshold = data[data.Cluster == 0].Value.max()
fast_progressors_threshold = data[data.Cluster == 1].Value.min()
threshold = (fast_progressors_threshold + slow_progressors_threshold) / 2
print(f'Threshold for fast progressors of AMD: {threshold:.2f} mm^2 / year')

# Plot lines for max of cluster 0, min of cluster 1, and margin
plt.axvline(slow_progressors_threshold, color='blue', linestyle='--', label='Max of Cluster 0')
plt.axvline(fast_progressors_threshold, color='green', linestyle='--', label='Min of Cluster 1')
plt.axvline(threshold, color='red', linestyle='-', label='Margin')
plt.legend()
plt.savefig('growth_rates.png')

# Using the threshold (calculated from AI), get the patient ids and eye's of people who are fast progressors based on manual measurements
fast_progressors_ground_truth = []
slow_progressors_ground_truth = []
df_grouth_truth = df[~df['GA Size Final'].isnull()]
for mrn, mrn_df in df_grouth_truth.groupby('PID'):
    for lat, lat_df in mrn_df.groupby('Laterality'):
        lat_df = lat_df.drop_duplicates(subset='ExamDate')
        lat_df.loc[:, 'ExamDate'] = pd.to_datetime(lat_df.loc[:, 'ExamDate'])
        lat_df = lat_df.sort_values(by='ExamDate')

        if len(lat_df.ExamDate) == 2:
            lat_df = lat_df.reset_index(drop=True)
            # compute the growth rate
            traj = [(lat_df.loc[0, 'ExamDate'], lat_df.loc[0, 'GA Size Final']), (lat_df.loc[1, 'ExamDate'], lat_df.loc[1, 'GA Size Final'])]
            growth_rate = get_acceleration(traj)
            if growth_rate > threshold:
                fast_progressors_ground_truth.append((mrn, lat))
            else:
                slow_progressors_ground_truth.append((mrn, lat))

# Using the threshold (calculated from AI), get the patient ids and eye's of people who are fast progressors based on AI measurements
mrns = df_grouth_truth[['PID', 'Laterality', 'ExamDate']].drop_duplicates(subset=['PID', 'Laterality', 'ExamDate'])

df_ai = df
fast_progressors_ai = []
slow_progressors_ai = []
for mrn, mrn_df in df.groupby('PID'):
    for lat, lat_df in mrn_df.groupby('Laterality'):
        lat_df = lat_df.drop_duplicates(subset='ExamDate')
        lat_df.loc[:, 'ExamDate'] = pd.to_datetime(lat_df.loc[:, 'ExamDate'])
        lat_df = lat_df.sort_values(by='ExamDate')

        # get monotonically increasing GA growth area
        traj = []
        step = 0
        for date, date_df in lat_df.groupby('ExamDate'):
            if step == 0:
                traj.append((date, date_df.mm_area.unique()[0].item()))
                step += 1
            else:
                if date_df.mm_area.unique()[0].item() >= traj[step-1][1]:
                    traj.append((date, date_df.mm_area.unique()[0].item()))
                    step += 1
            
        # only keep patients with atleast 3 timepoints
        if len(traj) >= 3:
            growth_rate = get_acceleration(traj)
            if growth_rate > threshold:
                fast_progressors_ai.append((mrn, lat))
                slow_progressors_ai.append((mrn, lat))

# Convert lists to sets
set1 = set(fast_progressors_ground_truth)
set2 = set(fast_progressors_ai)
set3 = set(slow_progressors_ground_truth)
set4 = set(slow_progressors_ai)

# Find the intersection (common elements) between the two sets
overlap_fast = list(set1 & set2)
overlap_slow = list(set3 & set4)

print(f'{len(overlap_fast)} eyes called fast progressors based on both AI and manual measurements.')
print(f'{len(overlap_slow)} eyes called slow progressors based on both AI and manual measurements.')
print(overlap_fast)
print(overlap_slow)

print(set1)
print(set2)