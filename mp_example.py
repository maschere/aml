# see https://stumpy.readthedocs.io/en/latest/tutorials.html
# %% imports

import pandas as pd
import stumpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
import datetime as dt

plt.style.use('ggplot')

# %% load data
steam_df = pd.read_csv("https://zenodo.org/record/4273921/files/STUMPY_Basics_steamgen.csv?download=1")
steam_df.head()
# %% plot timeseries
plt.suptitle('Steamgen Dataset', fontsize='24')
plt.xlabel('Time', fontsize ='16')
plt.ylabel('Steam Flow', fontsize='16')
plt.plot(steam_df['steam flow'].values)
plt.show()
# %% calc matrix profile
m = 640
mp = stumpy.stump(steam_df['steam flow'], m)
#mp = stumpy.gpu_stump(steam_df['steam flow'], m)
mp_pd = pd.DataFrame(mp)
mp_pd.columns = ["nn.distance", "nn.idx", "nn.left.idx", "nn.right.idx"]

# %% find motifs
mp_pd.sort_values("nn.distance")

motif_idx = mp_pd.sort_values("nn.distance").index[0]

print(f"The motif is located at index {motif_idx}")
nearest_neighbor_idx = mp[motif_idx, 1]

print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")
print(f"The (normalized) distance is {mp[motif_idx, 0]}")


# %% plot motifs
fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery', fontsize='24')

axs[0].plot(steam_df['steam flow'].values)
axs[0].set_ylabel('Steam Flow', fontsize='16')
rect = Rectangle((motif_idx, 0), m, 40, facecolor='yellow')
axs[0].add_patch(rect)
rect = Rectangle((nearest_neighbor_idx, 0), m, 40, facecolor='yellow')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='16')
axs[1].set_ylabel('Matrix Profile', fontsize='16')
axs[1].axvline(x=motif_idx, linestyle="dashed",color='yellow')
axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed",color='yellow')
axs[1].plot(mp[:, 0])
plt.show()

# %% anomaly detection
mp_pd.sort_values("nn.distance", ascending=False)

discord_idx = mp_pd.sort_values("nn.distance", ascending=False).index[0]

nearest_neighbor_distance = mp[discord_idx, 0]

print(f"The anomaly is located at index {discord_idx}")
print(f"The nearest neighbor subsequence to this anomaly is {nearest_neighbor_distance} units away")
# %% plot anomalies
fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Discord (Anomaly/Novelty) Discovery', fontsize='24')

axs[0].plot(steam_df['steam flow'].values)
axs[0].set_ylabel('Steam Flow', fontsize='16')
rect = Rectangle((discord_idx, 0), m, 40, facecolor='yellow')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='16')
axs[1].set_ylabel('Matrix Profile', fontsize='16')
axs[1].axvline(x=discord_idx, linestyle="dashed",color="yellow")
axs[1].plot(mp[:, 0])
plt.show()
# %%
