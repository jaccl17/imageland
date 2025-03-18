# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

# %%
# clustering log path
log_path = 'clustering_log_2025-03-18 12:01'
# %%
df = pd.read_csv(f'{log_path}/mdec_log_.csv')

plt.figure(figsize=(14, 6))

plt.subplot(2, 3, 1)
plt.plot(df.iloc[1:,0], df.iloc[1:,1],label=df.columns[1])
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(df.iloc[1:,0], df.iloc[1:,2],label=df.columns[2])
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(df.iloc[1:,0], df.iloc[1:,3],label=df.columns[3])
plt.legend()

for i in range (4,7):
    plt.subplot(2, 3, i)
    plt.plot(df.iloc[1:,0], df.iloc[1:,i],label=df.columns[i])
    plt.legend()

# plt.xlabel('iteration)
# plt.ylabel('metric value')
plt.show()

# %%
# make clustering gif
images = []
png_files = [f for f in os.listdir(log_path) if f.endswith('.png')]

# sort files numerically by converting filename (without extension) to integer
png_files.sort(key=lambda x: int(
    os.path.splitext(x)[0].split('_')[-1]  # Split on underscores and take last part
))

for filename in png_files:
    file_path = os.path.join(log_path, filename)
    images.append(imageio.imread(file_path))

imageio.mimsave(f'{log_path}/clustering_progress.gif', images) # create gif
# %%
