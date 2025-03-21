# %%
import logging, os
from pathlib import Path
import time
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import imageio
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn import metrics
from datetime import datetime

from tensorflow.keras.layers import Layer, InputSpec, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers.schedules import CosineDecay

from MDEC_autoencoder import ConvAutoencoder
from MDEC_clusteringlayer import ClusteringLayer
# %%
# clustering log path 
# this is the only thing that needs to be changed, all metrics below will be pulled from the specified log folder
log_path = 'clustering_log_2025-03-21_16:25'
# %%
# visualize how your model is performing in terms of accuracy and loss over time
# this script can be ran during training, as the logs are updated in specified intervals (see MDEC_main.py)
df = pd.read_csv(f'{log_path}/mdec_log_.csv')

fig = plt.figure(figsize=(14, 6))
fig.suptitle('Accuracy Metrics and Training Loss', fontsize=20)
fig.supxlabel('iteration', fontsize=16)
fig.supylabel('metric value', fontsize=16)

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
    plt.plot(df.iloc[1:,0], df.iloc[1:,i],label=df.columns[i], color='r')
    plt.legend()

plt.tight_layout()
plt.show()
# %%
# turn clustering epoch iamges into a .gif file for easier visualization
images = []
png_files = [f for f in os.listdir(log_path) if f.endswith('.png')]

png_files.sort(key=lambda x: int(
    os.path.splitext(x)[0].split('_')[-1] # sorts files based on the integer at the end of the filename
))

for filename in png_files:
    file_path = os.path.join(log_path, filename)
    images.append(imageio.v2.imread(file_path))

imageio.mimsave(f'{log_path}/clustering_progress.gif', images) # create gif
# %%
# Try out the autoencoder!
# run the current code block to train and test the performance of the autoencoder

# %%
