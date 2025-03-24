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
from MDEC_main import load_mnist, augmenter
# %%
# Clustering Log Path 
# this is the only thing that needs to be changed, the following metrics will populate based on the log path
# Note: this path is not required to try out the autoencoder
log_path = 'clustering_log_2025-03-21_21:29'
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
# turn clustering epoch imagges into a .gif file for easier visualization!
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

# start by running the first cell in this file to import all req libraries 
# then run this cell to train the autoencoder. adjust model parameters to experiment!

x_train, _, x_test, _ = load_mnist() # import train and test data
x_train = tf.map_fn(lambda pic: augmenter(pic), x_train) # augment the test data

autoencoder = ConvAutoencoder(shape=(28,28,1), kernel_size=5, padding='same', bottleneck_size=10)
autoencoder.summary()
optimizer = Adam(learning_rate=0.001, use_ema=True, ema_momentum=0.99)
# optimizer = SGD(learning_rate=0.001, momentum=0.9)
autoencoder.compile(optimizer=optimizer, loss='mse')
history = autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test)
                )
# autoencoder.save_weights('/home/jackwabbit/wabbit_world/imageland/nn/MDEC/test_bin/test_conv_ae.weights.h5')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
# %%
# after running the previous cell, run this one to see how the autoencoder encodes the
# a set of test images into a smaller representation and then decodes them back into their og size.

encoded_pics = autoencoder.predict(x_test)
decoded_pics = autoencoder.predict(encoded_pics)

# %%
# this block selects five random examples from 'x_test' set
# and diplays the orginal and reconstructed images.

# run this cell repeatedly to see different examples!

n = np.random.randint(0,x_test.shape[0],5)
plt.figure(figsize=(15, 4))
for i in range(len(n)):
    ax = plt.subplot(2, len(n), i + 1)
    plt.title(f'original {n[i]}')
    plt.imshow(x_test[n[i]], cmap='gray')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ax = plt.subplot(2, len(n), i + 1 + len(n))
    plt.title(f'reconstructed {n[i]}')
    plt.imshow(decoded_pics[n[i]], cmap='gray')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

plt.show()

# %%
n = np.random.randint(0,x_train.shape[0],5)
plt.imshow(x_train[n[i]], cmap='gray')

# %%
