# %%
import logging, csv, os
from pathlib import Path
import time
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from datetime import datetime
from scipy.optimize import linear_sum_assignment

import tensorflow as tf
print(tf.__version__)
sys_details = tf.sysconfig.get_build_info()
print(sys_details["cuda_version"])
from tensorflow.keras.layers import Layer, InputSpec, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist

from MDEC_autoencoder import ConvAutoencoder
from MDEC_clusteringlayer import ClusteringLayer
from MDEC_main import JDEC, load_mnist 
# %%
def align_cluster_labels(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_map = {row: col for row, col in zip(row_ind, col_ind)}
    return np.vectorize(label_map.get)(y_pred)
# %%
# 1. Define the SAME model architecture
input_shape = (28, 28, 1)  # Must match original training shape
bottleneck_size = 10         # Must match original training
n_clusters = 10              # Must match original training

# %%
# 2. Instantiate JDEC with the SAME parameters
jdec_new = JDEC(
    input_shape=input_shape,
    bottleneck_size=bottleneck_size,
    n_clusters=n_clusters,
    batch_size=128  # Match original batch size (optional for inference)
)
# %%
# 3. Load the pretrained weights
jdec_new.model.load_weights('/home/unitx/wabbit_playground/nn/clustering_log/JDEC_model.weights.h5')

# %%
# 4. Test on new data (example)
x, y, x_val, y_val = load_mnist()  # Load new data (shape: [N, 28, 28, 1])

# %%
# Predict clusters
cluster_labels = jdec_new.predict_clusters(x_val)
aligned_labels = align_cluster_labels(y_val, cluster_labels)
print("Cluster assignments:", cluster_labels)
print("Validation assignments:", x_val.shape)

print(x_val.shape)
print(aligned_labels.shape)
# %%
n = np.random.randint(0,x_val.shape[0])
plt.imshow(x_val[n], cmap='gray')
plt.show()

print(f'This is the number {aligned_labels[n]}')

# %%