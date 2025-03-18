# %% [markdown]
## Signal decomposition
# The idea is to use convultional neural networks to fit Gaussians and Lorentzians functions to signals while simultaneously fliltering out noise

# %% [markdown]
### Libraries

# %%
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Dense, Input, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist

# %% [markdown]
# NN Branching structure

# %%
class SigDecom(Model):
    def __init__(self):
        super(SigDecom, self).__init__()

        self.lorentzian_branch = tf.keras.Sequential([
            Conv1D(16, kernel_size=3, padding='same', activation='relu'),
            Conv1D(1, kernel_size=3, padding='same')
        ]),

        self.gaussian_branch = tf.keras.Sequential([
            Conv1D(16, kernel_size=3, padding='same', activation='relu'),
            Conv1D(1, kernel_size=3, padding='same')
        ]),

        self.noise_branch = tf.keras.Sequential([
            Conv1D(16, kernel_size=3, padding='same', activation='relu'),
            Conv1D(1, kernel_size=3, padding='same')
        ])

    def call(self, inputs):
        # Decompose the signal
        lorentzian = self.lorentzian_branch(inputs)
        gaussian = self.gaussian_branch(inputs)
        noise = self.noise_branch(inputs)
        
        return lorentzian, gaussian, noise
    

def decom_loss(y_true, y_pred):
    lorentzian, gaussian, noise = y_pred

    reconstruction_loss = tf.reduce_mean(tf.square(y_true - (lorentzian + gaussian + noise)))

    lorentzian_loss = tf.reduce_mean(tf.square(lorentzian - lorentzian_TRUE))

    gaussian_loss = tf.reduce_mean(tf.square(gaussian - gaussian_TRUE))

    lorentzian_loss = tf.reduce_mean(tf.square(lorentzian - lorentzian_TRUE))

    noise_loss = tf.reduce_mean(tf.square(noise))

    total_loss = reconstruction_loss + 0.1 * lorentzian_loss + 0.1 * gaussian_loss + 0.1 * noise_loss