import logging, csv, os
from pathlib import Path
import time
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.layers import Flatten, Layer, InputSpec, Dense, Input, Reshape, Conv2D, Conv2DTranspose, Conv1D, Conv1DTranspose, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist, fashion_mnist

def ConvAutoencoder( 
                shape = (28, 28, 1),
                kernel_size = 5,
                padding = 'same',
                bottleneck_size = 64):
    
    # encoder layers
    if shape[0] != shape[1]:
        raise Exception("Please ensure a square input tensor (ie: Height = Width)")
    else:
        downsized_shape = int(shape[0]/4)

        inputs = Input(shape=shape, name = 'en_input_layer')
        h = Conv2D(filters=32, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='en_conv1')(inputs)
        h = Conv2D(filters=64, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='en_conv2')(h)
        h = Conv2D(filters=128, kernel_size=3, activation='relu', padding=padding, strides=1, name='en_conv0')(h)
        h = Flatten(name='en_flatten')(h)

        # bottleneck
        bottleneck = Dense(bottleneck_size, activation='relu', name='bottleneck')(h)

        # decoder layers
        h = Dense(downsized_shape*downsized_shape*128, activation='relu', input_shape=(64,), name='decoder2')(bottleneck)
        h = Reshape((downsized_shape, downsized_shape, 128), name='de_reshape')(h)
        h = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', padding=padding, strides=2, name='de_deconv1')(h)
        h = Conv2DTranspose(filters=32, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='de_deconv2')(h)
        reconstruction = Conv2DTranspose(shape[2], kernel_size=kernel_size, activation='linear', padding=padding, name='reconstruction')(h)

    return Model(inputs, reconstruction, name='autoencoder')
    
# autoencoder = ConvAutoencoder(shape = (28,28,1), kernel_size=3, padding = 'same', pool_size = 2, bottle_neck_size = 10)
# autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
# history = autoencoder.fit(x, x,
#                 epochs = 20,
#                 shuffle=True,
#                 # validation_data=(x_test, x_test)
#                 )
# autoencoder.save_weights('conv_ae.weights.h5')
# autoencoder.summary()