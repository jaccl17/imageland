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
from tensorflow.keras.layers import Flatten, Layer, InputSpec, Dense, Input, Reshape, Conv2D, Conv2DTranspose, Conv1D, Conv1DTranspose, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist, fashion_mnist

###############################################################################

"""
The Convolutional Autoencoder

The following is a convolutional autoencoder that is used in the MDEC algorithm. This autoencoder applies and learns kernels with using convolutions
map to encode a higher dimensional input tensor (eg a 28x28x1-component 2D image) to a lower dimensional botleneck tensor (eg 10-component 1D vector). 
The function of an autoencoder is such that the bottleneck layer (or tensor of the same size) can be subsequently decoded to reconstruct the original
input image. Note that full-connected layers are used to compress and rebuild convolutional layers to and from the bottleneck layer. The autoencoder 
is symmetric, as is typical, which allows the model to learn encoding in decoding with the same tools.

The autoencoder is pretrained, and the bottleneck layer is used for clustering (size of 10 corresponds to 10 clusters); the clustering layer learns to
associate encoded images with 10 different labels (or clusters), each corresponding to a common identifier in the input image (ie: a number from 0-9). 
The decoder component of the autoencoder is used to calculate reconstruction loss (a comparison between the input and reconstructed image), 
which reinforces the algorithm to correctly cluster the images.

Input:
    This function must take the arguments listed below to properly construct the model, but the input of the initialised model is a 2D image (+ channels)
    that is encoded into a bottleneck layer and decoded into a reconstructed image. 


Output:
    A model of autoencoder that can be trained and used for predictions. The output of the input image is a reconstructed image.
    This model can operate in complete indepdence of the MDEC; a comented-out example is encluded below
 
Arguments:
    shape: tuple, shape of the input tensor (default to (28, 28, 1))
    kernel_size: int, HxW size of the convolutional kernel (default to 5 for 5x5)
    padding: str, type of padding to apply to the convolutional layers (default to 'same')
    bottleneck_size: int, size of the bottleneck layer (default to 10)


Example:
    autoencoder = ConvAutoencoder(shape = (28,28,1), kernel_size=3, padding = 'same', bottle_neck_size = 10)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    history = autoencoder.fit(x_train, x_train,
                    epochs = 20,
                    shuffle=True,
                    validation_data=(x_test, x_test)
                    )
    autoencoder.save_weights('conv_ae.weights.h5')
    autoencoder.summary()
"""

def ConvAutoencoder( 
                shape = (28, 28, 1),
                kernel_size = 5,
                padding = 'same',
                bottleneck_size = 10):
    
    # encoder layers
    if shape[0] != shape[1]:
        raise Exception("Please ensure a square input tensor (ie: Height = Width)")
    else:
        downsized_shape = int(shape[0]/4)

        inputs = Input(shape=shape, name = 'en_input_layer')
        h = Conv2D(filters=32, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='en_conv1')(inputs)
        h = Conv2D(filters=64, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='en_conv2')(h)
        # h = Dropout(0.3)(h)
        h = Conv2D(filters=128, kernel_size=3, activation='relu', padding=padding, strides=1, name='en_conv0')(h)
        h = Flatten(name='en_flatten')(h)

        # bottleneck
        bottleneck = Dense(bottleneck_size, activation='relu', name='bottleneck')(h)

        # decoder layers
        h = Dense(downsized_shape*downsized_shape*128, activation='relu', name='decoder2')(bottleneck)
        h = Reshape((downsized_shape, downsized_shape, 128), name='de_reshape')(h)
        h = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', padding=padding, strides=2, name='de_deconv1')(h)
        # h = Dropout(0.3)(h)
        h = Conv2DTranspose(filters=32, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='de_deconv2')(h)
        reconstruction = Conv2DTranspose(shape[2], kernel_size=kernel_size, activation='linear', padding=padding, name='reconstruction')(h)

    return Model(inputs, reconstruction, name='autoencoder')