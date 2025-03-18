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

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist

from MDEC_autoencoder import ConvAutoencoder

###############################################################################

@tf.keras.utils.register_keras_serializable(package='CustomLayers')
class ClusteringLayer(Layer):
    """
    This is a custom layer that is used by the deep learning algorithm. It is like a 'Dense' or 'Output' layer
    but serves it's own unique purpose. In this case, the 'Clustering' layer converts an input sample (llke an image)
    into a soft label that assigns cluster labels to that image using Student's t-distribution.

    Input:
        2D tensor with shape (n_samples, n_features)
            This tensor consists of a batch (n_samples) of images, where each 2D (or higher dim.) image
            is flattened into a 1D n_features vector.
    
    Output:
        2D tensor with shape (n_samples, n_clusters)
            The rows of the output tensor each correspond to the input image with the same index (n_samples index),
            but rather than be associated with features, each row has an associated vector of probabilities that 
            communicate the likely hood of that image being associated with each cluster

    Arguments:
        n_clusters: number of clusters
        weights: numpy array with shape (n_clusters, n_features); each n_clusters row represents a cluster
            in an n_features-dimensional feature space
        alpha: parameter in Student's t-distribution (default to 1.0)
    
    Example:
        model.add(ClusteringLayer(n_clusters=10))
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs) # calls the constructor (Keras 'Layer') of this class 
        # and ensures that all kwargs are properly handled

        self.n_clusters = n_clusters # stores n_clusters as an instance variable
        self.alpha = alpha # stores alpha for the Student's t-dist.
        self.initial_weights = weights # stores initial cluster centers if provided
        self.input_spec = InputSpec(ndim = 2)

    def build(self, input_shape):
        input_dim = input_shape[1]

        self.input_spec = InputSpec(dtype=tf.keras.backend.floatx(), shape=(None, input_dim)) # batch size can be
        # anything and the second dimension must match the input dimensions

        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        # adds a trainable weight called 'clusters' to the layer

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights) # updates all trainable weights in the layer (ie: clusters)
            del self.initial_weights # deletes to save space
        
        self.built = True # sets the layer as 'built'; it is properly initialised and ready to be used
    

    def call(self, inputs, **kwargs): # describes how the custom layer transforms its input into its output
        """
        Arguments:
            inputs: tensor containing data of shape (n_samples, n_features)

        Returns:
            q: the soft label (Student's t-distribution) for all samples of shape (n_samples, n_clusters)
        """
        sq_distance = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2)
        sq_distance = tf.maximum(sq_distance, 1e-10)  # prevent division by zero

        q = 1.0 / (1.0 + sq_distance / self.alpha)
        # input dims become (n_samples, 1, n_features); clusters dims automatically shift to (1, n_clusters, n_features)
        # subtraction broadcasts shape to (n_samples, n_clusters, n_features)
        # sum over axis=2 sums along n_featues and reduces dimensions to (n_samples, n_clusters)

        q **= (self.alpha + 1.0) / 2.0
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        # sum over axis=1 sums along n_cluster, but keepdims maintains the summed dimension as 1 for normalization

        return q # returns (n_clusters, n_features)
    
    def compute_output_shape(self, input_shape):
        if not input_shape or len(input_shape) != 2:
             raise ValueError(f"Input shape must be of rank 2. Received: {input_shape}")
        
        return input_shape[0], self.n_clusters
    
    def get_config(self):
        config = {
            'n_clusters': self.n_clusters,
            'alpha': self.alpha,
            'initial_weights': self.initial_weights.tolist() if self.initial_weights is not None else None
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        if config['initial_weights'] is not None:
            config['initial_weights'] = np.array(config['initial_weights'])
        return cls(**config)