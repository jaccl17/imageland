# %% [markdown]
## Clustering Neural Network for Automatic Defect Classification
# Trying to see if i can make a neural network that will look at defects and classify/categorize them
# based on which defects appear most similar
# %% [markdown]
### Libraries
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

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist

from nn.MDEC_autoencoder import autoencoder
# %%
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test), axis=0) # (60000, 28, 28) and (10000, 28, 28) become (70000, 28, 28) - this combines the train and test sets
    y = np.concatenate((y_train, y_test), axis=0) # (60000, ) and (10000, ) become (70000, ) - this combines the train and test sets
    x = x.reshape((x.shape[0],-1)) # (70000, 28, 28) becomes (70000, 28*28) or (70000, 784)
    x = x / 50.  # normalize as it does in DEC paper
    return x, y

    
# %%
test_image_path = '/home/unitx/wabbit_playground/nn/cluster_pattern_01.jpg'
test_image_file = tf.io.read_file(test_image_path)
test_image_tensor = tf.image.decode_image(test_image_file, channels=3)

x, y = load_mnist()
print(x.shape)

# %%
test_image_tensor = tf.image.resize(test_image_tensor, [224, 224]) # resize
test_image_tensor = test_image_tensor / 255.0 # normalize
test_image_tensor = tf.expand_dims(test_image_tensor, axis=0) # add batch dimension
# %%
X = test_image_tensor
K = 5 # number of clusters
T = 10 # target distribution update interval
delta = 0.01 # stopping threshold
# %% [markdown]
### Metrics
# These metrics will be used to evaluate the clustering. Note, there are custom accuracy  measurements that can
# be implemented as well, but I will use imported metrics for now

# %%
def cluster_accuracy(y_true, y_pred):
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    
    return print(f'NMI score: {nmi} \n ARI score: {ari} \n')
# %% [markdown]
### Autoencoder
# Structure for autoencoder (encoding and decoding)
# %%


# %% [markdown]
### Clustering Layer
# Used to cluster in the feature space after autoencoder processes encoding layers
# %%
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
        assert len(input_shape) == 2 # dimensions if of the input layer must be 2, otherwise assertion is raised
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

        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2)))
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
            # 'weights': self.weights
            }
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config): # helpful for layers that are difficult to initialize, may be unnecessary
        return cls(**config)
# %%
class JDEC(object):
    def __init__(self,
                 dimensions,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):
        
        super(JDEC, self).__init__()

        self.dimensions = dimensions # array of dimensions to go though as the autoencoder is trained
        self.input_dim = dimensions[0] # size of flattened input image is the first dimension
        self.n_stacks = len(self.dimensions) - 1 # number of layers not including the input layer

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder = autoencoder(self.dimensions) # makes sure the dimensions array is placed into the autoencoder

    def initialize_model(self, ae_weights=None, gamma=0.1, optimizer='adam'):
        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights) # if use pretrained weights for autoencoder if possible
            print('Pretrained weights have been loaded into the Autoencoder')
        else: 
            print('Please provide weights for the Autoencoder to proceed')
            exit()

        bottleneck = self.autoencoder.get_layer(name='bottleneck').output # extract the bottleneck layer from
        # the autoencoder to use for clustering

        self.encoder = Model(inputs=self.autoencoder.input, outputs=bottleneck) # this is the extracted encoder portion
        # of the autoencoder isolated from the decoder. this model can be used independently of the decoder.
        # essentially, you are bypassing the 'return' of the autoencoder function by creating a new model based on
        # the autoencoder input and the bottleneck layer that it produces before decoding. this encoder can now be used
        # for other applications if necessary

        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(bottleneck) # pass the bottleneck
        # layer as the input to the clustering layer. bottleneck layer = feature space
        # THIS IS WHERE THE CLUSTERING HAPPENS 

        self.model = Model(inputs=self.autoencoder.input, outputs=[clustering_layer, self.autoencoder.output])
        # this is another extremely important piece and this is where the DEC and IDEC differ. IDEC outputs both
        # the clustering layer AND the decoding portion of the autoencoder to calculate reconstruction loss to 
        # further optimize the model

        self.model.compile(loss={'clustering': 'kld', 'reconstruction': 'mse'}, 
                           loss_weights=[gamma, 1], 
                           optimizer=optimizer)
        # we have two losses: Loss_clustering, Loss_reconstruction. the former will be calculated used Kullback-Leibler divergence
        # and the latter will be calculated used Mean Square Error (typical for calculating decoding loss)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):
        encoder = Model(self.model.input, self.model.get_layer(name='bottleneck').output)
        # this is functionally the same as 'self.encoder' in the 'initialize_model' function, but is created
        # dynamically whenever feature extraction is required. 'self.encoder' is initialized only once and is tied 
        # to the autoencoder model specifically
        return encoder.predict(x)
    
    def predict_clusters(self, x): # predicts cluster labels based on the output of the clustering layer
        q, _ = self.model.predict(x) # '_' ignores the second output of the model (ie: the autoencoder output); se the 'self.model' assignment above
        return q.argmax(axis=1) # returns the index of the highest probability output along the columns

    @staticmethod
    def target_distribution(q):
        P_num =  tf.square(q) / tf.reduce_sum(q, axis=0)
        P_den = tf.reduce_sum(P_num, axis=1, keepdims=True) # ensures proper broadcasting during divison
        return P_num / P_den


    def clustering(self, x, 
                   y=None,
                   tol=1e-3,
                   update_interval=140,
                   save_interval=5,
                   maxiter=2e4):

        print(f'Update Interval: {update_interval}')
        print(f'Save Interval {save_interval}')       

        print('Initializing clusters with k-means')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20) # n_init is the number of times the algorithm is run with dif centroids, best run is output

        y_pred = kmeans.fit_predict(self.encoder.predict(x)) # performs k-means clustering on the pretrained bottleneck layer (feature space)
        # y_pred_last = y_pred # save the clusters
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_]) # set the weights according to the k-means clustering for good initialzation

        save_dir = '/home/unitx/wabbit_playground/nn/clustering_log'
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        rn = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file = save_dir_path / f'jdec_log_.csv_{rn}'

        header = ['iter', 'nmi', 'ari', 'L', 'Lr', 'Lc', 'timestamp']

        if not log_file.exists():
            pd.DataFrame(columns=header).to_csv(log_file, index=False)

        # using .fit() instead of .train_on_batch() because it has improved and automatic handling
        q, _ = self.model.predict(x, verbose=0) # recall that our model has 2 outputs, the clustering output, which is the input transformed by the student's t-dist (q) AND the autoencoder.
        # here, we only need the q output, so we use '_' to ignore the autoencoder output
        p = self.target_distribution(q)

        dataset = tf.data.Dataset.from_tensor_slices((x, (p, x))) # create a dataset consisting of inputs as well as an array with the target_distribution and it's associated input
        # the first argument 'x' is the input and the second arguments [p, x] are the targets that are used to calculate loss. loss between 'x' and 'p' refer to the  clustering loss 
        # (where 'p' is the target distribution) and loss between 'x' and 'x' refer to reconstruction loss (ie how 'x' compares to itself after encoding)
        # refer back to self.model.compile(...) above where we define how the loss is calculated
        dataset = dataset.batch(self.batch_size).repeat() # repeats the dataset for multiple epochs

        class ClusterCallbacks(tf.keras.callbacks.Callback):
            def __init__(self, x, y, tol, log_file, update_interval, target_distribution, predict_clusters):
                self.y = y
                self.x = x
                self.tol = tol
                self.log_file = log_file
                self.update_interval = update_interval
                self.y_pred_last = None
                self.target_distribution = target_distribution
                self.predict_clusters = predict_clusters
                print(f'\n {self.x.shape}')

                # super(ClusterCallbacks, self).__init__()
            
            def on_epoch_end(self, epoch, logs=None):
                if epoch % self.update_interval == 0: # execute this every update interval
                    print('\nUpdating Target Distribution')
                    print(f'\n {self.x.shape}')
                    q, _ = self.model.predict(self.x) 
                    p = self.target_distribution(q) # update target distribution, this distribution will be used in the following training epochs

                    y_pred = self.predict_clusters(self.x)
                    delta_metric = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0] # compares the values of the current and last predictions and spits out a boolean
                    # and then sums them all up. its divided by the size of the array, and this is compared to the set tolerance to determine if the model is converging

                    self.y_pred_last = y_pred # save current predictions for next assesment

                    if self.y is not None:
                        # metrics for each update
                        ari = np.round(metrics.adjusted_rand_score(self.y, y_pred), 5)
                        nmi = np.round(metrics.normalized_mutual_info_score(self.y, y_pred), 5)
                        # loss = np.round(np.sum(logs.values()), 5)
                        
                        new_row = { # new log entry for each update
                            'iter': epoch,
                            'nmi': nmi,
                            'ari': ari,
                            'L': logs['loss'],  # total loss
                            'Lc': logs['clustering_loss'],  # clustering loss - name defined in self.model.compile
                            'Lr': logs['reconstruction_loss'],  # reconstruction loss - - name defined in self.model.compile
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

                        new_entry = pd.DataFrame([new_row])
                        new_entry.to_csv(self.log_file, mode='a', header=False, index=False) # pandas handles write and close

                        print(f'Iteration {epoch} logged.')
                        print(f'nmi: {nmi}, ari: {ari}\n Total Loss: {logs["loss"]}, Clustering Loss: {logs["clustering_loss"]}, Reconstruction Loss: {logs["reconstruction_loss"]}')

                    if delta_metric < self.tol:
                        print(f'delta: {delta_metric} < tol: {self.tol}')
                        print('Tolerance reached, training will be stopped.')
                        self.model.stop_training = True

        self.model.fit( # the training procedure
            dataset, # define x inputs and y targets
            epochs = int(maxiter), # number of epochs / loop
            steps_per_epoch = int(x.shape[0] / self.batch_size), # essentially samples per epoch given a batch size: num_samples / batch_size
            callbacks = [ 
                ClusterCallbacks(x=x, y=y, tol=tol, log_file=log_file, update_interval=update_interval, target_distribution=JDEC.target_distribution, predict_clusters = self.predict_clusters), # updates y_pred and saves metrics at update intervals
                tf.keras.callbacks.ModelCheckpoint( # saves the model after each save_interval
                    filepath=os.path.join(save_dir, f'JDEC_model.weights.h5'),
                    save_weights_only=True,
                    save_freq=save_interval * int(x.shape[0] / self.batch_size)
                )
            ],
            verbose = 1 # progress bar
        )

        # y_pred = self.predict_clusters(x)

        return y_pred
    
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='mnist', choices=['mnist', 'usps', 'reutersidf10k'], help="Choose a training dataset")
    parser.add_argument('--n_clusters', default=10, help="How many clusters to fit", type=int)
    parser.add_argument('--batch_size', default=256, help="Samples per batch", type=int)
    parser.add_argument('--maxiter', default=2e4, help="", type=int)
    parser.add_argument('--gamma', default=0.1, help="Coefficient of clustering loss", type=float)
    parser.add_argument('--epochs', default=10, help="Number of training epochs", type=int)
    parser.add_argument('--update_interval', default=140, help="Number of epochs (interval) between target distribution P updates", type=int)
    parser.add_argument('--save_interval', default=5, help="Number of epochs (interval) between model saves", type=int)
    parser.add_argument('--tol', default=0.001, help="Model convergance tolerance", type=float)
    parser.add_argument('--ae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='/home/unitx/wabbit_playground/nn/clustering_log', help="Location for logs and model weights")
    args = parser.parse_args()
    print(args)

    optimizer = SGD(learning_rate=0.1, momentum=0.99)

    if args.dataset == 'mnist':
        x, y = load_mnist()
        optimizer = 'adam'

    
    jdec = JDEC(dimensions=[x.shape[-1], 500, 500, 2000, 8], n_clusters=args.n_clusters, batch_size=args.batch_size) # selects the last dimension of x (784) to by the input array size and 10 to be the bottleneck size
    jdec.initialize_model(ae_weights=args.ae_weights, gamma=args.gamma, optimizer=optimizer)
    plot_model(jdec.model, to_file='jdec_model.png', show_shapes=True)
    jdec.model.summary()

    t0 = time.time()
    y_pred = jdec.clustering(x, y=y, tol=args.tol, update_interval=args.update_interval, save_interval=args.save_interval, maxiter=args.maxiter)
    print(y_pred)
    print(("Clustering time: "), (time.time() - t0))



            

