# %% [markdown]
# # MDEC Algorithm (combined)
# Below is a scipt the conbines all of the MDEC components in one script. Please refer to
# the imageland/nn/MDEC repo at https://github.com/jaccl17 for the normal workflow
#
# Please refer to individual code blocks for details

# %% [markdown]
# ## Import libraries

# %%

import logging, os
from pathlib import Path
import time
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import umap

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn import metrics
from datetime import datetime

from tensorflow.keras.layers import Layer, InputSpec, Flatten, Dense, Input, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers.schedules import CosineDecay

print(f'Tensorflow Version: {tf.__version__}')
print(f'Numpy Version: {np.__version__}')
print(f'Pandas Version: {pd.__version__}')
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print(f'Number of GPUs Available: {num_gpus}')

# %% [markdown]
# # Useful Functions
#
# The following are functions used in the workflow of the MDEC algorithm, though are not directly related to the model itself. Their functions are
# briefly described below:
#
#    prediction_accuracy
# This function calculates the accuracy of the model's predictions using the Hungarian algorithm. Because the model is unsupervised, there is 
# not guarantee that a cluster label will be assigned to the same number as the true label (eg: cluster index 1 might actually describe the number 5).
# 
# The Hungarian algorithm uses a confusion matrix to determine the best possible mapping of cluster labels to true labels, and then calculates the
# accuracy of the model based on this mapping.
#
#    load_mnist
# This function loads the MNIST dataset from the tensorflow.keras.datasets module. The dataset is loaded as a tuple of numpy arrays, which are then
# normalized and reshaped to be used in the MDEC algorithm. The function returns the training and validation sets as well as their labels.

# %%
def prediction_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    confusion_matrix = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        confusion_matrix[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    return confusion_matrix[row_ind, col_ind].sum() / y_pred.size

def load_mnist():
    (x, y), (x_val, y_val) = mnist.load_data()
    x = x.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x = tf.expand_dims(x, axis = -1)
    x_val = tf.expand_dims(x_val, axis = -1)
    # y_train = tf.expand_dims(y_train, axis = -1)
    # x_test = tf.expand_dims(x_test, axis = -1)
    # y_test = tf.expand_dims(y_test, axis = -1)
    # x = np.concatenate((x_train, x_test), axis=0) # (60000, 28, 28) and (10000, 28, 28) become (70000, 28, 28) - this combines the train and test sets
    # y = np.concatenate((y_train, y_test), axis=0) # (60000, ) and (10000, ) become (70000, ) - this combines the train and test sets
    # x = x.reshape((x.shape[0],-1)) # (70000, 28, 28) becomes (70000, 28*28) or (70000, 784)  # normalize as it does in DEC paper
    return x, y, x_val, y_val

def augmenter(image):
    image = tf.image.random_flip_left_right(image)  # random horix flip
    image = tf.image.random_flip_up_down(image)  # random vert flip
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # randomly adjust contrast
    return image

# %% [markdown]
# ## The Convolutional Autoencoder
#
# The following is a convolutional autoencoder that is used in the MDEC algorithm. This autoencoder applies and learns kernels with using convolutions
# map to encode a higher dimensional input tensor (eg a 28x28x1-component 2D image) to a lower dimensional botleneck tensor (eg 10-component 1D vector). 
# The function of an autoencoder is such that the bottleneck layer (or tensor of the same size) can be subsequently decoded to reconstruct the original
# input image. Note that full-connected layers are used to compress and rebuild convolutional layers to and from the bottleneck layer. The autoencoder 
# is symmetric, as is typical, which allows the model to learn encoding in decoding with the same tools.
#
# The autoencoder is pretrained, and the bottleneck layer is used for clustering (size of 10 corresponds to 10 clusters); the clustering layer learns to
# associate encoded images with 10 different labels (or clusters), each corresponding to a common identifier in the input image (ie: a number from 0-9). 
# The decoder component of the autoencoder is used to calculate reconstruction loss (a comparison between the input and reconstructed image), 
# which reinforces the algorithm to correctly cluster the images.
#
# ### Input:
# This function must take the arguments listed below to properly construct the model, but the input of the initialised model is a 2D image (+ channels)
# that is encoded into a bottleneck layer and decoded into a reconstructed image.
#
# ### Output:
# A model of autoencoder that can be trained and used for predictions. The output of the input image is a reconstructed image.
# This model can operate in complete indepdence of the MDEC; a comented-out example is encluded below
#
# ### Arguments:
# - shape: tuple, shape of the input tensor (default to (28, 28, 1))
# - kernel_size: int, HxW size of the convolutional kernel (default to 5 for 5x5)
# - padding: str, type of padding to apply to the convolutional layers (default to 'same')
# - bottleneck_size: int, size of the bottleneck layer (default to 10)
#
# ### Example:
#    autoencoder = ConvAutoencoder(shape = (28,28,1), kernel_size=3, padding = 'same', bottle_neck_size = 10)
#    autoencoder.compile(optimizer='adam', loss='mse')
#    history = autoencoder.fit(x_train, x_train,
#                    epochs = 20,
#                    shuffle=True,
#                    validation_data=(x_test, x_test)
#                    )
#    autoencoder.save_weights('conv_ae.weights.h5')
#    autoencoder.summary()

# %%
def ConvAutoencoder( 
                shape = (28, 28, 1),
                kernel_size = 5,
                padding = 'same',
                bottleneck_size = 10):
    
    # encoder layers
    if shape[0] != shape[1]:
        raise Exception("Please ensure a square input tensor (ie: Height = Width)")
    else:
        downsized_shape = int(shape[0]/8)

        inputs = Input(shape=shape, name = 'en_input_layer')
        h = Conv2D(filters=32, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='en_conv1')(inputs)
        h = Conv2D(filters=64, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='en_conv2')(h)
        h = MaxPooling2D(pool_size=2, name='en_pool')(h)
        h = Conv2D(filters=128, kernel_size=3, activation='relu', padding='valid', strides=1, name='en_conv3')(h)
        h = Flatten(name='en_flatten')(h)
        # h = Dense(256, activation='relu', name='encoder1')(h)

        # bottleneck
        bottleneck = Dense(bottleneck_size, activation='relu', name='bottleneck')(h)

        # decoder layers
        # h = Dense(256, activation='relu', name='decoder1')(bottleneck)
        h = Dense(downsized_shape*downsized_shape*128, activation='relu', name='de_dense')(bottleneck)
        h = Reshape((downsized_shape, downsized_shape, 128), name='de_reshape')(h)
        h = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', padding='valid', strides=2, name='de_deconv1')(h)
        h = UpSampling2D(size=2, name='de_upsample')(h)
        h = Conv2DTranspose(filters=32, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='de_deconv2')(h)
        reconstruction = Conv2DTranspose(shape[2], kernel_size=kernel_size, activation='linear', padding=padding, name='reconstruction')(h)

    return Model(inputs, reconstruction, name='autoencoder')

# %%[markdown]
# ## The Clustering Layer

# This is a custom layer that is used by the MDEC algorithm. It is like a 'Conv2D', 'Dense', or 'Output' layer but serves it's own unique purpose. 
# In this case, the 'Clustering' layer is uses a soft assignment technique (Student's t-distribution) to assign the feature space (bottleneck layer) 
# of an input tensor, to 1 of 10 clusters. Each cluster is defined by a centroid that moves to centralize all of the images meant to describe the cluster.

# For example: All cluster centers are initialzed with K-means, a method that assigns centroids based on the feature space described by the pretrained 
# autoencoder. When an image of the number '1' are encoded and shown to the clustering layer, the cluster will adjust it's centroids to better represent 
# the feature space. A well-trained and converging model will move the cluster centroid associated with the number '1' closer to the encoded number '1'
# example in the feature space.

# ### Input:
# 2D tensor with shape (n_samples, n_features). This tensor consists of a batch (n_samples) of images, where each 2D (or higher dim.) image
# is flattened into a 1D vector of length n_features.

# ### Output:
# 2D tensor with shape (n_samples, n_clusters). The rows of the output tensor each correspond to the input image with the same index (n_samples index),
# but rather than be associated with features, each row has an associated vector of probabilities that communicate the likeliood of that image being 
# associated with each cluster.

# ### Arguments:
# - n_clusters: number of clusters
# - weights: numpy array with shape (n_clusters, n_features); each n_clusters row represents a cluster in an n_features-dimensional feature space
# - alpha: parameter in Student's t-distribution (default to 1.0)

# ### Example:
#    clustering_layer = ClusteringLayer(n_clusters=10, weights=clusterweights.weights.h5, name='clustering')(bottleneck)
#    # the bottleneck layer is the input tensor of shape (n_samples, n_features) that is passed to the clustering layer
#    # the clustering_layer variable is used to connect the clustering layer as an output to the bottleneck layer

# %%
@tf.keras.utils.register_keras_serializable(package='CustomLayers')

class ClusteringLayer(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs) # calls the constructor (Keras 'Layer') of this class 
        # and ensures that all kwargs are properly handled

        self.n_clusters = n_clusters # stores n_clusters as an instance variable
        self.alpha = alpha # stores alpha for the Student's t-dist.
        self.initial_weights = weights # stores initial cluster centers if provided
        self.input_spec = InputSpec(ndim = 2)

    def build(self, input_shape):
        # tf.print(f"Building ClusteringLayer with input shape: {input_shape}") # debug print
        
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
        This a vital component of the clutering layer. Since the cluster centers are used as trainable weights, this trainable weights uses 
        the Student's t-distribution to calculate the soft labels (q) for each sample in the input tensor. The soft labels are the associated probabilities
        that each sample (ie an image mapped into a feature space) belongs to each cluster. The cluster centroids (and thereby cluster assignments) are
        updated as the algorithm trains.

        Arguments:
            inputs: tensor containing data of shape (n_samples, n_features)

        Outputs:
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
    
    def compute_output_shape(self, input_shape): # used for debugging
        if not input_shape or len(input_shape) != 2:
             raise ValueError(f"Input shape must be of rank 2. Received: {input_shape}")
        
        return input_shape[0], self.n_clusters
    
    def get_config(self): # used when the model is saved; ensures proper organization of arguments
        config = {
            'n_clusters': self.n_clusters,
            'alpha': self.alpha,
            'initial_weights': self.initial_weights.tolist() if self.initial_weights is not None else None
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config): # also used when model is saved, oragnizes arguments
        if config['initial_weights'] is not None:
            config['initial_weights'] = np.array(config['initial_weights'])
        return cls(**config)

# %% [markdown]
# ## The Modernized Deep Embedded Clustering (MDEC) Algorithm

# The MDEC class unifies both the autoencoder and the clustering layer into an algoirthm for deep embedded clustering. It is a updated version of its DEC, IDEC, and DCEC 
# predecessors that offers better logging and visualisation tools, as well as an improved training workflow (which includes a validation set) and more debugging tools. 
# The algorithm begins with the optional to either pretrain the autoencoder, or input pretrained weights directly; the latter option allows for a reduced total training time,
# more consistant results, and easier debugging of the clustering component of the algorithm.
#
# The clustering function is next big component, and first creates the log file that tracks nmi and ari scores (accuracy metrics), as well as the accuracy from the Hungarian
# algorithm. Clustering loss (KL divergence between the target distribution and the soft labels) and reconstruction loss (difference between input and reconstructed image)
# are also tracked, alongside the total (combined) loss. Each tracked metric is associated with it's epoch and the metrics are taken at each update interval. The target
# distribution is used to improve the soft assignment (Student's t-distribution) by emphasizing high-confidence data-points, and normalizing the loss contributed by each
# centroid (prevents large clusters from overpowering feature space).
#
# The algorithm uses a train_on_batch() approach opposed to a fit() approach, predominantly because train_on_batch() is more conducive to our multi-component loss model, 
# and avoids having the complicated dataset structure that fit() requires to learn from these losses. We also have more flexibility over batching with this function.
#
# This class also includes functions for loading mdec model weights, predicting cluster assignments, encoding images to the feature space, dynamically tracking loss and
# accuracy, as well as turning the cluster mapping into a gif to visualize the training process.
#
# ### Input:
# The algorithm takes in a series of images that each contain a notable feature that indicates how they will be clustered (eg a number from 0-9).
#
# ### Output: 
# The algorithm outputs a model that categorizes an input image into a learned cluster that corresponds to one of the true labels for the image set.
#
# ### Arguments:
# - input_shape: tuple, shape of the input tensor (default to (28, 28, 1))
# - n_clusters: int, number of clusters to fit and the size of the bottleneck (default to 10)
# - batch_size: int, number of samples per batch (default to 256)
# - save_dir: str, location for logs and model weights (default to None)
# - **kwargs: any additional arguments to be passed to the model
#
# ### Example:
#     mdec = MDEC(input_shape=(28,28,1), n_clusters=10, batch_size=64, save_dir='MDEC_model')
#     mdec.pretrainer(x, x_val, batch_size=64, epochs=20, ae_weights=None, show_history=False)
#     plot_model(mdec.model, to_file='mdec_model.png', show_shapes=True)
#     mdec.model.summary()
#     mdec.compile(gamma=1.0, optimizer='adam')
#     y_pred = mdec.clustering(x, y=y, tol=1e-4, update_interval=100, save_interval=200, maxiter=2e4)
#     print('Clustering complete!')
# %%
class MDEC(object):
    def __init__(self,
                 input_shape,
                 n_clusters=10,
                 batch_size=256,
                 save_dir = 'nn/MDEC',
                 **kwargs):
        
        super(MDEC, self).__init__()

        self.input_shape = input_shape # size of flattened input image is the first dimension
        self.n_clusters = n_clusters # number of clusters to fit
        self.batch_size = batch_size # number of samples per batch
        self.save_dir = save_dir # location for logs and model weights
        self.kwargs = kwargs # any additional arguments to be passed to the model

        self.rn = datetime.now().strftime('%Y-%m-%d_%H:%M') # 'right now' timestamp for clustering log

        self.save_dir = f'{self.save_dir}/logs/clustering_log_{self.rn}'

        self.save_dir_path = Path(self.save_dir)
        self.save_dir_path.mkdir(parents=True, exist_ok=True)

        self.autoencoder = ConvAutoencoder(shape=input_shape, kernel_size=5, padding='same', bottleneck_size=n_clusters)
        # self.autoencoder = ConvAutoencoder(self.dimensions) # ensures the dimensions array is placed into the autoencoder

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

        self.model = Model(inputs=self.autoencoder.input, outputs=[clustering_layer, self.autoencoder.output], name='mdec_model')
        # this is another extremely important piece and this is where the DEC and IDEC differ. IDEC outputs both
        # the clustering layer AND the decoding portion of the autoencoder to calculate reconstruction loss to 
        # further optimize the model

    def pretrainer(self, x, x_val, batch_size=128, epochs=20, ae_weights=None, show_history=False):
        t_0 = time.time()

        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights) # use pretrained weights for autoencoder if possible
            print('Pretrained weights have been loaded into the Autoencoder')

        else:
            print('Begin Pretraining...')
            # optimizer = SGD(learning_rate=0.001, momentum=0.9)
            optimizer = Adam(learning_rate=0.001, use_ema=True, ema_momentum=0.99)
            self.autoencoder.compile(optimizer=optimizer, loss='mse')
            history = self.autoencoder.fit(
                x, x,  # input and target are the same (for reconstruction)
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val)
                )
            if show_history:
                train_loss = history.history['loss']
                val_loss = history.history['val_loss']
                window = 2

                ma_train_loss = np.convolve(train_loss, np.ones(window) / window, mode='valid')
                ma_val_loss = np.convolve(val_loss, np.ones(window) / window, mode='valid')

                plt.plot(train_loss, label='train_loss')
                plt.plot(val_loss, label='val_loss')
                plt.plot(range(window - 1, len(train_loss)), ma_train_loss, label=f'train_loss (moving_avg, window={window})', linestyle='--')
                plt.plot(range(window - 1, len(val_loss)), ma_val_loss, label=f'val_loss (moving_avg, window={window})', linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.savefig(f'{self.save_dir}/ae_pretraining_metrics.png')
                plt.show()
        
            print(f'Pretraining complete, took: {time.time() - t_0} seconds')
            self.autoencoder.save_weights(f'{self.save_dir}/cae_pretrain.weights.h5')
            print(f'Pretrained weights saved to log folder as cae_pretrain.weights.h5')

        # self.model.compile(loss={'clustering': 'kld', 'reconstruction': 'mse'}, 
        #                    loss_weights=[gamma, 1], 
        #                    optimizer=optimizer)
        # we have two losses: Loss_clustering, Loss_reconstruction. the former will be calculated used Kullback-Leibler divergence
        # and the latter will be calculated used Mean Square Error (typical for calculating decoding loss)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):
        # this is functionally the same as 'self.encoder' in the 'initialize_model' function, but is created
        # dynamically whenever feature extraction is required. 'self.encoder' is initialized only once and is tied 
        # to the autoencoder model specifically
        return self.encoder.predict(x)
    
    def predict_clusters(self, x): # predicts cluster labels based on the output of the clustering layer
        q, _ = self.model.predict(x) # '_' ignores the second output of the model (ie: the autoencoder output); se the 'self.model' assignment above
        return q.argmax(axis=1) # returns the index of the highest probability output along the columns
    
    def compile(self, gamma=0.1, optimizer='adam'):
        self.model.compile(loss={'clustering': 'kld', 'reconstruction': 'mse'}, 
                           loss_weights=[gamma, 1-gamma], 
                           optimizer=optimizer)

    @staticmethod
    def target_distribution(q):
        P_num =  tf.square(q) / tf.reduce_sum(q, axis=0) 
        P_den = tf.reduce_sum(P_num, axis=1, keepdims=True) # ensures proper broadcasting during divison
        return P_num / P_den

    def track_metrics(self, log_path=None):
        if log_path is None:
            raise Exception('Please provide a log path for accuracy and loss metric display')

        df = pd.read_csv(f'{log_path}/mdec_log.csv')

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
        
        plt.savefig(f'{log_path}/final_metrics.png')

        plt.show()


    def visualize_clustering(self, log_path=None):
        images = []
        png_files = [f for f in os.listdir(log_path) if f.endswith('.png')]

        png_files.sort(key=lambda x: int(
            os.path.splitext(x)[0].split('_')[-1] # sorts files based on the integer at the end of the filename
        ))

        for filename in png_files:
            file_path = os.path.join(log_path, filename)
            images.append(imageio.v2.imread(file_path))

        imageio.mimsave(f'{log_path}/clustering_progress.gif', images) # create gif
        print('Clustering progress gif created and saved to log folder.')

    def clustering(self, x, 
                   y=None,
                   tol=1e-3,
                   update_interval=140,
                   save_interval=140,
                   maxiter=2e4):

        print(f'Update Interval: {update_interval}')
        print(f'Save Interval {save_interval}')       

        print('Initializing clusters with k-means...')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20) # n_init is the number of times the algorithm is run with dif centroids, best run is output

        y_pred = kmeans.fit_predict(self.encoder.predict(x)) # performs k-means clustering on the pretrained bottleneck layer (feature space)
        y_pred_last = np.copy(y_pred) # save the clusters
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_]) # set the weights according to the k-means clustering for good initialzation

        log_file = self.save_dir_path / 'mdec_log_.csv'

        header = ['iter', 'nmi', 'acc', 'ari', 'L', 'Lc', 'Lr', 'timestamp']

        if not log_file.exists():
            pd.DataFrame(columns=header).to_csv(log_file, index=False)

        index = 0
        logs = {'loss': 0, 'clustering_loss': 0, 'reconstruction_loss': 0}

        q, _ = self.model.predict(x, verbose=0) # recall that our model has 2 outputs, the clustering output, which is the input transformed by the student's t-dist (q) AND the autoencoder.
                # here, we only need the q output, so we use '_' to ignore the autoencoder output
        p = self.target_distribution(q)

        for i in range(int(maxiter)):
            if i % update_interval == 0 and i > 0:
                q, _ = self.model.predict(x, verbose=0) # recall that our model has 2 outputs, the clustering output, which is the input transformed by the student's t-dist (q) AND the autoencoder.
                # here, we only need the q output, so we use '_' to ignore the autoencoder output
                p = self.target_distribution(q)

                y_pred = q.argmax(axis=1)
                
                if y is not None:
                    acc = np.round(prediction_accuracy(y, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                    # loss = np.round(np.sum(logs.values()), 5)
                    
                    new_row = { # new log entry for each update
                        'iter': i,
                        'nmi': nmi,
                        'acc': acc,
                        'ari': ari,
                        'L': logs['loss'],  # total loss
                        'Lc': logs['clustering_loss'],  # clustering loss - name defined in self.model.compile
                        'Lr': logs['reconstruction_loss'],  # reconstruction loss - - name defined in self.model.compile
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    new_entry = pd.DataFrame([new_row])
                    new_entry.to_csv(log_file, mode='a', header=False, index=False) # pandas handles write and close

                    print(f'Iteration {i} logged.')
                    print(f'nmi: {np.round(nmi*100, 5)}%, acc: {np.round(acc*100, 5)}%, ari: {np.round(ari*100, 5)}%\n Total Loss: {np.round(logs["loss"], 5)}, Clustering Loss: {np.round(logs["clustering_loss"], 5)}, Reconstruction Loss: {np.round(logs["reconstruction_loss"], 5)}')
                    delta_metric = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] # compares the values of the current and last predictions and spits out a boolean
                    # and then sums them all up. its divided by the size of the array, and this is compared to the set tolerance to determine if the model is converging

                    y_pred_last = np.copy(y_pred)

                    unique_clusters = np.unique(y_pred)
                    print(f"Unique clusters assigned: {unique_clusters}")

                if i > 0 and delta_metric < tol:
                        print(f'delta: {delta_metric} < tol: {tol}')
                        print('Tolerance reached, training will be stopped.')
                        break

            if (index + 1) * self.batch_size > x.shape[0]:
                train_loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                y=[p[index * self.batch_size::], x[index * self.batch_size::]])
                index = 0

            else:
                train_loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                y=[p[index * self.batch_size:(index + 1) * self.batch_size],
                                                x[index * self.batch_size:(index + 1) * self.batch_size]])
                
            logs = {'loss': train_loss[0], 'clustering_loss': train_loss[1], 'reconstruction_loss': train_loss[2]}
                
            if i % save_interval == 0 and i > 0:
                print(f'Saving model to {self.save_dir}/MDEC_model.weights.h5')
                self.model.save_weights(f'{self.save_dir}/MDEC_model.weights.h5')

                bottleneck_features = self.encoder.predict(x)
                features_2d = TSNE(n_components=2).fit_transform(bottleneck_features)
                # features_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(bottleneck_features)
                plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y_pred, cmap='tab10', s=5)
                plt.title(f'Clustering Visualization - Epoch {i}')
                plt.savefig(f'{self.save_dir}/clusters_epoch_{i}.png')
                plt.close()

            i += 1

        return y_pred
# %% [markdown]
# Main

# %%
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='mnist', choices=['mnist', 'usps', 'reutersidf10k'], help="Choose a training dataset")
    parser.add_argument('--n_clusters', default=10, help="Number of clusters to fit", type=int)
    parser.add_argument('--batch_size', default=256, help="Samples per batch", type=int)
    parser.add_argument('--maxiter', default=2e4, help="Train on batch iterations", type=int)
    parser.add_argument('--gamma', default=0.1, help="Coefficient of clustering loss", type=float)
    parser.add_argument('--ae_epochs', default=50, help="Number of autoencoder training epochs", type=int)
    parser.add_argument('--update_interval', default=140, help="Number of epochs (interval) between target distribution P updates", type=int)
    parser.add_argument('--save_interval', default=140, help="Number of epochs (interval) between model saves", type=int)
    parser.add_argument('--tol', default=0.001, help="Model convergance tolerance", type=float)
    parser.add_argument('--ae_weights', default=None, help="Path to pretrained autoencoder weights")
    parser.add_argument('--save_dir', default='nn/MDEC', help="Location for logs and model weights")
    parser.add_argument('--augment', default=False, help="Augment training data", type=bool)
    args = parser.parse_args()
    print(args)

    if args.dataset == 'mnist':
        x, y, x_val, _ = load_mnist()
        # print(x.shape)
        # print(y.shape)
        # print(x_val.shape)

    if args.augment:
        x = tf.map_fn(lambda pic: augmenter(pic), x)

    lr_schedule = CosineDecay(initial_learning_rate=0.001, decay_steps=2000, alpha=0.001)
    # optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    # optimizer = SGD(learning_rate=0.001, momentum=0.99)

    optimizer = Adam(learning_rate=lr_schedule, use_ema=True, ema_momentum=0.9)
    
    mdec = MDEC(input_shape=x.shape[1:], n_clusters=args.n_clusters, batch_size=args.batch_size, save_dir=args.save_dir) # selects the last dimension of x (784) to by the input array size and 10 to be the bottleneck size
    mdec.pretrainer(x, x_val, batch_size=args.batch_size, epochs=args.ae_epochs, ae_weights=args.ae_weights, show_history=True)
    plot_model(mdec.model, to_file=f'{mdec.save_dir}/mdec_model.png', show_shapes=True)
    mdec.model.summary()

    t0 = time.time()
    mdec.compile(gamma=args.gamma, optimizer=optimizer)
    y_pred = mdec.clustering(x, y=y, tol=args.tol, update_interval=args.update_interval, save_interval=args.save_interval, maxiter=args.maxiter)
    print('Clustering complete!')
    t1 = time.time() - t0
    print(f'Clustering time: {time.strftime("%H:%M:%S", time.gmtime(t1))}')

    mdec.track_metrics(log_path=mdec.save_dir) # plot metrics from the log file
    mdec.visualize_clustering(log_path=mdec.save_dir) # create a gif to visualize the clustering progress

# %%
