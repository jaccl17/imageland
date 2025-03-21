import logging, os
from pathlib import Path
import time
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
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

print(f'Tensorflow Version: {tf.__version__}')
print(f'Numpy Version: {np.__version__}')
print(f'Pandas Version: {pd.__version__}')
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print(f'Number of GPUs Available: {num_gpus}')

###############################################################################
"""
Useful Functions

The following are functions used in the workflow of the MDEC algorithm, though are not directly related to the model itself. Their functions are
briefly described below:

prediction_accuracy
    This function calculates the accuracy of the model's predictions using the Hungarian algorithm. Because the model is unsupervised, there is 
    not guarantee that a cluster label will be assigned to the same number as the true label (eg: cluster index 1 might actually describe the number 5).
    The Hungarian algorithm uses a confusion matrix to determine the best possible mapping of cluster labels to true labels, and then calculates the
    accuracy of the model based on this mapping.

load_mnist
    This function loads the MNIST dataset from the tensorflow.keras.datasets module. The dataset is loaded as a tuple of numpy arrays, which are then
    normalized and reshaped to be used in the MDEC algorithm. The function returns the training and validation sets as well as their labels.
"""

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

class MDEC(object):
    def __init__(self,
                 input_shape,
                 bottleneck_size,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256,
                 save_dir = None,
                 **kwargs):
        
        super(MDEC, self).__init__()

        self.input_shape = input_shape # size of flattened input image is the first dimension
        self.bottleneck_size = bottleneck_size # array of dimensions to go though as the autoencoder is trained
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.rn = datetime.now().strftime('%Y-%m-%d_%H:%M')
        self.save_dir = save_dir

        if self.save_dir is None:
            self.save_dir = f'/home/jackwabbit/wabbit_world/imageland/nn/MDEC/clustering_log_{self.rn}'

        self.save_dir_path = Path(self.save_dir)
        self.save_dir_path.mkdir(parents=True, exist_ok=True)

        self.autoencoder = ConvAutoencoder(shape=input_shape, kernel_size=5, padding='same', bottleneck_size=bottleneck_size)
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
        print('Begin Pretraining...')
        # print(x.shape)

        t_0 = time.time()

        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights) # use pretrained weights for autoencoder if possible
            print('Pretrained weights have been loaded into the Autoencoder')

        else:
            self.autoencoder.compile(optimizer='adam', loss='mse')
            history = self.autoencoder.fit(
                x, x,  # input and target are the same (for reconstruction)
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val)
                )
            if show_history:
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')  # Only if validation data is provided
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
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
                           loss_weights=[gamma, 1], 
                           optimizer=optimizer)

    @staticmethod
    def target_distribution(q):
        P_num =  tf.square(q) / tf.reduce_sum(q, axis=0) 
        P_den = tf.reduce_sum(P_num, axis=1, keepdims=True) # ensures proper broadcasting during divison
        return P_num / P_den


    def clustering(self, x, 
                   y=None,
                   tol=1e-3,
                   ae_weights=None,
                   update_interval=140,
                   save_interval=140,
                   maxiter=2e4):

        print(f'Update Interval: {update_interval}')
        print(f'Save Interval {save_interval}')       

        print('Initializing clusters with k-means')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20) # n_init is the number of times the algorithm is run with dif centroids, best run is output

        y_pred = kmeans.fit_predict(self.encoder.predict(x)) # performs k-means clustering on the pretrained bottleneck layer (feature space)
        y_pred_last = np.copy(y_pred) # save the clusters
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_]) # set the weights according to the k-means clustering for good initialzation

        log_file = self.save_dir_path / 'mdec_log_.csv'

        header = ['iter', 'nmi', 'acc', 'ari', 'L', 'Lr', 'Lc', 'timestamp']

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
                    print(f'nmi: {nmi}, acc: {acc}, ari: {ari}\n Total Loss: {logs["loss"]}, Clustering Loss: {logs["clustering_loss"]}, Reconstruction Loss: {logs["reconstruction_loss"]}')
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
                plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y_pred, cmap='tab10', s=5)
                plt.savefig(f'{self.save_dir}/clusters_epoch_{i}.png')
                plt.close()

            i += 1

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
    # parser.add_argument('--epochs', default=10, help="Number of training epochs", type=int)
    parser.add_argument('--update_interval', default=140, help="Number of epochs (interval) between target distribution P updates", type=int)
    parser.add_argument('--save_interval', default=140, help="Number of epochs (interval) between model saves", type=int)
    parser.add_argument('--tol', default=0.001, help="Model convergance tolerance", type=float)
    parser.add_argument('--ae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default=None, help="Location for logs and model weights")
    args = parser.parse_args()
    print(args)

    # optimizer = SGD(learning_rate=0.1, momentum=0.99)

    if args.dataset == 'mnist':
        x, y, x_val, _ = load_mnist()
        # print(x.shape)
        # print(y.shape)


    lr_schedule = CosineDecay(initial_learning_rate=0.005, decay_steps=1000, alpha=0.001)
    optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    # optimizer = SGD(learning_rate=0.001, momentum=0.99)

    # optimizer = Adam(learning_rate=0.001, use_ema=True, ema_momentum=0.9)
    
    mdec = MDEC(input_shape=x.shape[1:], bottleneck_size=10, n_clusters=args.n_clusters, batch_size=args.batch_size, save_dir=args.save_dir) # selects the last dimension of x (784) to by the input array size and 10 to be the bottleneck size
    mdec.pretrainer(x, x_val, batch_size=args.batch_size, epochs=50, ae_weights=args.ae_weights, show_history=True)
    plot_model(mdec.model, to_file='mdec_model.png', show_shapes=True)
    mdec.model.summary()

    t0 = time.time()
    mdec.compile(gamma=args.gamma, optimizer=optimizer)
    y_pred = mdec.clustering(x, y=y, tol=args.tol, update_interval=args.update_interval, save_interval=args.save_interval, maxiter=args.maxiter)
    print(y)
    print(y_pred)
    print(("Clustering time: "), (time.time() - t0))
