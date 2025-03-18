import logging, csv, os
from pathlib import Path
import time
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist

from nn.MDEC_autoencoder import ConvAutoencoder
from nn.MDEC_clusteringlayer import ClusteringLayer

###############################################################################

def load_mnist():
    (x, y), (x_test, y_test) = mnist.load_data()
    x = x.astype('float32') / 50.0
    x_val = x_test.astype('float32') / 50.0
    x = tf.expand_dims(x, axis = -1)
    x_val = tf.expand_dims(x_val, axis = -1)
    # y_train = tf.expand_dims(y_train, axis = -1)
    # x_test = tf.expand_dims(x_test, axis = -1)
    # y_test = tf.expand_dims(y_test, axis = -1)
    # x = np.concatenate((x_train, x_test), axis=0) # (60000, 28, 28) and (10000, 28, 28) become (70000, 28, 28) - this combines the train and test sets
    # y = np.concatenate((y_train, y_test), axis=0) # (60000, ) and (10000, ) become (70000, ) - this combines the train and test sets
    # x = x.reshape((x.shape[0],-1)) # (70000, 28, 28) becomes (70000, 28*28) or (70000, 784)  # normalize as it does in DEC paper
    return x, y, x_val
    
def align_clusters(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # Returns tuple (row, col)
    return np.array([col_ind[np.where(row_ind == i)][0] for i in range(len(row_ind))])

class MDEC(object):
    def __init__(self,
                 input_shape,
                 bottleneck_size,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):
        
        super(MDEC, self).__init__()

        self.bottleneck_size = bottleneck_size # array of dimensions to go though as the autoencoder is trained
        self.input_shape = input_shape # size of flattened input image is the first dimension

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder = ConvAutoencoder(shape=input_shape, kernel_size=5, padding='same', bottleneck_size=bottleneck_size)
        # self.autoencoder = ConvAutoencoder(self.dimensions) # makes sure the dimensions array is placed into the autoencoder

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

    def pretrainer(self, x, x_val, batch_size=128, epochs=20, ae_weights=None, show_history=False):
        print('Begin Pretraining...')
        print(x.shape)

        self.autoencoder.compile(optimizer='adam', loss='mse')

        t_0 = time.time()

        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights) # use pretrained weights for autoencoder if possible
            print('Pretrained weights have been loaded into the Autoencoder')
        
        history = self.autoencoder.fit(
            x, x,  # input and target are the same (for reconstruction)
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            validation_data=(x_val, x_val)
            )
        
        print(f'Pretraining complete, took: {time.time() - t_0} seconds')
        self.autoencoder.save_weights('cae_pretrain.weights.h5')
        print(f'Pretrained weights saved as cae_pretrain.weights.h5')
        
        if show_history:
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')  # Only if validation data is provided
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.show()

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
    
    def compile(self, gamma=0.1):
        self.model.compile(loss={'clustering': 'kld', 'reconstruction': 'mse'}, 
                           loss_weights=[gamma, 1], 
                           optimizer='adam')

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
                   save_interval=5,
                   maxiter=2e4):

        print(f'Update Interval: {update_interval}')
        print(f'Save Interval {save_interval}')       

        print('Initializing clusters with k-means')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20) # n_init is the number of times the algorithm is run with dif centroids, best run is output

        y_pred = kmeans.fit_predict(self.encoder.predict(x)) # performs k-means clustering on the pretrained bottleneck layer (feature space)
        y_pred_last = np.copy(y_pred) # save the clusters
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_]) # set the weights according to the k-means clustering for good initialzation

        save_dir = '/home/unitx/wabbit_playground/nn/clustering_log'
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        rn = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file = save_dir_path / f'mdec_log_.csv_{rn}'

        header = ['iter', 'nmi', 'acc', 'ari', 'L', 'Lr', 'Lc', 'timestamp']

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
            def __init__(self, mdec, x, y, tol, log_file, update_interval, target_distribution, predict_clusters):
                self.mdec = mdec
                self.y = y
                self.x = x
                self.tol = tol
                self.log_file = log_file
                self.update_interval = update_interval
                self.y_pred_last = None
                self.target_distribution = target_distribution
                self.predict_clusters = predict_clusters
                print(f'\n {self.x.shape}')

                super(ClusterCallbacks, self).__init__()
            
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

                    print(self.y.shape)
                    print(y_pred.shape)

                    if self.y is not None:
                        # metrics for each update
                        acc = np.round(metrics.accuracy_score(self.y, y_pred), 5)
                        ari = np.round(metrics.adjusted_rand_score(self.y, y_pred), 5)
                        nmi = np.round(metrics.normalized_mutual_info_score(self.y, y_pred), 5)
                        # loss = np.round(np.sum(logs.values()), 5)
                        
                        new_row = { # new log entry for each update
                            'iter': epoch,
                            'nmi': nmi,
                            'acc': acc,
                            'ari': ari,
                            'L': logs['loss'],  # total loss
                            'Lc': logs['clustering_loss'],  # clustering loss - name defined in self.model.compile
                            'Lr': logs['reconstruction_loss'],  # reconstruction loss - - name defined in self.model.compile
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

                        new_entry = pd.DataFrame([new_row])
                        new_entry.to_csv(self.log_file, mode='a', header=False, index=False) # pandas handles write and close

                        print(f'Iteration {epoch} logged.')
                        print(f'nmi: {nmi}, acc: {acc}, ari: {ari}\n Total Loss: {logs["loss"]}, Clustering Loss: {logs["clustering_loss"]}, Reconstruction Loss: {logs["reconstruction_loss"]}')

                    if delta_metric < self.tol:
                        print(f'delta: {delta_metric} < tol: {self.tol}')
                        print('Tolerance reached, training will be stopped.')
                        self.model.stop_training = True

                    bottleneck_features = self.mdec.encoder.predict(self.x)
                    features_2d = TSNE(n_components=2).fit_transform(bottleneck_features)
                    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y_pred, cmap='tab10', s=5)
                    plt.savefig(f'clusters_epoch_{epoch}.png')
                    plt.close()

        self.model.fit( # the training procedure
            dataset, # define x inputs and y targets
            epochs = int(maxiter), # number of epochs / loop
            steps_per_epoch = int(x.shape[0] / self.batch_size), # essentially samples per epoch given a batch size: num_samples / batch_size
            callbacks = [ 
                ClusterCallbacks(mdec=self, x=x, y=y, tol=tol, log_file=log_file, update_interval=update_interval, target_distribution=MDEC.target_distribution, predict_clusters = self.predict_clusters), # updates y_pred and saves metrics at update intervals
                tf.keras.callbacks.ModelCheckpoint( # saves the model after each save_interval
                    filepath=os.path.join(save_dir, f'MDEC_model.weights.h5'),
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
    # parser.add_argument('--epochs', default=10, help="Number of training epochs", type=int)
    parser.add_argument('--update_interval', default=140, help="Number of epochs (interval) between target distribution P updates", type=int)
    parser.add_argument('--save_interval', default=5, help="Number of epochs (interval) between model saves", type=int)
    parser.add_argument('--tol', default=0.001, help="Model convergance tolerance", type=float)
    parser.add_argument('--ae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='/home/unitx/wabbit_playground/nn/clustering_log', help="Location for logs and model weights")
    args = parser.parse_args()
    print(args)

    # optimizer = SGD(learning_rate=0.1, momentum=0.99)

    if args.dataset == 'mnist':
        x, y, x_val = load_mnist()
        optimizer = 'adam'
        print(x.shape)
        print(y.shape)

    
    mdec = MDEC(input_shape=x.shape[1:], bottleneck_size=10, n_clusters=args.n_clusters, batch_size=args.batch_size) # selects the last dimension of x (784) to by the input array size and 10 to be the bottleneck size
    mdec.pretrainer(x, x_val, batch_size=256, epochs=50, ae_weights=None, show_history=True)
    plot_model(mdec.model, to_file='mdec_model.png', show_shapes=True)
    mdec.model.summary()

    t0 = time.time()
    mdec.compile(gamma=0.9)
    y_pred = mdec.clustering(x, y=y, tol=args.tol, update_interval=args.update_interval, save_interval=args.save_interval, maxiter=args.maxiter)
    print(y)
    print(y_pred)
    print(("Clustering time: "), (time.time() - t0))
