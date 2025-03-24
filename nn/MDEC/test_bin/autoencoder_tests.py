# %%
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

# %%

def autoencoder(dimensions, act = 'relu'):
    """
    The following is a symmetric, under-complete autoencoder used for pretraining and measuring reconstruction loss.

    Arguments:
        dimensions: list of number of nodes in each layer; dimension[0] is the input dimension and dimension[-1]
            is the bottleneck layer. Because the autoencoder is symmetric about the bottleneck layer, the number of 
            layers in the autoencoder is (2 * len(dimensions) - 1)
        act: describes the activation function used at each node, not used in Input, Bottleneck, or Output layers

    Returns:
        Autoencoder model
    """
    n_stacks = len(dimensions) - 1 # number of encoding and decoding layers

    input_layer = Input(shape=(dimensions[0],), name='input')
    h = input_layer # h stores the input layer and makes it easy to pass this layer into the encoding section of the autoencoder
    # no activation function described means it defaults to linear activation

    for i in range(n_stacks - 1):
        h = Dense(dimensions[i + 1], activation=act, name=f'encoder{i}')(h) # connects subsequent h layers as the input for 
        # the following encoder layer (h = input then encoder_1 then encoder_2 then ...)
    
    bottleneck = Dense(dimensions[-1], name='bottleneck')(h) # uses last encoding layer as bottleneck input,
    # no activation function described means it defaults to linear activation
    h = bottleneck
    # h = Dense(dimensions[0], activation=act, name=f'decoder_prime')(h) # extra decoding layer
    # h = Dense(dimensions[-3], activation=act, name=f'decoder_primeprime')(h)

    for i in range(n_stacks-1, 0, -1):
        h = Dense(dimensions[i], activation=act, name=f'decoder_{i}')(h) # connects subsequent h layers to as the input for 
        # the following decoder layer (h = bottleneck then decoder_1 then decoder_2 then ...)

    output_layer = Dense(dimensions[0], activation='sigmoid', name='reconstruction')(h)
    # no activation function described means it defaults to linear activation

    return Model(inputs=input_layer, outputs=output_layer, name='autoencoder')
# %%
(x_train, _), (x_test, _) = mnist.load_data() # loads data from MNIST database
# by default, the training to validation set ratio is 6:1

x_train = x_train.astype('float32') / 255.0 # normalize to [0, 1]
x_test = x_test.astype('float32') / 255.0 # normalize to [0, 1]

x_train_flat = x_train.reshape((len(x_train), -1)) # flatten: from (60000, 28, 28) to (60000, 784)
x_test_flat = x_test.reshape((len(x_test), -1)) # flatten: from (10000, 28, 28) to (10000, 784)

# autoencoder architecture
dims = [784, 100, 64]  #  input dimension: 784, bottleneck dimension: 64, encoding and decoding layers are the others
autoencoder_model = autoencoder(dims, act='relu')

autoencoder_model.compile(optimizer='adam', loss='mse') # compile
autoencoder_model.save_weights('ae.weights.h5')

# train
history = autoencoder_model.fit(
    x_train_flat, x_train_flat,  # input and target are the same (for reconstruction)
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_flat, x_test_flat)
)
# %%
# Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')  # Only if validation data is provided
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#%%
# choose a random sample to test on
rand_idx = np.random.randint(0,10000)
sample_image = x_test[rand_idx] # shape: (28, 28)
sample_image_flat = sample_image.reshape(1, -1) # shape: (1, 784)

reconstructed_image_flat = autoencoder_model.predict(sample_image_flat) # shape: (1, 784)
reconstructed_image = reconstructed_image_flat.reshape(28, 28) # shape: (28, 28)

# compare

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(sample_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Reconstructed Image')
plt.imshow(reconstructed_image, cmap='gray')

plt.show()
# %%
# fashion data
(x_train, _), (x_test, _) = fashion_mnist.load_data() # loads data from MNIST database
# by default, the training to validation set ratio is 6:1
x_train = x_train.astype('float32') / 255.0 # normalize to [0, 1]
x_test = x_test.astype('float32') / 255.0 # normalize to [0, 1]
# %%
# digit data
(x_train, _), (x_test, _) = mnist.load_data() # loads data from MNIST database
# by default, the training to validation set ratio is 6:1
x_train = x_train.astype('float32') / 255.0 # normalize to [0, 1]
x_test = x_test.astype('float32') / 255.0 # normalize to [0, 1]
# %%
# expand dims (use for conv autoencoder)
x_test =  tf.expand_dims(x_test, axis = -1)
x_train =  tf.expand_dims(x_train, axis = -1)

# print(x_test.shape)
# %%
# Standard Autoencoder Class

shape = x_test.shape[1:] # image dimensions ie (24, 24)
dimensions = [500, 100, 64]

class AltAutoencoder(Model):
    def __init__(self, dimensions, shape):
        super(AltAutoencoder, self).__init__()
        self.dimensions = dimensions
        self.shape = shape

        in_out_size = tf.math.reduce_prod(shape).numpy() # size of input and output arrays after flattening / before reshaping

        encoder_layers = []
        decoder_layers = []

        if len(dimensions) < 1:
            raise Exception("Not enough dimensions, use at least 1!")
        
        elif len(dimensions) == 1:
            self.encoder = tf.keras.Sequential([ # denotes sequential procedure of layers
                Flatten(),
                Dense(dimensions[0], activation='relu', input_shape=(in_out_size,), name='bottleneck'), # encodes into a bottleneck layer the length of bottleneck_dim
                ])
            self.decoder = tf.keras.Sequential([
                Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid', name='reconstruction'), # takes the image dimensions (index 1,2 of the input array, see shape definition) and multiplies them
                # so that the output size is the same as the flattened dimension size 
                Reshape(shape), # shape to correct dimensions for decoded/reconstructed output
                ])
        else:
            encoder_layers.append(Flatten())
            for i in range(0, len(dimensions) - 1): # encodes up to bottleneck layer using intermediate hidden layers if given
                encoder_layers.append(
                    Dense(dimensions[i], activation='relu', name=f'encoder_layer{i}') 
                )
            encoder_layers.append(Dense(dimensions[-1], activation='relu', name='bottleneck')) # create bottleneck layer

            self.encoder = tf.keras.Sequential(encoder_layers) # denotes sequential procedure of layers

            for i in range(len(dimensions) - 2, -1, -1):
                decoder_layers.append(
                    Dense(dimensions[i], activation='relu', input_shape=(dimensions[-1],), name=f'decoder_layer{i}')
                )
                # print(f'Layer Size: {dimensions[i]}')
            decoder_layers.append(
                Dense(in_out_size, activation='sigmoid', name='reconstruction') # output later
            )
            decoder_layers.append(Reshape(shape)) # final layer is output

            self.decoder = tf.keras.Sequential(decoder_layers)

    def call(self, x): # define model output
        encoded_image = self.encoder(x)
        decoded_image = self.decoder(encoded_image)
        return decoded_image

# %%
# Conv Autoencoder Class

# shape = x_test.shape[1:]
filters = [16,16] # adjust padding in order to use more than 2 filters

class ConvAutoencoder(Model):
    def __init__(self, filters, 
                 shape = (28, 28, 1),
                 kernel_size = 3,
                 padding = 'same',
                 strides = 2,
                 pool_size = 2):
        super(ConvAutoencoder, self).__init__()
        self.filters = filters
        self.shape = shape
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.pool_size = pool_size

        if len(filters) != 2:
            raise Exception('Filters array must be of length 2')
        
        else:

            self.encoder = tf.keras.Sequential([ # denotes sequential procedure of layers
                Input(shape=shape),
                # Conv2D(filters = filters[0], kernel_size = kernel_size, activation='relu', padding = padding, strides = 1),
                Conv2D(filters = filters[1], kernel_size = kernel_size, activation='relu', padding = padding, strides = strides),
                # MaxPool2D(pool_size=pool_size), # padding = same, strides = 1, pool_size = 2, output + (14, 14, 1)
                # Conv2D(filters = filters[1], kernel_size = kernel_size, activation='relu', padding = padding, strides = strides),
                # MaxPool2D(pool_size=pool_size),
                # Flatten(),
                # Reshape((14*14*filters[0], 1)),
                # Conv1D(filters = filters[1], kernel_size = kernel_size, padding = 'same'),
                # Flatten(),
                # Reshape((14, 14, filters[1])),
                Conv2D(filters = filters[0], kernel_size = kernel_size, activation='relu', padding = padding, strides = 1),
                Flatten(),
                # Dense(500, activation='relu', name='bottleneck'),
                Dense(64, activation='relu', name='bottleneck')
                ])
            
            self.decoder = tf.keras.Sequential([
                Dense(14*14*filters[1], activation='relu', name='decoder1', input_shape=(64,)),
                # Reshape((1568, 1)),
                # Conv1DTranspose(filters = filters[1], kernel_size = kernel_size, padding = 'valid'),
                Reshape((14, 14, filters[1])),
                Conv2DTranspose(filters = filters[0], kernel_size = kernel_size, activation='relu', padding = padding, strides = 1),
                # Flatten(),
                # Reshape((14, 14, filters[0])),
                Conv2DTranspose(filters = filters[1], kernel_size = kernel_size, activation='relu', padding = padding, strides = strides), # takes the image dimensions (index 1,2 of the input array, see shape definition) and multiplies them
                # Conv2DTranspose(filters = filters[0], kernel_size = kernel_size, activation='relu', padding = padding, strides = 1),
                # so that the output size is the same as the flattened dimension size 
                Conv2D(1, kernel_size=kernel_size, activation='sigmoid', padding='same')
                ])

    def call(self, x):
        encoded_image = self.encoder(x)
        decoded_image = self.decoder(encoded_image)
        return decoded_image
# %%
class AltConvAutoencoder(Model):
    def __init__(self, 
                 shape = (28, 28, 1),
                 kernel_size = 3,
                 padding = 'same',
                 pool_size = 2,
                 bottle_neck_size = 64):
        super(AltConvAutoencoder, self).__init__()
        self.shape = shape
        self.kernel_size = kernel_size
        self.padding = padding
        self.pool_size = pool_size
        self.bottle_neck_size = bottle_neck_size
        
        # encoder layers
        self.input_layer = Input(shape=shape, name = 'en_input_layer')
        self.conv1 = Conv2D(filters=16, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='en_conv1')
        self.conv2 = Conv2D(filters=16, kernel_size=kernel_size, activation='relu', padding=padding, strides=1, name='en_conv2')
        self.flatten = Flatten(name='en_flatten')
        self.bottleneck = Dense(bottle_neck_size, activation='relu', name='bottleneck')

        # decoder layers
        self.decoder1 = Dense(14*14*16, activation='relu', input_shape=(bottle_neck_size,), name='de_decoder1')
        self.reshape = Reshape((14, 14, 16), name='de_reshape')
        self.deconv1 = Conv2DTranspose(filters=16, kernel_size=kernel_size, activation='relu', padding=padding, strides=1, name='de_deconv1')
        self.deconv2 = Conv2DTranspose(filters=16, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='de_deconv2')
        self.reconstruction = Conv2D(1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='reconstruction')

        self.build_graph()

    def build_graph(self): # build model
        x = self.input_layer
        self(x)

    def call(self, inputs, training=None):
        # encoder
        h = self.conv1(inputs)
        h = self.conv2(h)
        h = self.flatten(h)
        encoded = self.bottleneck(h)

        # decoder
        h = self.decoder1(encoded)
        h = self.reshape(h)
        h = self.deconv1(h)
        h = self.deconv2(h)
        decoded = self.reconstruction(h)
        return decoded
    
    def get_encoder(self):
        inputs = self.input_layer
        h = self.conv1(inputs)
        h = self.conv2(h)
        h = self.flatten(h)
        encoded = self.bottleneck(h)
        return Model(inputs=inputs, outputs=encoded)
    
    def get_decoder(self):
        en_inputs = Input((self.bottle_neck_size,))
        h = self.decoder1(en_inputs)
        h = self.reshape(h)
        h = self.deconv1(h)
        h = self.deconv2(h)
        decoded = self.reconstruction(h)
        return Model(inputs=en_inputs, outputs=decoded)

# %%
class AltConvAutoencoder(Model):
    def __init__(self, 
                 shape = (28, 28, 1),
                 kernel_size = 3,
                 padding = 'same',
                 pool_size = 2,
                 bottle_neck_size = 64):
        super(AltConvAutoencoder, self).__init__()
        self.shape = shape
        self.kernel_size = kernel_size
        self.padding = padding
        self.pool_size = pool_size
        self.bottle_neck_size = bottle_neck_size
        self.model = self.build_model()
        
    def build_model(self):
        # encoder layers
        inputs = Input(shape=self.shape, name = 'en_input_layer')
        h = Conv2D(filters=16, kernel_size=self.kernel_size, activation='relu', padding=self.padding, strides=2, name='en_conv1')(inputs)
        h = Conv2D(filters=32, kernel_size=self.kernel_size, activation='relu', padding=self.padding, strides=2, name='en_conv2')(h)
        h = Conv2D(filters=128, kernel_size=3, activation='relu', padding=self.padding, strides=1, name='en_conv0')(h)
        h = Flatten(name='en_flatten')(h)

        # bottleneck
        bottleneck = Dense(self.bottle_neck_size, activation='relu', name='bottleneck')(h)

        # decoder layers
        h = Dense(7*7*128, activation='relu', input_shape=(64,), name='decoder2')(bottleneck)
        h = Reshape((7, 7, 128), name='de_reshape')(h)
        h = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding=self.padding, strides=2, name='de_deconv1')(h)
        h = Conv2DTranspose(filters=16, kernel_size=self.kernel_size, activation='relu', padding=self.padding, strides=2, name='de_deconv2')(h)
        reconstruction = Conv2D(self.shape[2], kernel_size=self.kernel_size, activation='sigmoid', padding=self.padding, name='reconstruction')(h)

        autoencoder = Model(inputs, reconstruction, name='autoencoder')

        return autoencoder

    def compile(self, **kwargs):
        self.model.compile(**kwargs)

    def fit(self, x, y, **kwargs):
        return self.model.fit(x, y, **kwargs)  # Delegate to the inner model

    def save_weights(self, path):
        self.model.save_weights(path)  # Delegate to the inner model

    def summary(self):
        self.model.summary()

    def get_encoder(self):
        return Model(inputs=self.model.input, outputs=self.model.get_layer('bottleneck').output)
    
    def get_decoder(self):
        de_inputs = Input((self.bottle_neck_size,))
        h = self.model.get_layer('decoder2')(de_inputs)
        h = self.model.get_layer('de_reshape')(h)
        h = self.model.get_layer('de_deconv1')(h)
        h = self.model.get_layer('de_deconv2')(h)
        decoded = self.model.get_layer('reconstruction')(h)
        return Model(inputs=de_inputs, outputs=decoded)
# %%
# x = tf.constant([[1,2],[2,3]])
# print(tf.math.reduce_prod(x))

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_test = tf.expand_dims(x_test, axis = -1)
print(x_test.shape)
# %%
# init standard autoencoder
autoencoder = AltAutoencoder(dimensions, shape)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
history = autoencoder.fit(x_train, x_train,
                epochs = 10,
                shuffle=True,
                validation_data=(x_test, x_test))
autoencoder.summary()

# %%
# init conv autoencoder
autoencoder = ConvAutoencoder(filters = filters, shape = (28,28,1), kernel_size=3, padding = 'same', strides = 2, pool_size = 2, bottle_neck_size = 64)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
history = autoencoder.fit(x_train, x_train,
                epochs = 20,
                shuffle=True,
                validation_data=(x_test, x_test))
autoencoder.save_weights('conv_ae.weights.h5')
autoencoder.summary()

# %%
# init ALT conv autoencoder
autoencoder_model = AltConvAutoencoder(shape = (28,28,1), kernel_size=5, padding = 'same', pool_size = 2, bottle_neck_size = 10)
autoencoder_model.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder_model.summary()
history = autoencoder_model.fit(x_train, x_train,
                epochs = 20,
                shuffle=True,
                validation_data=(x_test, x_test))
autoencoder_model.save_weights('conv2_ae.weights.h5')
# %%
def cocoa( 
        shape = (28, 28, 1),
        kernel_size = 5,
        padding = 'same',
        bottle_neck_size = 64):
    
    # encoder layers
    if shape[0] != shape[1]:
        raise Exception("Please ensure a square input tensor (ie: Height = Width)")
    else:
        downsized_shape = 7 #int(shape[0]/4)

        inputs = Input(shape=shape, name = 'en_input_layer')
        h = Conv2D(filters=16, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='en_conv1')(inputs)
        h = Conv2D(filters=32, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='en_conv2')(h)
        h = Conv2D(filters=128, kernel_size=3, activation='relu', padding=padding, strides=1, name='en_conv0')(h)
        h = Flatten(name='en_flatten')(h)

        # bottleneck
        bottleneck = Dense(bottle_neck_size, activation='relu', name='bottleneck')(h)

        # decoder layers
        h = Dense(downsized_shape*downsized_shape*128, activation='relu', input_shape=(64,), name='decoder2')(bottleneck)
        h = Reshape((downsized_shape, downsized_shape, 128), name='de_reshape')(h)
        h = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding=padding, strides=2, name='de_deconv1')(h)
        h = Conv2DTranspose(filters=16, kernel_size=kernel_size, activation='relu', padding=padding, strides=2, name='de_deconv2')(h)
        reconstruction = Conv2D(shape[2], kernel_size=kernel_size, activation='sigmoid', padding=padding, name='reconstruction')(h)

    return Model(inputs, reconstruction, name='autoencoder')

# %%
autoencoder_model = cocoa(shape = (28, 28, 1),
        kernel_size = 5,
        padding = 'same',
        bottle_neck_size = 10)

autoencoder_model.compile(optimizer='adam', loss='mse') # compile

history = autoencoder_model.fit(
    x_train, x_train,  # input and target are the same (for reconstruction)
    epochs=20,
    batch_size=128,
    shuffle=False
    # validation_data=(x_test, x_test)
)

autoencoder_model.save_weights('conv3_ae.weights.h5')
# %%
# Plot history
num_epochs = history.params['epochs']

plt.plot(np.arange(0,num_epochs,1), history.history['val_loss'], history.history['loss'])
plt.legend(['val_loss', 'train_loss'])
plt.title(f'Loss over {num_epochs} Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()
# %%
# init test for altconvautoencoder
encoder = autoencoder_model.get_encoder()
decoder = autoencoder_model.get_decoder()
# %%
encoded_pics = autoencoder_model.predict(x_test)
decoded_pics = autoencoder_model.predict(encoded_pics)

# %%
# test it!
# encoded_pics = autoencoder.get_encoder.x_test
# decoded_pics = autoencoder.get_decoder.encoded_pics

# encoded_pics = encoder.predict(x_test)
# decoded_pics = decoder.predict(encoded_pics)

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
for i in range(0,0,-1):
    print(i) 
# %%
