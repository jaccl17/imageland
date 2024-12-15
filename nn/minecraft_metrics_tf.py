# %% [markdown]
## Minecraft Metrics
# I wanted to provide an example of my literacy with tensorflow for my grad applications, so I designed this simple 
# neural network. It looks at screenshots from Minecraft, one of my favorite video games, and provides feedback on 
# what the NN sees.
# The images for this exploration can be found at 
# https://www.kaggle.com/datasets/sqdartemy/minecraft-screenshots-dataset-with-features
# I have provided a correctly organized CSV (minecraft_features_and_decisions.csv) that you should use
# to make my model train and load correctly
# 
# It looks at each image and predicts a yes (1) or no (0) to the following four quesitons
# - Is the player on land?
# - Is the player holding an item?
# - Is the player in a location where mobs (monsters) could spawn?
# - Is the player at full health?
#
# These questions are simple, but very important for anyone playing Minecraft as conditions, reactions, 
# and playthrough change depending on the answers. And in the future, if a computer wanted to play minecraft itself,
# these would absolutely be metrics it needed to assess.
#
# Note: this is not a simple classification network, but a **multi-label classification** network, as it studies and predicts
# multiple classifications on each image

# %% [markdown]
### Import libraries and packages
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
import random
import os
import logging
from minecraft_utils import *
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
# %% [markdown]
# Here we're just going to check for GPUs and configure them. These help speed up training and prediction.

# %%
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

print("\nTensorFlow Build Information:")
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("Is GPU available:", tf.test.is_built_with_gpu_support())
# %% [markdown]
# Next I'm just gonna define some global variables, these will likely not be adjusted, but can be if necessary

# %%
img_size = 224
autotune = tf.data.experimental.AUTOTUNE
batch_size = 500
decision_tree = np.array([['On land', 'In Water'],
                             ['Holding an Item', 'Empty-handed'], 
                             ['Mob Spawn Risk!','Safe from Mob Spawns'],
                             ['Full Health', 'Low Health']])
# %% [markdown]
# Next I want to set up data organisation. This is a crucial yet often overlooked step. In order to have a function NN
# it's important to have well-organized data. Though this works differently than PyTorch, Tensorflow has some helpful
# data organizing tools
# %%
minecraft_data = pd.read_csv("/home/unitx/wabbit_playground/nn/minecraft_features_and_decisions.csv")
minecraft_data.head(3)
label_names = list(minecraft_data.columns[1:])
print(label_names)
# %%
image_names = minecraft_data.iloc[:,0] # keep image names as reference
X = image_names
y = minecraft_data.iloc[:,1:] # all other columns are the labels
# %%
# Allocate 60% of data to training
X_train, X_, y_train, y_ = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Split remaining 40% into validation and test sets (20% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_, y_, test_size=0.5, random_state=42
)

n_labels = len(y_train.values[1])
# %% [markdown]
# I tend to perform a lot of 'sanity checks' so I'll put some here to make sure everything is well-sorted
# %%
# Check data sorting
print(f'No. of training samples: {X_train.size}')
print(f'No. of validation samples: {X_val.size}')
print(f'No. of test samples: {X_test.size}')
print(f'All input data allocated correctly: {X.size == X_test.size + X_val.size + X_train.size}')

# %%
# Double check we can match up our data correctly, I cross-refernce a handful with the input csv to make sure
for image_name, labels in zip(X_train, y_train.values):
    print(f"Image: {image_name}, Labels: {labels}")
# %% [markdown]
# Now we're gonna make the datasets. I always like to have a train, validation, and test set.
# Though all three are common practice, sometimes the validation set isn't used.

# %%
train_dataset = make_dataset(X_train, y_train.values.tolist())
val_dataset = make_dataset(X_val, y_val.values.tolist(), is_training=False)
test_dataset = make_dataset(X_test, y_test.values.tolist(), is_training=False)
# %%
# Just another check to make sure that we have batched our data and the image sizes are correct
# features (batch_size, img_size, img_size, channels)
# labels (batch_size, # labels)
try:
    for features, labels in train_dataset.take(1):
        print("Shape of features:", features.numpy().shape)
        print("Shape of labels:", labels.numpy().shape)
except Exception as e:
    print("Error iterating dataset:", e)
# %% [markdown]
# It's time now to build the model!
# We're gonna use a pretrained model from ImageNet bc it has so much feature data already trained into it and it's free to use
# one of the faster models is MobileNet V2, we'll use that.
#
# We're also gonna build a scheduler which will decrease the training rate at the model learns. This helps avoid overfitting,
# as the model is less likely to fall into local minima associated with optimized parameters.
#
# Our model will consist of a pretrained layer, a Relu activation layer, a dropout layer to make the model work harder to find details
# and a linear output layer. We don't use a sigmoid output layer, because we are using a Binary Cross Entropy layer that will perform
# the sigmoid actitvation for us.
#
# We will save loss and accuracy metrics to a 'history' valuable to make plotting easier.

# %%
pretrained_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'
pretrained_layer = hub.KerasLayer(pretrained_url, input_shape=(img_size,img_size,3))
pretrained_layer.trainable = False # freezes the pretrained layer

# %%
# Scheduler 
lr_schedule = tf_keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,
    decay_steps=20,
    decay_rate=0.8
)
# %%
# Model
model = tf_keras.Sequential([
    pretrained_layer,
    tf_keras.layers.Dense(1024, activation='relu', name='L1'),
    tf_keras.layers.Dropout(0.6),
    tf_keras.layers.Dense(n_labels, activation ='linear', name ='output')
])
# model.summary() # use this to check all layers are correct

# %%
# Train and fit
model.compile(
    loss=tf_keras.losses.BinaryCrossentropy(from_logits=True),
    # optimizer=tf_keras.optimizers.Adam(learning_rate=0.0008),
    optimizer = tf_keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=[tf_keras.metrics.AUC(name='auc'), tf_keras.metrics.BinaryAccuracy()]
)

history = model.fit(train_dataset,
                            epochs=30,
                            validation_data=val_dataset)

# %% [markdown]
# We call our function to plot our metrics
# %%
plot_metrics(history)
# %% [markdown]
# Next comes the most fun part, we make a prediction!!!!!
# Below I have few lines of code to pick a random image from our Test Dataset, plot it, and compare the predicted answers to 
# the actual answers. After the model has compiled, this cell can be run repeatedly and indepedently to make new predictions!
# %%
for features, labels in test_dataset.take(1):
    rN = random.randint(0, len(features) - 1)

    logits = model.predict(features)
    probabilities = tf.nn.sigmoid(logits).numpy()
    
    print(f'Actual metrics: {decider(labels[rN].numpy())}')
    print(f'Predicted metrics: {decider(probabilities[rN])}')
    plt.imshow(features[rN].numpy())
# %% [markdown]
# And that's it! This model isn't my most robust, but it's an NDA-safe example of my DL experience!
