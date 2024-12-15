
# %% [markdown]
# Here are four functions for my minecraft NN
# - image_parser: This is going to read in all of the images, transform them into TF tensors, resize them for
#       optimized training, and normalize the pixel values, and augment data a bit so out NN has more variety
#       to learn from.
#
# - make_dataset: This guy is gonna do the tedious part of keeping track of our tensors and their associated labels.
#       It will also cache our training dataset so that we don't have to reprocess each batch with every epoch.
#       This function also batches our data.
#
# - plot_metrics: This won't be called until after we train, but this is one of the most important aspects.
#       Here we plot the loss and accuracy of our training and validation sets. This is the best way we can assess
#       model performance and decide whether we need to make adjusts (ie: learning rate, dropout, scheduler, etc).
#
# - decider: This is a simple function to make visualizing the results of our model a bit easier. It will output the actual and 
#       predicted answers to the questions we're asking.

# %%
def image_parser(image_name, label):
    try:
        image_read = tf.io.read_file(image_name)
        image_decode = tf.image.decode_png(image_read, channels=3)
        image_resize = tf.image.resize(image_decode, [img_size, img_size])
        image_normalize = image_resize / 255.0
        # image_augmented = tf.image.random_flip_left_right(image_normalize)
        # image_augmented = tf.image.random_brightness(image_augmented, 0.2)
        return image_normalize, label
    except Exception as e:
        print(f"Error parsing image {image_name}: {e}")
        # Return a placeholder or handle the error appropriately
        return tf.zeros([img_size, img_size, 3]), label

def make_dataset(image_names, labels, is_training=True):
    image_names = np.array([os.path.join('/home/unitx/wabbit_playground/nn/minecraft/', str(f)) for f in image_names])
    labels = np.array(labels)

    minecraft_dataset = tf.data.Dataset.from_tensor_slices((image_names, labels))
    minecraft_dataset = minecraft_dataset.map(image_parser, num_parallel_calls=autotune)
    if is_training == True:
        minecraft_dataset = minecraft_dataset.take(batch_size).cache() # maintain dataset in memory
        minecraft_dataset = minecraft_dataset.shuffle(buffer_size = 1000)

    minecraft_dataset = minecraft_dataset.batch(batch_size)
    minecraft_dataset = minecraft_dataset.prefetch(buffer_size=autotune)

    return minecraft_dataset

def plot_metrics(history):
    metrics = history.history.keys()  # metrics from history
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics, 1):
        plt.subplot((len(metrics) + 1) // 2, 2, i)
        plt.plot(epochs, history.history[metric], label=f"Training {metric}")
        if f"val_{metric}" in history.history:
            plt.plot(epochs, history.history[f"val_{metric}"], label=f"Validation {metric}")
        plt.title(metric.capitalize())
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.xticks(epochs)
        plt.legend()
        plt.xlim(1,len(epochs))
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def decider(array):
    decisions = []
    for i in range(len(array)):
        if array[i] > 0.8:
            decisions.append(decision_tree[i,0])
        else:
            decisions.append(decision_tree[i,1])
    return decisions