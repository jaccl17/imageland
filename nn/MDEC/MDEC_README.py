# %% [markdown]
# # Modernized Deep Embedded Clustering (MDEC)
# ## Overview
# The Modernised Deep Embedded Clustering (MDEC) is a framework and tool that I developed at UnitX as a Vision Technician 
# in order to improve our capabilities in detecting defects in customer products. The algorithm employs is comprised of two deep
# learning models, a convolutional autoencoder that maps images into a 10-dimensional feature space and a clustering that learns
# to categorize this feature layer by associating the learned nodes with the inputs images. Note that the size of the feature
# layer is equivalent to the number of clusters (aka categories or labels) that describe each image
#
# For example, this document is trained on the MNIST digit dataset, a collection of handwritten digits from 0-9. The algorithm 
# splits the dataset into training and validation sets, studies the training images, and then groups the images based on their
# lower-dimensional representation. 
#
# However in it's practical application, this algorithm is used to cluster masked defects in larger images of manufactures parts. 
# A more in-depth explantaion of the motivation and application of this algorithm is provided below.
# 
# A Note for KIPAC: I'm not sure if this algorithm in particular will be of interest to any of the research groups, but at the very least 
# it serves as a strong example of my hard work
#
# %% [markdown]
# ## Required Dependencies
#
# - python 3.11.8
# - tensorflow 2.19.0
# - numpy 1.26.4
# - pandas 2.2.2
# - scipy 1.13.0
# - matplotlib 3.8.3
# - scikit-learn 1.4.1.post1
#
# I recommend working in a virtual environment with libraries installed using 'pip'. The 'conda' and 'conda-forge' package managers tend to have
# outdated packages that don't always work well with tensorflow.
#
# %% [markdown]
# ## Why did I develop this?
# Throughout my time at UnitX, I have been tasked with developing machien vision solutions to inspecting manufactured parts for prospective
# customers. This process begins with a Proof of Concept (POC) where I create a small inspection station, involving lights, camera, robotic arms, 
# and anything else we have available to create the best automated solution. 
#
# The next step, and the reason most customer come to us, involves training a neural network to detect defects in the images we captured of their defective
# parts. Though this process is very repeatable across customers, parts, and applications, it relies predominantly on two main factors:
# - The quality of the training data (ie the images)
# - And the consistancy of the labeled defects
#
# Because we measure success not by the number of units sold, but the number of units sucessfully deployed, we rely on the percentage of 
# FAs (False Acceptances) and FRs (False Rejections), which are usually taken to be less than 1%. This means, that a working, deployed, system must
# Falsely accept and falsely reject no more than 1$ of the parts inspected during a deployment (typically 300-500 parts).
#
# So what does that have to do with clustering? Well, recall the two factors I mentioned. The first is dependent on the quality of the solution I develop
# (ie: how well I do designing a camera and light system, which is something I've gotten good at). The second factor, the consistancy of labeling,
# is trickier to perfect, mostly due to the fact that defects of the same type can vary widely across applications, or even the same part. A scratch on the 
# top of a rotor looks mutch different from one on the side teeth. Discoloration on a plastic part varies in color and texture depending on the cause of
# the defect. And more often than not, most customers don't even know what they're looking for or how to diagnose their defects, but instead care more
# about the size, location, and frequency of the defect.
#
# This is why I decided to develop the MDEC algorithm. I allows me to label defects of various geometries, textures, and colors that I know are of concern
# to the customer, and then let the algoirthm decide how to best categorize them. This way, when I train the detection model, it has the best chance at
# finding defects in new samples, because the categories it knows to look for have been expertly outlined by the MDEC deep learning algorithm.
#
# In fewer words, the MDEC algorithm is a tool that allows me to cluster or categorize defects in images, that can then be used to train a detection model
# for the best outcome.
#
# %% [markdown]
# ## How does it work?
#
# A practical application of MDEC works similarly to this example, though the datatset is comprised of defect examples rather than handwritten digits.
#
# First, I upload images into a GUI (made by someone else) that allows me to draw a boundary around each defect; this region is padded and saved as a square,
# usually not much larger than 100x100 pixels. The masking selects only a small component of the image that is relevant to the defect; this is repeated for
# all defects in the image. The masked defects are either scaled to the same size or padded to the same size (I haven't decided yet which is better, it's
# a tradeoff between information compression and computational efficiency), and fed to the MDEC algorithm. The algorithm studies the defects, than clusters
# them based on similarities. Specifics about each component of the algorithm are provided in the algorithm's documentation (ie: MDEC_main.py).
#
# What results is a set of clusters, each containing a set of defects that contain the most similarities. These labels and masked defects are then fed to 
# our deep learning algorithm, which is trained to detect these defects in new images!
#
# %% [markdown]
# ## How is this relevant to KIPAC or astrophysics in general?
#
# Simply put, it's an unsupervised data classification algorithm that is extremely generalizable. The key here is unsupervised, meaning it does not rely 
# on a specific, or even sizable, training set in order to work sucessfully. In it's most minimal form, it studies single or muti-dimensional tensors for 
# trends that are not easily interpretable by humans, that can be used to draw conclusions about the data. I'll provide an astrophysical hypothical.
#
# Image you have an apparently uniform image of a cloud of gas. The image could also be a dataset that decribes local densities, temperatures, or even
# scattering coefficients for a particualr wavelength. It may looks like a bunch of numbers, and it may even be interpretable. But you may want a second 
# opinion, a more detailed analysis, or a tool to verify your conclusions. So you segment the data tensor into smaller components, keeping track of where
# you draw your masks, and feed it to the MDEC algorith.
#
# The algorithm groups these segments, perhaps into 12 different groups. Upon further inspection of these grouped areas, you uncover a larger, less obvious
# structure to the cloud of gas. You may find that the gas is actually a collection of smaller clouds, or that the gas is moving in a particular direction, 
# or that the gas is utterly and entirely uninteresting!
#
# The goal of this algorithm is not to replace human interpretation, to write a paper for you, or to even be useful in all scenarios. But it is an example 
# or artificial intelligence as a use scientific tool that can lead us in new, sometimes surprising directions.
#
# %% [markdown]
# ## The state of the algorithm
#
# This is not a new idea, nor is it finished - the use of deep embedded clustering (DEC) may never reach an end. It is an ongoing project that I develop daily,
# though unfortunately I am unable to share all aspects of my progress due to an NDA. Regardless, the algorithm I have designed is a unique interpretation of 
# a project started in 2015 by Dr. Junyuan Xie et al. that has since been improved (IDEC), varied (VaDE), and now modernized (MDEC). 
#
# The qualifier "Modernized" was chosen first and foremost to emphasize the flexibility of my algorithm compared to its predecessors, but also because it
# uses the most up-to-date packages avaiable for tensorflow. The algorithm works with the current line of GPUs (5000 series), though I run a 4000 series.
# In my algorithm you'll find:
# - a convolutional autoencoder that preserves 2D data structure, with max pooling, as it translates a image to the feature space
# - more metrics for both the autoencoder and cluster training 
# - increased visibility into the clustering progress through epoch graphics and gifs
# - a learning rate scheduler that decreases the likelihood of cluster convergence on local minima
# - the added option of augmenting input data (random rotations and contrast adjustments) for increased training variety
# - expanded warning and notifications of all processes (helpful for debugging)
# - updated and organized logging with pandas
# - and a 'test_station' script that allow the user to visualize live (or historical) metrics, as well try out and adjust the autoencoder paramters (start here, it's fun)
#
# Most importantly, I have decoupled the autoencoder and the clustering algorithm, so that technically any autoencoder can be used with the clustering algorithm, 
# as long as it was trained on data of the same size as it to be clustered, though this is intuitive.
