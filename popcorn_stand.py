# %% [markdown]
## Kernels and image processing
# Below will be an exploration and application into kernels as an image processing tool.
# I will be using the dandy bridges as a test subject and will attempt to eliminate glare from the crowns while 
# still resolving defects
#%%
import numpy as np
import cv2
import matplotlib.pylab as plt
import scipy
from skimage import io, img_as_ubyte 
import pandas as pd
from glob import glob
# %% [markdown]

#### Let's first import all of the images. 
# I will have 2 paths, 1 for images captured with the outer OptiX V6 Mini ring, 
# one for the inner ring
# %%
inner_pics = glob('./Desktop/images_capture/innerring_103us/*.png')
outer_pics = glob('./Desktop/images_capture/outerring_103us/*.png')

# %%
