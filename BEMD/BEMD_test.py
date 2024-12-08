import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from emd import sift

image = io.imread('doublebarlight_02_cropped_grayscale.png', as_gray=True) # 2D image

def corpsify(image, decomp_type):
    rows, cols = image.shape
    if decomp_type == 'R': # Row decomposition
        mode1 = np.zeros((rows, cols))
        mode2 = np.zeros((rows, cols))
        mode3 = np.zeros((rows, cols))
        for i in range(rows):
            imfs = sift.sift(image[i, :], max_imfs=5)
            mode1[i, :] = imfs[:, 0]
            mode2[i, :] = imfs[:, 1]
            mode3[i, :] = imfs[:, 2]
    
    elif decomp_type == 'C': # Column decomposition
        mode1 = np.zeros((rows, cols))
        mode2 = np.zeros((rows, cols))
        mode3 = np.zeros((rows, cols))
        for j in range(cols):
            imfs = sift.sift(image[:, j], max_imfs=5)
            if imfs.shape[1] == 2:  # If it's (N, 2)
                imfs = np.hstack((imfs, np.zeros((imfs.shape[0], 1))))  # Add one column of zeros
            else:
                imfs = imfs
            # print(imfs.shape)
            mode1[:, j] = imfs[:, 0]
            mode2[:, j] = imfs[:, 1]
            mode3[:, j] = imfs[:, 2]

    return mode1, mode2, mode3

# Row decomposition
mode1ij, mode2ij, mode3ij = corpsify(image, 'R')

# Column decomposition
CMode11ij, CMode12ij, CMode13ij = corpsify(mode1ij, 'C')
CMode21ij, CMode22ij, CMode23ij = corpsify(mode2ij, 'C')
CMode31ij, CMode32ij, CMode33ij = corpsify(mode3ij, 'C')

C1 = CMode11ij + CMode21ij + CMode31ij + CMode12ij + CMode13ij
C2 = CMode22ij + CMode32ij + CMode23ij
C3 = CMode33ij

def save_as_png(array, filename):
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    uint8_array = img_as_ubyte(normalized_array)
    
    io.imsave(filename, uint8_array)
C4 = C2 + C3
save_as_png(C1, 'C1_image.png')
save_as_png(C2, 'C2_image.png')
save_as_png(C3, 'C3_image.png')
save_as_png(C4, 'C4_image.png')


