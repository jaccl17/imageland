import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
import emd
import time

start_time = time.time()

image = io.imread('testcoin_crop.png', as_gray=True) # 2D image

# imf = emd.sift.sift(image[:,0], imf_opts={'sd_thresh': 0.1})

def corpsify(image, decomp_type):
    rows, cols = image.shape
    if decomp_type == 'R': # Row decomposition
        mode1 = np.zeros((rows, cols))
        mode2 = np.zeros((rows, cols))
        mode3 = np.zeros((rows, cols))
        mode4 = np.zeros((rows, cols))
        for i in range(rows):
            imfs = emd.sift.sift(image[i, :])
            mode1[i, :] = imfs[:, 0]
            mode2[i, :] = imfs[:, 1]
            mode3[i, :] = imfs[:, 2]
            mode4[i, :] = imfs[:, 2]
    
    elif decomp_type == 'C': # Column decomposition
        mode1 = np.zeros((rows, cols))
        mode2 = np.zeros((rows, cols))
        mode3 = np.zeros((rows, cols))
        mode4 = np.zeros((rows, cols))
        for j in range(cols):
            imfs = emd.sift.sift(image[:, j])
            mode1[:, j] = imfs[:, 0]
            mode2[:, j] = imfs[:, 1]
            mode3[:, j] = imfs[:, 2]
            mode4[:, j] = imfs[:, 2]

    return mode1, mode2, mode3, mode4

# Row decomposition
mode1ij, mode2ij, mode3ij, mode4ij = corpsify(image, 'R')

# Column decomposition
CMode11ij, CMode12ij, CMode13ij, CMode14ij = corpsify(mode1ij, 'C')
CMode21ij, CMode22ij, CMode23ij, CMode24ij = corpsify(mode2ij, 'C')
CMode31ij, CMode32ij, CMode33ij, CMode34ij = corpsify(mode3ij, 'C')
CMode41ij, CMode42ij, CMode43ij, CMode44ij = corpsify(mode4ij, 'C')

C1 = CMode11ij + CMode21ij + CMode31ij + CMode12ij + CMode13ij + CMode41ij + CMode14ij
C2 = CMode22ij + CMode32ij + CMode23ij + CMode24ij + CMode42ij
C3 = CMode33ij + CMode34ij + CMode43ij
C4 = CMode44ij

def save_as_png(array, filename):
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    uint8_array = img_as_ubyte(normalized_array)
    
    io.imsave(filename, uint8_array)

save_as_png(C1, 'C1_image.png')
save_as_png(C2, 'C2_image.png')
save_as_png(C3, 'C3_image.png')
save_as_png(C4, 'C4_image.png')

end_time = time.time()
print(f'elapsed time: {end_time - start_time}')