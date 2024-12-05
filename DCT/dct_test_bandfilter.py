import numpy as np
from scipy.fftpack import idct, dct
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte

T_D = 18
F_D = 17
S_D = 0
the_norm = 'backward'
the_type = 1
WD = 12# Width of the deletion range in pixels, you can adjust this based on the desired range

# Define the angles in degrees
angles = [79.7,70.6,62,54.5,48.2,43,35]

def save_as_png(array, filename):
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    uint8_array = img_as_ubyte(normalized_array)
    io.imsave(filename, uint8_array)

def apply_dct(image):
    dct_rows = dct(image, type=the_type, axis=0, norm=the_norm)
    dct_image = dct(dct_rows, type=the_type, axis=1, norm=the_norm)
    return dct_image

def rebuild_image(dct_image):
    idct_rows = idct(dct_image, type=the_type, axis=1, norm=the_norm)
    idct_image = idct(idct_rows, type=the_type, axis=0, norm=the_norm)
    return idct_image

def filter_by_angle(dct_image, angle, WD):
    u_max, v_max = dct_image.shape
    mask = np.zeros_like(dct_image, dtype=bool)
    
    theta = np.radians(angle)

    for u in range(u_max):
        for v in range(v_max):
            if theta < np.pi/9:
                condition_upper = (u * np.sin(theta) - v * np.cos(theta) <= 0)
                condition_lower = (u * np.sin(theta) - v * np.cos(theta) + WD >= 0)
            elif np.pi/9 <= theta < 7*np.pi/18:
                condition_upper = (u * np.sin(theta) - v * np.cos(theta) - WD / 2 <= 0)
                condition_lower = (u * np.sin(theta) - v * np.cos(theta) + WD / 2 >= 0)
            elif theta >= 7*np.pi/18:
                condition_upper = (u * np.sin(theta) - v * np.cos(theta) - WD <= 0)
                condition_lower = (u * np.sin(theta) - v * np.cos(theta) >= 0)
            if condition_upper and condition_lower:
                mask[u, v] = True
    
    return mask

def three_way_band_filter(dct_image, angles, WD):
    mask = np.zeros_like(dct_image, dtype=bool)
    magnitude_spectrum = np.log1p( 1 + (np.abs(dct_image))**2)
    filtered_dct = np.zeros_like(dct_image)
    filtered_magspec = np.zeros_like(dct_image)

    # Combine masks for each angle
    for angle in angles:
        angle_mask = filter_by_angle(dct_image, angle, WD)
        mask |= angle_mask

    # Apply the mask to filter out the frequencies in the band
    filtered_dct[mask] = 0  # Set frequency values in the band to 0
    filtered_dct[~mask] = dct_image[~mask]

    filtered_magspec[mask] = 255
    filtered_magspec[~mask] = magnitude_spectrum[~mask]

    return filtered_dct, filtered_magspec

def overhaul_with_band_filter(image_path, angles, WD):
    image = io.imread(image_path, as_gray=True)
    dct_image = apply_dct(image)

    filtered_dct, filtered_magspec = three_way_band_filter(dct_image, angles, WD)
    
    filtered_image = rebuild_image(filtered_dct)
    
    return dct_image, filtered_dct, filtered_magspec, filtered_image

image_path1 = '/home/unitx/Downloads/macbook_images/grid2.jpg'
image_path2 = '/home/unitx/Downloads/macbook_images/grid2_btmright.jpg'
dct_NG, filtered_dct_NG, filtered_magspec_NG, filtered_NG = overhaul_with_band_filter(image_path1, angles, WD)
dct_OK, filtered_dct_OK, filtered_magspec_OK, filtered_OK = overhaul_with_band_filter(image_path2, angles, WD)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

axes[0,0].imshow(io.imread(image_path1, as_gray=True), cmap='gray')
axes[0,0].set_title('raw_lens1')

axes[0,1].imshow(io.imread(image_path2, as_gray=True), cmap='gray')
axes[0,1].set_title('raw_lens2')

axes[0,2].imshow(filtered_NG, cmap='gray')
axes[0,2].set_title('filtered_lens1')

axes[0,3].imshow(filtered_OK, cmap='gray')
axes[0,3].set_title('filtered_lens2')

axes[1,0].imshow(np.log1p(1 + (np.abs(dct_NG))**2), cmap='gray')
axes[1,0].set_title('DCT_lens1')

axes[1,1].imshow(np.log1p(1 + (np.abs(dct_OK))**2), cmap='gray')
axes[1,1].set_title('DCT_lens2')

axes[1,2].imshow(filtered_magspec_NG, cmap='gray')
axes[1,2].set_title('filtered_DCT_lens1')

axes[1,3].imshow(filtered_magspec_OK, cmap='gray')
axes[1,3].set_title('filtered_DCT_lens2')

plt.tight_layout()
plt.show()
