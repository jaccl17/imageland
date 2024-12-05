import numpy as np
import argparse
from scipy import stats as st
from scipy.fftpack import idct, dct, fft, ifft
import matplotlib.pyplot as plt

from skimage import io, img_as_ubyte

T_D = .8
F_D = 0
S_D = 0

angles = [1]
the_norm = 'backward'
the_type  = 2

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

def filter_by_angle(dct_image, angle, W_D):
    u_max, v_max = dct_image.shape
    mask = np.zeros_like(dct_image, dtype=bool)
    theta = np.radians(angle)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    u_values = np.arange(u_max).reshape(-1, 1)  # Column vector
    v_values = np.arange(v_max).reshape(1, -1)  # Row vector

    u_cos_v = u_values * sin_theta - v_values * cos_theta

    if theta < np.pi/9:
        condition_upper = (u_cos_v <= 0) #& (v_values <= 50)
        condition_lower = (u_cos_v + W_D >= 0)

    elif np.pi/9 <= theta < 7*np.pi/18:
        condition_upper = (u_cos_v - W_D / 2 <= 0) #& (v_values <= 50)
        condition_lower = (u_cos_v + W_D / 2 >= 0)

    else:
        condition_upper = (u_cos_v - W_D <= -0.5) #& (v_values <= 50) # pixel at position 0 expands from -0.5 to +0.5 so the image is offset by -0.5px so i set this offset as my starting/min condition instead of using 0
        condition_lower = (u_cos_v >= -0.5) # pixel at position 0 expands from -0.5 to +0.5 so the image is offset by -0.5px so i set this offset as my starting/min condition instead of using 0

    mask = condition_upper & condition_lower
    
    return mask

def filter(dct_image, filter_type, angles, W_D, T_D, S_D, F_D):
    magnitude_spectrum = np.log1p((np.abs(dct_image))**2)
    band_mask = np.zeros_like(dct_image, dtype=bool)
    threshold_mask = np.zeros_like(dct_image, dtype=bool)
    total_mask = np.zeros_like(dct_image, dtype=bool)
    filtered_dct = np.zeros_like(dct_image)
    filtered_magspec = np.zeros_like(dct_image)

    if filter_type == 'band':
        for angle in angles:
            angle_mask = filter_by_angle(dct_image, angle, W_D)
            band_mask |= angle_mask

        filtered_dct[band_mask] = np.max(magnitude_spectrum) + 0.1 # Set frequency values in the band to 0
        filtered_dct[~band_mask] = dct_image[~band_mask]

        filtered_magspec[band_mask] = np.max(magnitude_spectrum) + 0.1
        filtered_magspec[~band_mask] = magnitude_spectrum[~band_mask]

    elif filter_type == 'threshold':
        gate_mask = np.full(np.shape(dct_image), False)
        gate_mask[5:3200,5:4500] = True

        threshold_mask = (magnitude_spectrum <= F_D) | (magnitude_spectrum >= T_D) # sets all cells that meet this condition (elimination condition) as TRUE
        tot =  threshold_mask & gate_mask

        filtered_dct[tot] = S_D # all cells to be eliminated are set to S_D (zero)
        filtered_dct[~tot] = dct_image[~tot] # all other cells to kept are given the value of dct_image

        filtered_magspec[tot] = np.max(magnitude_spectrum) + 0.1 # all mask cells are set to white (note: they will all be black if every cell is set to 255)
        filtered_magspec[~tot] = magnitude_spectrum[~tot] # all other cells will be their respective mag spectrum value

    elif filter_type == 'none':
        filtered_dct = dct_image
        filtered_magspec = magnitude_spectrum

    elif filter_type == 'both':
        for angle in angles:
            angle_mask = filter_by_angle(dct_image, angle, W_D)
            band_mask |= angle_mask

        # gate_mask = np.full(np.shape(dct_image), False)
        # gate_mask[20:3000,20:4000] = True

        threshold_mask = (magnitude_spectrum <= F_D) | (magnitude_spectrum >= T_D)

        total_mask = threshold_mask & band_mask #& gate_mask

        filtered_dct[total_mask] = magnitude_spectrum[0,0] 
        filtered_dct[~total_mask] = dct_image[~total_mask]

        filtered_magspec[total_mask] = np.max(magnitude_spectrum) + 0.1
        filtered_magspec[~total_mask] = magnitude_spectrum[~total_mask]

    return magnitude_spectrum, filtered_dct, filtered_magspec

def overhaul(image_path, W_D, filter_type):
    image = io.imread(image_path, as_gray=True)
    dct_image = apply_dct(image)
    magnitude_spectrum, filtered_dct, filtered_magspec = filter(dct_image, filter_type, angles, W_D, T_D, S_D, F_D)
    filtered_image = rebuild_image(filtered_dct)
    return dct_image, magnitude_spectrum, filtered_dct, filtered_magspec, filtered_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DCT image filtering')
    parser.add_argument('W_D', help="This is relevent for 'band' and 'both' filters; select a width to apply the filter at each given angle", type=int)
    parser.add_argument('filter_type', choices=['band', 'threshold', 'both', 'none'], help="Type of filter to apply: 'band', 'threshold', 'both' or 'none'")
    args = parser.parse_args()

    image_path1 = '/home/unitx/Downloads/macbook_images/gradient2.jpg'
    image_path2 = '/home/unitx/Downloads/macbook_images/doublebarlight_03_cropped.png'

    dct_01, magnitude_spectrum_01, filtered_dct_01, filtered_magspec_01, filtered_01 = overhaul(image_path1, args.W_D, args.filter_type)
    dct_02, magnitude_spectrum_02, filtered_dct_02, filtered_magspec_02, filtered_02 = overhaul(image_path2, args.W_D, args.filter_type)

    round_magspec_01 = np.round(magnitude_spectrum_01, 5)
    flat_round_magspec_01 = round_magspec_01.flatten()

    round_magspec_02 = np.round(magnitude_spectrum_02, 5)
    flat_round_magspec_02 = round_magspec_02.flatten()

    # Plotting
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]})

    image1 = io.imread(image_path1, as_gray=True)
    image2 = io.imread(image_path2, as_gray=True)

    axes[0,0].imshow(image1, cmap='gray')
    axes[0,0].set_title('raw_lens1')

    axes[0,1].imshow(filtered_01, cmap='gray')
    axes[0,1].set_title('filtered_lens1')

    axes[0,2].imshow(magnitude_spectrum_01, cmap='gray')
    axes[0,2].set_title('DCT_lens1')

    axes[0,3].imshow(filtered_magspec_01, cmap='gray')
    axes[0,3].set_title('filtered_DCT_lens1')

    axes[0,4].text(0.0, 0.7, f' size mag_spec: {len(flat_round_magspec_01)}\n mean mag_spec: {np.mean(magnitude_spectrum_01)}\n mode mag_spec: {st.mode(flat_round_magspec_01).mode}\n mode_count mag_spec: {st.mode(flat_round_magspec_01).count}\n max mag_spec: {np.max(magnitude_spectrum_01)}\n min mag_spec: {np.min(magnitude_spectrum_01)}', horizontalalignment='left', verticalalignment='top', fontsize=12)
    axes[0,4].axis('off')


    axes[1,0].imshow(image2, cmap='gray')
    axes[1,0].set_title('raw_lens2')

    axes[1,1].imshow(filtered_02, cmap='gray')
    axes[1,1].set_title('filtered_lens2')

    axes[1,2].imshow(magnitude_spectrum_02, cmap='gray')
    axes[1,2].set_title('DCT_lens2')

    axes[1,3].imshow(filtered_magspec_02, cmap='gray')
    axes[1,3].set_title('filtered_DCT_lens2')

    axes[1,4].text(0.0, 0.7, f' size mag_spec: {len(flat_round_magspec_02)}\n mean mag_spec: {np.mean(magnitude_spectrum_02)}\n mode mag_spec: {st.mode(flat_round_magspec_02).mode}\n mode_count mag_spec: {st.mode(flat_round_magspec_02).count}\n max mag_spec: {np.max(magnitude_spectrum_02)}\n min mag_spec: {np.min(magnitude_spectrum_02)}', horizontalalignment='left', verticalalignment='top', fontsize=12)
    axes[1,4].axis('off')

    plt.tight_layout()
    plt.show()

# save_as_png(filtered_01, 'DCT_filtered_01.png')
# save_as_png(filtered_02, 'DCT_filtered_02.png')