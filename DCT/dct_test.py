import numpy as np
import argparse
from scipy import stats as st
from scipy.fftpack import idct, dct, fft, ifft
import matplotlib.pyplot as plt

from skimage import io, img_as_ubyte

T_D = 14.5
F_D = 11.5
S_D = 0
# W_D = 10
# angles = [90, 80, 70.6, 62, 54.5, 48.2, 43, 35, 25, 13, 0]
angles = [4,8,12,16,20,24,28,32,36,48,52,56,60,64,68,72,76,80,84,88]
# angles = [0,1,2,3,4,5,6,7,8,9,10,12,13,15,17,20,22,25,27,30,31,32,33,34,35,36,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,55,60,62,65,68,70,75,86,89,90]
the_norm = 'backward'
the_type  = 2

def save_as_png(array, filename):
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    uint8_array = img_as_ubyte(normalized_array)
    io.imsave(filename, uint8_array)

def apply_dct(image):
    dct_rows = dct(image, type=the_type, axis=0, norm=the_norm)
    dct_image = dct(dct_rows, type=the_type, axis=1, norm=the_norm)
    # dct_rows = fft(image)
    # dct_image = fft(dct_rows)
    return dct_image

def rebuild_image(dct_image):
    idct_rows = idct(dct_image, type=the_type, axis=1, norm=the_norm)
    idct_image = idct(idct_rows, type=the_type, axis=0, norm=the_norm)
    # idct_rows = ifft(dct_image)
    # idct_image = ifft(idct_rows)
    return idct_image

def filter_by_angle(dct_image, angle, W_D):
    u_max, v_max = dct_image.shape
    mask = np.zeros_like(dct_image, dtype=bool)
    theta = np.radians(angle)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    u_values = np.arange(u_max).reshape(-1, 1)  # Column vector
    # print(u_values)
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

def threshold_filter(dct_image, T_D, S_D, F_D):
    magnitude_spectrum = np.log1p( 1 + (np.abs(dct_image))**2)
    mask = (magnitude_spectrum <= F_D) | (magnitude_spectrum >= T_D)
    filtered_dct = np.zeros_like(dct_image)
    filtered_magspec = np.zeros_like(dct_image)
    round_magspec = np.round(magnitude_spectrum, 5)
    flat_round_magspec = round_magspec.flatten()

    
    print(f'size mag_spec: {len(flat_round_magspec)}')
    print(f'mean mag_spec: {np.mean(magnitude_spectrum)}')
    print(f'mode mag_spec: {st.mode(flat_round_magspec).mode}')
    print(f'mode_count mag_spec: {st.mode(flat_round_magspec).count}')
    print(f'max mag_spec: {np.max(magnitude_spectrum)}')
    print(f'min mag_spec: {np.min(magnitude_spectrum)}\n')
    
    filtered_dct[mask] = S_D
    filtered_dct[~mask] = dct_image[~mask]

    filtered_magspec[mask] = 255
    filtered_magspec[~mask] = magnitude_spectrum[~mask]

    return filtered_dct, filtered_magspec

def adjust_dct_energy(dct_image, filtered_dct):
    original_energy = np.sum(np.abs(dct_image))
    filtered_energy = np.sum(np.abs(filtered_dct))
    # print(f'og: {original_energy}')
    # print(f'filtered: {filtered_energy}')

    energy_ratio = original_energy / filtered_energy
    adjusted_dct = filtered_dct * energy_ratio

    # print(f'adjusted: {np.sum(np.abs(adjusted_dct))}')
    # else:
    #     adjusted_dct = filtered_dct

    return adjusted_dct

def rescale_image(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val > 0:
        scaled_image = (image - min_val) / (max_val - min_val) * 255
    else:
        scaled_image = image

    return scaled_image.astype(np.uint8)

def filter(dct_image, filter_type, angles, W_D, T_D, S_D, F_D):
    magnitude_spectrum = np.log1p((np.abs(dct_image))**2)
    # magnitude_spectrum = np.log1p(1 + np.abs(dct_image))
    band_mask = np.zeros_like(dct_image, dtype=bool)
    threshold_mask = np.zeros_like(dct_image, dtype=bool)
    total_mask = np.zeros_like(dct_image, dtype=bool)
    filtered_dct = np.zeros_like(dct_image)
    filtered_magspec = np.zeros_like(dct_image)

    if filter_type == 'band':
        for angle in angles:
            angle_mask = filter_by_angle(dct_image, angle, W_D)
            band_mask |= angle_mask

        # print(np.mean(magnitude_spectrum[band_mask]))
        filtered_dct[band_mask] = np.max(magnitude_spectrum) + 0.1 # Set frequency values in the band to 0
        filtered_dct[~band_mask] = dct_image[~band_mask]

        filtered_magspec[band_mask] = np.max(magnitude_spectrum) + 0.1
        filtered_magspec[~band_mask] = magnitude_spectrum[~band_mask]

    elif filter_type == 'threshold':
        threshold_mask = (magnitude_spectrum <= F_D) | (magnitude_spectrum >= T_D) # sets all cells that meet this condition (elimination condition) as TRUE 
        # print(f'thresh: + {threshold_mask}')

        filtered_dct[threshold_mask] = S_D # all cells to be eliminated are set to S_D (zero)
        filtered_dct[~threshold_mask] = dct_image[~threshold_mask] # all other cells to kept are given the value of dct_image

        filtered_magspec[threshold_mask] = np.max(magnitude_spectrum) + 0.1 # all mask cells are set to white (note: they will all be black if every cell is set to 255)
        filtered_magspec[~threshold_mask] = magnitude_spectrum[~threshold_mask] # all other cells will be their respective mag spectrum value
        # print(np.max(magnitude_spectrum))

    elif filter_type == 'none':

        filtered_dct = dct_image
        filtered_magspec = magnitude_spectrum

    elif filter_type == 'soft':
        decay_factor=0.5
        threshold_mask = (magnitude_spectrum <= F_D) | (magnitude_spectrum >= T_D) # sets all cells that meet this condition (elimination condition) as TRUE 
        # print(f'thresh: + {threshold_mask}')

        filtered_dct[threshold_mask] = S_D # all cells to be eliminated are set to S_D (zero)
        filtered_dct[~threshold_mask] = dct_image[~threshold_mask]*decay_factor # all other cells to kept are given the value of dct_image

        filtered_magspec[threshold_mask] = np.max(magnitude_spectrum) + 0.1 # all mask cells are set to white (note: they will all be black if every cell is set to 255)
        filtered_magspec[~threshold_mask] = magnitude_spectrum[~threshold_mask]/decay_factor # all other cells will be their respective mag spectrum value
        # print(np.max(magnitude_spectrum))

    elif filter_type == 'both':
        for angle in angles:
            angle_mask = filter_by_angle(dct_image, angle, W_D)
            band_mask |= angle_mask

        threshold_mask = (magnitude_spectrum <= F_D) | (magnitude_spectrum >= T_D)
        # print(f'thresh: {threshold_mask}')
        # print(f'band: {band_mask}')

        gate_mask = np.full(np.shape(dct_image), True)
        gate_mask[:1200,:1200] = False

        total_mask = threshold_mask & band_mask & gate_mask

        # print(f'mean: {np.min(dct_image)}')
        # print[magnitude_spectrum]
        filtered_dct[total_mask] = magnitude_spectrum[0,0] 
        filtered_dct[~total_mask] = dct_image[~total_mask]
        # filtered_dct[321,450] = np.max(dct_image)

        filtered_magspec[total_mask] = np.max(magnitude_spectrum) + 0.1
        filtered_magspec[~total_mask] = magnitude_spectrum[~total_mask]

        # print(np.mean(filtered_magspec))
    # print(f'filtered1: {np.mean(filtered_dct)}')
    # filtered_dct = adjust_dct_energy(dct_image, filtered_dct)
    # print(f'filtered2: {np.mean(filtered_dct)}')

    elif filter_type == 'test':
        for angle in angles:
            angle_mask = filter_by_angle(dct_image, angle, W_D)
            band_mask |= angle_mask

        # round_dct = np.round(dct_image, 1)
        # flat_round_dct = round_dct.flatten()

        gate_mask = np.full(np.shape(dct_image), False)
        gate_mask[40:3800,30:5200] = True

        threshold_mask = (magnitude_spectrum <= F_D) | (magnitude_spectrum >= T_D)

        mask1 = threshold_mask & band_mask
        mask2 = threshold_mask & gate_mask

        total_mask = mask1 | mask2
        # print(f'size mag_spec: {len(flat_round_dct)}')
        # print(f'mean mag_spec: {np.mean(dct_image)}')
        # print(f'mode mag_spec: {st.mode(flat_round_dct).mode}')
        # print(f'mode_count mag_spec: {st.mode(flat_round_dct).count}')
        # print(f'max mag_spec: {np.max(dct_image)}')
        # print(f'min mag_spec: {np.min(dct_image)}\n')

        filtered_dct[total_mask] = 0 # all cells to be eliminated are set to S_D (zero)
        filtered_dct[~total_mask] = dct_image[~total_mask] # all other cells to kept are given the value of dct_image

        filtered_magspec[total_mask] = np.max(magnitude_spectrum) + 0.1 # all mask cells are set to white (note: they will all be black if every cell is set to 255)
        filtered_magspec[~total_mask] = magnitude_spectrum[~total_mask] # all other cells will be their respective mag spectrum value
        # print(np.max(magnitude_spectrum))

    return magnitude_spectrum, filtered_dct, filtered_magspec

def overhaul(image_path, W_D, filter_type):
    image = io.imread(image_path, as_gray=True)
    # print(type(image))
    # print(type(np.ones((10,10))))
    dct_image = apply_dct(image)
    magnitude_spectrum, filtered_dct, filtered_magspec = filter(dct_image, filter_type, angles, W_D, T_D, S_D, F_D)
    # filtered_dct, filtered_magspec = threshold_filter(dct_image, T_D, S_D, F_D)
    #print(filtered_dct)
    filtered_image = rebuild_image(filtered_dct)
    # filtered_image[310:335,435:465] = dct_image[310:335,435:465]
    # rescaled_image = rescale_image(filtered_image)
    return dct_image, magnitude_spectrum, filtered_dct, filtered_magspec, filtered_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DCT image filtering')
    parser.add_argument('W_D', help="This is relevent for 'band' and 'both' filters; select a width to apply the filter at each given angle", type=int)
    parser.add_argument('filter_type', choices=['band', 'threshold', 'both', 'soft', 'none', 'test'], help="Type of filter to apply: 'band', 'threshold', 'soft', 'both' or 'none'")
    args = parser.parse_args()

    # Example image paths (you can modify them)
    image_path1 = '/home/unitx/Downloads/macbook_images/gradient2.jpg'
    image_path2 = '/home/unitx/Downloads/macbook_images/doublebarlight_02_cropped.png'

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
save_as_png(image2, 'doublebarlight_02_cropped_grayscale.png')
save_as_png(magnitude_spectrum_02, 'doublebarlight_02_cropped_MAGSPEC.png')
save_as_png(filtered_magspec_02, 'doublebarlight_02_cropped_FILTEREDMAGSPEC.png')
save_as_png(filtered_02, 'doublebarlight_02_cropped_FILTERED.png')
# save_as_png(C2, 'C2_image.png')
# save_as_png(C3, 'C3_image.png')
# save_as_png(C4, 'C4_image.png')