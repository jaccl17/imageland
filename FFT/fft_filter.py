# %% [markdown]

## Fast Fourier Transform (2) for filtering and processing images
# The primary goal of this exploration is to develop a script that moves a 2D array (ie: an image) into fourier space. with the pixel arrays (rows and columns) translated to frequencies, I can apply thresholds and masks to filter out noise,
# highlight geometries, and reconstruct the images to illuminate difficult-to-see defects
#
# This is similar to a DCT but allows me filter asymmetric signals, fitler in phase space, and work with negtive signals.
#%% [markdown]

### Import libaries and packages
# %%
import numpy as np
import argparse
from scipy import stats as st
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt

from skimage import io, img_as_ubyte

# %% [markdown]

### FFT function definitions

# %%

def apply_fft2(image):
    # fft2_image = np.real(fft2(image, axes=(0,1)))
    fft2_image = fft2(image, axes=(0,1))
    shifted_fft2_image = np.fft.fftshift(fft2_image) # moves zero frequency component to the orgin instead of the 4 corners
    return shifted_fft2_image

def rebuild_image(fft2_image):
    # ifft2_image = np.real(ifft2(fft2_image, axes=(0,1)))
    ifft2_image = ifft2(fft2_image)
    return ifft2_image

def rebuild_image_specs(magnitude_spectrum, phase_spectrum):
    complex_spectrum = magnitude_spectrum * np.exp(1j * phase_spectrum)
    unshifted_fft2_image = np.fft.ifftshift(complex_spectrum)
    ifft2_image = ifft2(unshifted_fft2_image)
    return ifft2_image


def spectrum_extract(fft2_image):
    magnitude_spectrum = np.abs(fft2_image) # for filtering
    log_magnitude_spectrum = np.log1p((np.abs(fft2_image))) # for visualization
    phase_spectrum = np.angle(fft2_image) # for filtering
    return magnitude_spectrum, log_magnitude_spectrum, phase_spectrum

# %% [markdown]

### Filter functions (called with terminal args)

# %%

def star_filter(dct_image, angle, thickness):
    u_max, v_max = dct_image.shape
    center_u, center_v = u_max // 2, v_max // 2
    mask = np.zeros_like(dct_image, dtype=bool)
    theta = np.radians(angle)

    u_values = np.arange(center_u, u_max).reshape(-1, 1)  
    v_values = np.arange(center_v, v_max).reshape(1, -1) 

    if theta == np.pi / 4:
        m = 1
        condition_upper = (v_values >= m * (u_values - center_u) + center_v - thickness / 2)
        condition_lower = (v_values <= m * (u_values - center_u) + center_v + thickness / 2)

    elif 0 < theta < np.pi / 4:
        m = np.tan(theta)
        condition_upper = (v_values >= m * (u_values - center_u) + center_v - thickness / 2)
        condition_lower = (v_values <= m * (u_values - center_u) + center_v + thickness / 2)

    elif np.pi / 2 > theta > np.pi / 4:
        m = 1 / np.tan(theta)
        condition_upper = (u_values >= m * (v_values - center_v) + center_u - thickness / 2)
        condition_lower = (u_values <= m * (v_values - center_v) + center_u + thickness / 2)

    elif theta == 0:
        condition_upper = v_values >= center_v - thickness / 2
        condition_lower = v_values <= center_v + thickness / 2

    elif theta == np.pi / 2:
        condition_upper = u_values >= center_u - thickness / 2
        condition_lower = u_values <= center_u + thickness / 2

    mask[center_u:, center_v:] = condition_upper & condition_lower

    mask[:center_u, :] = np.flip(mask[center_u + (u_max % 2):, :], axis=0)  # correct the reflection when u_max is odd
    mask[:, :center_v] = np.flip(mask[:, center_v + (v_max % 2):], axis=1)

    return mask

def filter_box(fft2_image, width, height):
    u_max, v_max = fft2_image.shape
    mask = np.zeros_like(fft2_image, dtype=bool)

    u_values = np.arange(u_max).reshape(-1, 1) 
    v_values = np.arange(v_max).reshape(1, -1)

    u_center = u_max / 2
    v_center = v_max / 2

    width_condition = (v_center - width/2 < v_values) & (v_values <= v_center + width/2)
    height_condition = (u_center - height/2 < u_values) & (u_values <= u_center + height/2)

    mask = width_condition & height_condition

    return mask


def filter(raw_spectrum, filter_type, width, height, T_D, S_D, F_D):
    threshold_mask = np.zeros_like(raw_spectrum, dtype=bool)
    total_mask = np.zeros_like(raw_spectrum, dtype=bool)
    filtered_fft2 = np.zeros_like(raw_spectrum)
    filtered_spectrum = np.zeros_like(raw_spectrum)

    if filter_type == 'box':
        box_mask = filter_box(raw_spectrum, width, height)

        filtered_spectrum[box_mask] = np.max(raw_spectrum) + 0.1
        filtered_spectrum[~box_mask] = raw_spectrum[~box_mask]

    elif filter_type == 'threshold':
        # gate_mask = np.full(np.shape(raw_spectrum), False)
        # gate_mask[5:3200,5:4500] = True

        threshold_mask = (raw_spectrum <= F_D) | (raw_spectrum >= T_D) # sets all cells that meet this condition (elimination condition) as TRUE
        tot =  threshold_mask

        filtered_spectrum[tot] = S_D #np.max(magnitude_spectrum) + 0.1 # all mask cells are set to white (note: they will all be black if every cell is set to 255)
        # print(filtered_spectrum)
        filtered_spectrum[~tot] = raw_spectrum[~tot] # all other cells will be their respective spectrum value

        # print(filtered_spectrum)
    elif filter_type == 'threshold_i':
        # gate_mask = np.full(np.shape(raw_spectrum), False)
        # gate_mask[5:3200,5:4500] = True

        threshold_mask = (raw_spectrum >= F_D) | (raw_spectrum <= T_D) # sets all cells that meet this condition (elimination condition) as TRUE
        tot =  threshold_mask

        filtered_spectrum[tot] = S_D #np.max(magnitude_spectrum) + 0.1 # all mask cells are set to white (note: they will all be black if every cell is set to 255)
        # print(filtered_spectrum)
        filtered_spectrum[~tot] = raw_spectrum[~tot] # all other cells will be their respective spectrum value

        # print(filtered_spectrum)

    elif filter_type == 'none':
        filtered_spectrum = raw_spectrum

    elif filter_type == 'boxhold':
        box_mask = filter_box(raw_spectrum, width, height)

        # gate_mask = np.full(np.shape(raw_spectrum), False)
        # gate_mask[20:3000,20:4000] = True

        threshold_mask = (raw_spectrum <= F_D) | (raw_spectrum >= T_D)

        total_mask = threshold_mask & box_mask #& gate_mask

        filtered_spectrum[total_mask] = np.max(raw_spectrum) + 0.1
        filtered_spectrum[~total_mask] = raw_spectrum[~total_mask]

    elif filter_type == 'star':
        star_mask = np.zeros_like(raw_spectrum, dtype=bool)
        
        for angle in angles:
            angle_mask = star_filter(raw_spectrum,angle,thickness)
            star_mask |= angle_mask

        filtered_spectrum[star_mask] = np.max(raw_spectrum) + 0.1
        filtered_spectrum[~star_mask] = raw_spectrum[~star_mask]

    elif filter_type == 'starhold':
        star_mask = np.zeros_like(raw_spectrum, dtype=bool)
        
        for angle in angles:
            angle_mask = star_filter(raw_spectrum,angle,thickness)
            star_mask |= angle_mask

        threshold_mask = (raw_spectrum <= F_D) | (raw_spectrum >= T_D) # sets all cells that meet this condition (elimination condition) as TRUE
        tot =  threshold_mask & star_mask

        filtered_spectrum[tot] = np.max(raw_spectrum) + 0.1
        filtered_spectrum[~tot] = raw_spectrum[~tot]
    elif filter_type == 'zap':
        threshold_mask = (raw_spectrum == F_D) | (raw_spectrum == T_D)
        tot =  threshold_mask

        filtered_spectrum[tot] = S_D #np.max(magnitude_spectrum) + 0.1 # all mask cells are set to white (note: they will all be black if every cell is set to 255)
        # print(filtered_spectrum)
        filtered_spectrum[~tot] = raw_spectrum[~tot] # all other cells will be their respective spectrum value
    return filtered_spectrum

#%% [markdown]

### Overhaul and Save Image functions
#  The first function reads the input image, applies the transforms and filter(s), rebuilds and outputs the raw + new image
#  as well as the filtered + non-filtered magnitude spectrum

#  The second function just allows for an array to be saved as a png

# %%

def overhaul(image_path, filter_type1, filter_type2, width, height):
    image = io.imread(image_path, as_gray=True)
    fft2_image = apply_fft2(image)
    magnitude_spectrum, log_magnitude_spectrum, phase_spectrum = spectrum_extract(fft2_image)
    # print(st.mode(np.round(phase_spectrum,1)))
    filtered_magspec = filter(magnitude_spectrum, filter_type1, width, height, T_D_mag, S_D, F_D_mag)
    filtered_phasespec = filter(phase_spectrum, filter_type2, width, height, T_D_phase, S_D, F_D_phase)
    filtered_image = np.real(rebuild_image_specs(filtered_magspec, filtered_phasespec))
    # print(filtered_image)
    return image, log_magnitude_spectrum, phase_spectrum, filtered_image, np.log1p(filtered_magspec), filtered_phasespec

def save_as_png(array, filename):
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    uint8_array = img_as_ubyte(normalized_array)
    io.imsave(filename, uint8_array)

#%% [markdown]

## Main
#  sets up the parser, defines image paths and variables, and outputs graphs + images

# %%

if __name__ == "__main__":

    # main vars

    image_path1 = '/home/unitx/wabbit_playground/FFT/testbook.png'
    image_path2 = '/home/unitx/Downloads/macbook_images/doublebarlight_03_cropped.png'

    S_D = 0

    # mag spec avrs
    T_D_mag = 5000 # upper
    F_D_mag = 0 # lower

    # phase spec avrs
    T_D_phase = -3.1 # upper
    F_D_phase = -3 # lower

    # angles = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
    angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # angles = np.arange(0,84,3)

    thickness = 60

    parser = argparse.ArgumentParser(description='FFT2 image filtering')
    parser.add_argument('filter_type1', choices=['box', 'threshold', 'threshold_i', 'boxhold', 'none', 'star', 'starhold', 'zap'], help="Type of filter to apply: 'box', 'threshold', 'boxhold' or 'none'")
    parser.add_argument('filter_type2', choices=['box', 'threshold', 'threshold_i', 'boxhold', 'none', 'star', 'starhold', 'zap'], help="Type of filter to apply: 'box', 'threshold', 'boxhold' or 'none'")
    parser.add_argument('-wd', '--width', help="This is relevent for 'box' and 'boxhold' filters; select a width to apply the filter at each given angle", type=int)
    parser.add_argument('-ht', '--height', help="This is relevent for 'box' and 'boxhold' filters; select a width to apply the filter at each given angle", type=int)
    args = parser.parse_args()

    image01, log_magnitude_spectrum01, phase_spectrum01, filtered_image01, log_filtered_magspec01, filtered_phasespec01 = overhaul(image_path1, args.filter_type1, args.filter_type2, args.width, args.height)
    # fft2_02, magnitude_spectrum_02, filtered_fft2_02, filtered_magspec_02, filtered_02, phase_spectrum02 = overhaul(image_path2, args.filter_type, args.width, args.height)

    # round_magspec_01 = np.round(magnitude_spectrum_01, 5)
    # flat_round_magspec_01 = round_magspec_01.flatten()

    # round_magspec_02 = np.round(magnitude_spectrum_02, 5)
    # flat_round_magspec_02 = round_magspec_02.flatten()

    image1 = io.imread(image_path1, as_gray=True)
    # image2 = io.imread(image_path2, as_gray=True)

    # print(filtered_magspec_01)

    # Plotting
    fig1, axes1 = plt.subplots(2, 3, figsize=(33, 18), gridspec_kw={'width_ratios': [1, 1, 1]})

    im1 = axes1[0, 0].imshow(image1, cmap='gray')
    axes1[0, 0].set_title('Raw Image')

    im2 = axes1[0, 1].imshow(phase_spectrum01, cmap='gray')
    axes1[0, 1].set_title('Phase Spectrum')

    im3 = axes1[0, 2].imshow(log_magnitude_spectrum01, cmap='gray')
    axes1[0, 2].set_title('Magnitude Spectrum')

    im4 = axes1[1, 0].imshow(filtered_image01, cmap='gray')
    axes1[1, 0].set_title('Filtered Image')

    im5 = axes1[1, 1].imshow(filtered_phasespec01, cmap='gray')
    axes1[1, 1].set_title('Filtered Phase Spectrum')

    im6 = axes1[1, 2].imshow(log_filtered_magspec01, cmap='gray')
    axes1[1, 2].set_title('Filtered Magnitude Spectrum')

    # plt.colorbar(im1, ax=axes1[0, 0])
    # plt.colorbar(im2, ax=axes1[0, 1])
    # plt.colorbar(im3, ax=axes1[0, 2])
    # plt.colorbar(im4, ax=axes1[0, 3])
    # plt.colorbar(im5, ax=axes1[1, 0])
    # plt.colorbar(im6, ax=axes1[1, 1])



    # fig2, axes2 = plt.subplots(2, 3, figsize=(25, 10), gridspec_kw={'width_ratios': [1, 1, 1]})


    # flat_mag_01 = magnitude_spectrum_01.flatten()
    # axes2[0,0].set_ylim(0,40)
    # axes2[0,0].plot(np.arange(0,len(flat_mag_01),1),flat_mag_01)
    # axes2[0,0].set_title('magnitudes_01')

    # flat_mag_filtered_01 = filtered_magspec_01.flatten()
    # axes2[0,1].set_ylim(0,40)
    # axes2[0,1].plot(np.arange(0,len(flat_mag_filtered_01),1),flat_mag_filtered_01)
    # axes2[0,1].set_title('filtered_magnitudes_01')

    # axes2[0,2].text(0.0, 0.7, f' size mag_spec: {len(flat_round_magspec_01)}\n mean mag_spec: {np.mean(magnitude_spectrum_01)}\n mode mag_spec: {st.mode(flat_round_magspec_01).mode}\n mode_count mag_spec: {st.mode(flat_round_magspec_01).count}\n max mag_spec: {np.max(magnitude_spectrum_01)}\n min mag_spec: {np.min(magnitude_spectrum_01)}', horizontalalignment='left', verticalalignment='top', fontsize=12)
    # axes2[0,2].axis('off')


    # flat_mag_02 = magnitude_spectrum_02.flatten()
    # axes2[1,0].set_ylim(0,40)
    # axes2[1,0].plot(np.arange(0,len(flat_mag_02),1),flat_mag_02)
    # axes2[1,0].set_title('magnitudes_02')

    # flat_mag_filtered_02 = filtered_magspec_02.flatten()
    # axes2[1,1].set_ylim(0,40)
    # axes2[1,1].plot(np.arange(0,len(flat_mag_filtered_02),1),flat_mag_filtered_02)
    # axes2[1,1].set_title('filtered_magnitudes_02')

    # axes2[1,2].text(0.0, 0.7, f' size mag_spec: {len(flat_round_magspec_02)}\n mean mag_spec: {np.mean(magnitude_spectrum_02)}\n mode mag_spec: {st.mode(flat_round_magspec_02).mode}\n mode_count mag_spec: {st.mode(flat_round_magspec_02).count}\n max mag_spec: {np.max(magnitude_spectrum_02)}\n min mag_spec: {np.min(magnitude_spectrum_02)}', horizontalalignment='left', verticalalignment='top', fontsize=12)
    # axes2[1,2].axis('off')

    plt.tight_layout()
    plt.show()

# save_as_png(filtered_01, 'FFT2_filtered_01.png')
# save_as_png(filtered_02, 'FFT2_filtered_02.png')