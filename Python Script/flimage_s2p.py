"""
NOTE: .flim file required formats:
1. multi-file: each .flim file contains data from a particular timepoint, can contain z-stack
2. single-file: each .flim file contains data from ALL timepoints; Z dimension encodes time, so no z-capabilities

"""

import tkinter as tk
from tkinter import filedialog
from FLIMageFileReader import FileReader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import suite2p
import torch
import os

import matplotlib.animation as animation
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

lifetimeLimit = [1.4, 2]
intensityLimit = [3, 8]
z_plane = 1
lifetime_offset = 1.1
spc_start_idx = 2
single_file = True

def to_int16(
    data: np.ndarray,
    input_range: tuple = None,
) -> np.ndarray:
    """
    Convert any numpy array to int16, with scaling appropriate for 2-photon imaging.

    Args:
        data:         Input array of any dtype
        input_range:  (min, max) to use for float normalization. If None, auto-detects
                      whether data is pre-normalized to [-1, 1], otherwise uses actual min/max.
    Returns:
        int16 array
    Raises:
        ValueError: If the input array contains NaNs
        TypeError:  If the input dtype is not supported
    """
    INT16_MIN, INT16_MAX = -32768, 32767

    if data.dtype == np.int16:
        return data.copy()

    # --- NaN check ---
    if np.issubdtype(data.dtype, np.floating) and np.isnan(data).any():
        raise ValueError("Input array contains NaNs. Please clean your data before converting.")

    # --- Float types ---
    if np.issubdtype(data.dtype, np.floating):
        if input_range is not None:
            low, high = input_range
        elif data.min() >= -1.0 and data.max() <= 1.0:
            low, high = -1.0, 1.0
        else:
            low, high = data.min(), data.max()

        normalized = (data - low) / (high - low)
        scaled = np.round(normalized * INT16_MAX).clip(INT16_MIN, INT16_MAX)
        return scaled.astype(np.int16)

    # --- Unsigned integer types ---
    elif np.issubdtype(data.dtype, np.unsignedinteger):
        info = np.iinfo(data.dtype)
        scaled = np.round(data.astype(np.float64) * (INT16_MAX / info.max))
        return scaled.clip(0, INT16_MAX).astype(np.int16)

    # --- Signed integer types ---
    elif np.issubdtype(data.dtype, np.signedinteger):
        info = np.iinfo(data.dtype)
        if info.min == INT16_MIN and info.max == INT16_MAX:
            return data.astype(np.int16)
        normalized = (data.astype(np.float64) - info.min) / (info.max - info.min)
        scaled = np.round(normalized * (INT16_MAX - INT16_MIN) + INT16_MIN)
        return scaled.clip(INT16_MIN, INT16_MAX).astype(np.int16)

    else:
        raise TypeError(f"Unsupported dtype: {data.dtype}")


### APPLY offsets on raw data (for validation) and plot
def apply_offsets(data_in, offsets_y, offsets_x):
    # data_in should be a (frame * y * x) 3D array
    # offsets should be a vector the same length as num frames
    
    data_mc = np.full_like(data_in, np.nan, dtype=np.float32)
    
    for i, (frame, dy, dx) in enumerate(zip(data_in, offsets_y, offsets_x)):
        data_mc[i] = np.roll(np.roll(frame, -dy, axis=0), -dx, axis=1)
    
        # NaN out wrapped edges
        if dy > 0: data_mc[i, -dy:, :] = np.nan
        elif dy < 0: data_mc[i, :-dy, :] = np.nan
        if dx > 0: data_mc[i, :, -dx:] = np.nan
        elif dx < 0: data_mc[i, :, :-dx] = np.nan

    return data_mc


def imshow_raw_mc(raw, mc, title_addon, cbar_label='',
                  cmap_='gray', lifetime_limit=None, invert_cmap=False):
    """
    Display raw and motion-corrected images side by side with colorbars.

    Args:
        raw:            Raw image (2D intensity or RGB 3D array for lifetime)
        mc:             Motion-corrected image (same format as raw)
        title_addon:    String appended to subplot titles
        cbar_label:     Colorbar label
        cmap_:          Colormap for intensity images
        lifetime_limit: If provided, treats images as lifetime RGB and uses fake colorbar
        invert_cmap:    If True, reverses colormap direction (also flips RGB channels for lifetime)
    """

    from matplotlib import gridspec

    # --------------------------------------------------
    # Create 4-axis layout: [raw | cbar | mc | cbar]
    # --------------------------------------------------
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[20, 1, 20, 1], figure=fig)

    ax_raw   = fig.add_subplot(gs[0, 0])
    ax_cbar1 = fig.add_subplot(gs[0, 1])
    ax_mc    = fig.add_subplot(gs[0, 2])
    ax_cbar2 = fig.add_subplot(gs[0, 3])

    # --------------------------------------------------
    # Lifetime mode: RGB + fake colorbar
    # --------------------------------------------------
    if lifetime_limit is not None:
        if invert_cmap:
            raw = raw[..., ::-1].copy()
            mc  = mc[..., ::-1].copy()

        # --- Plot images ---
        ax_raw.imshow(raw)
        ax_mc.imshow(mc)

        for ax in [ax_raw, ax_mc]:
            ax.set_xticks([])
            ax.set_yticks([])

        # --- Colormap subset like MATLAB jet(64)[9:56] ---
        cmap = plt.cm.jet(np.linspace(0, 1, 64))[:, :3]
        cmap2 = cmap[8:56, :]
        if invert_cmap:
            cmap2 = cmap2[::-1]

        cmap_img = cmap2.reshape((cmap2.shape[0], 1, 3))

        # --- Draw colorbars as images ---
        for ax_cbar in [ax_cbar1, ax_cbar2]:
            ax_cbar.imshow(cmap_img, aspect='auto')
            ax_cbar.set_xticks([])
            ax_cbar.set_ylim(0.5, cmap2.shape[0] + 0.5)
            ax_cbar.set_yticks([0, cmap2.shape[0]-1])
            ax_cbar.set_yticklabels(lifetime_limit[::-1])
            ax_cbar.yaxis.tick_right()

    # --------------------------------------------------
    # Intensity mode
    # --------------------------------------------------
    else:
        vmin = np.nanpercentile([raw, mc], 2)
        vmax = np.nanpercentile([raw, mc], 99)

        im1 = ax_raw.imshow(raw, cmap=cmap_, vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=ax_raw, label=cbar_label)

        im2 = ax_mc.imshow(mc, cmap=cmap_, vmin=vmin, vmax=vmax)
        plt.colorbar(im2, ax=ax_mc, label=cbar_label)

        # Remove unused colorbar axes
        ax_cbar1.remove()
        ax_cbar2.remove()

    # --- Titles ---
    ax_raw.set_title("Raw image: " + title_addon)
    ax_mc.set_title("Manually offset image: " + title_addon)

    plt.tight_layout()
    plt.show()

# user select file to load and get root directory
file_path = filedialog.askopenfilename()
p = Path(file_path)
root_dir = p.parent

if single_file:
    flim_files = [p]
else:
    # based on selected file, find similarly named .flim files
    prefix = re.sub(r'\d+\.flim$', '', p.name) # remove trailing digits before .flim to make template to find files
    flim_files = sorted(root_dir.glob(prefix + "*.flim"))

# load first file to get num planes and x/y dims
# for single-file mode, this first instance of reading .flim files loads the entire dataset
iminfo = FileReader()
iminfo.read_imageFile(str(flim_files[0]), True)
if iminfo.ZStack:
    num_z = len(iminfo.acqTime) # CZ THIS IS NOT THE BEST WAY TO FIND num_Z planes I THINK
else:
    num_z = 1
if single_file:
    n_times = len(iminfo.acqTime)
else:
    n_times = len(flim_files)
iminfo.calculatePage(0, 0, 0, [spc_start_idx, iminfo.n_time[0]], intensityLimit, lifetimeLimit, lifetime_offset)
x, y = iminfo.rgbLifetime.shape[:2]

# load all files and make grayscale image
group_lifetime = np.zeros((num_z, n_times, x, y))
group_rgblifetime = np.zeros((num_z, n_times, x, y, 3))
group_lifetimemap = np.zeros((num_z, n_times, x, y))
group_intensity = np.zeros((num_z, n_times, x, y))
for time_idx in range(n_times):

    # if multi-file, load each .flim timepoint
    if not single_file:
        iminfo = FileReader()
        iminfo.read_imageFile(str(flim_files[time_idx]), True)

    for z_plane in range(num_z):

        # i here is a dynamic variable; when in single-file mode, i is referencing the time sample
        # when in multi-file mode, i encodes the z plane
        if single_file:
            i = time_idx
        else:
            i = z_plane

        iminfo.calculatePage(
            i,
            0,
            0,
            [spc_start_idx, iminfo.n_time[0]],
            intensityLimit,
            lifetimeLimit,
            lifetime_offset
        )

        # reverse engineered calculateRGBLifetimeMap in FLIMageFileReader.py
        lifetime_norm = (iminfo.lifetimeMap - lifetimeLimit[0]) / (lifetimeLimit[1] - lifetimeLimit[0])
        lifetime_norm = 1 - lifetime_norm
        lifetime_norm = np.clip(lifetime_norm, 0, 1)

        intensity_norm = (iminfo.intensity - intensityLimit[0]) / (intensityLimit[1] - intensityLimit[0])
        intensity_norm = np.clip(intensity_norm, 0, 1)

        grayImage = lifetime_norm * intensity_norm

        group_lifetime[z_plane, time_idx, :, :] = grayImage
        iminfo.calculateRGBLifetimeMap(lifetimeLimit = lifetimeLimit, intensityLimit = intensityLimit)
        group_rgblifetime[z_plane, time_idx, :, :, :] = iminfo.rgbLifetime
        group_lifetimemap[z_plane, time_idx, :, :] = iminfo.lifetimeMap
        group_intensity[z_plane, time_idx, :, :] = iminfo.intensity


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SUITE2P PORTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# s2p expects data to have dimensions: n_time, Ly, Lx

from suite2p import registration
from suite2p.io import BinaryFile

data = np.squeeze(group_intensity[z_plane, :, :, :])
data_rgb = np.squeeze(group_rgblifetime[z_plane, :, :, :, :]) # dims: z_plane, frame, y, x, RBG_chan

fname = prefix
Lx, Ly = group_lifetime.shape[-2:] 

db = {
    "data_path": [root_dir], # Directory where your input files are located
    "save_path0": '/content/suite2p_output', # Directory where you want suite2p to write output files.
    "file_list": [fname], # Specify files you'd like to specifically use in the data_path
    "input_format": "tif",
    "nplanes": 1, # each tiff has these many planes in sequence
    "nchannels": 1, # each tiff has these many channels per plane
    "keep_movie_raw": True,
    "batch_size": 200, # we will decrease the batch_size in case low RAM on computer
}

settings = suite2p.default_settings()
settings['detection']['threshold_scaling'] = 2.0 # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
settings['fs'] = 13 # sampling rate of recording, determines binning for cell detection
settings['tau'] = 1.25 # timescale of gcamp to use for deconvolution
settings['device'] = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available for faster processing
settings['registration']['reg_tif'] = True
settings['registration']['nonrigid'] = False

raw_bin_path = os.path.join(root_dir, 'raw_data.bin')
reg_bin_path = os.path.join(root_dir, 'registered_data.bin')
to_int16(data).tofile(raw_bin_path)
f_raw = BinaryFile(Ly=Ly, Lx=Lx, filename=raw_bin_path)

# Create a binary file we will write our registered image to
f_reg = suite2p.io.BinaryFile(
    Ly=Ly, Lx=Lx, filename=reg_bin_path,
    n_frames = f_raw.shape[0], write=True
) # Set registered binary file to have same n_frames

settings['device'] = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available for faster processing
device = torch.device(settings['device'])

reg_outputs = registration.registration_wrapper(
  f_reg, f_raw=f_raw, f_reg_chan2=None, f_raw_chan2=None,
  align_by_chan2=None, save_path= db['save_path0'],
  badframes=None, settings=settings["registration"],
  device=device
)

f_reg.close()

# np.save(os.path.join(db['save_path0'], "reg_outputs.npy"), reg_outputs)

#### PLOT raw and S2P MC grayscale image

# Compute both images
img1 = np.squeeze(np.mean(to_int16(data), axis=0))
img2 = reg_outputs['meanImg']

# Set shared range using percentiles for saturation
vmin = np.percentile([img1, img2], 2)
vmax = np.percentile([img1, img2], 98)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im1 = axes[0].imshow(img1, cmap='gray', vmin=vmin, vmax=vmax)
axes[0].set_title("Raw intensity image")
plt.colorbar(im1, ax=axes[0], label='Intensity (int16)')

im2 = axes[1].imshow(img2, cmap='gray', vmin=vmin, vmax=vmax)
axes[1].set_title("Motion-corrected intensity image")
plt.colorbar(im2, ax=axes[1], label='Intensity')

axes[0].set_axis_off()
axes[1].set_axis_off()

plt.tight_layout()
plt.show()

#~~~~~~~~~ Plot raw and manually-corrected (using s2p offsets) images ~~~~~~~~~~~~

manual_mc = apply_offsets(data, reg_outputs['yoff'], reg_outputs['xoff'])
imshow_raw_mc(np.squeeze(np.nanmean(data, axis=0)), np.squeeze(np.nanmean(manual_mc, axis=0)), 'Intensity', cbar_label='Intensity', cmap_='gray')

### do the same but for RGBlifetime image
manual_mc_rgb = apply_offsets(data_rgb, reg_outputs['yoff'], reg_outputs['xoff'])
imshow_raw_mc(np.squeeze(np.nanmean(data_rgb, axis=0)), np.squeeze(np.nanmean(manual_mc_rgb, axis=0)), "Lifetime", cmap_='turbo', lifetime_limit=lifetimeLimit, invert_cmap=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# data_rgb and manual_mc_rgb already exist
# shape: (time, y, x)

T = data_rgb.shape[0]

# global color limits so both videos share the same scale
vmin = np.nanmin([data_rgb.min(), manual_mc_rgb.min()])
vmax = np.nanmax([data_rgb.max(), manual_mc_rgb.max()])

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

ax1, ax2 = axes

# remove axes
for ax in axes:
    ax.axis("off")

# initial frames
im1 = ax1.imshow(data_rgb[0], cmap='turbo')
im2 = ax2.imshow(manual_mc_rgb[0], cmap='turbo')

# titles
ax1.set_title("Raw")
ax2.set_title("MC Manually Offset")

# colorbar shared
cbar = fig.colorbar(im2, ax=axes, fraction=0.046, pad=0.04)
cbar.set_label("RGB lifetime")

def update(frame):
    im1.set_data(data_rgb[frame])
    im2.set_data(manual_mc_rgb[frame])
    return [im1, im2]

anim = FuncAnimation(
    fig,
    update,
    frames=T,
    interval=1000/15,  # for preview speed
    blit=True
)

# save video
writer = FFMpegWriter(fps=15)
anim.save(os.path.join(root_dir,"rgb_lifetime_comparison.mp4"), writer=writer, dpi=200)

plt.close(fig)



