import tkinter as tk
from tkinter import filedialog
import FLIMageFileReader
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

lifetimeLimit = [1, 2]
intensityLimit = [0, 50]
z_plane = 4
lifetime_offset = 1

# user select file to load and get root directory
file_path = filedialog.askopenfilename()
p = Path(file_path)
root_dir = p.parent

# based on selected file, find similarly named .flim files
prefix = re.sub(r'\d+\.flim$', '', p.name) # remove trailing digits before .flim to make template to find files
flim_files = sorted(root_dir.glob(prefix + "*.flim"))
n_files = len(flim_files)

# load first file to get num planes and x/y dims
iminfo = FileReader()
iminfo.read_imageFile(str(flim_files[0]), True)
num_z = len(iminfo.acqTime)
iminfo.calculatePage(0, 0, 0, [2, iminfo.n_time[0]], intensityLimit, lifetimeLimit, lifetime_offset)
x, y = iminfo.rgbLifetime.shape[:2]

# load all files and make grayscale image
group_lifetime = np.zeros((num_z, n_files, x, y))
group_rgblifetime = np.zeros((num_z, n_files, x, y))
group_lifetimemap = np.zeros((num_z, n_files, x, y))
group_intensity = np.zeros((num_z, n_files, x, y))
for f, flim_file in enumerate(flim_files):

    iminfo = FileReader()
    iminfo.read_imageFile(str(flim_file), True)

    for z_plane in range(num_z):

        iminfo.calculatePage(
            z_plane,
            0,
            0,
            [2, iminfo.n_time[0]],
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

        group_lifetime[z_plane, f, :, :] = grayImage
        group_rgblifetime[z_plane, f, :, :] = iminfo.calculateRGBLifetimeMap(lifetimeLimit = [1.0, 2.0], intensityLimit = [3, 20])
        group_lifetimemap[z_plane, f, :, :] = iminfo.lifetimeMap
        group_intensity[z_plane, f, :, :] = iminfo.intensity


plt.figure
img = plt.imshow(grayImage) # lifetimeMap rgbLifetime
#img.set_clim(4,10)
plt.title('Gray image of last frame')
plt.colorbar(img)
plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SUITE2P PORTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# s2p expects data to have dimensions: n_time, Ly, Lx

from suite2p import registration
from suite2p.io import BinaryFile

z_plane = 4
data = np.squeeze(group_lifetime[z_plane, :, :, :])
data_rgb = np.squeeze(group_lifetime[z_plane, :, :, :])

fname = prefix
n_time = n_files
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


# Convert our example tif file into a binary file
if data.dtype != np.int16:
    if np.issubdtype(data.dtype, np.floating):
        # Scale float (assumed 0-1) to int16 range
        data = (data * 32767).astype(np.int16)
    elif data.dtype == np.uint16:
        data = (data // 2).astype(np.int16)
    elif data.dtype == np.uint8:
        data = (data.astype(np.int16) * 128)
    else:
        # General case: clip and cast
        data = data.astype(np.int16)# Write to binary

raw_bin_path = os.path.join(root_dir, 'raw_data.bin')
reg_bin_path = os.path.join(root_dir, 'registered_data.bin')
data.tofile(raw_bin_path)
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
img1 = np.squeeze(np.mean(data, axis=0))
img2 = reg_outputs['meanImg']

# Set shared range using percentiles for saturation
vmin = np.percentile([img1, img2], 2)
vmax = np.percentile([img1, img2], 98)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im1 = axes[0].imshow(img1, cmap='gray', vmin=vmin, vmax=vmax)
axes[0].set_title("Raw intensity image")
plt.colorbar(im1, ax=axes[0], label='Intensity')

im2 = axes[1].imshow(img2, cmap='gray', vmin=vmin, vmax=vmax)
axes[1].set_title("Motion-corrected intensity image")
plt.colorbar(im2, ax=axes[1], label='Intensity')

axes[0].set_axis_off()
axes[1].set_axis_off()

plt.tight_layout()
plt.show()


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


def imshow_raw_mc(raw, mc, title_addon, cbar_label='', cmap_='gray'):

    img1 = raw
    img2 = mc

    vmin = np.nanpercentile([img1, img2], 2)
    vmax = np.nanpercentile([img1, img2], 99)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axes[0].imshow(img1, cmap=cmap_, vmin=vmin, vmax=vmax)
    axes[0].set_title("Raw image: " + title_addon)
    plt.colorbar(im1, ax=axes[0], label=cbar_label)

    im2 = axes[1].imshow(img2, cmap=cmap_, vmin=vmin, vmax=vmax)
    axes[1].set_title("Manually offset image: " + title_addon)
    plt.colorbar(im2, ax=axes[1], label=cbar_label)

    axes[0].set_axis_off()
    axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()

manual_mc = apply_offsets(data, reg_outputs['yoff'], reg_outputs['xoff'])
imshow_raw_mc(np.squeeze(np.nanmean(data, axis=0)), np.squeeze(np.nanmean(manual_mc, axis=0)), 'grayscale', cbar_label='Intensity')

### do the same but for RGBlifetime image
manual_mc_rgb = apply_offsets(data_rgb, reg_outputs['yoff'], reg_outputs['xoff'])
imshow_raw_mc(np.squeeze(np.nanmean(data_rgb, axis=0)), np.squeeze(np.nanmean(manual_mc_rgb, axis=0)), 'rgb', cbar_label='RGB Lifetime', cmap_='turbo')


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



