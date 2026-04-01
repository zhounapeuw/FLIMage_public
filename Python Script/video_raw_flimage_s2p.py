import tkinter as tk
from tkinter import filedialog
from FLIMageFileReader import FileReader
import matplotlib as mpl
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

data_type = 'lifetime'
lifetimeLimit = [1.6, 2] # first entry will be the upper bound (red) of the colorbar, 2nd is the lower bound (blue)
intensityLimit = [3, 300]# [3, 100] for pure intensity
# semi-static vars
spc_start_idx = 2 
lifetime_offset = 1.1

if data_type == 'intensity':
    norm = mpl.colors.Normalize(vmin=intensityLimit[0], vmax=intensityLimit[1])
    colormap_ = 'gray'
    cbar_label = 'Intensity'
elif data_type == 'lifetime':
    norm = mpl.colors.Normalize(vmin=lifetimeLimit[0], vmax=lifetimeLimit[1])
    colormap_ = 'turbo'
    cbar_label = 'Lifetime (ns)'
sm = mpl.cm.ScalarMappable(cmap=colormap_, norm=norm)
sm.set_array([])


root_dir = r'C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen'

# intensity
raw_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen\s2p_analysis\intensity_raw.npy"
flimage_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen\flimage_mc\060825ST01F00T1_allSumaF.flim"
s2p_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen\s2p_analysis\intensity_rig_nonrig_half.npy"

# lifetime
raw_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen\s2p_analysis\rbglifetimemap_raw.npy"
flimage_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen\flimage_mc\060825ST01F00T1_allSumaF.flim"
s2p_rig_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen\s2p_analysis\rbglifetimemap_s2p_mc.npy"
s2p_rig_nonrig_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen\s2p_analysis\rbglifetimemap_rig_nonrig_half.npy"

raw_data = np.squeeze(np.load(raw_path))
s2p_rig_data = np.load(s2p_rig_path)
s2p_rig_nonrig_data = np.load(s2p_rig_nonrig_path)

iminfo = FileReader()
iminfo.read_imageFile(flimage_path, True)

iminfo.calculatePage(0, 0, 0, [spc_start_idx, iminfo.n_time[0]], intensityLimit, lifetimeLimit, lifetime_offset)
x, y = iminfo.rgbLifetime.shape[:2]
n_times = len(iminfo.acqTime)

group_rgblifetime = np.zeros((n_times, x, y, 3))
group_intensity = np.zeros((n_times, x, y))
for time_idx in range(n_times):

    iminfo.calculatePage(
            time_idx,
            0,
            0,
            [spc_start_idx, iminfo.n_time[0]],
            intensityLimit,
            lifetimeLimit,
            lifetime_offset
        )
    
    group_rgblifetime[time_idx, :, :, :] = iminfo.rgbLifetime
    group_intensity[time_idx, :, :] = iminfo.intensity

if data_type == 'intensity':
    flimage_to_plot = group_intensity
elif data_type == 'lifetime':
    flimage_to_plot = group_rgblifetime

T = raw_data.shape[0]

#~~~~~~~~~~~~~~~~~~~~~~``
import math

# ---- define what you want to plot ----
data_list = [raw_data, flimage_to_plot, s2p_rig_data, s2p_rig_nonrig_data]  # add/remove freely
titles = ["Raw", "FLIMage MC", "Suite2p Rig Only", "Suite2p Rig-NonRig"]

n_plots = len(data_list)

#~~~~~~~~~~~~~~~~~~~~ make mean images


# ---- average across time (axis=0 assumes [T, Y, X]) ----
avg_data_list = [np.mean(data, axis=0) for data in data_list]

# ---- choose layout ----
if n_plots <= 3:
    rows, cols = 1, n_plots
else:
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)

# ---- create subplots ----
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True)

# flatten safely
if isinstance(axes, np.ndarray):
    axes = axes.ravel()
else:
    axes = [axes]

# ---- remove unused axes ----
for ax in axes[n_plots:]:
    ax.axis("off")

# ---- remove axes visuals ----
for ax in axes[:n_plots]:
    ax.axis("off")

# ---- plot averaged images ----
ims = []
for ax, data in zip(axes, avg_data_list):
    im = ax.imshow(data, cmap=colormap_)
    ims.append(im)

# ---- titles ----
for ax, title in zip(axes, titles):
    ax.set_title(title)

# ---- shared colorbar ----
cbar = fig.colorbar(sm, ax=axes[:n_plots], fraction=0.046, pad=0.04)
cbar.set_label(cbar_label)

if lifetimeLimit[0] < lifetimeLimit[1]:
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks[::-1]])

# ---- save static image instead of animation ----
plt.savefig(
    os.path.join(root_dir, "lifetime_comparison_mean.png"),
    dpi=200
)

plt.close(fig)

#~~~~~~~~~~~~~~~~~~~~ make movie

# ---- choose layout ----
if n_plots <= 3:
    rows, cols = 1, n_plots
else:
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)

# ---- create subplots ----
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

# flatten safely
if isinstance(axes, np.ndarray):
    axes = axes.ravel()
else:
    axes = [axes]

# ---- remove unused axes ----
for ax in axes[n_plots:]:
    ax.axis("off")

# ---- remove axes visuals ----
for ax in axes[:n_plots]:
    ax.axis("off")

# ---- initial frames ----
ims = []
for ax, data in zip(axes, data_list):
    im = ax.imshow(data[0], cmap=colormap_)
    ims.append(im)

# ---- titles ----
for ax, title in zip(axes, titles):
    ax.set_title(title)

# ---- shared colorbar ----
cbar = fig.colorbar(sm, ax=axes[:n_plots], fraction=0.046, pad=0.04)
cbar.set_label(cbar_label)

if lifetimeLimit[0] < lifetimeLimit[1]:
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks[::-1]])

# ---- animation update ----
def update(frame):
    for im, data in zip(ims, data_list):
        im.set_data(data[frame])
    return ims

anim = FuncAnimation(
    fig,
    update,
    frames=T,
    interval=1000/15,
    blit=True
)

# ---- save ----
writer = FFMpegWriter(fps=15)
anim.save(
    os.path.join(root_dir, "lifetime_comparison_dynamic.mp4"),
    writer=writer,
    dpi=200
)

plt.close(fig)