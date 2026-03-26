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

lifetimeLimit = [1.6, 2] # first entry will be the upper bound (red) of the colorbar, 2nd is the lower bound (blue)
intensityLimit = [3, 300]
# semi-static vars
spc_start_idx = 2 
lifetime_offset = 1.1

norm = mpl.colors.Normalize(vmin=lifetimeLimit[0], vmax=lifetimeLimit[1])
sm = mpl.cm.ScalarMappable(cmap='turbo', norm=norm)
sm.set_array([])


root_dir = r'C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen'
raw_path = r'C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen\rbglifetimemap.npy'
flimage_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen\flimage_mc\060825ST01F00T1_allSumaF.flim"
s2p_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\helen\rbglifetimemap_s2p_mc.npy"

raw_data = np.squeeze(np.load(raw_path))
s2p_data = np.load(s2p_path)


iminfo = FileReader()
iminfo.read_imageFile(flimage_path, True)

iminfo.calculatePage(0, 0, 0, [spc_start_idx, iminfo.n_time[0]], intensityLimit, lifetimeLimit, lifetime_offset)
x, y = iminfo.rgbLifetime.shape[:2]
n_times = len(iminfo.acqTime)

group_rgblifetime = np.zeros((n_times, x, y, 3))
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

T = raw_data.shape[0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ax1, ax2, ax3 = axes

# remove axes
for ax in axes:
    ax.axis("off")

# initial frames
im1 = ax1.imshow(raw_data[0], cmap='turbo')
im2 = ax2.imshow(group_rgblifetime[0], cmap='turbo')
im3 = ax3.imshow(s2p_data[0], cmap='turbo')

# titles
ax1.set_title("Raw")
ax2.set_title("FLIMage MC")
ax3.set_title("Suite2p MC")

# colorbar shared
cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
cbar.set_label("Lifetime (ns)")

if lifetimeLimit[0] < lifetimeLimit[1]:
    # get current ticks
    ticks = cbar.get_ticks()
    # reverse only the labels
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks[::-1]])

def update(frame):
    im1.set_data(raw_data[frame])
    im2.set_data(group_rgblifetime[frame])
    im3.set_data(s2p_data[frame])
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
anim.save(os.path.join(root_dir,"rgb_lifetime_comparison_raw_flimage_suite2p.mp4"), writer=writer, dpi=200)

plt.close(fig)