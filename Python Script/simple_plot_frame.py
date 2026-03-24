import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from FLIMageFileReader import FileReader
from pathlib import Path
import re

def plot_rgb_with_colorbar(rgbimage, lifetime_limits):
    fig, (ax_img, ax_cbar) = plt.subplots(
        1, 2, figsize=(6, 4),
        gridspec_kw={'width_ratios': [20, 1]}
    )

    # --- Main image (equivalent to image(rgbimage)) ---
    ax_img.imshow(rgbimage)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # --- Colormap (jet(64) → subset 9:56) ---
    cmap = plt.cm.jet(np.linspace(0, 1, 64))[:, :3]
    cmap2 = cmap[8:56, :]  # MATLAB 9:56 → Python 8:56

    # --- Colorbar as image (FV_colorbar equivalent) ---
    cmap_img = cmap2.reshape((cmap2.shape[0], 1, 3))

    ax_cbar.imshow(cmap_img, aspect='auto')

    ax_cbar.set_xticks([])
    ax_cbar.set_ylim(0.5, cmap2.shape[0] + 0.5)
    ax_cbar.set_yticks([0, cmap2.shape[0] - 1])

    # MATLAB: label(end:-1:1)
    labels = lifetime_limits[::-1]
    ax_cbar.set_yticklabels(labels)

    ax_cbar.yaxis.tick_right()

    plt.tight_layout()
    return fig, ax_img, ax_cbar


# --- Example usage ---
lifetimeLimit = [1.4, 2]
intensityLimit = [5, 20]
z_plane = 4
lifetime_offset = 1.1

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
iminfo.calculatePage(z_plane, 0, 0, [2, iminfo.n_time[0]], intensityLimit, lifetimeLimit, lifetime_offset)


iminfo.calculateRGBLifetimeMap(lifetimeLimit = lifetimeLimit, intensityLimit = intensityLimit)

plot_rgb_with_colorbar(iminfo.rgbLifetime, lifetimeLimit)
plt.show()
adsdas