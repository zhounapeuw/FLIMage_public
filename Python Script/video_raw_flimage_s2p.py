import tkinter as tk
from tkinter import filedialog
from FLIMageFileReader import FileReader
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

import matplotlib.animation as animation
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

data_type = 'lifetime'
lifetimeLimit = [1.4, 2] # first entry will be the upper bound (red) of the colorbar, 2nd is the lower bound (blue)
intensityLimit = [0, 15]# [3, 100] for pure intensity
make_movie = False
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


root_dir = r'C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\landon\42\slice5'
output_path = os.path.join(root_dir, 's2p_analysis')

# intensity
raw_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\landon\42\slice5\s2p_analysis\intensity_raw.npy"

# lifetime
raw_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\landon\42\slice5\s2p_analysis\rbglifetimemap_raw.npy"
s2p_rig_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\landon\42\slice5\s2p_analysis\rbglifetimemap_rig.npy"
s2p_rig_nonrig_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\landon\42\slice5\s2p_analysis\rbglifetimemap_rig_nonrig_half.npy"

# flimage path (contains both lifetime and intensity)
flimage_path = r"C:\Users\charl\OHSU Dropbox\Charles Zhou\CZ\2pFLIM\landon\42\slice5\42_morphine001_concat_aligned.flim"


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
avg_data_list = [np.nanmean(data, axis=0) for data in data_list]

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

if make_movie:
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



#~~~~~~~~~~~~~ quantification

import numpy as np
import cv2
from scipy.ndimage import center_of_mass

# ---------- helpers ----------

def to_grayscale(movie):
    """
    Convert [T, Y, X, C] RGB movie to grayscale [T, Y, X]
    """
    if movie.shape[-1] == 3:
        gray = np.dot(movie[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = movie.squeeze()
    return gray


def nanmean_image(movie_gray):
    """NaN-safe mean image"""
    return np.nanmean(movie_gray, axis=0)


def compute_sharpness_mean(movie_gray):
    """
    Compute sharpness of mean image using variance of Laplacian (NaN-safe)
    """
    mean_img = nanmean_image(movie_gray)

    # Replace remaining NaNs (if an entire pixel column was NaN)
    mean_img = np.nan_to_num(mean_img, nan=0.0)

    lap = cv2.Laplacian(mean_img.astype(np.float64), cv2.CV_64F)
    return lap.var()


def compute_com_motion(movie_gray):
    """
    Compute center-of-mass per frame and motion stats (NaN-safe)
    """
    coms = []

    for frame in movie_gray:
        # Mask valid pixels
        valid_mask = ~np.isnan(frame)

        if not np.any(valid_mask):
            # If frame is all NaN, append NaNs
            coms.append([np.nan, np.nan])
            continue

        # Replace NaNs with 0 ONLY for CoM computation
        frame_clean = np.where(valid_mask, frame, 0)

        # Shift to positive to avoid weird CoM behavior
        min_val = np.min(frame_clean[valid_mask])
        frame_clean = frame_clean - min_val

        # If all values become zero → undefined CoM
        if np.all(frame_clean == 0):
            coms.append([np.nan, np.nan])
            continue

        coms.append(center_of_mass(frame_clean))

    coms = np.array(coms)

    # Remove NaN frames for displacement calc
    valid_idx = ~np.isnan(coms).any(axis=1)
    coms_valid = coms[valid_idx]

    if len(coms_valid) < 2:
        displacements = np.array([np.nan])
    else:
        displacements = np.sqrt(
            np.sum(np.diff(coms_valid, axis=0)**2, axis=1)
        )

    results = {
        "com_positions": coms,  # includes NaNs where undefined
        "mean_displacement": np.nanmean(displacements),
        "std_displacement": np.nanstd(displacements),
        "max_displacement": np.nanmax(displacements),
        "total_path_length": np.nansum(displacements)
    }

    return results


# ---------- main ----------

def analyze_motion_quality(data_list, titles):
    results = {}

    for data, title in zip(data_list, titles):
        movie_gray = to_grayscale(data)

        sharpness = compute_sharpness_mean(movie_gray)
        com_results = compute_com_motion(movie_gray)

        results[title] = {
            "sharpness_mean_image": sharpness,
            **com_results
        }

    return results


# ---------- run ----------

results = analyze_motion_quality(data_list, titles)

def save_results_npz(results, filename="motion_metrics.npz"):
    save_dict = {}

    for key, res in results.items():
        prefix = key.replace(" ", "_")

        for subkey, val in res.items():
            save_dict[f"{prefix}__{subkey}"] = val

    np.savez_compressed(filename, **save_dict)


# save
save_results_npz(results, filename=os.path.join(output_path, "motion_metrics.npz"))