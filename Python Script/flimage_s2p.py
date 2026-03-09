import tkinter as tk
from tkinter import filedialog
import FLIMageFileReader
from FLIMageFileReader import FileReader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

lifetimeLimit = [1, 2]
intensityLimit = [0, 50]
z_plane = 4
lifetime_offset = 1

file_path = filedialog.askopenfilename()

p = Path(file_path)
root_dir = p.parent

# remove trailing digits before .flim
prefix = re.sub(r'\d+\.flim$', '', p.name)
flim_files = sorted(root_dir.glob(prefix + "*.flim"))
n_files = len(flim_files)

# load first file to get num planes and x/y dims
iminfo = FileReader()
iminfo.read_imageFile(str(flim_files[0]), True)
num_z = len(iminfo.acqTime)
iminfo.calculatePage(0, 0, 0, [2, iminfo.n_time[0]], intensityLimit, lifetimeLimit, lifetime_offset)
x, y = iminfo.rgbLifetime.shape[:2]

group_lifetime = np.zeros((num_z, x, y, n_files))

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

        lifetime_norm = (iminfo.lifetimeMap - lifetimeLimit[0]) / (lifetimeLimit[1] - lifetimeLimit[0])
        lifetime_norm = 1 - lifetime_norm
        lifetime_norm = np.clip(lifetime_norm, 0, 1)

        intensity_norm = (iminfo.intensity - intensityLimit[0]) / (intensityLimit[1] - intensityLimit[0])
        intensity_norm = np.clip(intensity_norm, 0, 1)

        grayImage = lifetime_norm * intensity_norm


        group_lifetime[z_plane, :, :, f] = grayImage

# iminfo = FileReader()
# iminfo.read_imageFile(file_path, True)
# num_z_slices = len(iminfo.acqTime) # might be a better way to get this

# iminfo.calculatePage(z_plane, 0, 0, [2, iminfo.n_time[0]], intensityLimit, lifetimeLimit, lifetime_offset = 1)
#  page = 0, fastZpage = 0, channel = 0,  lifetimeRange = [0, 64], intensityLimit = [0, 20], lifetimeLimit = [1.6, 2.0], lifetimeOffset = 0.5)




plt.figure
img = plt.imshow(grayImage) # lifetimeMap rgbLifetime
#img.set_clim(4,10)
plt.colorbar(img)
plt.show()
