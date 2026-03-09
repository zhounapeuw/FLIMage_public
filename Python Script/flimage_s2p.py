import tkinter as tk
from tkinter import filedialog
import FLIMageFileReader
from FLIMageFileReader import FileReader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import suite2p

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


        group_lifetime[z_plane, f, :, :] = grayImage



plt.figure
img = plt.imshow(grayImage) # lifetimeMap rgbLifetime
#img.set_clim(4,10)
plt.colorbar(img)
plt.show()

##########

# s2p expects data to have dimensions: n_time, Ly, Lx

from suite2p import registration
from suite2p.io import BinaryFile

z_plane = 4
data = np.squeeze(group_lifetime[z_plane, f, :, :])

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
print(settings)
     

# Convert our example tif file into a binary file
if data.dtype == np.uint16:
    data = (data // 2).astype(np.int16)
# Write to binary
data.tofile('raw_data.bin')
f_raw = BinaryFile(Ly=Ly, Lx=Lx, filename='raw_data.bin')
print(f_raw.shape)

# Create a binary file we will write our registered image to
f_reg = suite2p.io.BinaryFile(
    Ly=Ly, Lx=Lx, filename='registered_data.bin',
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

     

     


