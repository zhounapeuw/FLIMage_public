import tkinter as tk
from tkinter import filedialog
import FLIMageFileReader
from FLIMageFileReader import FileReader
import matplotlib.pyplot as plt
import numpy as np

file_path = filedialog.askopenfilename()
iminfo = FileReader()
iminfo.read_imageFile(file_path, True)
num_z_slices = len(iminfo.acqTime) # might be a better way to get this

iminfo.calculatePage(4, 0, 0, [2, iminfo.n_time[0]], [0, 50], [1, 2], 1)

#  page = 0, fastZpage = 0, channel = 0,  lifetimeRange = [0, 64], intensityLimit = [0, 20], lifetimeLimit = [1.6, 2.0], lifetimeOffset = 0.5)

plt.figure
img = plt.imshow(iminfo.rgbLifetime) # lifetimeMap rgbLifetime
print(iminfo.rgbLifetime)
#img.set_clim(4,10)
plt.colorbar(img)
plt.show()
