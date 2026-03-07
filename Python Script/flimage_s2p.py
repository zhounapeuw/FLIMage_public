import tkinter as tk
from tkinter import filedialog
import FLIMageFileReader


plotWindow = tk.Tk()
plotWindow.wm_title('Fluorescence lifetime')                
plotWindow.withdraw()

file_path = filedialog.askopenfilename()
iminfo = FLIMageFileReader.FileReader()
iminfo.read_imageFile(file_path, True)
iminfo.calculatePage(0, 0, 0, [0, iminfo.n_time[0]], [0, 10], [1.6, 3], 1.5)

print('finished reading in "iminfo class"')
