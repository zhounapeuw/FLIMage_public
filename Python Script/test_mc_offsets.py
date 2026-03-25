import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# Motion correction function (yours)
# -----------------------------
def apply_motion_correction(frame, dy, dx):
    corrected = np.roll(np.roll(frame, -dy, axis=0), -dx, axis=1)

    # NaN out wrapped edges
    corrected = corrected.astype(float)

    if dy > 0:
        corrected[-dy:, :] = np.nan
    elif dy < 0:
        corrected[:-dy, :] = np.nan

    if dx > 0:
        corrected[:, -dx:] = np.nan
    elif dx < 0:
        corrected[:, :-dx] = np.nan

    return corrected

# -----------------------------
# Synthetic "cell" generator
# -----------------------------
def make_gaussian(size=100, center=(50, 50), sigma=10):
    y = np.arange(size)
    x = np.arange(size)
    X, Y = np.meshgrid(x, y)

    cx, cy = center
    img = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    return img

# -----------------------------
# Create base + shifted image
# -----------------------------
size = 100
base = make_gaussian(size=size)

# TRUE shift (simulated motion)
true_dy = 7
true_dx = -5

shifted = np.roll(np.roll(base, true_dy, axis=0), true_dx, axis=1)

# -----------------------------
# Tkinter window
# -----------------------------
root = tk.Tk()
root.title("dx/dy Motion Correction Tester")

# -----------------------------
# Matplotlib Figure
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
plt.subplots_adjust(bottom=0.25)

ax_orig, ax_shifted, ax_corrected = axes

im0 = ax_orig.imshow(base, cmap='hot')
ax_orig.set_title("Original")

im1 = ax_shifted.imshow(shifted, cmap='hot')
ax_shifted.set_title(f"Shifted (true dy={true_dy}, dx={true_dx})")

corrected_init = apply_motion_correction(shifted, 0, 0)
im2 = ax_corrected.imshow(corrected_init, cmap='hot')
ax_corrected.set_title("Corrected")

for ax in axes:
    ax.axis('off')

# -----------------------------
# Sliders
# -----------------------------
ax_dy = plt.axes([0.2, 0.1, 0.6, 0.03])
ax_dx = plt.axes([0.2, 0.05, 0.6, 0.03])

slider_dy = Slider(ax_dy, 'dy', -20, 20, valinit=0, valstep=1)
slider_dx = Slider(ax_dx, 'dx', -20, 20, valinit=0, valstep=1)

# -----------------------------
# Update function
# -----------------------------
def update(val):
    dy = int(slider_dy.val)
    dx = int(slider_dx.val)

    corrected = apply_motion_correction(shifted, dy, dx)

    im2.set_data(corrected)
    ax_corrected.set_title(f"Corrected (dy={dy}, dx={dx})")

    fig.canvas.draw_idle()

slider_dy.on_changed(update)
slider_dx.on_changed(update)

# -----------------------------
# Embed in Tkinter
# -----------------------------
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# -----------------------------
# Run app
# -----------------------------
root.mainloop()