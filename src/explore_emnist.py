"""
explore_emnist.py - First look at EMNIST: download the data, check shapes,
and visualize the transpose issue.
"""
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import EMNIST

# EMNIST ByClass label map: 0-9 = digits, 10-35 = uppercase A-Z, 36-61 = lowercase a-z
EMNIST_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Download the dataset (~500 MB on first run, may take a few minutes)
test = EMNIST(root="data", split="byclass", train=False, download=True)
x = test.data.numpy()      # shape (N, 28, 28), pixel values 0-255
y = test.targets.numpy()   # shape (N,), labels 0-61

print("Image array shape:", x.shape)
print("Label array shape:", y.shape)
print("Pixel range:", x.min(), "to", x.max())

# Pick a few characters whose orientation is easy to judge
chars = ["1", "2", "7", "A", "E", "F", "L", "R"]

fig, axes = plt.subplots(2, len(chars), figsize=(12, 3.5))
for i, ch in enumerate(chars):
    label = EMNIST_MAP.index(ch)          # character -> label index
    img = x[np.where(y == label)[0][0]]   # first image of this class

    axes[0, i].imshow(img, cmap="gray")       # top row: raw data as stored
    axes[0, i].set_title(f"raw '{ch}'")
    axes[1, i].imshow(img.T, cmap="gray")     # bottom row: transposed
    axes[1, i].set_title(f"fixed '{ch}'")
    axes[0, i].axis("off")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("data/emnist_transpose_check.png", dpi=120)
print("Saved figure: data/emnist_transpose_check.png")