"""
dataset.py - Training data for the character classifier(s), ready for PyTorch.

Reads data/emnist36.npz (made by src/emnist_loader.py) and serves it per
character GROUP, because Form ML-7 tells us which boxes hold digits and which
hold letters:

    group="digits"   -> labels 0-9    (characters '0'..'9')
    group="letters"  -> labels 0-25   (characters 'A'..'Z')
    group="all"      -> labels 0-35   (one model for everything)

Images come out exactly like normalize_cell() output at inference time:
float32 in [0, 1], white ink on black, shape (1, 28, 28). The only change made
to training images is division by 255 - no extra resizing or centring - so
training and inference see the same kind of input.

Usage (in the training script):

    from dataset import make_loaders, decode
    train_dl, val_dl, info = make_loaders("letters", batch_size=256, balanced=True)
    for x, y in train_dl: ...                 # x: (B, 1, 28, 28) float, y: (B,) long
    decode("letters", 3)                      # -> 'D'

Run this file to check it:  python src/dataset.py
"""
import numpy as np

NPZ_PATH = "data/emnist36.npz"
EMNIST_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GROUPS = {
    "digits": (0, 10),      # (first label, last label + 1) in the 36-class scheme
    "letters": (10, 36),
    "all": (0, 36),
}


# --------------------------------------------------------------------------- numpy core
def load_group(group, split, npz_path=NPZ_PATH):
    """
    Return (x, y) for one character group and one split ('train' or 'val').
    x: float32 (N, 1, 28, 28) in [0, 1]; y: int64 labels starting at 0 for the group.
    The 'test' split is refused on purpose: it is reserved for the synthetic test logs.
    """
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val' - the test split is reserved for the test logs")
    lo, hi = GROUPS[group]
    data = np.load(npz_path)
    x, y = data[f"x_{split}"], data[f"y_{split}"]
    keep = (y >= lo) & (y < hi)
    x = x[keep].astype(np.float32)[:, None, :, :] / 255.0
    y = (y[keep] - lo).astype(np.int64)
    return x, y


def decode(group, label):
    """Group label -> the character it stands for."""
    return EMNIST_MAP[GROUPS[group][0] + int(label)]


def class_counts(y, n_classes):
    return np.bincount(y, minlength=n_classes)


def sample_weights(y, n_classes):
    """
    Per-image weights so every class is drawn equally often (weight = 1 / class size).
    Used for the letters, where K, Q, X, Z have ~10x fewer images than O or S
    but appear just as often in client codes.
    """
    counts = class_counts(y, n_classes)
    return (1.0 / counts)[y]


def class_weights(y, n_classes):
    """Alternative to the sampler: loss weights (inverse frequency, mean 1) for nn.CrossEntropyLoss."""
    counts = class_counts(y, n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()


# --------------------------------------------------------------------------- torch wrapper
def make_loaders(group, batch_size=256, balanced=False, augment=False, npz_path=NPZ_PATH, num_workers=0):
    """
    Build PyTorch DataLoaders for one group.
      balanced=True -> WeightedRandomSampler, each class drawn equally often (training only)
      augment=True  -> small random rotation / shift / scale on training images,
                       similar to the jitter of handwriting placed in a box
    Returns (train_loader, val_loader, info) where info holds counts and the class weights.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

    x_tr, y_tr = load_group(group, "train", npz_path)
    x_va, y_va = load_group(group, "val", npz_path)
    n_classes = GROUPS[group][1] - GROUPS[group][0]

    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va))

    if balanced:
        w = torch.as_tensor(sample_weights(y_tr, n_classes), dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(y_tr), replacement=True)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

    if augment:
        train_dl = _AugmentedLoader(train_dl)

    info = {
        "group": group,
        "n_classes": n_classes,
        "classes": [decode(group, i) for i in range(n_classes)],
        "train_counts": class_counts(y_tr, n_classes).tolist(),
        "val_counts": class_counts(y_va, n_classes).tolist(),
        "class_weights": class_weights(y_tr, n_classes).tolist(),
    }
    return train_dl, val_dl, info


class _AugmentedLoader:
    """Wraps a DataLoader and applies a random affine jitter to every training batch."""

    def __init__(self, loader, max_deg=10, max_shift=0.08, scale=(0.9, 1.1)):
        self.loader, self.max_deg, self.max_shift, self.scale = loader, max_deg, max_shift, scale

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        import math
        import torch
        import torch.nn.functional as F
        for x, y in self.loader:
            b = x.shape[0]
            ang = (torch.rand(b) * 2 - 1) * math.radians(self.max_deg)
            s = torch.empty(b).uniform_(*self.scale)
            t = (torch.rand(b, 2) * 2 - 1) * self.max_shift * 2
            cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
            theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1),
                                 torch.stack([sin, cos, t[:, 1]], 1)], 1)
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            yield F.grid_sample(x, grid, align_corners=False, padding_mode="zeros"), y


# --------------------------------------------------------------------------- self-check
def main():
    for group in ("digits", "letters"):
        x, y = load_group(group, "train")
        n = GROUPS[group][1] - GROUPS[group][0]
        counts = class_counts(y, n)
        assert x.shape[1:] == (1, 28, 28) and x.dtype == np.float32
        assert 0.0 <= x.min() and x.max() <= 1.0, "pixels must be in [0, 1], like normalize_cell()"
        assert y.min() == 0 and y.max() == n - 1, "labels must start at 0 for each group"
        print(f"{group:8s} train {len(y):>7,}  classes {n}  "
              f"smallest '{decode(group, counts.argmin())}' {counts.min():,}  "
              f"largest '{decode(group, counts.argmax())}' {counts.max():,}")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not installed - numpy part OK, skipping the DataLoader check")
        return

    train_dl, val_dl, info = make_loaders("letters", batch_size=512, balanced=True, augment=True)
    xb, yb = next(iter(train_dl))
    print(f"\nletters batch: x {tuple(xb.shape)} {xb.dtype}, y {tuple(yb.shape)} {yb.dtype}")
    drawn = np.bincount(yb.numpy(), minlength=info["n_classes"])
    print(f"balanced sampler, one batch of 512: min {drawn.min()} / max {drawn.max()} per letter "
          f"(about {512 // 26} each expected)")
    print("DataLoader check passed")


if __name__ == "__main__":
    main()