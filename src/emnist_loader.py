"""
emnist_loader.py - Load EMNIST ByClass, fix the orientation, keep only the
36 classes used on Form ML-7 (digits 0-9 and capitals A-Z), split train/val,
and save everything to one .npz file.

Run:  python src/emnist_loader.py
"""
import numpy as np
from torchvision.datasets import EMNIST

EMNIST_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
N_CLASSES = 36            # labels 0-35 = digits + uppercase; 36-61 (lowercase) are dropped
DATA_DIR = "data"
OUT_PATH = "data/emnist36.npz"
VAL_FRACTION = 0.10       # 10% of the official train split becomes validation
SEED = 42                 # fixed seed so every teammate gets the same split


def load_raw(train):
    """Load one official EMNIST ByClass split as numpy arrays (uint8 images, int labels)."""
    ds = EMNIST(root=DATA_DIR, split="byclass", train=train, download=True)
    return ds.data.numpy(), ds.targets.numpy()


def fix_orientation(x):
    """EMNIST is stored transposed relative to MNIST; swap the row and column axes."""
    return np.ascontiguousarray(np.transpose(x, (0, 2, 1)))


def keep_form_classes(x, y):
    """Keep only digits and uppercase letters - the only characters allowed on Form ML-7."""
    mask = y < N_CLASSES
    return x[mask], y[mask]


def check_orientation(x, y):
    """
    Automated guard against the transpose bug.
    The average upright '1' is a vertical stroke, so its ink spreads much more
    vertically than horizontally. If the images were still transposed, the
    stroke would be horizontal and this assertion would fail.
    """
    mean_one = x[y == EMNIST_MAP.index("1")].astype(np.float32).mean(axis=0)
    w = mean_one / mean_one.sum()                      # ink weights
    rows, cols = np.mgrid[0:28, 0:28]
    spread_v = np.sqrt((w * (rows - (w * rows).sum()) ** 2).sum())
    spread_h = np.sqrt((w * (cols - (w * cols).sum()) ** 2).sum())
    assert spread_v > 1.5 * spread_h, (
        f"Orientation check failed: vertical spread {spread_v:.2f}, "
        f"horizontal spread {spread_h:.2f}. Images may still be transposed."
    )
    print(f"Orientation check passed (vertical {spread_v:.2f} vs horizontal {spread_h:.2f})")


def train_val_split(x, y, frac, seed):
    """Random split of the official train set into train and validation parts."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def print_class_counts(y, name):
    """Show how many images each class has - this is the class-imbalance evidence."""
    counts = np.bincount(y, minlength=N_CLASSES)
    print(f"\n{name}: {len(y):,} images")
    for start in range(0, N_CLASSES, 6):
        print("   ".join(f"{EMNIST_MAP[i]}:{counts[i]:>6,}" for i in range(start, start + 6)))
    digits, letters = counts[:10].sum(), counts[10:].sum()
    print(f"digits {digits:,} vs letters {letters:,} | "
          f"largest class / smallest class = {counts.max() / counts.min():.1f}x")


def main():
    # 1. load both official splits
    x_tr, y_tr = load_raw(train=True)
    x_te, y_te = load_raw(train=False)

    # 2. fix orientation, 3. keep the 36 form classes
    x_tr, y_tr = keep_form_classes(fix_orientation(x_tr), y_tr)
    x_te, y_te = keep_form_classes(fix_orientation(x_te), y_te)

    # 4. make sure the orientation is right before anything is saved
    check_orientation(x_tr, y_tr)

    # 5. train / validation split (the official test split stays untouched;
    #    it is reserved for the synthetic mileage logs)
    x_tr, y_tr, x_va, y_va = train_val_split(x_tr, y_tr, VAL_FRACTION, SEED)

    # 6. report class balance
    print_class_counts(y_tr, "TRAIN")
    print_class_counts(y_va, "VALIDATION")
    print_class_counts(y_te, "TEST (reserved for synthetic logs)")

    np.savez(OUT_PATH, x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va,
             x_test=x_te, y_test=y_te)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()