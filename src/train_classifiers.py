"""
train_classifiers.py -- SCRUM-14: train a DIGIT model (0-9) and a LETTER model (A-Z).

Run (from the repo folder):  python src/train_classifiers.py --emnist data/emnist36.npz
Or auto-download:            python src/train_classifiers.py
Saves models/digit_model.pt and models/letter_model.pt.
Needs: pip install torch torchvision numpy matplotlib

emnist36.npz holds x_train/y_train, x_val/y_val, x_test/y_test. Labels: 0-9 digits,
10-35 A-Z (upper/lower case already merged). Its images are ALREADY upright, so no
transpose is applied. The auto-download (torchvision) images ARE sideways, so those
are transposed.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMNIST_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"  # first 36 = 0-9, A-Z


# ---------------------------------------------------------------- 1. load data
def load(path=None):
    """Return dict split -> (x, y). Uses the .npz if given, else downloads EMNIST ByClass."""
    out = {}
    if path:
        z = np.load(path)
        splits = [sp for sp in ("train", "val", "test") if f"x_{sp}" in z.files]
        for sp in splits:
            x = z[f"x_{sp}"].reshape(-1, 28, 28)                              # uint8, already upright
            out[sp] = (x, z[f"y_{sp}"].astype(np.int64))
    else:
        from torchvision import datasets
        for sp in ("train", "test"):
            d = datasets.EMNIST("emnist_data", split="byclass", train=(sp == "train"), download=True)
            x = d.data.numpy().reshape(-1, 28, 28)
            x = np.ascontiguousarray(np.transpose(x, (0, 2, 1)))                                   # torchvision is sideways
            out[sp] = (x, d.targets.numpy().astype(np.int64))
    return out


def save_preview(x, y, name="preview.png"):
    """Look at this picture! Characters must be UPRIGHT. If sideways, the fix above is wrong."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 8, figsize=(10, 3))
    for a, img, lab in zip(axs.ravel(), x[:16], y[:16]):
        a.imshow(img, cmap="gray"); a.set_title(EMNIST_MAP[lab]); a.axis("off")
    fig.savefig(name); plt.close(fig)


# ---------------------------------------------------------------- 2. pick classes
def make_digit_set(x, y):
    keep = y < 10
    return x[keep], y[keep]                       # labels stay 0-9


def make_letter_set(x, y):
    keep = y >= 10
    x, y = x[keep], y[keep]
    return x, (y - 10) % 26                       # a and A both become class 0, etc.


# ---------------------------------------------------------------- 3. the model
class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   # 14 -> 7
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


# ---------------------------------------------------------------- 4. train + test
def accuracy(model, x, y, dev):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            xb = torch.from_numpy(x[i:i + 1024]).float().div(255).unsqueeze(1).to(dev)
            pred = model(xb).argmax(1).cpu().numpy()
            correct += (pred == y[i:i + 1024]).sum()
    return correct / len(x)


def train(name, train_xy, val_xy, test_xy, n_classes, epochs=6, lr=1e-3, seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    if val_xy is not None:                        # emnist36.npz already has a validation split
        xtr, ytr = train_xy
        xval, yval = val_xy
    else:                                         # otherwise keep 10% of TRAIN for validation
        x, y = train_xy
        idx = rng.permutation(len(x))
        n_val = len(x) // 10
        val, tr = idx[:n_val], idx[n_val:]
        xtr, ytr, xval, yval = x[tr], y[tr], x[val], y[val]
    print(f"[{name}] train={len(xtr)} val={len(xval)} test={len(test_xy[0])}")
    print(f"[{name}] train class counts:", np.bincount(ytr, minlength=n_classes).tolist())

    model = SmallCNN(n_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(xtr))
        for i in range(0, len(order), 128):
            b = order[i:i + 128]
            xb = torch.from_numpy(xtr[b]).float().div(255).unsqueeze(1).to(dev)
            yb = torch.from_numpy(ytr[b]).to(dev)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        print(f"[{name}] epoch {ep + 1}/{epochs}  val acc = {accuracy(model, xval, yval, dev):.4f}")

    print(f"[{name}] TEST accuracy = {accuracy(model, *test_xy, dev):.4f}")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", f"{name}_model.pt"))
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emnist", default=None, help="optional .npz file; omit to auto-download")
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    data = load(args.emnist)
    save_preview(*data["train"])                  # open preview.png and check it
    print("Saved preview.png -- characters must look upright.")

    val = data.get("val")
    mk = lambda f, sp: f(*data[sp])
    train("digit", mk(make_digit_set, "train"), mk(make_digit_set, "val") if val else None,
          mk(make_digit_set, "test"), 10, args.epochs)
    train("letter", mk(make_letter_set, "train"), mk(make_letter_set, "val") if val else None,
          mk(make_letter_set, "test"), 26, args.epochs)
