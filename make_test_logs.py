"""
make_test_logs.py -- build labelled test logs (image + ground-truth JSON) at three quality levels,
using the instructor's make_log_samples.py. Put both files in the same folder.

Run:
  python make_test_logs.py --emnist emnist36.npz --n 10 --out test_logs

Makes   clean_00.png / .json ...   phone_00.jpg / .json ...   field_00.jpg / .json ...

Two things this script fixes for you:
  * make_log_samples.py transposes EMNIST on load, but emnist36.npz is already upright,
    so we pre-transpose a temporary copy (otherwise every character would be sideways).
  * make_log_samples.py hard-codes Linux font paths; we point it at a font that exists on your computer.
Uses only the EMNIST TEST split, so no test character was seen in training.
"""
import argparse
import json
import os
import tempfile

import numpy as np

import make_log_samples as g

FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LEVELS = {"clean": dict(messy=0.1, degrade=None),
          "phone": dict(messy=0.4, degrade=0.5),
          "field": dict(messy=0.75, degrade=1.0)}


def pick_font():
    for regular, bold in FONT_CANDIDATES:
        if os.path.exists(regular) and os.path.exists(bold):
            g.FONT, g.FONTB = regular, bold
            return regular
    raise SystemExit("No suitable font found. Edit FONT_CANDIDATES in make_test_logs.py with a .ttf path on your computer.")


def random_log(rng):
    n_rows = int(rng.integers(5, 10))
    odo, day, rows = int(rng.integers(1000, 990000)), 21, []
    for _ in range(n_rows):
        odo += int(rng.integers(0, 15))
        start, miles = odo, int(rng.integers(5, 95))
        odo = start + miles
        client = "".join(rng.choice(list(LETTERS), 3))
        rows.append((f"09{day:02d}", client, f"{start:06d}", f"{odo:06d}", f"{miles:03d}"))
        if rng.random() < 0.4:
            day = min(day + 1, 27)
    emp = "".join(rng.choice(list(LETTERS), 2)) + f"{int(rng.integers(0, 10000)):04d}"
    return rows, emp, "092726"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emnist", default="emnist36.npz")
    ap.add_argument("--n", type=int, default=10, help="logs per quality level")
    ap.add_argument("--out", default="test_logs")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    print("font:", pick_font())
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.emnist)
    tmp = os.path.join(tempfile.mkdtemp(), "emnist_for_generator.npz")
    np.savez(tmp, x_test=np.transpose(z["x_test"], (0, 2, 1)), y_test=z["y_test"])   # cancel the generator's transpose

    rng = np.random.default_rng(a.seed)
    bank = g.GlyphBank(tmp, seed=a.seed)
    for level, cfg in LEVELS.items():
        for i in range(a.n):
            rows, emp, week = random_log(rng)
            form, ink, truth = g.fill_log(bank, rows, emp, week, rng, messy=cfg["messy"])
            img = g.compose(form, ink, (20, 20, 20) if cfg["degrade"] else (25, 35, 90))
            if cfg["degrade"]:
                img = g.degrade(img, rng, cfg["degrade"])
                name = f"{level}_{i:02d}.jpg"
                img.save(os.path.join(a.out, name), quality=90)
            else:
                name = f"{level}_{i:02d}.png"
                img.save(os.path.join(a.out, name))
            json.dump(truth, open(os.path.join(a.out, f"{level}_{i:02d}.json"), "w"))
        print(f"made {a.n} {level} logs")
    print("done ->", a.out)
