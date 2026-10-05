"""
check_normalize.py - Verify that src/normalize.py produces EMNIST-style images.

What it checks:
  1. Size match: the longest side of characters in EMNIST vs in our normalized
     cells. If they differ, TARGET_SIZE in normalize.py should be changed.
  2. Round trip: put known EMNIST test characters into Form ML-7 boxes, render a
     clean scan and a phone photo, cut the boxes out again, normalize them, and
     compare with the original character (correlation, 1.0 = identical).
  3. Empty cells: boxes in unused rows must come back as None.
Also saves data/normalize_check.png (original | cell crop | normalized).

Run:  python tests/check_normalize.py
"""
import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from make_logs import GlyphBank, QUALITY, random_trip_data, pick_hazards, render_log, degrade  # noqa: E402
from normalize import normalize_cell, fit_to_28, TARGET_SIZE  # noqa: E402


class RecordingBank(GlyphBank):
    """Same as GlyphBank, but remembers every glyph it hands out, in order."""

    def __init__(self):
        super().__init__()
        self.log = []

    def get(self, ch, rng):
        g = super().get(ch, rng)
        self.log.append((ch, g))
        return g


def longest_side(img, thr=0.1):
    ys, xs = np.nonzero(img > thr)
    return max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) if len(ys) else 0


def crop_box(img, box, warp, pad=4):
    """Cut a box (flat-form coordinates) out of the page, following the photo's perspective warp."""
    x0, y0, x1, y1 = box
    corners = np.float32([[x0 - pad, y0 - pad], [x1 + pad, y0 - pad], [x1 + pad, y1 + pad], [x0 - pad, y1 + pad]])
    pts = cv2.perspectiveTransform(corners[None], np.asarray(warp, np.float64))[0]
    # straighten the (slightly skewed) box back to a square crop
    side = int(x1 - x0 + 2 * pad)
    dst = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (side, side), borderValue=(255, 255, 255))


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run(quality, n_logs, rng_seed, examples):
    preset = QUALITY[quality]
    data_rng, draw_rng = np.random.default_rng(rng_seed), np.random.default_rng(rng_seed + 1)
    bank = RecordingBank()
    scores, sizes, missed, hazard_skipped = [], [], 0, 0
    empty_total, empty_wrong = 0, 0

    for _ in range(n_logs):
        bank.log.clear()
        data = random_trip_data(data_rng)
        hazards = pick_hazards(data, draw_rng, preset["hazard_rate"])
        page, fields = render_log(data, bank, draw_rng, preset["messy"], preset["ink"], hazards)
        page, warp = degrade(page, draw_rng, preset["strength"])
        page = np.asarray(page)

        glyphs = iter(bank.log)
        for name, f in fields.items():
            for box in f["boxes"]:
                ch, original = next(glyphs)
                if name in hazards:                 # segmentation hazard: not a normalization test
                    hazard_skipped += 1
                    continue
                cell = crop_box(page, box, warp)
                out = normalize_cell(cell)
                if out is None:
                    missed += 1
                    continue
                scores.append(corr(out, fit_to_28(original)))
                sizes.append(longest_side(out))
                if sum(e[0] == quality for e in examples) < 6 and draw_rng.random() < 0.05:
                    examples.append((quality, ch, original, cell, out))

        # unused rows must be detected as empty
        for r in range(len(data["rows"]), 9):
            box = [170, 420 + r * 86, 232, 482 + r * 86]       # first DATE box of row r
            empty_total += 1
            empty_wrong += normalize_cell(crop_box(page, box, warp)) is not None

    total = len(scores) + missed
    print(f"\n[{quality}]  {total} written cells checked ({hazard_skipped} skipped: segmentation hazards)")
    print(f"  mean correlation with original EMNIST glyph : {np.mean(scores):.3f}  (1.0 = identical)")
    print(f"  cells with correlation < 0.6                : {np.mean(np.array(scores) < 0.6):.1%}")
    print(f"  written cells wrongly called empty          : {missed / total:.1%}")
    print(f"  empty cells wrongly called written          : {empty_wrong}/{empty_total}")
    print(f"  median longest side of normalized character : {np.median(sizes):.0f} px")


def main():
    x = np.load("data/emnist36.npz")["x_test"][:3000].astype(np.float32) / 255
    ref = np.median([longest_side(im) for im in x])
    print(f"EMNIST reference: median longest side of a character = {ref:.0f} px (TARGET_SIZE = {TARGET_SIZE})")
    if abs(ref - TARGET_SIZE) > 2:
        print(f"  -> MISMATCH: set TARGET_SIZE = {ref:.0f} in src/normalize.py so inference matches training")

    examples = []
    for quality in ["clean", "phone"]:
        run(quality, n_logs=10, rng_seed=7, examples=examples)

    fig, axes = plt.subplots(3, len(examples), figsize=(1.4 * len(examples), 4.4))
    for j, (q, ch, orig, cell, out) in enumerate(examples):
        for i, (img, cmap) in enumerate([(orig, "gray"), (cell, None), (out, "gray")]):
            axes[i, j].imshow(img, cmap=cmap)
            axes[i, j].axis("off")
        axes[0, j].set_title(f"{q}\n'{ch}'", fontsize=8)
    axes[0, 0].text(-12, 14, "EMNIST", ha="right", fontsize=8)
    axes[1, 0].text(-12, 30, "cell crop", ha="right", fontsize=8)
    axes[2, 0].text(-12, 14, "normalized", ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig("data/normalize_check.png", dpi=130)
    print("\nSaved data/normalize_check.png")


if __name__ == "__main__":
    main()