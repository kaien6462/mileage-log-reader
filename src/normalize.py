"""
FIRST VERSION, kept for reference: the pipeline now uses read_form.normalize_cell, which centres
the bounding box like EMNIST (centre-of-mass centring here cost about 25 points on 7 and 9).

normalize.py - Turn one cell crop from a Form ML-7 photo into an EMNIST-style
28x28 image, so the classifier sees the same kind of input at inference as in
training.

EMNIST characters are: white ink on black, cropped to the character, scaled to
fit a fixed-size box (aspect ratio kept), and centred in a 28x28 frame.
Our cells are: dark ink on light paper, off-centre, with fragments of the
printed box lines around them. This module closes that gap.

    from normalize import normalize_cell
    img28 = normalize_cell(crop)        # None if the cell is empty

Steps:
  1. grayscale; estimate paper and ink brightness from the crop itself
     (works under uneven light and with any pen colour)
  2. convert to an "ink map": 0 = paper, 1 = darkest ink (white-on-black)
  3. remove printed box-line fragments near the crop border
  4. drop tiny specks (JPEG noise, dust); decide whether the cell is empty
  5. crop to the character, scale the longest side to TARGET_SIZE pixels,
     place in 28x28 and shift so the centre of mass is at the centre
"""
import numpy as np
import cv2
from scipy import ndimage

TARGET_SIZE = 24        # longest side of the character inside the 28x28 frame.
                        # EMNIST (unlike MNIST's 20x20 box) pads each character by
                        # 2 px and resizes to 28x28, so characters span ~24 px.
                        # tests/check_normalize.py measures this on the real data
                        # and reports a mismatch - inference MUST match training.
INK_THRESHOLD = 0.35    # ink-map value above which a pixel counts as ink
MIN_CONTRAST = 40       # paper-vs-darkest difference (0-255) below which a cell is blank
MIN_INK_FRACTION = 0.03 # share of the cell CENTRE that must be ink for the cell to be non-empty
BORDER_BAND = 0.18      # outer share of the crop where box-line fragments are searched for


def to_ink_map(cell):
    """Grayscale crop (any colour, any lighting) -> float map, 0 = paper, 1 = ink."""
    g = cell if cell.ndim == 2 else cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
    g = g.astype(np.float32)
    paper = np.percentile(g, 90)            # most of the cell is paper
    darkest = np.percentile(g, 1)
    if paper - darkest < MIN_CONTRAST:       # nothing darker than the paper -> blank
        return None
    return np.clip((paper - g) / (paper - darkest), 0, 1)


def remove_box_lines(ink):
    """
    Erase the printed box lines that frame the crop, without cutting the
    handwriting that touches or crosses them.
      - A box line is a row (or column) in the outer border band of the crop
        that is mostly ink along its whole length (projection test).
      - Each line pixel is erased, EXCEPT where ink continues on the inner side
        of the line: there a handwritten stroke crosses the line, so it is kept.
    Lines are only searched for near the border, so a character's own straight
    strokes in the middle of the cell (the bar of a 7 or a T) are never touched.
    """
    h, w = ink.shape
    mask = ink > INK_THRESHOLD
    out = ink.copy()
    bh, bw = max(2, int(h * BORDER_BAND)), max(2, int(w * BORDER_BAND))

    def runs(idx):
        """Group consecutive indices into (first, last) runs."""
        groups = []
        for i in idx:
            if groups and i == groups[-1][1] + 1:
                groups[-1][1] = i
            else:
                groups.append([i, i])
        return groups

    # vertical lines: columns in the left/right band that are mostly ink
    col_full = mask.mean(axis=0) > 0.5
    cols = [c for c in range(w) if col_full[c] and (c < bw or c >= w - bw)]
    for c0, c1 in runs(cols):
        c0, c1 = max(0, c0 - 2), min(w - 1, c1 + 2)          # blurred / anti-aliased edge
        step = 1 if c0 < w // 2 else -1                        # direction towards the box centre
        a, b = (c1 + 2, c1 + 4) if step == 1 else (c0 - 2, c0 - 4)
        a, b = min(max(a, 0), w - 1), min(max(b, 0), w - 1)
        crossing = mask[:, a] & mask[:, b]                     # ink continues inside -> real stroke
        out[~crossing, c0:c1 + 1] = 0

    # horizontal lines: rows in the top/bottom band that are mostly ink
    row_full = mask.mean(axis=1) > 0.5
    rows = [r for r in range(h) if row_full[r] and (r < bh or r >= h - bh)]
    for r0, r1 in runs(rows):
        r0, r1 = max(0, r0 - 2), min(h - 1, r1 + 2)
        step = 1 if r0 < h // 2 else -1
        a, b = (r1 + 2, r1 + 4) if step == 1 else (r0 - 2, r0 - 4)
        a, b = min(max(a, 0), h - 1), min(max(b, 0), h - 1)
        crossing = mask[a, :] & mask[b, :]
        out[r0:r1 + 1, ~crossing] = 0
    return out


def clean_components(ink):
    """Keep real strokes; drop specks and thin line leftovers. Returns None if the cell is empty."""
    h, w = ink.shape
    mask = (ink > INK_THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    min_area = max(4, int(0.004 * h * w))
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue                                        # speck / JPEG noise
        if min(bw, bh) <= 3 and (x == 0 or y == 0 or x + bw >= w or y + bh >= h):
            continue                                        # thin sliver left over from a box line
        keep[labels == i] = 1
    # Empty-cell test on the CENTRE of the cell only: leftover fringes of the
    # box lines sit at the border, while every handwritten character crosses the centre.
    ch, cw = h // 4, w // 4
    if keep[ch:h - ch, cw:w - cw].mean() < MIN_INK_FRACTION:
        return None
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))      # keep the soft edge of each stroke
    return ink * keep


def fit_to_28(ink, target=TARGET_SIZE):
    """Crop to the character, scale longest side to `target`, centre by mass in 28x28 (MNIST/EMNIST style)."""
    ys, xs = np.nonzero(ink > 0.1)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    s = target / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_AREA)

    out = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    cy, cx = ndimage.center_of_mass(out)
    out = ndimage.shift(out, (14 - cy, 14 - cx), order=1, mode="constant")
    out = np.clip(out, 0, None)
    return out / out.max() if out.max() > 0 else out


def normalize_cell(cell, target=TARGET_SIZE):
    """
    One cell crop (H x W x 3 RGB or H x W gray, uint8) -> 28x28 float32 in [0, 1],
    white character on black, EMNIST style. Returns None if the cell is empty.
    """
    ink = to_ink_map(cell)
    if ink is None:
        return None
    ink = clean_components(remove_box_lines(ink))
    if ink is None:
        return None
    return fit_to_28(ink, target).astype(np.float32)