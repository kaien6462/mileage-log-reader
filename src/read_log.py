"""
read_log.py - End-to-end reader for one Form ML-7 image (SCRUM-22).

    photo -> 1 register -> 2 extract cells -> 3 normalize -> 4 classify
          -> 5 assemble fields and rows -> (6 validation, SCRUM-16: validate.py)

Stages 1-4 are Tenaj's read_form.py (SCRUM-14/15): page corners + template alignment,
pen-ink assignment to boxes, EMNIST-style bounding-box normalization, digit / letter CNNs.
This file is the ONE entry point the rest of the project uses (validate.py, app.py,
demo_utils.py, tests/), and it shapes the output for them:

    {"employee_id": "WE0634", "week_ending": "092726", "total_written": "0085",
     "rows": [{"row": 1, "date": "0921", "client": "KMR", "odo_start": "048213",
               "odo_end": "048238", "miles_written": "025"}, ...],
     "fields": {"r0_odo_start": {"text": "048213", "filled": [True, ...],
                                 "confidence": [0.99, ...]}, ...},
     "problems": ["r3_miles: 1 of 3 boxes empty", ...],     # segmentation, apart from recognition
     "registration": {"page_found": True, ...}}

An empty box reads as '_' (validate.py routes the row as "not fully read").

Run:
    python src/read_log.py data/test_official/phone/log_0001.png
    python src/read_log.py photo.jpg --json
"""
import hashlib
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_form as rf  # noqa: E402
from make_logs import Form  # noqa: E402

FORM = Form()                          # the form our generator draws; same box positions as rf.layout()
N_ROWS = FORM.n_rows
PAD = 4                                # pixels drawn around each box in the demo overlay
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MIN_ROW_BOXES = 3                      # a row counts as written only if at least this many of its 22 boxes have ink
ROW_FIELDS = [("date", "date"), ("client", "client"), ("odo_start", "odo_start"),
              ("odo_end", "odo_end"), ("miles", "miles_written")]   # (form field, output key)


# --------------------------------------------------------------------------- 1 register
def to_gray(img):
    """RGB / RGBA / gray array -> gray uint8 (PIL loads RGB, read_form expects gray or BGR)."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2GRAY)


_LAST = {"key": None}                  # the demo asks for the same page several times: register it once

# A form PRINTED on letter paper has margins, so the paper's corners are not the form's corners and
# read_form.register_page() lands the boxes in the wrong place. When the printed box outlines fit poorly,
# we find the printed boxes themselves and map them onto the layout. All synthetic test logs fit at
# 0.40 or better (this step never runs for them); real printed-form photos fit below 0.20.
PRINTED_FIT = 0.30
_ON_LINE = None
_ALL = np.array([r for rects in rf.layout().values() for r in rects], float)
_LAY_C = ((_ALL[:, 0] + _ALL[:, 2]).min() / 2, (_ALL[:, 1] + _ALL[:, 3]).min() / 2,
          (_ALL[:, 0] + _ALL[:, 2]).max() / 2, (_ALL[:, 1] + _ALL[:, 3]).max() / 2)   # extent of box centres


def outline_fit(page):
    """Share of ink under the layout's printed box outlines (higher = boxes sit where we expect them)."""
    global _ON_LINE
    if _ON_LINE is None:
        _ON_LINE = rf._outline_template() > 0.5
    return float(cv2.dilate(rf.ink_map(page), np.ones((5, 5), np.uint8))[_ON_LINE].mean())


def find_printed_boxes(page):
    """Centres of printed boxes: square holes enclosed by thin straight walls, all about the same size."""
    ink = (rf.ink_map(page) > 0.15).astype(np.uint8)
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21)))
    walls = cv2.dilate(h | v, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(1 - walls, connectivity=4)
    b = np.array([(x + w / 2, y + hh / 2, w, hh) for x, y, w, hh, a in st[1:]
                  if 20 <= w <= 110 and 20 <= hh <= 110 and 0.7 < w / hh < 1.4 and a > 0.75 * w * hh])
    if len(b) == 0:
        return b
    mw, mh = np.median(b[:, 2]), np.median(b[:, 3])
    return b[(abs(b[:, 2] - mw) < 0.2 * mw) & (abs(b[:, 3] - mh) < 0.2 * mh)]   # the common box size only


def align_printed_form(page):
    """Map the extent of the detected boxes onto the layout, then read_form's local refinement (wider first)."""
    b = find_printed_boxes(page)
    if len(b) < 30:
        return None
    x0, y0 = np.percentile(b[:, 0], 1), np.percentile(b[:, 1], 1)
    x1, y1 = np.percentile(b[:, 0], 99), np.percentile(b[:, 1], 99)
    A = cv2.getAffineTransform(np.float32([[x0, y0], [x1, y0], [x1, y1]]),
                               np.float32([[_LAY_C[0], _LAY_C[1]], [_LAY_C[2], _LAY_C[1]], [_LAY_C[2], _LAY_C[3]]]))
    # perspective: match every detected box to the nearest layout box, fit a homography (RANSAC), twice
    src = b[:, :2].astype(np.float32)
    H = np.vstack([A, [0, 0, 1]]).astype(np.float64)
    lay = np.stack([(_ALL[:, 0] + _ALL[:, 2]) / 2, (_ALL[:, 1] + _ALL[:, 3]) / 2], 1)
    for tol in (30, 12):
        mapped = cv2.perspectiveTransform(src[None], H)[0]
        d = np.linalg.norm(mapped[:, None] - lay[None], axis=2)
        j, ok = d.argmin(1), d.min(1) < tol
        if ok.sum() < 30:
            break
        Hn, _ = cv2.findHomography(src[ok], lay[j[ok]].astype(np.float32), cv2.RANSAC, 6.0)
        if Hn is None:
            break
        H = Hn
    out = cv2.warpPerspective(page, H, (rf.W, rf.H), flags=cv2.INTER_LINEAR, borderValue=255)
    for search in (12, 6):
        M = rf._local_homography(rf.ink_map(out), rf._outline_template(), search=search)
        if M is not None:
            out = cv2.warpPerspective(out, M, (rf.W, rf.H), flags=cv2.INTER_LINEAR, borderValue=255)
    return out


def _register_one(gray):
    """read_form.register_page; if the boxes fit poorly (a form printed with margins), align_printed_form."""
    page, info = rf.register_page(gray)
    info["fit"] = round(outline_fit(page), 3)
    info["printed_form_alignment"] = False
    if info["fit"] < PRINTED_FIT:
        alt = align_printed_form(page)
        if alt is not None and outline_fit(alt) > info["fit"] + 0.03:
            page = alt
            info.update(fit=round(outline_fit(alt), 3), printed_form_alignment=True)
    return page, info


def register_gray(img):
    """Photo -> (flat 2200x1400 gray page aligned to the form, info).
    Form ML-7 is landscape: a portrait photo (phone held upright) is tried turned both ways and the
    turn whose printed boxes fit best is kept. Landscape photos are used as they are."""
    key = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest() + str(img.shape)
    if _LAST["key"] != key:
        gray = to_gray(img)
        if gray.shape[0] > gray.shape[1]:
            tries = [(_register_one(np.ascontiguousarray(np.rot90(gray, k))), k) for k in (-1, 1)]  # clockwise, counter
            (page, info), k = max(tries, key=lambda t: t[0][1]["fit"])
            info["rotated"] = "clockwise" if k == -1 else "counter-clockwise"
        else:
            page, info = _register_one(gray)
        _LAST.update(key=key, page=page, info=info)
    return _LAST["page"], _LAST["info"]


def register(img):
    """Photo -> registered page as RGB (for display in the demo)."""
    page, _ = register_gray(img)
    return np.stack([page] * 3, axis=-1)


# --------------------------------------------------------------------------- 2 + 3
def extract_cells(page):
    """Registered page (gray or RGB) -> {field: [ink window per box]} (read_form.extract_cells)."""
    return rf.extract_cells(page if page.ndim == 2 else page[..., 0])


normalize_cell = rf.normalize_cell     # window -> 28x28 EMNIST-style image, or None if the box is empty


# --------------------------------------------------------------------------- 4 classify
class Classifier:
    """The two trained CNNs in models/ (digits 0-9, letters A-Z), via read_form.TorchClassifier."""

    def __init__(self, digit_path=None, letter_path=None):
        digit_path = digit_path or os.path.join(MODEL_DIR, "digit_model.pt")
        letter_path = letter_path or os.path.join(MODEL_DIR, "letter_model.pt")
        for p in (digit_path, letter_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} not found - run 'git pull' (the models live in models/)")
        self.model = rf.TorchClassifier(digit_path, letter_path)

    def predict(self, images, group):
        """images: list of 28x28 arrays, group 'digits' | 'letters' -> (chars, confidences)."""
        if not images:
            return [], []
        kind, alphabet = ("D", rf.DIGITS) if group == "digits" else ("L", rf.LETTERS)
        idx, conf = self.model.predict(np.stack(images).astype(np.float32), kind)
        return [alphabet[int(i)] for i in idx], [float(p) for p in conf]


# --------------------------------------------------------------------------- 5 assemble
def _field(reads):
    """read_form reads for one field [(char, conf) | None, ...] -> {"text", "filled", "confidence"}."""
    return {"text": "".join(r[0] if r else "_" for r in reads),
            "filled": [r is not None for r in reads],
            "confidence": [float(r[1]) if r else None for r in reads]}


def read_log(img, clf=None):
    """Full read of one log image (RGB array). Returns a dict shaped like the ground-truth JSON."""
    clf = clf or Classifier()
    page, info = register_gray(img)
    cells = rf.extract_cells(page)
    norm = {f: [rf.normalize_cell(c) for c in cs] for f, cs in cells.items()}
    reads = rf.classify_cells(norm, clf.model)

    out = {"employee_id": None, "week_ending": None, "rows": [], "total_written": None,
           "fields": {}, "problems": [], "registration": info}
    for name, key in [("emp_id", "employee_id"), ("week_ending", "week_ending"), ("total_miles", "total_written")]:
        out["fields"][name] = _field(reads[name])
        out[key] = out["fields"][name]["text"]

    for r in range(N_ROWS):
        row_reads = {name: _field(reads[f"r{r}_{name}"]) for name, _ in ROW_FIELDS}
        if sum(sum(f["filled"]) for f in row_reads.values()) < MIN_ROW_BOXES:
            continue                                    # unused row (a speck or shadow in one box is not a row)
        row = {"row": r + 1}
        for name, key in ROW_FIELDS:
            row[key] = row_reads[name]["text"]
            out["fields"][f"r{r}_{name}"] = row_reads[name]
        out["rows"].append(row)

    # segmentation check: every used field must have all of its boxes filled
    for name, f in out["fields"].items():
        missing = f["filled"].count(False)
        if missing:
            out["problems"].append(f"{name}: {missing} of {len(f['filled'])} boxes empty")
    if not info["page_found"]:
        out["problems"].append("page edges not found - assumed the photo is the whole page")
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: python src/read_log.py <image> [--json]")
        return
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
    res = read_log(img)
    print(f"Employee {res['employee_id']}   week ending {res['week_ending']}   total {res['total_written']}")
    print(f"{'#':>2}  {'date':6} {'client':6} {'odo start':9} {'odo end':9} miles")
    for r in res["rows"]:
        print(f"{r['row']:>2}  {r['date']:6} {r['client']:6} {r['odo_start']:9} {r['odo_end']:9} {r['miles_written']}")
    print("\nSegmentation problems:" if res["problems"] else "\nNo segmentation problems found.")
    for p in res["problems"]:
        print("  -", p)
    if "--json" in sys.argv:
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
