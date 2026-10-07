"""
read_log.py - End-to-end reader for one Form ML-7 image (SCRUM-22).

    photo -> 1 register -> 2 extract cells -> 3 normalize -> 4 classify
          -> 5 assemble fields and rows -> (6 validation, SCRUM-16)

This is the skeleton that wires the stages together. Two stages are
PLACEHOLDERS that teammates replace without touching anything else:

    register()     SCRUM-15 (Tenaj): straighten a photo onto the 2200x1400 form.
                   Placeholder = plain resize, correct only for flat clean scans.
    Classifier     SCRUM-14 (Tenaj): the trained model(s).
                   Placeholder = returns '?' with confidence 0 for every cell.

Output has the same keys as the ground-truth JSON from make_logs.py, plus a
confidence per character and a list of segmentation problems, so the
evaluation script (SCRUM-26) can compare the two directly.

Run:
    python src/read_log.py data/test_official/clean/log_0001.png
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from make_logs import Form  # noqa: E402
from normalize import normalize_cell  # noqa: E402

FORM = Form()                         # the form our own generator draws: box positions are exact
N_ROWS = FORM.n_rows
PAD = 4                               # pixels kept around each box when cutting it out

# Which character group each box may contain (Form ML-7 field specification).
# employee ID = 2 letters + 4 digits; client code = 3 letters; everything else digits.
FIELD_GROUPS = {"emp_id": ["letters"] * 2 + ["digits"] * 4, "client": ["letters"] * 3}
ROW_FIELDS = [("date", "date"), ("client", "client"), ("odo_start", "odo_start"),
              ("odo_end", "odo_end"), ("miles", "miles_written")]   # (form field, output key)


# --------------------------------------------------------------------------- 1 register
def register(img):
    """
    PLACEHOLDER for SCRUM-15. Must return the page as an RGB array of exactly
    Form.H x Form.W (1400 x 2200), straightened so the printed boxes sit at the
    coordinates in FORM.fields. A plain resize only works for flat, uncropped scans.
    """
    return np.asarray(Image.fromarray(img).resize((Form.W, Form.H)))


# --------------------------------------------------------------------------- 2 extract cells
def extract_cells(page):
    """Cut every box out of the registered page: {field name: [crop, crop, ...]}."""
    cells = {}
    for name, boxes in FORM.fields.items():
        cells[name] = [page[max(0, y0 - PAD):y1 + PAD, max(0, x0 - PAD):x1 + PAD] for x0, y0, x1, y1 in boxes]
    return cells


# --------------------------------------------------------------------------- 4 classify
class Classifier:
    """
    PLACEHOLDER for SCRUM-14. The real version loads the trained model(s) and
    implements predict(). Interface the rest of the pipeline relies on:

        predict(images, group) -> (chars, confidences)
            images : list of 28x28 float32 arrays (normalize_cell output)
            group  : "digits" or "letters"  (the model may only answer from this group)
            chars  : list of single characters, e.g. ['0', '4', '8']
            confidences : list of floats in [0, 1] (e.g. top softmax probability)
    """

    def predict(self, images, group):
        return ["?"] * len(images), [0.0] * len(images)


def group_of(field, i):
    """Character group allowed in box i of a field."""
    base = field.split("_", 1)[1] if field.startswith("r") and field[1].isdigit() else field
    return FIELD_GROUPS.get(base, ["digits"] * 10)[i]


# --------------------------------------------------------------------------- 3 + 4 + 5
def read_field(field, crops, clf):
    """Normalize every crop of one field, classify the non-empty ones, return text + details."""
    imgs = [normalize_cell(c) for c in crops]
    chars, conf = [""] * len(crops), [None] * len(crops)
    for group in ("digits", "letters"):
        idx = [i for i, im in enumerate(imgs) if im is not None and group_of(field, i) == group]
        if idx:
            c, p = clf.predict([imgs[i] for i in idx], group)
            for i, ch, pr in zip(idx, c, p):
                chars[i], conf[i] = ch, float(pr)
    filled = [im is not None for im in imgs]
    return {"text": "".join(ch if f else "_" for ch, f in zip(chars, filled)),
            "filled": filled, "confidence": conf}


def read_log(img, clf=None):
    """Full read of one log image (RGB array). Returns a dict shaped like the ground-truth JSON."""
    clf = clf or Classifier()
    page = register(img)
    cells = extract_cells(page)

    def field(name):
        return read_field(name, cells[name], clf)

    out = {"employee_id": None, "week_ending": None, "rows": [], "total_written": None,
           "fields": {}, "problems": []}
    for name, key in [("emp_id", "employee_id"), ("week_ending", "week_ending"), ("total_miles", "total_written")]:
        f = field(name)
        out["fields"][name] = f
        out[key] = f["text"]

    for r in range(N_ROWS):
        reads = {key: field(f"r{r}_{name}") for name, key in ROW_FIELDS}
        if not any(any(f["filled"]) for f in reads.values()):
            continue                                    # unused row
        row = {"row": r + 1}
        for (name, key), f in zip(ROW_FIELDS, reads.values()):
            row[key] = f["text"]
            out["fields"][f"r{r}_{name}"] = f
        out["rows"].append(row)

    # segmentation check: every used field must have all of its boxes filled
    for name, f in out["fields"].items():
        missing = f["filled"].count(False)
        if missing:
            out["problems"].append(f"{name}: {missing} of {len(f['filled'])} boxes empty")
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: python src/read_log.py <image>")
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
        print(json.dumps({k: v for k, v in res.items() if k != "fields"}, indent=1))


if __name__ == "__main__":
    main()