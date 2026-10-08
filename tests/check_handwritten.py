"""
check_handwritten.py - Read REAL handwritten Form ML-7 photos and compare with their answer keys.

Put pairs in data/handwritten/:   hw_01.jpg + hw_01.json   (answer key in the make_logs.py format)
(iPhone HEIC photos: convert first, e.g.  sips -s format jpeg photo.heic --out data/handwritten/hw_01.jpg)

Boxes listed under "out_of_scope_boxes" in a key (e.g. a crossed-out correction) are shown but not scored.
For each photo it prints the read next to the truth, character accuracy (digits / letters),
field accuracy, and the auto-post / needs-review decision per row (lookups simulated from the key).

Run:  python tests/check_handwritten.py
      python tests/check_handwritten.py --out results/results_handwritten.txt
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from read_log import Classifier, read_log  # noqa: E402
from validate import validate, reference_from_truth  # noqa: E402

ROW_KEYS = [("date", "date"), ("client", "client"), ("odo_start", "odo_start"),
            ("odo_end", "odo_end"), ("miles", "miles_written")]


def truth_fields(t):
    f = {"emp_id": t["employee_id"], "week_ending": t["week_ending"], "total_miles": t["total_written"]}
    for i, r in enumerate(t["rows"]):
        for name, key in ROW_KEYS:
            f[f"r{i}_{name}"] = r[key]
    return f


def check(img_path, json_path, clf):
    t = json.load(open(json_path))
    img = np.asarray(ImageOps.exif_transpose(Image.open(img_path)).convert("RGB"))
    res = read_log(img, clf)
    lines = [f"=== {os.path.basename(img_path)}   (fit {res['registration'].get('fit')}, "
             f"printed-form alignment: {res['registration'].get('printed_form_alignment')})"]
    c = dict(d=0, d_ok=0, l=0, l_ok=0, missing=0, fields=0, fields_ok=0)
    skip = t.get("out_of_scope_boxes", {})                 # e.g. a crossed-out correction: reported, not scored
    for name, want in truth_fields(t).items():
        f = res["fields"].get(name, {"text": "_" * len(want)})
        got = f["text"]
        c["fields"] += 1
        c["fields_ok"] += got == want
        for i, (g, w) in enumerate(zip(got, want)):
            if i in skip.get(name, []):
                lines.append(f"  (out of scope) {name} box {i + 1}: read {g!r}, truth {w!r} - not counted")
                continue
            if g == "_":
                c["missing"] += 1
                continue
            k = "d" if w.isdigit() else "l"
            c[k] += 1
            c[k + "_ok"] += g == w
        if got != want:
            lines.append(f"  {name:14s} read {got:8s} truth {want}")
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "n/a"
    lines.append(f"  digits {pct(c['d_ok'], c['d'])} ({c['d']})  letters {pct(c['l_ok'], c['l'])} ({c['l']})  "
                 f"empty-read boxes {c['missing']}  fields {c['fields_ok']}/{c['fields']} exactly right")
    for d in validate(res, reference_from_truth(t)):
        lines.append(f"  row {d['row']}: {d['decision']:12s} {'; '.join(d['reasons'])[:110]}")
    return "\n".join(lines)


def main():
    pairs = []
    for jp in sorted(glob.glob(os.path.join("data", "handwritten", "*.json"))):
        img = next((jp[:-5] + e for e in (".jpg", ".jpeg", ".png", ".JPG", ".PNG") if os.path.exists(jp[:-5] + e)), None)
        if img:
            pairs.append((img, jp))
    if not pairs:
        print("No photo + answer-key pairs in data/handwritten/ (e.g. hw_01.jpg + hw_01.json)")
        return
    clf = Classifier()
    text = "\n\n".join(check(i, j, clf) for i, j in pairs)
    print(text)
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        open(out, "w").write(text + "\n")
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
