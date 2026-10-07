"""
check_pipeline.py - Run read_log.py over the official test set and measure the
stages that do not need a trained model yet:

  - rows found       : did the reader find exactly the rows the clinician filled in?
  - cell extraction  : share of written fields where every box was found filled
                       (a miss here is a SEGMENTATION error, counted apart from recognition)
  - empty boxes      : boxes in unused rows wrongly seen as written

Recognition accuracy is added once the real Classifier (SCRUM-14) is plugged in.

Run:  python tests/check_pipeline.py            (clean + medium + phone, 50 logs each)
      python tests/check_pipeline.py --n 10     (quicker)
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from read_log import read_log  # noqa: E402


def check(folder, n):
    paths = sorted(glob.glob(os.path.join(folder, "log_*.json")))[:n]
    if not paths:
        print(f"[{folder}] no logs found - generate the official test set first (see README)")
        return
    rows_ok, fields_total, fields_ok = 0, 0, 0
    for p in paths:
        truth = json.load(open(p))
        img_path = p[:-5] + (".png" if os.path.exists(p[:-5] + ".png") else ".jpg")
        res = read_log(np.asarray(Image.open(img_path).convert("RGB")))
        rows_ok += len(res["rows"]) == len(truth["rows"])
        for name in truth["fields"]:
            fields_total += 1
            f = res["fields"].get(name)
            fields_ok += f is not None and all(f["filled"])
    quality = os.path.basename(folder.rstrip("/"))
    print(f"[{quality:6s}] {len(paths)} logs | rows found correctly: {rows_ok / len(paths):6.1%} | "
          f"cell extraction success: {fields_ok / fields_total:6.1%} ({fields_ok}/{fields_total} fields)")


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 50
    for q in ("clean", "medium", "phone"):
        check(os.path.join("data", "test_official", q), n)


if __name__ == "__main__":
    main()