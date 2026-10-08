"""
check_pipeline.py - End-to-end check of the path the demo uses:
    image -> read_log.read_log() -> validate.validate() -> auto-post / needs review

Per condition (clean / medium / phone) it reports, as SEPARATE numbers:
  - rows found         : did the reader find exactly the rows that were written?
  - cell extraction    : written fields with every box found (a miss = SEGMENTATION error)
  - field accuracy     : written fields read exactly right (RECOGNITION, given extraction)
  - row raw error      : rows with any field misread, before any rule
                         (a misread employee ID / week ending / total counts against every row)
  - row auto-post rate : rows validate() posts with no clerk
  - wrong & posted     : auto-posted rows that are still wrong (misread OR a clinician slip)
Lookups = validate.reference_from_truth (HR list + visit schedule simulated from the answer key).
Recognition details per digit / letter / field type: src/check_reader.py (Tenaj).

Run:  python tests/check_pipeline.py                 (50 logs per condition)
      python tests/check_pipeline.py --n 10          (quicker)
      python tests/check_pipeline.py --out results/results_pipeline.txt
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import read_form as rf  # noqa: E402
from read_log import FORM, Classifier, read_log  # noqa: E402
from validate import validate, reference_from_truth  # noqa: E402

KEYS = ["date", "client", "odo_start", "odo_end", "miles_written"]


def check_layout():
    """read_form's hard-coded box layout must match the form make_logs.py draws."""
    ours = {k: [tuple(b) for b in v] for k, v in FORM.fields.items()}
    assert ours == rf.layout(FORM.n_rows), "read_form.layout() and make_logs.Form disagree"


def truth_text(truth):
    t = {"emp_id": truth["employee_id"], "week_ending": truth["week_ending"],
         "total_miles": truth["total_written"]}
    for i, r in enumerate(truth["rows"]):
        for name, key in [("date", "date"), ("client", "client"), ("odo_start", "odo_start"),
                          ("odo_end", "odo_end"), ("miles", "miles_written")]:
            t[f"r{i}_{name}"] = r[key]
    return t


def check(folder, n, clf):
    paths = sorted(glob.glob(os.path.join(folder, "log_*.json")))[:n]
    quality = os.path.basename(folder.rstrip("/"))
    if not paths:
        return f"[{quality}] no logs found - generate the official test set first (see README)"
    c = dict(logs=0, rows_ok=0, fields=0, extracted=0, correct=0, rows=0, raw_bad=0, posted=0, posted_bad=0)
    for p in paths:
        truth = json.load(open(p))
        img_path = next(p[:-5] + e for e in (".png", ".jpg", ".jpeg") if os.path.exists(p[:-5] + e))
        res = read_log(np.asarray(Image.open(img_path).convert("RGB")), clf)
        c["logs"] += 1
        c["rows_ok"] += len(res["rows"]) == len(truth["rows"])

        for name, want in truth_text(truth).items():
            f = res["fields"].get(name)
            c["fields"] += 1
            c["extracted"] += f is not None and all(f["filled"])
            c["correct"] += f is not None and f["text"] == want

        header_ok = all(res[k] == truth[k] for k in ("employee_id", "week_ending", "total_written"))
        decisions = {d["row"]: d["decision"] for d in validate(res, reference_from_truth(truth))}
        read_rows = {r["row"]: r for r in res["rows"]}
        for i, t in enumerate(truth["rows"]):
            got = read_rows.get(i + 1)
            misread = got is None or not header_ok or any(got[k] != t[k] for k in KEYS)
            wrong = misread or t["clinician_error"]          # posting a clinician slip is also a wrong payment
            posted = decisions.get(i + 1) == "auto-post"
            c["rows"] += 1
            c["raw_bad"] += misread
            c["posted"] += posted
            c["posted_bad"] += posted and wrong

    pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "  n/a "
    return (f"[{quality:6s}] {c['logs']} logs | rows found {pct(c['rows_ok'], c['logs'])} | "
            f"cell extraction {pct(c['extracted'], c['fields'])} | field accuracy {pct(c['correct'], c['fields'])} | "
            f"row raw error {pct(c['raw_bad'], c['rows'])} | row auto-post {pct(c['posted'], c['rows'])} | "
            f"wrong & posted {c['posted_bad']}/{c['posted']} rows")


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 50
    check_layout()
    clf = Classifier()                                   # load the models once
    lines = []
    for q in ("clean", "medium", "phone"):
        lines.append(check(os.path.join("data", "test_official", q), n, clf))
        print(lines[-1], flush=True)
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        open(out, "w").write("\n".join(lines) + "\n")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
