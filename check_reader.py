"""
check_reader.py -- measure how well read_form.py reads generated logs with known answers.

Works on the team's official layout made by make_logs.py:
  data/test_official/clean/log_0001.png + log_0001.json, .../medium/..., .../phone/...
The CONDITION is the name of the folder the log is in (clean / medium / phone).
(Older flat folders with names like phone_03.jpg still work: condition = text before the first "_".)

Run:
  python check_reader.py data/test_official --digit-model digit_model.pt --letter-model letter_model.pt

Reports, per condition, as SEPARATE numbers (the assignment asks for this):
  * cell-extraction rate : boxes that should hold a character and were found non-empty
  * false-ink boxes      : boxes that should be empty but were read as holding something
  * character accuracy   : digits and letters separately (only over boxes that were found)
  * field accuracy       : every character in the field correct, per field type
  * row count correct    : did we find exactly the rows that were written
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

import read_form as rf


def truth_cells(truth):
    """Ground-truth JSON -> {field: string} for every field on the form (blank rows are absent)."""
    t = {"emp_id": truth["employee_id"], "week_ending": truth["week_ending"],
         "total_miles": f"{int(truth['total_written']):04d}"}
    for r, row in enumerate(truth["rows"]):
        t[f"r{r}_date"] = row["date"]
        t[f"r{r}_client"] = row["client"]
        t[f"r{r}_odo_start"] = row["odo_start"]
        t[f"r{r}_odo_end"] = row["odo_end"]
        t[f"r{r}_miles"] = row["miles_written"]
    return t


def read_cells_only(img, clf):
    """Run the pipeline up to classification and return the per-box reads (None = empty box)."""
    page, info = rf.register_page(img)
    cells = rf.extract_cells(page)
    norm = {f: [rf.normalize_cell(c) for c in cs] for f, cs in cells.items()}
    return rf.classify_cells(norm, clf), info


def field_type(name):
    if name in ("emp_id", "week_ending", "total_miles"):
        return {"emp_id": "employee_id", "week_ending": "week_ending", "total_miles": "total"}[name]
    if name.endswith("_client"):
        return "client"
    if name.endswith("_odo_start") or name.endswith("_odo_end"):
        return "odometer"
    if name.endswith("_miles"):
        return "miles"
    if name.endswith("_date"):
        return "date"
    return {"emp_id": "employee_id", "week_ending": "week_ending", "total_miles": "total"}[name]


def evaluate(folder, clf, verbose=True):
    stats = defaultdict(lambda: defaultdict(int))
    files = sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True))
    for jp in files:
        base = os.path.splitext(jp)[0]
        imgp = next((base + e for e in (".png", ".jpg", ".jpeg") if os.path.exists(base + e)), None)
        if imgp is None:
            continue
        parent = os.path.basename(os.path.dirname(base))
        cond = parent if os.path.abspath(os.path.dirname(base)) != os.path.abspath(folder) \
            else os.path.basename(base).split("_")[0]
        S = stats[cond]
        truth = json.load(open(jp))
        hz = truth.get("hazards") or {}                        # fields the generator made touching / overflowing
        if isinstance(hz, dict):
            hazard_fields = set(hz)
        else:
            hazard_fields = {h if isinstance(h, str) else h.get("field", "") for h in hz}
        tc = truth_cells(truth)
        reads, info = read_cells_only(cv2.imread(imgp), clf)
        S["logs"] += 1
        S["page_found"] += int(info["page_found"])
        n_rows_found = sum(any(r is not None for k in ("date", "client", "odo_start", "odo_end", "miles")
                               for r in reads[f"r{i}_{k}"]) for i in range(rf.N_ROWS))
        S["rows_correct"] += int(n_rows_found == len(truth["rows"]))
        for field, got in reads.items():
            want = tc.get(field)
            kinds = rf.kinds_for(field)
            if want is None:                                   # box group that should be empty
                S["empty_boxes"] += len(got)
                S["false_ink"] += sum(g is not None for g in got)
                continue
            ft = field_type(field)
            S[f"fields_{ft}"] += 1
            all_ok = True
            for i, ch in enumerate(want):
                g = got[i]
                S["chars_expected"] += 1
                if g is None:
                    S["chars_missing"] += 1
                    all_ok = False
                    continue
                k = "digit" if kinds[i] == "D" else "letter"
                S[f"{k}_found"] += 1
                if g[0] == ch:
                    S[f"{k}_correct"] += 1
                else:
                    all_ok = False
            S[f"fields_{ft}_ok"] += int(all_ok)
            grp = "hazard" if field in hazard_fields else "plain"   # hazard = touching / overflow written on purpose
            if ft in ("odometer", "miles", "date", "total"):
                S[f"fields_{grp}"] += 1
                S[f"fields_{grp}_ok"] += int(all_ok)

    lines = []
    for cond, S in stats.items():
        pct = lambda a, b: f"{100 * a / b:6.2f}%" if b else "   n/a "
        lines.append(f"--- condition: {cond}  ({S['logs']} logs) ---")
        lines.append(f"page found                : {pct(S['page_found'], S['logs'])}")
        lines.append(f"row count correct         : {pct(S['rows_correct'], S['logs'])}")
        lines.append(f"cell extraction rate      : {pct(S['chars_expected'] - S['chars_missing'], S['chars_expected'])}"
                     f"   ({S['chars_missing']} of {S['chars_expected']} written boxes read as empty)")
        lines.append(f"false-ink boxes           : {S['false_ink']} of {S['empty_boxes']} blank boxes")
        lines.append(f"digit accuracy (found)    : {pct(S['digit_correct'], S['digit_found'])}   ({S['digit_found']} digits)")
        lines.append(f"letter accuracy (found)   : {pct(S['letter_correct'], S['letter_found'])}   ({S['letter_found']} letters)")
        for ft in ("employee_id", "week_ending", "date", "client", "odometer", "miles", "total"):
            lines.append(f"field accuracy {ft:<12}: {pct(S[f'fields_{ft}_ok'], S[f'fields_{ft}'])}   ({S[f'fields_{ft}']} fields)")
        lines.append(f"numeric fields WITHOUT hazard : {pct(S['fields_plain_ok'], S['fields_plain'])}   ({S['fields_plain']} fields)")
        lines.append(f"numeric fields WITH touching/overflow : {pct(S['fields_hazard_ok'], S['fields_hazard'])}   ({S['fields_hazard']} fields)")
        lines.append("")
    text = "\n".join(lines)
    if verbose:
        print(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--digit-model", default="digit_model.pt")
    ap.add_argument("--letter-model", default="letter_model.pt")
    ap.add_argument("--out", default="results_official.txt")
    a = ap.parse_args()
    text = evaluate(a.folder, rf.TorchClassifier(a.digit_model, a.letter_model))
    open(a.out, "w").write(text)
    print(f"Saved {a.out}")
