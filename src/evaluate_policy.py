"""
evaluate_policy.py -- SCRUM-18: measure the auto-post policy on the official test set.

Reads every log with read_form.py (ONE pass), then applies the post-or-route rules at several
confidence thresholds and reports, separately:
  * RAW model error      : rows with any character misread, before any rule is applied
  * ERROR AFTER RULES    : rows that were auto-posted AND still wrong (the dangerous ones)
  * AUTO-POST RATE       : share of logs (and of rows) posted with no clerk
  * COST per 10,000 logs : clerk keying ($3.40 per routed log) + priced misreads on auto-posted logs

Uses the TEAM rules in validate.py (SCRUM-16, Kaien) when it is in the same folder.

Run (from the repo folder; models from models/):
  python src/evaluate_policy.py data/test_official      (saves results/results_policy.txt)

Rules (Form ML-7 spec, same list as validate.py's docstring):
  1. every box read (no '?')
  2. miles = odo_end - odo_start
  3. odo_end > odo_start, and odo_start >= previous row's odo_end
  4. week ending is a real Sunday, and each trip date is inside that week
  5. total = sum of the MILES column
  6. every character's confidence >= threshold (letters in client / employee ID use their own threshold)
A LOG is auto-posted only if every row passes and the header (employee ID, week ending, total) passes.
"""
import argparse
import datetime as dt
import glob
import json
import os
from collections import defaultdict

import cv2

import read_form as rf

RATE = 0.62            # $ per mile (assignment): misread cost = RATE * 10**digit_position
KEYING = 3.40          # $ per log keyed by hand
LOGS_PER_MONTH = 10_000


# ----------------------------------------------------------------------------- the rules
def _int(s):
    return int(s) if s and s.isdigit() else None


def _week_start_end(week_ending):
    """'092726' -> (monday_date, sunday_date) or None if not a real Sunday."""
    try:
        d = dt.date(2000 + int(week_ending[4:6]), int(week_ending[0:2]), int(week_ending[2:4]))
    except (ValueError, TypeError):
        return None
    return (d - dt.timedelta(days=6), d) if d.weekday() == 6 else None



# ----------------------------------------------------------------------------- scoring
def priced_miles_error(read_miles, true_miles):
    """$ cost of a wrong posted MILES value: RATE * 10**p for every wrong digit position p (0 = units)."""
    r, t = str(read_miles).zfill(4)[::-1], str(true_miles).zfill(4)[::-1]
    return sum(RATE * 10 ** p for p, (a, b) in enumerate(zip(r, t)) if a != b)


def score_log(result, truth, decisions):
    n = len(truth["rows"])
    found = len(result["rows"]) == n
    header_ok = (result["employee_id"] == truth["employee_id"]
                 and result["week_ending"] == truth["week_ending"]
                 and result["total_written"] == truth["total_written"])
    emp_bad = result["employee_id"] != truth["employee_id"]
    raw_wrong, row_bad, row_letters, row_digits, cost = [], [], [], [], 0.0
    for r, t in zip(result["rows"], truth["rows"]):
        digits_bad = (any(r[k] != t[k] for k in ("date", "odo_start", "odo_end"))
                      or _int(r["miles_written"]) != t["miles_true"])   # paid miles = TRUE miles
        letters_bad = r["client"] != t["client"] or emp_bad
        raw_wrong.append(any(r[k] != t[k] for k in ("date", "client", "odo_start", "odo_end"))
                         or r["miles_written"] != t["miles_written"])
        row_digits.append(digits_bad or result["week_ending"] != truth["week_ending"])
        row_letters.append(letters_bad)
        row_bad.append(digits_bad or letters_bad)
        if _int(r["miles_written"]) is not None and _int(r["miles_written"]) != t["miles_true"]:
            cost += priced_miles_error(_int(r["miles_written"]), t["miles_true"])
    auto_rows = [d["decision"] == "auto-post" for d in decisions]
    auto = found and all(auto_rows)
    return {"raw_wrong": raw_wrong, "any_raw": (not header_ok) or any(raw_wrong), "auto": auto,
            "posted_bad": (not found) or any(row_bad), "letters": any(row_letters),
            "digits": any(row_digits), "cost": cost, "auto_rows": auto_rows if found else [False] * n,
            "row_bad": row_bad, "row_letters": row_letters, "row_digits": row_digits}


def summarize(scored, label):
    n = len(scored)
    auto = [s for s in scored if s["auto"]]
    bad = [s for s in auto if s["posted_bad"]]
    only_l = sum(s["letters"] and not s["digits"] for s in bad)
    dig = sum(s["digits"] for s in bad)
    priced = sum(s["cost"] for s in auto) * LOGS_PER_MONTH / n
    keying = (n - len(auto)) * KEYING * LOGS_PER_MONTH / n          # every routed log is keyed by a clerk
    rows_total = sum(len(s["auto_rows"]) for s in scored)
    rows_auto = sum(sum(s["auto_rows"]) for s in scored)
    rows_auto_bad = sum(b and a for s in scored for a, b in zip(s["auto_rows"], s["row_bad"]))
    return (f"{label:<34}| LOG-level: auto-post {100 * len(auto) / n:5.1f}%  "
            f"wrong {len(bad):>2}/{len(auto):<2} (letters-only {only_l}, digit-related {dig})  priced misread ${priced:>7,.0f}/mo  keying ${keying:>7,.0f}/mo  TOTAL ${priced + keying:>7,.0f}/mo"
            f" | ROW-level: auto-post {100 * rows_auto / rows_total:5.1f}%  wrong {rows_auto_bad}/{rows_auto}"
            f" ({100 * rows_auto_bad / max(rows_auto, 1):.1f}%)")


def collect(folder, clf, cache):
    """Read every log once -> {condition: [(result, truth), ...]}; saved so sweeps never re-read images."""
    import pickle
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    data = defaultdict(list)
    keep = ("employee_id", "employee_id_conf", "week_ending", "week_ending_conf",
            "total_written", "total_written_conf", "rows", "fields")
    for jp in sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)):
        base = os.path.splitext(jp)[0]
        imgp = next((base + e for e in (".png", ".jpg", ".jpeg") if os.path.exists(base + e)), None)
        if imgp is None:
            continue
        cond = os.path.basename(os.path.dirname(base))
        res = rf.read_log(cv2.imread(imgp), clf)
        data[cond].append(({k: res[k] for k in keep}, json.load(open(jp))))
    if cache:
        pickle.dump(dict(data), open(cache, "wb"))
    return data


def evaluate(folder, clf, cache="reads_cache.pkl"):
    import random
    import validate as team                              # Kaien's SCRUM-16 rules
    data = collect(folder, clf, cache)
    rng = random.Random(2026)
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    others = {rng.choice(L) + rng.choice(L) + f"{rng.randrange(10000):04d}" for _ in range(5000)}

    def reference(truth, mode):
        if mode == "none":
            return None
        ref = team.reference_from_truth(truth)           # validate.py's simulation: HR list = the ONE true ID
        if mode == "realistic":
            ref["employees"] = ref["employees"] | others  # HR list also holds 5,000 other valid IDs
        return ref

    lines = ["Post-or-route policy on the official test set (150 logs), using the team's validate.py (SCRUM-16).",
             "RAW model error and error AFTER rules are separate.",
             "LOG-level: a log is auto-posted only if every row passes. ROW-level: each row is decided on its own.",
             "'wrong' = auto-posted but still wrong against the TRUE answer. 'priced misread' = wrong MILES values only",
             "(0.62 x 10^digit position), per 10,000 logs; letter errors are counted but NOT priced.",
             "Lookups: none = arithmetic/date/confidence rules only | HR5000 = employee list with 5,000 other valid IDs",
             "plus the true visit schedule | HR1 = validate.py's own simulation (HR list holds only the true ID).", ""]
    for cond in ("clean", "medium", "phone"):
        if cond not in data:
            continue
        pairs = data[cond]
        rows = sum(len(t["rows"]) for _, t in pairs)
        raw = sum(sum(score_log(r, t, [])["raw_wrong"]) for r, t in pairs)
        lines.append(f"--- {cond} ({len(pairs)} logs, {rows} rows) | RAW row error {100 * raw / rows:.1f}% "
                     f"| logs with any raw error {100 * sum(score_log(r, t, [])['any_raw'] for r, t in pairs) / len(pairs):.0f}% ---")
        for mode, name in (("none", "no lookups"), ("realistic", "HR5000+schedule"), ("sim", "HR1+schedule")):
            for cm in (0.0, 0.8, 0.95, 0.99):
                if mode == "sim" and cm != 0.8:
                    continue
                team.CONF_MIN = cm
                scored = [score_log(r, t, team.validate(r, reference(t, mode))) for r, t in pairs]
                lines.append(summarize(scored, f"{name}, confidence>={cm}"))
        lines.append("")
    team.CONF_MIN = 0.80
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--digit-model", default="models/digit_model.pt")
    ap.add_argument("--letter-model", default="models/letter_model.pt")
    ap.add_argument("--cache", default="reads_cache.pkl", help="reads are saved here; delete it to re-read the images")
    ap.add_argument("--out", default="results/results_policy.txt")
    a = ap.parse_args()
    clf = rf.TorchClassifier(a.digit_model, a.letter_model) if not os.path.exists(a.cache) else None
    text = evaluate(a.folder, clf, a.cache)
    open(a.out, "w").write(text)
    print(f"Saved {a.out}")
