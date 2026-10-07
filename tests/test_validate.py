"""
test_validate.py - Checks for src/validate.py (SCRUM-16).

Part 1: hand-made cases, one per rule (incl. Susan's hand-filled log DALL 2, row 2).
Part 2: every log of the official clean test set, read PERFECTLY (ground truth fed
        straight in): rows with a clinician slip must be routed, all others posted.

Run:  python tests/test_validate.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from validate import validate, reference_from_truth  # noqa: E402


def make_result(emp, week, total, rows, low=None):
    """Build a read_log-style result. low = {field name: [confidences]} to fake unsure characters."""
    fields = {}
    out_rows = []
    for i, (date, client, s, e, m) in enumerate(rows):
        out_rows.append({"row": i + 1, "date": date, "client": client,
                         "odo_start": s, "odo_end": e, "miles_written": m})
        for name, text in [("date", date), ("client", client), ("odo_start", s), ("odo_end", e), ("miles", m)]:
            fields[f"r{i}_{name}"] = {"text": text, "filled": [True] * len(text), "confidence": [0.99] * len(text)}
    for name, text in [("emp_id", emp), ("week_ending", week), ("total_miles", total)]:
        fields[name] = {"text": text, "filled": [True] * len(text), "confidence": [0.99] * len(text)}
    for name, conf in (low or {}).items():
        fields[name]["confidence"] = conf
    return {"employee_id": emp, "week_ending": week, "total_written": total,
            "rows": out_rows, "fields": fields}


def decisions(result, reference=None):
    return [(d["decision"], d["reasons"]) for d in validate(result, reference)]


CHECKS = []


def case(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


GOOD = [("0921", "KMR", "048213", "048238", "025"),
        ("0921", "PDL", "048238", "048251", "013"),
        ("0922", "AVT", "048262", "048309", "047")]
REF = {"employees": {"RC5107"}, "schedule": {"0921": {"KMR", "PDL"}, "0922": {"AVT"}}}


@case("clean log, all lookups pass -> every row auto-posted")
def _():
    d = decisions(make_result("RC5107", "092726", "0085", GOOD), REF)
    assert all(x[0] == "auto-post" for x in d), d


@case("Susan's DALL 2, row 2: miles 73 but 125435 - 125412 = 23 -> only row 2 routed")
def _():
    rows = [("1130", "WQX", "125348", "125412", "064"),
            ("1130", "BIC", "125412", "125435", "073"),
            ("1201", "SXW", "125485", "125547", "062")]
    d = decisions(make_result("BA8034", "120626", "0199", rows))
    assert [x[0] for x in d] == ["auto-post", "needs review", "auto-post"], d
    assert "end - start = 23" in d[1][1][0]


@case("total does not match the miles column -> every row routed")
def _():
    d = decisions(make_result("RC5107", "092726", "0095", GOOD), REF)
    assert all(x[0] == "needs review" and "total" in x[1][0] for x in d), d


@case("week ending is not a Sunday -> every row routed")
def _():
    d = decisions(make_result("RC5107", "092626", "0085", GOOD), REF)
    assert all("Sunday" in x[1][0] for x in d), d


@case("trip date outside the week -> that row routed")
def _():
    rows = list(GOOD)
    rows[2] = ("1002",) + GOOD[2][1:]
    d = decisions(make_result("RC5107", "092726", "0085", rows))
    assert d[2][0] == "needs review" and "outside the week" in d[2][1][0], d


@case("odometer runs backwards from the previous row -> that row routed")
def _():
    rows = list(GOOD)
    rows[2] = ("0922", "AVT", "048200", "048247", "047")
    d = decisions(make_result("RC5107", "092726", "0085", rows))
    assert d[2][0] == "needs review" and "previous row" in " ".join(d[2][1]), d


@case("misread employee ID (WS0634 for WE0634) -> every row routed by the HR list")
def _():
    d = decisions(make_result("WS0634", "092726", "0085", GOOD),
                  {"employees": {"WE0634"}, "schedule": REF["schedule"]})
    assert all("HR list" in x[1][0] for x in d), d


@case("confidently misread client (EWO for EWQ) -> caught only by the schedule")
def _():
    rows = list(GOOD)
    rows[1] = ("0921", "PDO", "048238", "048251", "013")
    no_lookup = decisions(make_result("RC5107", "092726", "0085", rows))
    with_lookup = decisions(make_result("RC5107", "092726", "0085", rows), REF)
    assert no_lookup[1][0] == "auto-post"                       # arithmetic alone misses it
    assert with_lookup[1][0] == "needs review" and "schedule" in with_lookup[1][1][0]


@case("low-confidence character -> that row routed")
def _():
    d = decisions(make_result("RC5107", "092726", "0085", GOOD,
                              low={"r0_odo_end": [0.99, 0.99, 0.99, 0.55, 0.99, 0.99]}), REF)
    assert d[0][0] == "needs review" and "low confidence" in d[0][1][0], d
    assert d[1][0] == "auto-post"


@case("empty / unknown box -> that row routed")
def _():
    rows = list(GOOD)
    rows[0] = ("0921", "KMR", "0482_3", "048238", "025")
    d = decisions(make_result("RC5107", "092726", "0085", rows))
    assert d[0][0] == "needs review" and "not fully read" in d[0][1][0], d


def official_clean_set():
    paths = sorted(glob.glob("data/test_official/clean/log_*.json"))
    if not paths:
        print("  (skipped: generate data/test_official/clean first, see README)")
        return
    rows = slips = posted_ok = slips_caught = 0
    for p in paths:
        t = json.load(open(p))
        res = make_result(t["employee_id"], t["week_ending"], t["total_written"],
                          [(r["date"], r["client"], r["odo_start"], r["odo_end"], r["miles_written"])
                           for r in t["rows"]])
        for r, d in zip(t["rows"], validate(res, reference_from_truth(t))):
            rows += 1
            if r["clinician_error"]:
                slips += 1
                slips_caught += d["decision"] == "needs review"
            else:
                posted_ok += d["decision"] == "auto-post"
    print(f"  {len(paths)} logs, {rows} rows read perfectly: "
          f"{posted_ok}/{rows - slips} clean rows auto-posted, {slips_caught}/{slips} clinician slips routed")
    assert posted_ok == rows - slips and slips_caught == slips


def main():
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}\n      {e}")
    print("\nOfficial clean test set (ground truth as input):")
    official_clean_set()
    print("\nAll checks passed" if not failed else f"\n{failed} check(s) failed")


if __name__ == "__main__":
    main()