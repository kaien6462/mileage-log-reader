"""
validate.py - Business rules and the post-or-route decision (SCRUM-16).

    validate(result, reference=None) -> one decision per row:
        {"row": 1, "decision": "auto-post" | "needs review",
         "reasons": ["miles written 073 but end - start = 23", ...]}

    result    = the dict returned by read_log.read_log()
    reference = optional lookups the company already has (simulated for our test logs):
                {"employees": {"WE0634", ...},                    # HR master list
                 "schedule": {"0923": {"ZSX"}, "0925": {...}}}     # clinician's visit schedule by date
                Without it the ID / client lookups are skipped and say so.

A row is auto-posted only if EVERY check below passes; any failure routes it to a clerk,
with the reason shown. A misread and a clinician's own slip look the same to the rules
(both break miles = end - start), and both need a person, so both are routed.

Log-level checks (a failure routes every row, because the whole log is affected):
  L1  employee ID read completely, 2 letters + 4 digits, and on the HR list
  L2  week ending is a real date and a Sunday
  L3  total miles = sum of the MILES column as written
  L4  every character of these fields read with confidence >= CONF_MIN
Row-level checks:
  R1  every box read (no empty '_' or unknown '?')
  R2  trip date is a real date inside the stated week
  R3  odometer end > start
  R4  odometer start >= previous row's end (odometers only run forward)
  R5  miles written = end - start
  R6  client code on the clinician's schedule for that date
  R7  every character in the row read with confidence >= CONF_MIN
Why R6 and L1 matter: on our first real read, two client-code letters were misread
with high confidence and every arithmetic check still passed - only a lookup catches that.
"""
import datetime as dt
import re

CONF_MIN = 0.80                     # characters below this confidence send their row to a clerk
ROW_FIELDS = [("date", "date"), ("client", "client"), ("odo_start", "odo_start"),
              ("odo_end", "odo_end"), ("miles", "miles_written")]   # (form field, result key)


def _complete(text):
    return bool(text) and "_" not in text and "?" not in text


def _low_conf(field):
    """Characters of one field read below CONF_MIN, e.g. ['box 2 (40%)']."""
    if not field:
        return []
    return [f"box {i + 1} ({p:.0%})" for i, p in enumerate(field["confidence"])
            if p is not None and p < CONF_MIN]


def _parse_week_end(text):
    try:
        return dt.date(2000 + int(text[4:6]), int(text[0:2]), int(text[2:4]))
    except (ValueError, TypeError):
        return None


def _trip_date(mmdd, week_end):
    """Trip date as a date, choosing the year so it falls next to the week ending (Dec/Jan safe)."""
    for year in (week_end.year, week_end.year - 1):
        try:
            d = dt.date(year, int(mmdd[0:2]), int(mmdd[2:4]))
        except ValueError:
            return None
        if d <= week_end:
            return d
    return None


def check_log(result, reference=None):
    """Log-level problems (list of strings). Any problem here routes every row."""
    problems = []
    fields = result.get("fields", {})

    emp = result.get("employee_id") or ""
    if not _complete(emp):
        problems.append(f"employee ID not fully read ({emp})")
    elif not re.fullmatch(r"[A-Z]{2}[0-9]{4}", emp):
        problems.append(f"employee ID {emp} is not 2 letters + 4 digits")
    elif reference and "employees" in reference and emp not in reference["employees"]:
        problems.append(f"employee ID {emp} not on the HR list")

    week = result.get("week_ending") or ""
    week_end = _parse_week_end(week) if _complete(week) else None
    if week_end is None:
        problems.append(f"week ending {week} is not a valid date")
    elif week_end.weekday() != 6:
        problems.append(f"week ending {week_end:%m/%d/%y} is a {week_end:%A}, not a Sunday")

    total = result.get("total_written") or ""
    miles = [r.get("miles_written", "") for r in result.get("rows", [])]
    if not _complete(total):
        problems.append(f"total miles not fully read ({total})")
    elif all(_complete(m) for m in miles) and int(total) != sum(int(m) for m in miles):
        problems.append(f"total {int(total)} != sum of the miles column {sum(int(m) for m in miles)}")

    for name, label in [("emp_id", "employee ID"), ("week_ending", "week ending"), ("total_miles", "total")]:
        low = _low_conf(fields.get(name))
        if low:
            problems.append(f"low confidence in {label}: {', '.join(low)}")
    return problems


def validate(result, reference=None):
    """One post-or-route decision per row (see module docstring for the checks)."""
    log_problems = check_log(result, reference)
    week_end = _parse_week_end(result.get("week_ending") or "")
    schedule = (reference or {}).get("schedule")
    fields = result.get("fields", {})

    decisions, prev_end = [], None
    for row in result.get("rows", []):
        r = row["row"] - 1                                  # 0-based index used in field names
        reasons = [f"log: {p}" for p in log_problems]

        # R1 every box read
        missing = [key for _, key in ROW_FIELDS if not _complete(row.get(key, ""))]
        if missing:
            reasons.append("not fully read: " + ", ".join(missing))

        # R2 date inside the week
        if "date" not in missing and week_end:
            d = _trip_date(row["date"], week_end)
            if d is None:
                reasons.append(f"date {row['date']} is not a valid date")
            elif not (week_end - dt.timedelta(days=6) <= d <= week_end):
                reasons.append(f"date {d:%m/%d} is outside the week ending {week_end:%m/%d}")

        # R3-R5 odometer arithmetic
        start = int(row["odo_start"]) if "odo_start" not in missing else None
        end = int(row["odo_end"]) if "odo_end" not in missing else None
        if start is not None and end is not None:
            if end <= start:
                reasons.append(f"odometer end {end} is not greater than start {start}")
            elif "miles_written" not in missing and int(row["miles_written"]) != end - start:
                reasons.append(f"miles written {row['miles_written']} but end - start = {end - start}")
        if start is not None and prev_end is not None and start < prev_end:
            reasons.append(f"odometer start {start} is below the previous row's end {prev_end}")
        prev_end = end if end is not None else prev_end

        # R6 client on the schedule for that date
        if schedule is not None and "client" not in missing and "date" not in missing:
            if row["client"] not in schedule.get(row["date"], set()):
                reasons.append(f"client {row['client']} not on the schedule for {row['date'][:2]}/{row['date'][2:]}")

        # R7 confidence of every character in the row
        for name, key in ROW_FIELDS:
            low = _low_conf(fields.get(f"r{r}_{name}"))
            if low:
                reasons.append(f"low confidence in {key}: {', '.join(low)}")

        decisions.append({"row": row["row"],
                          "decision": "needs review" if reasons else "auto-post",
                          "reasons": reasons or ["all checks passed"]})
    return decisions


def reference_from_truth(truth):
    """
    Simulated company lookups for a synthetic test log, built from its ground-truth JSON:
    the HR list contains the true employee ID, the schedule the true client for each trip date.
    (In production these would come from the HR system and the scheduling system.)
    """
    schedule = {}
    for r in truth["rows"]:
        schedule.setdefault(r["date"], set()).add(r["client"])
    return {"employees": {truth["employee_id"]}, "schedule": schedule}