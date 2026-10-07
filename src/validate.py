"""
validate.py - Business rules and the post-or-route decision (SCRUM-16, Susan).

PLACEHOLDER: defines the interface the demo (app.py) and the evaluation script
use. Every row is routed to a clerk until the real rules are written.

    validate(result) -> list of decisions, one per row:
        {"row": 1, "decision": "auto-post" | "needs review", "reasons": ["...", ...]}

    result = the dict returned by read_log.read_log()
             (employee_id, week_ending, total_written, rows=[{row, date, client,
              odo_start, odo_end, miles_written}], fields={...confidence...})

Rules to implement (Form ML-7 spec):
  - every box read (no '_' empty box, no '?')
  - miles = odo_end - odo_start
  - odo_end > odo_start, and odo_start >= previous row's odo_end
  - trip date inside the week; week ending is a Sunday
  - total = sum of the MILES column
  - (optional) low model confidence on any character -> needs review
"""


def validate(result):
    return [{"row": r["row"], "decision": "needs review",
             "reasons": ["validation rules not implemented yet (SCRUM-16)"]}
            for r in result["rows"]]