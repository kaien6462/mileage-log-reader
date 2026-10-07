"""
make_blank_form.py - Printable blank Form ML-7 + an empty ground-truth JSON
template for hand-filled test logs (SCRUM-25).

The blank form is drawn by the same Form class the generator uses, so the box
positions on paper are exactly the ones our pipeline expects.

Run:
    python src/make_blank_form.py
Creates:
    data/handwritten/blank_form_ML7.pdf      (print landscape, "fit to page")
    data/handwritten/TEMPLATE.json           (copy once per photo, fill in the answers)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_logs import Form  # noqa: E402

OUT = os.path.join("data", "handwritten")


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. printable blank form (2200x1400 px at 200 dpi = 11 x 7 in, fits US Letter landscape)
    Form().img.save(os.path.join(OUT, "blank_form_ML7.pdf"), resolution=200)

    # 2. ground-truth template in the same format as the generator's JSON
    template = {
        "employee_id": "AB1234",
        "week_ending": "092726",
        "rows": [
            {"date": "0921", "client": "KMR", "odo_start": "048213",
             "odo_end": "048238", "miles_written": "025"},
            {"date": "0921", "client": "PDL", "odo_start": "048238",
             "odo_end": "048251", "miles_written": "013"},
        ],
        "total_written": "0038",
        "hazards": {},
        "conditions": {"quality": "handwritten", "writer": "KH", "phone": "iPhone",
                       "notes": "e.g. shadow on right edge, row 2 odometer written over the box line"},
    }
    with open(os.path.join(OUT, "TEMPLATE.json"), "w") as fh:
        json.dump(template, fh, indent=1)

    print(f"Saved {OUT}/blank_form_ML7.pdf and {OUT}/TEMPLATE.json")


if __name__ == "__main__":
    main()