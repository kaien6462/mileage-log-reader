"""
demo_utils.py - Helpers for the demo app (app.py), kept separate so they can be
tested without Streamlit.
"""
import numpy as np
from PIL import Image, ImageDraw

from read_log import FORM, PAD, extract_cells, register
from normalize import normalize_cell

LOW_CONFIDENCE = 0.80        # characters below this are highlighted in the demo


def overlay_boxes(page, result):
    """
    Draw every box of the registered page in a colour that tells what the reader saw:
      green  = written box in a field the reader used
      orange = a used field with an empty box (segmentation problem)
      grey   = unused box
    """
    img = Image.fromarray(page).convert("RGB")
    d = ImageDraw.Draw(img)
    for name, boxes in FORM.fields.items():
        f = result["fields"].get(name)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            if f is None:
                colour, w = (170, 170, 170), 1
            elif f["filled"][i]:
                colour, w = (30, 160, 70), 3
            else:
                colour, w = (235, 130, 20), 4
            d.rectangle((x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD), outline=colour, width=w)
    return img


def rows_table(result, decisions):
    """One dict per row for the results table: the read fields + the post-or-route decision."""
    by_row = {d["row"]: d for d in decisions}
    table = []
    for r in result["rows"]:
        d = by_row.get(r["row"], {"decision": "needs review", "reasons": ["no decision"]})
        table.append({"#": r["row"], "Date": r["date"], "Client": r["client"],
                      "Odometer start": r["odo_start"], "Odometer end": r["odo_end"],
                      "Miles": r["miles_written"],
                      "Decision": ("✅ auto-post" if d["decision"] == "auto-post" else "⚠️ needs review"),
                      "Reasons": "; ".join(d["reasons"])})
    return table


def low_confidence(result, threshold=LOW_CONFIDENCE):
    """List of (field, box index, character, confidence) the model was unsure about."""
    out = []
    for name, f in result["fields"].items():
        for i, (ch, p) in enumerate(zip(f["text"], f["confidence"])):
            if p is not None and p < threshold:
                out.append((name, i + 1, ch, p))
    return out


def field_stages(img, field):
    """For one field: the raw crops and their normalized 28x28 versions (None = empty box)."""
    crops = extract_cells(register(img))[field]
    return crops, [normalize_cell(c) for c in crops]


def to_display(img28, scale=4):
    """28x28 float -> enlarged uint8 image for display (black square if empty)."""
    a = np.zeros((28, 28), np.uint8) if img28 is None else (img28 * 255).astype(np.uint8)
    return np.kron(a, np.ones((scale, scale), np.uint8))