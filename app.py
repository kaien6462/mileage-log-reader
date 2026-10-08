"""
app.py - Live demo for the Form ML-7 mileage-log reader (SCRUM-19).

    python -m streamlit run app.py

Load a log image (upload, file path, or pick a test log), then see every pipeline stage,
the read of every row, the checks, and the auto-post / needs-review decision per row.
Every section has a short explanation underneath so the audience can follow along.
"""
import glob
import json
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "src"))
from read_log import read_log, register, Classifier, FORM  # noqa: E402
from validate import validate, reference_from_truth, CONF_MIN  # noqa: E402
from demo_utils import overlay_boxes, low_confidence, field_stages, to_display  # noqa: E402

st.set_page_config(page_title="Mileage Log Reader", layout="wide")


def explain(text):
    """Grey explanation text under a section."""
    st.caption(text)


FIELD_LABELS = {"emp_id": "Employee ID", "week_ending": "Week ending", "total_miles": "Total miles",
                "date": "Date", "client": "Client", "odo_start": "Odometer start",
                "odo_end": "Odometer end", "miles": "Miles"}


def pretty(field):
    """Internal field name -> what a person calls it: 'r5_client' -> 'Row 6 · Client'."""
    if field in FIELD_LABELS:
        return FIELD_LABELS[field]
    row, name = field.split("_", 1)
    return f"Row {int(row[1:]) + 1} · {FIELD_LABELS.get(name, name)}"


def mark(text, i):
    """Show which character of a field is meant: ('POL', 1) -> 'P[O]L'."""
    return text[:i] + "[" + text[i] + "]" + text[i + 1:]


@st.cache_resource
def load_classifier():
    return Classifier()


def test_logs():
    """Images that have an answer key (.json) next to them, relative to the app folder."""
    found = []
    for pattern in ("data/handwritten/*.jpg", "data/handwritten/*.png",
                    "data/test_official/*/log_000[1-5].png", "data/test_official/*/log_000[1-5].jpg"):
        for p in sorted(glob.glob(os.path.join(APP_DIR, pattern))):
            if os.path.exists(os.path.splitext(p)[0] + ".json"):
                found.append(os.path.relpath(p, APP_DIR))
    return found


# ======================================================================= header
st.title("Mileage Log Reader – Form ML-7")
st.markdown("**Sabal Coast Home Health · ISM 6642 Group 5**  \n"
            "Photo of a handwritten mileage log in → every box read → business rules checked → "
            "each row is **auto-posted** to payroll or **sent to a clerk** with the reason.")
explain("The reader only uses models we trained ourselves on EMNIST handwritten characters "
        "(one CNN for digits 0-9, one for letters A-Z). No OCR service or language model reads the form.")

# ======================================================================= 1. input
with st.sidebar:
    st.header("1. Load a log")
    upload = st.file_uploader("Upload a photo or scan", type=["png", "jpg", "jpeg"])
    path = st.text_input("…or a file path", placeholder="data/handwritten/hw_03.jpg")
    samples = ["(none)"] + test_logs()
    sample = st.selectbox("…or pick a test log", samples)
    explain("Test logs have an answer key next to them. For those, the demo also simulates the "
            "company's HR employee list and the clinician's visit schedule, so the lookup checks run. "
            "data/handwritten = real photos of forms we filled in by hand; "
            "data/test_official = synthetic logs built from unseen EMNIST characters.")
    st.divider()
    st.markdown("**Box colours in the overlay**  \n🟩 box read as written  \n"
                "🟧 empty box inside a used field (segmentation problem)  \n⬜ unused box")

src_path = None
if upload is not None:
    img, source = np.asarray(ImageOps.exif_transpose(Image.open(upload)).convert("RGB")), upload.name
elif path:
    full_path = path if os.path.isabs(path) else os.path.join(APP_DIR, path)
    if not os.path.exists(full_path):
        st.error(f"File not found: {path}  (paths are relative to the project folder)")
        st.stop()
    img, source, src_path = np.asarray(ImageOps.exif_transpose(Image.open(full_path)).convert("RGB")), path, full_path
elif sample != "(none)":
    src_path = os.path.join(APP_DIR, sample)
    img, source = np.asarray(ImageOps.exif_transpose(Image.open(src_path)).convert("RGB")), sample
else:
    st.info("Upload a mileage-log photo, type a file path, or pick a test log in the sidebar to start.")
    st.stop()

truth_path = os.path.splitext(src_path)[0] + ".json" if src_path else None
reference = (reference_from_truth(json.load(open(truth_path)))
             if truth_path and os.path.exists(truth_path) else None)

# ======================================================================= 2. read + check
with st.spinner("Reading the log…"):
    result = read_log(img, load_classifier())
    decisions = validate(result, reference)
    page = register(img)

n_rows = len(result["rows"])
n_post = sum(d["decision"] == "auto-post" for d in decisions)

st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Employee ID", result["employee_id"])
c2.metric("Week ending", result["week_ending"])
c3.metric("Total miles (written)", result["total_written"])
c4.metric("Auto-posted rows", f"{n_post} of {n_rows}")
explain("What the reader found in the header boxes, and how many trip rows passed every check and can "
        "be paid without a clerk. Rows that fail any check are not rejected: they go to a clerk "
        "together with the reason.")
if reference is None:
    st.info("No answer key next to this image, so the HR-list and visit-schedule lookups are skipped. "
            "Arithmetic, date and confidence checks still run.")

# ======================================================================= 3. pipeline stages
st.subheader("Pipeline stages")
left, right = st.columns(2)
left.image(img, caption=f"Input: {source}", use_container_width=True)
right.image(overlay_boxes(page, result), caption="Registered to the form + boxes found", use_container_width=True)
reg = result.get("registration", {})
how = []
if reg.get("rotated"):
    how.append(f"the photo was turned {reg['rotated']}")
if reg.get("printed_form_alignment"):
    how.append("the form was printed on paper with margins, so the printed boxes themselves were "
               "located and a perspective correction was fitted to them")
explain("**Registration:** the reader finds the page, flattens the photo (removes tilt and perspective) and "
        "lines the printed boxes up with the known Form ML-7 layout, so every box sits at a known position. "
        + (f"For this image, {'; '.join(how)}. " if how else "")
        + f"Alignment score: {reg.get('fit', 'n/a')} (share of printed box outlines that land on ink). "
        "**Cell extraction:** each box is cut out with a small margin, and only the pen ink that belongs to "
        "that box is kept, so strokes that touch a neighbouring box are split along the box border.")

# ======================================================================= 4. log-level checks
st.subheader("Log-level checks")
log_reasons = []
for d in decisions:
    for r in d["reasons"]:
        if r.startswith("log: ") and r[5:] not in log_reasons:
            log_reasons.append(r[5:])
if log_reasons:
    st.warning("**These problems affect the whole log, so every row is sent to a clerk:**\n\n- "
               + "\n- ".join(log_reasons))
else:
    st.success("Header checks passed: employee ID, week ending and total are all fine.")
explain("Checked once per log: the employee ID is 2 letters + 4 digits and is on the HR list; "
        "the week ending is a real date and a Sunday; the total equals the sum of the MILES column; "
        f"and every header character was read with confidence of at least {CONF_MIN:.0%}. "
        "A problem here means the whole log cannot be trusted, so no row is auto-posted.")

# ======================================================================= 5. rows
st.subheader("Rows read, checked and routed")
by_row = {d["row"]: d for d in decisions}
table = []
for r in result["rows"]:
    d = by_row.get(r["row"], {"decision": "needs review", "reasons": ["no decision"]})
    own = [x for x in d["reasons"] if not x.startswith("log: ") and x != "all checks passed"]
    if own:
        why = "; ".join(own)
    elif d["decision"] == "auto-post":
        why = "all checks passed"
    else:
        why = "row itself is fine – held back by the log-level problems above"
    table.append({"#": r["row"], "Date": r["date"], "Client": r["client"],
                  "Odometer start": r["odo_start"], "Odometer end": r["odo_end"],
                  "Miles": r["miles_written"],
                  "Decision": "✅ auto-post" if d["decision"] == "auto-post" else "⚠️ needs review",
                  "Reasons for this row": why})
st.dataframe(table, use_container_width=True, hide_index=True)
explain("Each trip row is checked on its own: every box was read; the trip date falls inside the week; "
        "odometer end is greater than start; odometer start is not below the previous row's end; "
        "miles written = end − start; the client code is on the clinician's visit schedule for that date; "
        f"and every character in the row was read with at least {CONF_MIN:.0%} confidence. "
        "A row is auto-posted only if all of these pass and the log-level checks pass. "
        "A misread and a clinician's own mistake look the same to the rules, and both need a person.")

if result["problems"]:
    st.warning("**Segmentation problems** (counted separately from recognition errors):\n\n- "
               + "\n- ".join(result["problems"]))
    explain("A used field with an empty box usually means a character was missed when the box was cut out, "
            "not that the model misread it. The assignment asks us to report these separately.")

# ======================================================================= 6. low confidence
unsure = low_confidence(result)
with st.expander(f"Low-confidence characters ({len(unsure)})", expanded=bool(unsure)):
    if unsure:
        st.table([{"Where on the form": pretty(f), "Field as read": mark(result["fields"][f]["text"], i - 1),
                   "Character": f"#{i} from the left", "Read as": ch, "Confidence": f"{p:.0%}"}
                  for f, i, ch, p in unsure])
        st.markdown("**What the model saw for each of these characters** (photo → model input):")
        shown = unsure[:8]
        cols = st.columns(max(len(shown), 6))          # at least 6 slots, so a few images stay small
        stages = {}
        for col, (f, i, ch, p) in zip(cols, shown):
            if f not in stages:
                stages[f] = field_stages(img, f)
            crops, norm = stages[f]
            col.image(crops[i - 1], use_container_width=True)
            col.image(to_display(norm[i - 1]), caption=f"{pretty(f)}, #{i}: read {ch} ({p:.0%})",
                      use_container_width=True)
    else:
        st.write("Every character was read with high confidence.")
    explain("'Field as read' shows the whole field with the doubtful character in [brackets]; "
            "'#2 from the left' means the second box of that field. "
            f"Confidence is the model's own probability for the character it chose. Below {CONF_MIN:.0%} "
            "the row (or, for header boxes, the whole log) goes to a clerk. On our tests most misreads had low "
            "confidence, but some correct characters do too: this threshold trades clerk time against safety.")

# ======================================================================= 7. inside one field
with st.expander("Look inside one field: cell crop → normalized 28×28 → read"):
    used = [name for name in FORM.fields if name in result["fields"]]
    field = st.selectbox("Field", used, index=used.index("r0_odo_start") if "r0_odo_start" in used else 0,
                         format_func=lambda f: f"{pretty(f)}  –  read: {result['fields'][f]['text']}")
    crops, norm = field_stages(img, field)
    f = result["fields"][field]
    cols = st.columns(len(crops))
    for i, col in enumerate(cols):
        conf = f["confidence"][i]
        col.image(crops[i], caption=f"box {i + 1}", use_container_width=True)
        col.image(to_display(norm[i]),
                  caption=f"read: {f['text'][i]}" + (f" ({conf:.0%})" if conf is not None else ""),
                  use_container_width=True)
    explain("Top row: the box as photographed, with a margin (bits of neighbouring boxes can be visible). "
            "Bottom row: what the model actually sees – only this box's ink, white on black, scaled so the "
            "longest side is 24 pixels and centred in a 28×28 square, exactly like the EMNIST training images. "
            "Digit boxes go to the digit model, letter boxes to the letter model, so a digit box can never be "
            "read as a letter.")

# ======================================================================= footer
st.divider()
explain("**Scope:** one capital letter or digit per box, the whole form visible in the photo, written in pen. "
        "Not handled: crossed-out corrections, two characters in one box, faint pencil. "
        "Lookups (HR list, visit schedule) are simulated from the answer key for the demo; in production they "
        "would come from the HR and scheduling systems.")
