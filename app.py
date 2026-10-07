"""
app.py - Live demo for the Form ML-7 mileage-log reader (SCRUM-19).

    streamlit run app.py

Load ANY log image (upload, or a file path - e.g. a photo taken at demo time),
see each pipeline stage, the read for every row, the checks, and the
auto-post / needs-review decision per row.
"""
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from read_log import read_log, register, Classifier, FORM  # noqa: E402
from validate import validate  # noqa: E402
from demo_utils import overlay_boxes, rows_table, low_confidence, field_stages, to_display  # noqa: E402

st.set_page_config(page_title="Mileage Log Reader", layout="wide")
st.title("Mileage Log Reader – Form ML-7")
st.caption("Sabal Coast Home Health · ISM 6642 Group 5 · photo in → rows read → checked → auto-post or clerk")


@st.cache_resource
def load_classifier():
    return Classifier()


# ------------------------------------------------------------------ 1. input
# ------------------------------------------------------------------ 1. input
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES = sorted(
    os.path.relpath(p, APP_DIR)
    for q in ("clean", "medium", "phone")
    for p in __import__("glob").glob(os.path.join(APP_DIR, "data", "test_official", q, "log_000[1-5].*"))
    if not p.endswith(".json")
)

with st.sidebar:
    st.header("1. Load a log")
    upload = st.file_uploader("Upload a photo or scan", type=["png", "jpg", "jpeg"])
    path = st.text_input("…or a file path", placeholder="data/test_official/phone/log_0001.jpg").strip()
    sample = st.selectbox("…or pick a test log", ["(none)"] + SAMPLES)
    st.divider()
    st.markdown("**Legend for the boxes**  \n🟩 written · 🟧 empty box in a used field · ⬜ unused")

full_path = os.path.join(APP_DIR, path) if path and not os.path.isabs(path) else path
if upload is not None:
    img, source = np.asarray(Image.open(upload).convert("RGB")), upload.name
elif path:
    if not os.path.exists(full_path):
        st.error(f"File not found: `{full_path}`")
        st.stop()
    img, source = np.asarray(Image.open(full_path).convert("RGB")), path
elif sample != "(none)":
    img, source = np.asarray(Image.open(os.path.join(APP_DIR, sample)).convert("RGB")), sample
else:
    st.info("Upload a mileage-log image, enter a file path, or pick a test log in the sidebar to start.")
    st.stop()

# ------------------------------------------------------------------ 2. read
with st.spinner("Reading the log…"):
    result = read_log(img, load_classifier())
    decisions = validate(result)
    page = register(img)

n_rows = len(result["rows"])
n_post = sum(d["decision"] == "auto-post" for d in decisions)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Employee ID", result["employee_id"])
c2.metric("Week ending", result["week_ending"])
c3.metric("Total miles (written)", result["total_written"])
c4.metric("Auto-posted rows", f"{n_post} of {n_rows}")

# ------------------------------------------------------------------ 3. stages
st.subheader("Pipeline stages")
left, right = st.columns(2)
left.image(img, caption=f"Input: {source}", use_container_width=True)
right.image(overlay_boxes(page, result), caption="Registered to the form + boxes found", use_container_width=True)

# ------------------------------------------------------------------ 4. rows + decision
st.subheader("Rows read, checked and routed")
st.dataframe(rows_table(result, decisions), use_container_width=True, hide_index=True)

if result["problems"]:
    st.warning("Segmentation problems (counted apart from recognition errors):\n\n- "
               + "\n- ".join(result["problems"]))

unsure = low_confidence(result)
if unsure:
    with st.expander(f"Low-confidence characters ({len(unsure)})"):
        st.table([{"Field": f, "Box": i, "Read as": ch, "Confidence": f"{p:.0%}"} for f, i, ch, p in unsure])

# ------------------------------------------------------------------ 5. look inside one field
with st.expander("Look inside one field: cell crop → normalized 28×28 → read"):
    used = [name for name in FORM.fields if name in result["fields"]]
    field = st.selectbox("Field", used, index=used.index("r0_odo_start") if "r0_odo_start" in used else 0)
    crops, norm = field_stages(img, field)
    f = result["fields"][field]
    cols = st.columns(len(crops))
    for i, col in enumerate(cols):
        col.image(crops[i], caption=f"box {i + 1}", use_container_width=True)
        col.image(to_display(norm[i]), caption=f"read: {f['text'][i]}", use_container_width=True)