# Mileage Log Reader – Form ML-7

**ISM 6642 · Group 5 · Sabal Coast Home Health**

Clinicians hand in a weekly paper mileage log (Form ML-7). Today a clerk keys every log by hand
($3.40 per log). This project reads a **photo** of the form, reads every handwritten box with our own
EMNIST-trained models, checks the business rules, and decides **per row** whether the row can be
**auto-posted** or must go to a clerk, together with the reason.

> No OCR service, Tesseract or language model reads the form: only the two small CNNs we trained on
> EMNIST (digits 0-9, letters A-Z). No real mileage logs are used anywhere, only synthetic logs and forms
> we filled in ourselves with made-up data.

| Team member | Role |
|---|---|
| Kaien Huang | Scrum Lead · data workflow, integration, demo |
| Tenaj Lloyd | Product Lead · models, form reader, cost evaluation |
| Susan Rollins | QA · testing, evaluation, results summary |

---

## How it works

```
photo ─► 1 register ─► 2 cut out boxes ─► 3 normalize ─► 4 read ─► 5 assemble ─► 6 check ─► auto-post / clerk
```

1. **Register** – find the page, remove tilt and perspective, line the printed boxes up with the known
   Form ML-7 layout. Portrait photos are turned automatically; forms printed on letter paper (with margins)
   are aligned on their printed boxes.
2. **Cut out boxes** – one image per box; pen ink is assigned to the box that holds it, so touching
   characters are split along the box border.
3. **Normalize** – make each box look like an EMNIST training image (white on black, bounding box
   centred, longest side 24 px in a 28×28 square).
4. **Read** – digit boxes go to the digit CNN, letter boxes to the letter CNN; each character gets a
   confidence.
5. **Assemble** – characters → fields → rows (date, client, odometer start/end, miles).
6. **Check** (`src/validate.py`) – a row is auto-posted only if **every** check passes:
   - log level: employee ID is 2 letters + 4 digits and on the HR list; week ending is a Sunday;
     total = sum of the MILES column; header read with ≥ 80 % confidence
   - row level: every box read; date inside the week; odometer end > start; start ≥ previous row's end;
     miles = end − start; client on the clinician's visit schedule; every character ≥ 80 % confidence

---

## Setup (macOS / Linux)

```bash
git clone https://github.com/kaien6462/mileage-log-reader.git
cd mileage-log-reader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The trained models are already in `models/` (`digit_model.pt`, `letter_model.pt`), so you do **not**
need to train anything to run the demo.

> If you use Anaconda and see `(base)` in your prompt, run `conda deactivate` first, and always start
> the demo with `python -m streamlit …` so the virtual environment's Streamlit is used.

---

## Run the demo

```bash
python -m streamlit run app.py
```

Open http://localhost:8501, then in the sidebar either upload a photo, type a file path, or pick a
test log. Every section of the page has a short explanation underneath.

Good demo cases (real handwritten forms, see below):

| Log | What it shows |
|---|---|
| `data/handwritten/hw_03.jpg` | clinician's own slip on row 2 (miles 073, odometers say 23) is routed by the odometer check |
| `data/handwritten/hw_01.jpg` | employee ID misread (H→W, 37 % confidence) – the HR list holds back the whole log |
| `data/handwritten/hw_02.jpg` | rows 1–4 auto-posted; row 5 client misread (K→L) is caught by the schedule |

Read one image from the command line instead:

```bash
python src/read_log.py data/handwritten/hw_02.jpg
```

---

## Data

`data/` is **not** in git (too large). Recreate it like this:

```bash
python src/emnist_loader.py        # EMNIST → data/emnist36.npz (36 classes, orientation fixed, train/val/test)

# official test set: 50 logs × 3 image qualities, built ONLY from the EMNIST test split, seed 2026
python src/make_logs.py --n 50 --quality clean  --seed 2026 --out data/test_official/clean
python src/make_logs.py --n 50 --quality medium --seed 2026 --out data/test_official/medium
python src/make_logs.py --n 50 --quality phone  --seed 2026 --out data/test_official/phone

python src/make_blank_form.py      # printable blank Form ML-7 (PDF) for handwritten tests
```

**Real handwritten test forms** (`data/handwritten/hw_01 … hw_06.jpg` + answer keys `.json`) are shared
by the team separately (`handwritten_forms.zip`); unzip into `data/`. They were filled in by hand with
made-up data, printed on letter paper and photographed with a phone. iPhone HEIC photos must be converted
first: `sips -s format jpeg photo.heic --out data/handwritten/hw_07.jpg`.

---

## Tests and evaluation

| Command | What it measures | Output |
|---|---|---|
| `python tests/test_validate.py` | every business rule on hand-made cases + all clean test logs read perfectly | prints PASS / FAIL |
| `python tests/check_pipeline.py --out results/results_pipeline.txt` | end to end (read → validate) on the official test set: segmentation, field accuracy, auto-post rate, wrong rows posted | `results/results_pipeline.txt` |
| `python src/check_reader.py data/test_official` | recognition per condition: digits, letters, each field type, touching / overflow boxes | `results/results_official.txt` |
| `python src/evaluate_policy.py data/test_official` | post-or-route policy at several confidence thresholds, with and without lookups, monthly cost | `results/results_policy.txt` |
| `python tests/check_handwritten.py --out results/results_handwritten.txt` | the 6 real handwritten forms against their answer keys | `results/results_handwritten.txt` |

---

## Results (details in `results/`)

**Synthetic official test set** (150 logs, 978 rows; lookups simulated, 80 % threshold)

| | clean | medium | phone |
|---|---|---|---|
| digit accuracy | 99.4 % | 99.0 % | 97.6 % |
| letter accuracy | 95.6 % | 95.2 % | 90.7 % |
| rows auto-posted | 50.0 % | 25.5 % | 12.0 % |
| **wrong rows auto-posted** | **0 / 163** | **0 / 83** | **0 / 39** |

**Real handwritten forms** (6 forms, 843 characters): digits **99.6 %** (727/730), letters **94.7 %**
(107/113), all boxes found. 10 of 34 rows auto-posted, all correct; each of the 9 misreads was caught by
the HR list (1), the schedule (5) or the odometer check (3).

Without the HR-list and schedule lookups, 7–22 % of auto-posted rows were wrong (misread letters pass
every arithmetic check), so **the lookups are required** for auto-posting.

---

## Scope and limitations

- One capital letter or digit per box; the whole form visible in the photo; written in **pen**.
- Not handled: crossed-out corrections, two characters in one box, faint pencil.
- HR list and visit schedule are simulated from the answer keys; in production they come from the HR and
  scheduling systems.
- Synthetic test logs use EMNIST characters; the real-handwriting test is 6 forms from one writer –
  a pilot on real logs is needed before production.

---

## Repository layout

```
app.py                     Streamlit demo
src/
  read_log.py              end-to-end reader used by the demo and tests (registration, assembly)
  read_form.py             form reader: page registration, cell extraction, normalization, CNN classifier
  validate.py              business rules and the auto-post / needs-review decision
  demo_utils.py            helpers for the demo (box overlay, low-confidence list, cell inspector)
  train_classifiers.py     trains the digit and letter CNNs → models/
  evaluate_classifiers.py  classifier accuracy on the EMNIST test split
  check_reader.py          recognition evaluation on the official test set
  evaluate_policy.py       policy and cost evaluation
  emnist_loader.py         EMNIST loading, orientation fix, 36 classes, splits
  dataset.py               PyTorch data loaders (balanced sampling, augmentation)
  make_logs.py             synthetic Form ML-7 generator with ground truth
  make_blank_form.py       printable blank form
  explore_emnist.py        EMNIST orientation exploration
  normalize.py             first normalization version (kept for reference; replaced by read_form)
tests/                     test_validate, check_pipeline, check_handwritten, check_normalize
models/                    digit_model.pt, letter_model.pt
notebooks/                 single-character error spike (Susan)
results/                   saved evaluation outputs
```

## Workflow

Every change goes through a branch → pull request → review by a teammate → merge, as agreed in our team
contract. Stories are tracked in JIRA (SCRUM-xx); each PR names its story.
