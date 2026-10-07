# Mileage Log Reader

ISM 6642 Final Project, Group 5. Reads handwritten Form ML-7 mileage logs and decides
which rows can be auto-posted and which go to a clerk.

## Setup (from a clean clone)

```bash
git clone https://github.com/kaien6462/mileage-log-reader.git
cd mileage-log-reader
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## Run the demo

```bash
python -m streamlit run app.py
```

Upload any Form ML-7 photo (or pick a test log in the sidebar) to see each row read, checked, and marked auto-post or needs review.

## Data pipeline

The `data/` folder is not stored in Git. Recreate it with:

```bash
# 1. Download EMNIST ByClass, fix orientation, keep digits + capitals (36 classes),
#    split train/val, save data/emnist36.npz
python src/emnist_loader.py

# 2. Generate synthetic Form ML-7 logs with ground truth (EMNIST test split only)
python src/make_logs.py --n 50 --quality clean
python src/make_logs.py --n 50 --quality medium
python src/make_logs.py --n 50 --quality phone
```

### Official test set

All accuracy numbers in our reports are measured on this fixed set
(50 logs per quality level, seed 2026, EMNIST test split only):

```bash
python src/make_logs.py --n 50 --quality clean  --seed 2026 --out data/test_official/clean
python src/make_logs.py --n 50 --quality medium --seed 2026 --out data/test_official/medium
python src/make_logs.py --n 50 --quality phone  --seed 2026 --out data/test_official/phone
```

Each log is saved as an image plus a JSON file with the correct answer for every field,
the segmentation hazards in that log, and the image conditions it was made under.
The same `--seed` gives the same trip data at every quality level.

| Quality | What it simulates |
|---|---|
| clean  | flat scan, neat handwriting |
| medium | mild phone-photo effects, some writing over box lines |
| phone  | skewed phone photo, shadow, blur, JPEG, messy handwriting, touching / overflowing digits |

## Files

| File | Purpose |
|---|---|
| `src/explore_emnist.py` | First look at EMNIST; shows the transposed-image issue |
| `src/emnist_loader.py`  | Loads EMNIST, fixes orientation (with automated check), splits data |
| `src/make_logs.py`      | Synthetic mileage-log generator with ground truth |

## Team

| Member | Role |
|---|---|
| Kaien Huang | Scrum Lead, Data Workflow |
| Tenaj Lloyd | Product Lead, Engineer |
| Susan Rollins | QA |