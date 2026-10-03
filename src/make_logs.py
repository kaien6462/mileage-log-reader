"""
make_logs.py - Synthetic Form ML-7 mileage-log generator with ground truth.

Part 1: random but valid trip data, handwritten EMNIST *test-split* characters
placed into the printed boxes of Form ML-7, saved as a clean scan (PNG) plus a
ground-truth JSON file for every log.

The form layout (page size, box size, box positions) follows Form ML-7 as shown
in the project brief, so our reading pipeline is built against the same form the
instructor will bring to the demo.

Run:
    python src/make_logs.py --n 5 --out data/synthetic/clean
"""
import argparse
import datetime as dt
import json
import os

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

EMNIST_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NPZ_PATH = "data/emnist36.npz"           # produced by src/emnist_loader.py
RATE_PER_MILE = 0.62
LETTERS = EMNIST_MAP[10:36]
DIGITS = EMNIST_MAP[:10]

# Fonts for the printed parts of the form (macOS, Linux, Windows). First one found is used.
FONT_REGULAR = ["/System/Library/Fonts/Supplemental/Arial.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:/Windows/Fonts/arial.ttf"]
FONT_BOLD = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "C:/Windows/Fonts/arialbd.ttf"]


def get_font(size, bold=False):
    """Return a TrueType font from the first path that exists; fall back to Pillow's default."""
    for path in (FONT_BOLD if bold else FONT_REGULAR):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


# --------------------------------------------------------------------------- glyphs
class GlyphBank:
    """
    Handwritten character images, taken ONLY from the EMNIST test split.
    The training split is never loaded here, so no character the model was
    trained on can appear in a synthetic test log.
    """

    def __init__(self, npz_path=NPZ_PATH):
        data = np.load(npz_path)
        x, y = data["x_test"], data["y_test"]          # already orientation-fixed by the loader
        self.bank = {EMNIST_MAP[c]: x[y == c] for c in range(36)}

    def get(self, ch, rng):
        """Random 28x28 image of character ch, as floats in [0, 1] (1 = ink)."""
        imgs = self.bank[ch]
        return imgs[rng.integers(len(imgs))].astype(np.float32) / 255.0


# --------------------------------------------------------------------------- form
class Form:
    """Draws an empty Form ML-7 and remembers where every handwriting box is."""
    W, H = 2200, 1400
    BOX = 62                                    # side length of one character box, in pixels
    LINE = (70, 90, 120)                        # colour of printed box lines

    def __init__(self, n_rows=9):
        self.img = Image.new("RGB", (self.W, self.H), (250, 249, 244))
        self.d = ImageDraw.Draw(self.img)
        self.fields = {}                        # field name -> list of box rectangles
        self.n_rows = n_rows
        self._draw()

    def _boxes(self, name, x, y, n, sep_after=()):
        """Draw n boxes in a row starting at (x, y); a printed '/' follows indices in sep_after."""
        rects = []
        for i in range(n):
            rect = (x, y, x + self.BOX, y + self.BOX)
            self.d.rectangle(rect, outline=self.LINE, width=2)
            rects.append(rect)
            x += self.BOX
            if i in sep_after:
                self.d.text((x + 6, y + 8), "/", font=get_font(40), fill=self.LINE)
                x += 30
        self.fields[name] = rects

    def _draw(self):
        d, f = self.d, get_font
        d.text((80, 50), "SABAL COAST HOME HEALTH", font=f(48, True), fill=(20, 45, 90))
        d.text((80, 112), "Weekly Field Mileage Log  -  Form ML-7", font=f(32), fill=(40, 40, 40))
        d.text((1420, 60), "Submit to Accounts Payable by Monday 10:00", font=f(26), fill=(90, 90, 90))
        d.text((1420, 98), "Reimbursement rate: $0.62 per business mile", font=f(26), fill=(90, 90, 90))

        d.text((80, 206), "EMPLOYEE ID", font=f(28, True), fill=(30, 30, 30))
        self._boxes("emp_id", 290, 190, 6)
        d.text((820, 206), "WEEK ENDING (MM/DD/YY)", font=f(28, True), fill=(30, 30, 30))
        self._boxes("week_ending", 1190, 190, 6, sep_after=(1, 3))
        d.text((80, 285), "Print ONE capital letter or digit per box.  Odometer: whole miles, "
                          "all six digits (include leading zeros).", font=f(24), fill=(90, 90, 90))

        # table header
        ty = 350
        d.rectangle((70, ty, 2130, ty + 50), fill=(225, 231, 240))
        for label, x in [("#", 80), ("DATE (MM/DD)", 170), ("CLIENT", 520),
                         ("ODOMETER START", 760), ("ODOMETER END", 1200), ("MILES", 1640)]:
            d.text((x, ty + 12), label, font=f(24, True), fill=(20, 45, 90))

        # trip rows
        y = ty + 70
        for r in range(self.n_rows):
            d.text((90, y + 14), str(r + 1), font=f(28), fill=(90, 90, 90))
            self._boxes(f"r{r}_date", 170, y, 4, sep_after=(1,))
            self._boxes(f"r{r}_client", 520, y, 3)
            self._boxes(f"r{r}_odo_start", 760, y, 6)
            self._boxes(f"r{r}_odo_end", 1200, y, 6)
            self._boxes(f"r{r}_miles", 1640, y, 3)
            d.line((70, y + self.BOX + 12, 2130, y + self.BOX + 12), fill=(210, 210, 210), width=1)
            y += self.BOX + 24

        d.text((1300, y + 18), "TOTAL MILES", font=f(28, True), fill=(30, 30, 30))
        self._boxes("total_miles", 1640 - self.BOX, y, 4)
        d.text((80, y + 12), "I certify these miles were driven on company business.",
               font=f(24), fill=(90, 90, 90))
        d.text((80, y + 60), "Office use only:   Keyed by ______   Approved ______",
               font=f(24), fill=(150, 150, 150))


# --------------------------------------------------------------------------- trip data
def random_trip_data(rng, slip_rate=0.10):
    """
    Generate one week of valid trip data that obeys every Form ML-7 business rule:
      - employee ID = 2 letters + 4 digits
      - week ending is a Sunday; every trip date falls inside that week
      - odometer runs forward: end > start, start >= previous row's end
      - miles = end - start, total = sum of the MILES column
    With probability slip_rate, the clinician makes one arithmetic slip in the
    MILES column (like row 5 of Figure 1). The TOTAL then matches the clinician's
    own (wrong) column, not the odometers.
    """
    emp_id = "".join(rng.choice(list(LETTERS), 2)) + "".join(rng.choice(list(DIGITS), 4))

    anchor = dt.date(2026, 9, 27)                                  # a Sunday
    week_end = anchor - dt.timedelta(weeks=int(rng.integers(0, 9)))  # within the last ~60 days
    days = sorted(int(d) for d in rng.integers(0, 7, size=int(rng.integers(4, 10))))
    dates = [week_end - dt.timedelta(days=6 - d) for d in days]   # Monday..Sunday of that week

    clients = ["".join(rng.choice(list(LETTERS), 3)) for _ in range(int(rng.integers(3, 7)))]

    odo = int(rng.integers(5_000, 900_000))
    rows = []
    for date in dates:
        odo += int(rng.choice([0, 0, rng.integers(1, 15)]))      # personal miles between trips
        start = odo
        miles = int(rng.integers(3, 86))
        odo = start + miles
        rows.append({"date": date.strftime("%m%d"),
                     "client": str(rng.choice(clients)),
                     "odo_start": f"{start:06d}",
                     "odo_end": f"{odo:06d}",
                     "miles_true": miles,
                     "miles_written": f"{miles:03d}",
                     "clinician_error": False})

    if rng.random() < slip_rate:                                   # one arithmetic slip
        r = rows[int(rng.integers(len(rows)))]
        wrong = r["miles_true"] + int(rng.choice([-1, 1]) * rng.integers(1, 10))
        r["miles_written"] = f"{max(1, wrong):03d}"
        r["clinician_error"] = r["miles_written"] != f"{r['miles_true']:03d}"

    total = sum(int(r["miles_written"]) for r in rows)
    return {"employee_id": emp_id,
            "week_ending": week_end.strftime("%m%d%y"),
            "rows": rows,
            "total_written": f"{total:04d}",
            "total_true": sum(r["miles_true"] for r in rows)}


# --------------------------------------------------------------------------- handwriting
def write_field(form, ink, bank, field, text, rng, messy):
    """
    Place one handwritten glyph per box. 'messy' (0..1) increases size,
    rotation and position jitter. Returns the box of every character written.
    """
    boxes = []
    for i, ch in enumerate(text):
        x0, y0, x1, y1 = form.fields[field][i]
        size = int(form.BOX * rng.uniform(0.86, 1.0 + 0.25 * messy))
        g = cv2.resize(bank.get(ch, rng), (size, size), interpolation=cv2.INTER_CUBIC)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), rng.normal(0, 4 + 6 * messy), 1.0)
        g = np.clip(cv2.warpAffine(g, M, (size, size)), 0, 1)

        cx = (x0 + x1) / 2 + rng.normal(0, 2 + 7 * messy)
        cy = (y0 + y1) / 2 + rng.normal(0, 2 + 7 * messy)
        px, py = max(0, int(cx - size / 2)), max(0, int(cy - size / 2))
        region = ink[py:py + size, px:px + size]
        ink[py:py + size, px:px + size] = np.maximum(region, g[:region.shape[0], :region.shape[1]])
        boxes.append([int(v) for v in (x0, y0, x1, y1)])
    return {"text": text, "boxes": boxes}


def render_log(data, bank, rng, messy=0.1, ink_rgb=(25, 35, 90)):
    """Fill an empty form with the trip data. Returns the page image and per-field ground truth."""
    form = Form()
    ink = np.zeros((form.H, form.W), np.float32)               # 1 = full ink

    fields = {"emp_id": write_field(form, ink, bank, "emp_id", data["employee_id"], rng, messy),
              "week_ending": write_field(form, ink, bank, "week_ending", data["week_ending"], rng, messy)}
    for r, row in enumerate(data["rows"]):
        for key, name in [("date", "date"), ("client", "client"), ("odo_start", "odo_start"),
                          ("odo_end", "odo_end"), ("miles_written", "miles")]:
            fields[f"r{r}_{name}"] = write_field(form, ink, bank, f"r{r}_{name}", row[key], rng, messy)
    fields["total_miles"] = write_field(form, ink, bank, "total_miles", data["total_written"], rng, messy)

    # blend the ink onto the paper
    paper = np.asarray(form.img, np.float32)
    alpha = np.clip(ink * 1.15, 0, 1)[..., None]
    page = paper * (1 - alpha) + np.array(ink_rgb, np.float32) * alpha
    return Image.fromarray(page.astype(np.uint8)), fields


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Generate synthetic Form ML-7 mileage logs.")
    ap.add_argument("--n", type=int, default=5, help="number of logs to generate")
    ap.add_argument("--out", default="data/synthetic/clean", help="output folder")
    ap.add_argument("--seed", type=int, default=0, help="random seed (same seed = same logs)")
    ap.add_argument("--messy", type=float, default=0.1, help="handwriting messiness, 0 to 1")
    ap.add_argument("--slip-rate", type=float, default=0.10,
                    help="share of logs with one clinician arithmetic slip")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bank = GlyphBank()

    for i in range(1, args.n + 1):
        data = random_trip_data(rng, args.slip_rate)
        img, fields = render_log(data, bank, rng, messy=args.messy)
        name = f"log_{i:04d}"
        img.save(os.path.join(args.out, name + ".png"))
        truth = {**data, "fields": fields,
                 "conditions": {"quality": "clean_scan", "messy": args.messy, "seed": args.seed}}
        with open(os.path.join(args.out, name + ".json"), "w") as fh:
            json.dump(truth, fh, indent=1)
        slip = " (contains clinician slip)" if any(r["clinician_error"] for r in data["rows"]) else ""
        print(f"{name}: {data['employee_id']}, {len(data['rows'])} rows, "
              f"total {data['total_written']}{slip}")

    print(f"Saved {args.n} logs to {args.out}")


if __name__ == "__main__":
    main()