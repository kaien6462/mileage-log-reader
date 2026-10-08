"""
read_form.py -- SCRUM-15: photo of a Form ML-7 in, fields out.

Pipeline (each stage is one function, so you can show them in the demo):
  1. register_page   find the 4 corners of the paper, flatten the photo to 2200x1400,
                     then nudge it so the printed boxes line up with the known layout
  2. extract_cells   cut one small image per printed box
  3. normalize_cell  make each crop look like an EMNIST training image
                     (white ink on black, longest side 24 px, bounding box centred in 28x28)
  4. classify        digit boxes -> digit model, letter boxes -> letter model
  5. assemble        join the characters of each box group into a field

Run:
  python read_form.py photo.jpg --digit-model digit_model.pt --letter-model letter_model.pt
  python read_form.py photo.jpg ... --debug debug_out      (also saves stage pictures)

Scope (say this in the approach document):
  * the whole page is visible, roughly upright (a few degrees of tilt, perspective is fine)
  * one character per box, written inside or close to the box
  * crossed-out corrections and two characters in one box are NOT handled
"""
import argparse
import json
import os

import cv2
import numpy as np

W, H, BOX = 2200, 1400, 62          # registered page size and box size (same as the form generator)
DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
N_ROWS = 9


# ======================================================================= layout
def layout(n_rows=N_ROWS):
    """Box rectangles (x0, y0, x1, y1) for every field on Form ML-7, in registered-page pixels."""
    fields = {}

    def boxes(name, x, y, n, sep_after=()):
        rects = []
        for i in range(n):
            rects.append((x, y, x + BOX, y + BOX))
            x += BOX
            if i in sep_after:
                x += 30                      # the printed "/" between date parts
        fields[name] = rects

    boxes("emp_id", 290, 190, 6)
    boxes("week_ending", 1190, 190, 6, (1, 3))
    y = 420
    for r in range(n_rows):
        boxes(f"r{r}_date", 170, y, 4, (1,))
        boxes(f"r{r}_client", 520, y, 3)
        boxes(f"r{r}_odo_start", 760, y, 6)
        boxes(f"r{r}_odo_end", 1200, y, 6)
        boxes(f"r{r}_miles", 1640, y, 3)
        y += BOX + 24
    boxes("total_miles", 1640 - BOX, y, 4)
    return fields


def kinds_for(field):
    """'D' = digit box, 'L' = letter box, one per box in the field."""
    if field == "emp_id":
        return "LLDDDD"                      # 2 letters + 4 digits
    if field.endswith("_client"):
        return "LLL"
    return "D" * len(layout()[field])        # dates, odometers, miles, total: digits only


# ============================================================ 1. page registration
def _order_corners(pts):
    pts = np.array(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)  # tl tr br bl


def find_page_corners(gray):
    """Return (corners, found). corners are tl, tr, br, bl in image pixels."""
    h, w = gray.shape
    full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    scale = 800.0 / w
    small = cv2.resize(gray, (800, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2)
    ring = np.concatenate([small[:6].ravel(), small[-6:].ravel(), small[:, :6].ravel(), small[:, -6:].ravel()])
    bg, paper = np.percentile(ring, 10), np.percentile(small, 95)
    if bg < 0.6 * paper:
        thr = bg + 0.25 * (paper - bg)                     # background visible around the page: cut low, so a shadowed edge stays paper
    else:
        thr, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # page fills the photo
    mask = ((small > thr) * 255).astype(np.uint8)
    pad = 10
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return full, False
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.4 * small.shape[0] * small.shape[1]:
        return full, False                   # no clear page: assume the photo IS the page
    peri = cv2.arcLength(c, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            quad = approx
            break
    pts = quad.reshape(4, 2) if quad is not None else cv2.boxPoints(cv2.minAreaRect(c))
    pts = (np.array(pts, np.float32) - pad) / scale
    return _order_corners(pts), True


def ink_map(gray):
    """Float image, 1 = dark ink/lines, 0 = paper. Divides out uneven lighting and shadows."""
    g = gray.astype(np.float32)
    bg = cv2.GaussianBlur(cv2.dilate(g, np.ones((31, 31), np.uint8)), (0, 0), 21)
    return np.clip(1.0 - g / np.maximum(bg, 1.0), 0, 1)


_TEMPLATE = None


def _outline_template():
    global _TEMPLATE
    if _TEMPLATE is None:
        t = np.zeros((H, W), np.float32)
        for rects in layout().values():
            for (x0, y0, x1, y1) in rects:
                cv2.rectangle(t, (x0, y0), (x1, y1), 1.0, 3)
        _TEMPLATE = t
    return _TEMPLATE


def _best_shift(ink, template, search, scale):
    """How far (dx, dy, in page pixels) the printed boxes in `ink` sit from the template."""
    if scale != 1.0:
        a = cv2.resize(ink, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        a, t = ink, template
    p = int(round(search * scale))
    a = cv2.copyMakeBorder(a, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    res = cv2.matchTemplate(a, t, cv2.TM_CCORR_NORMED)
    _, _, _, (mx, my) = cv2.minMaxLoc(res)
    return (mx - p) / scale, (my - p) / scale


def _local_homography(ink, template, nx=3, ny=2, search=6):
    """Measure how far the printed boxes sit from the layout in 6 regions of the page and fit a small
    perspective correction through those 6 measurements. Fixes drift that one global shift cannot."""
    x0, x1, y0, y1 = 150, 1850, 170, 1290
    xs = np.linspace(x0, x1, nx + 1).astype(int)
    ys = np.linspace(y0, y1, ny + 1).astype(int)
    src, dst = [], []
    for j in range(ny):
        for i in range(nx):
            ta = template[ys[j]:ys[j + 1], xs[i]:xs[i + 1]]
            ia = ink[ys[j] - search:ys[j + 1] + search, xs[i] - search:xs[i + 1] + search]
            res = cv2.matchTemplate(ia, ta, cv2.TM_CCORR_NORMED)
            _, _, _, (mx, my) = cv2.minMaxLoc(res)
            c = np.array([(xs[i] + xs[i + 1]) / 2.0, (ys[j] + ys[j + 1]) / 2.0])
            dst.append(c)
            src.append(c + [mx - search, my - search])        # where the template's box pattern really is
    M, _ = cv2.findHomography(np.array(src, np.float32), np.array(dst, np.float32), 0)
    return M


def register_page(img):
    """Photo (BGR or gray) -> flat 2200x1400 gray page aligned to the form layout.
    Returns (page_gray, info)."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, found = find_page_corners(gray)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    page = cv2.warpPerspective(gray, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)

    t = _outline_template()
    dx, dy = _best_shift(ink_map(page), t, search=16, scale=0.5)    # coarse: one global shift
    page = cv2.warpAffine(page, np.float32([[1, 0, -dx], [0, 1, -dy]]), (W, H),
                          flags=cv2.INTER_LINEAR, borderValue=255)
    for _ in range(2):                                              # fine: local drift, two passes
        Mh = _local_homography(ink_map(page), t)
        if Mh is None:
            break
        page = cv2.warpPerspective(page, Mh, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)
    return page, {"page_found": bool(found), "coarse_shift_px": (float(dx), float(dy))}


# ============================================================ 2. cell extraction
PEN_LEVEL = 0.40          # ink darker than this counts as pen (lines/shadows/coffee rings are fainter)
WINDOW = 20               # each cell image is the box plus this many pixels around it (room for overflow)
_BOXMAP = None


def _box_index():
    """Page-sized map: which box (index into the flat box list) owns each pixel, -1 = none."""
    global _BOXMAP
    if _BOXMAP is None:
        m = np.full((H, W), -1, np.int16)
        i = 0
        for rects in layout().values():
            for (x0, y0, x1, y1) in rects:
                m[y0:y1, x0:x1] = i
                i += 1
        _BOXMAP = m
    return _BOXMAP


def pen_mask(ink):
    """Binary mask of handwriting only. Near a printed box line, ink only counts if it is clearly darker
    than the printed line itself (pen is darker than the form's printing), so lines vanish but strokes
    that cross them survive. The line darkness is measured on this page, so faint photos work too."""
    on_line = _outline_template() > 0.5
    band = cv2.dilate(on_line.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    line_level = float(np.percentile(ink[on_line], 90))          # upper end of the printed lines (shared dividers are the darkest)
    pen = ink[ink > PEN_LEVEL]
    pen_level = float(np.percentile(pen, 95)) if pen.size else 1.0   # nothing dark on the page at all   # how dark the pen is on this page
    band_thr = max(PEN_LEVEL, line_level + 0.5 * (pen_level - line_level))
    return np.where(band, ink > band_thr, ink > PEN_LEVEL).astype(np.uint8)


def assign_ink_to_boxes(pen):
    """Give every blob of pen ink to the box that holds most of it. A blob spread over several boxes
    (two digits touching) is cut along the box borders. Returns a page-sized array of box indices."""
    boxmap = _box_index()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pen, connectivity=8)
    assign = np.full((H, W), -1, np.int16)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_BLOB:
            continue
        sub = lab[y:y + h, x:x + w] == i
        ids = boxmap[y:y + h, x:x + w][sub]
        ids = ids[ids >= 0]
        if ids.size == 0:
            continue                                           # printed text, not inside any box
        counts = np.bincount(ids)
        top = counts.argmax()
        region = assign[y:y + h, x:x + w]
        if counts[top] >= 0.8 * counts.sum():
            region[sub] = top                                  # whole blob belongs to one box (overflow included)
        else:
            bm = boxmap[y:y + h, x:x + w]
            region[sub] = bm[sub]                              # touching neighbours: split at the box borders
    return assign


def extract_cells(page):
    """-> {field: [window, window, ...]}; each window is a float ink image (1 = ink) of one box
    plus a margin, containing only the handwriting that belongs to that box."""
    ink = ink_map(page)
    assign = assign_ink_to_boxes(pen_mask(ink))
    cells, i = {}, 0
    for name, rects in layout().items():
        crops = []
        for (x0, y0, x1, y1) in rects:
            ya, yb, xa, xb = y0 - WINDOW, y1 + WINDOW, x0 - WINDOW, x1 + WINDOW
            mine = (assign[ya:yb, xa:xb] == i).astype(np.uint8)
            mine = cv2.dilate(mine, np.ones((3, 3), np.uint8))   # keep the soft anti-aliased edge of the strokes
            crops.append(ink[ya:yb, xa:xb] * mine)
            i += 1
        cells[name] = crops
    return cells


# ============================================================ 3. normalization
MIN_BLOB = 15             # ignore specks smaller than this many pixels
MIN_INK_PIXELS = 40       # a box needs at least this many pen pixels to count as written


CHAR_SIDE = 24            # EMNIST characters are about 24 px on their longest side (measured on the data)


def normalize_cell(crop):
    """Window (1 = ink, handwriting of ONE box only) -> 28x28 float32 in the EMNIST style, or None if empty.
    EMNIST convention (measured on emnist36.npz): crop to the bounding box, scale so the longest side is
    ~24 px keeping the aspect ratio, centre the BOUNDING BOX in 28x28. (Not centre of mass, not 20 px:
    both of those cost ~25 points of accuracy on digits like 7 and 9.)
    THIS FUNCTION MUST DO THE SAME THING TO EVERY CELL AT TEST TIME AS EMNIST DID TO TRAINING IMAGES."""
    if (crop > PEN_LEVEL).sum() < MIN_INK_PIXELS:
        return None
    c = crop / crop.max()                                    # strongest ink = 1.0, like EMNIST
    ys, xs = np.nonzero(c > 0.15)
    c = c[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = c.shape
    s = CHAR_SIDE / max(hh, ww)
    c = cv2.resize(c, (max(1, int(round(ww * s))), max(1, int(round(hh * s)))), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.float32)
    hh, ww = c.shape
    y0, x0 = (28 - hh) // 2, (28 - ww) // 2
    out[y0:y0 + hh, x0:x0 + ww] = c
    return out


# ============================================================ 4. classification
class TorchClassifier:
    """Loads digit_model.pt and letter_model.pt made by train_classifiers.py."""

    def __init__(self, digit_path, letter_path):
        import torch
        from train_classifiers import SmallCNN
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        for kind, path, n in (("D", digit_path, 10), ("L", letter_path, 26)):
            m = SmallCNN(n).to(self.dev)
            m.load_state_dict(torch.load(path, map_location=self.dev))
            m.eval()
            self.models[kind] = m

    def predict(self, cells, kind):
        """cells: (N, 28, 28) float 0..1  ->  (indices, confidences)"""
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(cells).float().unsqueeze(1).to(self.dev)
            p = torch.softmax(self.models[kind](x), dim=1)
            conf, idx = p.max(1)
        return idx.cpu().numpy(), conf.cpu().numpy()


def classify_cells(norm, clf):
    """norm: {field: [28x28 or None]} -> {field: [(char, confidence) or None]}"""
    out = {f: [None] * len(v) for f, v in norm.items()}
    for kind, alphabet in (("D", DIGITS), ("L", LETTERS)):
        keys, imgs = [], []
        for f, cells in norm.items():
            ks = kinds_for(f)
            for i, c in enumerate(cells):
                if c is not None and ks[i] == kind:
                    keys.append((f, i))
                    imgs.append(c)
        if imgs:
            idx, conf = clf.predict(np.stack(imgs), kind)
            for (f, i), j, p in zip(keys, idx, conf):
                out[f][i] = (alphabet[int(j)], float(p))
    return out


# ============================================================ 5. field assembly
def _field(reads):
    """list of (char, conf) / None -> (text, min_conf, n_empty). Empty boxes show as '?'."""
    text = "".join(r[0] if r else "?" for r in reads)
    confs = [r[1] for r in reads if r]
    return text, (min(confs) if confs else 0.0), sum(r is None for r in reads)


def assemble(reads):
    res = {"rows": []}
    for key, name in (("employee_id", "emp_id"), ("week_ending", "week_ending"), ("total_written", "total_miles")):
        t, c, e = _field(reads[name])
        res[key], res[key + "_conf"] = t, c
    for r in range(N_ROWS):
        names = {k: f"r{r}_{k}" for k in ("date", "client", "odo_start", "odo_end", "miles")}
        if all(x is None for k in names.values() for x in reads[k]):
            continue                                         # blank row
        row = {"row": r + 1}
        for k, f in names.items():
            t, c, e = _field(reads[f])
            row["miles_written" if k == "miles" else k] = t
            row[(("miles_written" if k == "miles" else k) + "_conf")] = c
            row[(("miles_written" if k == "miles" else k) + "_empty")] = e
        res["rows"].append(row)
    # per-box confidences in the shape validate.py (SCRUM-16) reads: fields[name] = {"text", "confidence": [p or None per box]}
    res["fields"] = {name: {"text": "".join(r[0] if r else "?" for r in rs),
                            "confidence": [float(r[1]) if r else None for r in rs]}
                     for name, rs in reads.items()}
    return res


# ================================================================== whole pipeline
def read_log(img, clf, debug_dir=None):
    page, info = register_page(img)
    cells = extract_cells(page)
    norm = {f: [normalize_cell(c) for c in cs] for f, cs in cells.items()}
    reads = classify_cells(norm, clf)
    result = assemble(reads)
    n_total = sum(len(v) for v in norm.values())
    n_ink = sum(c is not None for v in norm.values() for c in v)
    result["registration"] = info
    result["cells_with_ink"] = n_ink
    result["cells_total"] = n_total
    if debug_dir:
        save_debug(debug_dir, page, norm, reads)
    return result


def save_debug(out_dir, page, norm, reads):
    """Stage pictures for the demo: the flattened page and every normalized cell with its read."""
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "1_registered_page.png"), page)
    tile, pad = 56, 22
    fields = ["date", "client", "odo_start", "odo_end", "miles"]
    rows = [r for r in range(N_ROWS) if any(c is not None for f in fields for c in norm[f"r{r}_{f}"])]
    sheets = []
    for r in rows:
        strip = []
        for f in fields:
            for i, c in enumerate(norm[f"r{r}_{f}"]):
                img = np.zeros((tile + pad, tile), np.uint8)
                if c is not None:
                    img[:tile] = cv2.resize((c * 255).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST)
                    ch, conf = reads[f"r{r}_{f}"][i]
                    col = 255 if conf >= 0.9 else 120
                    cv2.putText(img, ch, (4, tile + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    cv2.putText(img, f"{int(conf * 100)}", (22, tile + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
                img[:, -1] = 90
                strip.append(img)
            strip.append(np.full((tile + pad, 10), 90, np.uint8))
        sheets.append(np.hstack(strip))
    if sheets:
        cv2.imwrite(os.path.join(out_dir, "2_normalized_cells_with_reads.png"), np.vstack(sheets))


# ======================================================================= command line
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--digit-model", default="models/digit_model.pt")
    ap.add_argument("--letter-model", default="models/letter_model.pt")
    ap.add_argument("--debug", default=None, help="folder to save stage pictures")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"Could not open image: {a.image}")
    res = read_log(img, TorchClassifier(a.digit_model, a.letter_model), a.debug)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"Employee ID : {res['employee_id']}   Week ending: {res['week_ending']}   Total written: {res['total_written']}")
        print(f"{'row':>3} {'date':>5} {'client':>6} {'odo start':>10} {'odo end':>10} {'miles':>6}")
        for r in res["rows"]:
            print(f"{r['row']:>3} {r['date']:>5} {r['client']:>6} {r['odo_start']:>10} {r['odo_end']:>10} {r['miles_written']:>6}")
        print(f"Cells with ink: {res['cells_with_ink']} of {res['cells_total']}  |  page found: {res['registration']['page_found']}")
