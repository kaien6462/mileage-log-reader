"""
evaluate_classifiers.py -- SCRUM-14 error analysis.

Run (same folder as train_classifiers.py, digit_model.pt, letter_model.pt, emnist36.npz):
    python evaluate_classifiers.py --emnist emnist36.npz

Prints and saves (results_classifiers.txt):
  - overall TEST accuracy for the digit and letter models
  - accuracy for EACH digit / letter, worst first
  - the most common mix-ups (true -> predicted)
Also saves confusion_digit.csv and confusion_letter.csv.
"""
import argparse
import numpy as np
import torch

from train_classifiers import SmallCNN, load, make_digit_set, make_letter_set

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def predict(model, x, dev):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            xb = torch.from_numpy(x[i:i + 1024]).float().div(255).unsqueeze(1).to(dev)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)          # rows = true class, columns = predicted
    return cm


def report(name, names, cm, out):
    n = len(names)
    total = cm.sum()
    overall = np.trace(cm) / total
    out.append(f"=== {name} model: TEST accuracy {overall:.4f} on {total} images ===")

    per_class = np.diag(cm) / np.maximum(cm.sum(1), 1)
    out.append(f"\nAccuracy per {name} (worst first):  char | accuracy | test images")
    for c in np.argsort(per_class):
        out.append(f"   {names[c]}  |  {per_class[c]:.4f}  |  {cm[c].sum()}")

    out.append(f"\nTop 10 mix-ups for {name} (true -> predicted : count):")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off.ravel())[::-1][:10]
    for f in flat:
        t, p = divmod(int(f), n)
        if off[t, p] == 0:
            break
        out.append(f"   {names[t]} -> {names[p]} : {off[t, p]}")
    out.append("")
    np.savetxt(f"confusion_{name}.csv", cm, fmt="%d", delimiter=",",
               header="rows=true, cols=predicted: " + ",".join(names), comments="")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emnist", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load(args.emnist)
    out = []
    for name, names, make, path in [("digit", DIGITS, make_digit_set, "digit_model.pt"),
                                    ("letter", LETTERS, make_letter_set, "letter_model.pt")]:
        x, y = make(*data["test"])
        model = SmallCNN(len(names)).to(dev)
        model.load_state_dict(torch.load(path, map_location=dev))
        cm = confusion(y, predict(model, x, dev), len(names))
        report(name, names, cm, out)

    text = "\n".join(out)
    print(text)
    open("results_classifiers.txt", "w").write(text)
    print("Saved results_classifiers.txt, confusion_digit.csv, confusion_letter.csv")
