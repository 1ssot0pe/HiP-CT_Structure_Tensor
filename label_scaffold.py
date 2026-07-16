"""
label_scaffold.py -- step 4 (B3) of the gate fix plan: the grounding step.

Steps 1-3 established that NO cheap signal isolates crossings on real i74, and
every threshold/method was validated only on synthetic phantoms that do not
transfer. So the honest blocker is: we have no real-data ground truth. This
builds the scaffold to get some -- a small, human-labelled set of real i74
patches -- and then scores every candidate gate against it.

The labels MUST be independent of HOST (else validating HOST with them is
circular), so the contact sheet shows ONLY the raw anatomy; HOST is never run
here. An expert labels each patch from the image:
    single       -- one coherent fibre direction
    crossing     -- two (or more) interleaved directions in the same tissue
    disorganised -- no clear direction (isotropic / blurry / out-of-tissue)
    unclear      -- genuinely ambiguous, exclude from scoring

Patches are STRATIFIED across the lambda3 range so the labelled set spans the
gate's decision boundary (not just easy extremes). build() writes the sheet +
a CSV template; score(csv) reads the filled CSV and reports, for each gate,
catch (recall on true crossings) and false-fire (on non-crossings) -- the first
REAL-DATA numbers in this whole investigation.

Reuses patch_feats (selfcal_gate), load_interior (selfcalib_threshold).
"""

import warnings
warnings.filterwarnings("ignore")

import csv
import os

import numpy as np

from selfcal_gate import patch_feats
from gate_battery import HALF

EPS = 1e-12
SHEET = "label_sheet.png"
TEMPLATE = "labels_template.csv"
N_PATCHES = 12
Z_SLICE = 100
STEP = 14


# ---------------------------------------------------------------------------
# build: stratified patch selection + contact sheet + CSV template
# ---------------------------------------------------------------------------

def _grid_feats(sub, step=STEP):
    centers = list(range(HALF + 8, sub.shape[1] - HALF - 5, step))
    rows = []
    for cy in centers:
        for cx in centers:
            patch = sub[:, cy - HALF:cy + HALF + 1, cx - HALF:cx + HALF + 1]
            f = patch_feats(patch)
            if f is None:
                continue
            rows.append(dict(cy=cy, cx=cx, **f))
    return rows


def _stratified(rows, n):
    """Pick n patches evenly spaced along the lambda3 rank -> spans the boundary."""
    rows = sorted(rows, key=lambda r: r["lam3"])
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    return [rows[i] for i in idx]


def build(z_slice=Z_SLICE, n=N_PATCHES, sheet=SHEET, template=TEMPLATE):
    from selfcalib_threshold import load_interior
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = load_interior()
    sub = vol[z_slice - HALF:z_slice + HALF + 1].astype(np.float32)
    sub = (sub - sub.min()) / (np.ptp(sub) + EPS)
    chosen = _stratified(_grid_feats(sub), n)
    for i, r in enumerate(chosen):
        r["patch_id"] = i
        r["z"] = z_slice

    # contact sheet: three orthogonal mid-slices per patch so a 3D crossing is
    # visible, not just one plane. rows = patches, cols = (z-mid, y-mid, x-mid).
    ncol = 3
    fig, axes = plt.subplots(n, ncol, figsize=(ncol * 2.6, n * 2.6))
    for i, r in enumerate(chosen):
        cy, cx = r["cy"], r["cx"]
        patch = sub[:, cy - HALF:cy + HALF + 1, cx - HALF:cx + HALF + 1]
        views = [patch[HALF], patch[:, HALF], patch[:, :, HALF]]
        names = ["z-mid (in-plane)", "y-mid", "x-mid"]
        for j, (v, nm) in enumerate(zip(views, names)):
            ax = axes[i, j] if n > 1 else axes[j]
            ax.imshow(v, cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"#{i}", fontsize=12, rotation=0, labelpad=18,
                              va="center")
            if i == 0:
                ax.set_title(nm, fontsize=9)
        # metrics as a faint annotation on the last column (reference, not a hint)
        axc = axes[i, ncol - 1] if n > 1 else axes[ncol - 1]
        axc.text(1.05, 0.5, f"lam3={r['lam3']:.3f}\ncs={r['cs']:.2f}\ncp={r['cp']:.2f}",
                 transform=axc.transAxes, fontsize=8, va="center", color="#555")
    fig.suptitle("i74 patches to label (independent of HOST): single / crossing / "
                 "disorganised / unclear", fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 0.93, 0.98))
    fig.savefig(sheet, dpi=130)
    print(f"Saved {sheet}  ({n} patches, stratified by lambda3)")

    # CSV template with everything the scorer needs + a blank label column
    with open(template, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch_id", "z", "cy", "cx", "lam3", "cs", "cp", "label"])
        for r in chosen:
            w.writerow([r["patch_id"], r["z"], r["cy"], r["cx"],
                        f"{r['lam3']:.5f}", f"{r['cs']:.4f}", f"{r['cp']:.4f}", ""])
    print(f"Saved {template}  -- fill the 'label' column "
          f"(single/crossing/disorganised/unclear) from {sheet}, then run: "
          f"python label_scaffold.py score")
    return chosen


# ---------------------------------------------------------------------------
# score: read the filled CSV and report REAL catch / false-fire per gate
# ---------------------------------------------------------------------------

def _otsu(vals):
    from skimage.filters import threshold_otsu
    return float(threshold_otsu(np.asarray(vals)))


def score(csv_path=TEMPLATE):
    from gate_protocol import load_gate

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            lab = (r.get("label") or "").strip().lower()
            if not lab:
                continue
            rows.append(dict(lam3=float(r["lam3"]), cs=float(r["cs"]),
                             cp=float(r["cp"]), label=lab))
    labelled = [r for r in rows if r["label"] in
                ("single", "crossing", "disorganised", "unclear")]
    scored = [r for r in labelled if r["label"] != "unclear"]
    if not scored:
        print(f"No usable labels in {csv_path}. Fill the 'label' column first "
              f"(single/crossing/disorganised/unclear).")
        return
    cross = [r for r in scored if r["label"] == "crossing"]
    noncross = [r for r in scored if r["label"] != "crossing"]
    lam3s = np.array([r["lam3"] for r in scored])
    print("=" * 74)
    print(f"REAL-DATA gate scoring on {len(scored)} labelled patches "
          f"({len(cross)} crossing, {len(noncross)} non-crossing)")
    if not cross or not noncross:
        print("  need both crossing and non-crossing labels to score. "
              f"(have crossing={len(cross)}, non={len(noncross)})")
        print("=" * 74)
        return

    def report(name, thr, key, fire_high=True):
        fire = (lambda v: v > thr) if fire_high else (lambda v: v < thr)
        catch = np.mean([fire(r[key]) for r in cross])
        fp = np.mean([fire(r[key]) for r in noncross])
        print(f"  {name:>34}: catch={catch:.0%}  false-fire={fp:.0%}  (thr={thr:.4g})")

    # gate 1: fire-fraction lambda3 (the wired-in gate) -- fire top 65% of lambda3
    ff = float(load_gate()["train_fire_fraction"])
    report(f"lambda3 fire-fraction ({ff:.0%})",
           float(np.percentile(lam3s, 100 * (1 - ff))), "lam3", True)
    # gate 2: lambda3 Otsu valley (self-calibrated on THIS real set)
    report("lambda3 Otsu (self-cal)", _otsu(lam3s), "lam3", True)
    # gate 3: planarity c_p Otsu (fire if LOW = not single)
    report("planarity c_p Otsu (fire if low)",
           _otsu([r["cp"] for r in scored]), "cp", False)
    print("-" * 74)
    print("  (first real-data catch/false-fire numbers; phantom validation only "
          "predicted these -- now they are measured)")
    print("=" * 74)


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"
    if stage == "build":
        build()
    elif stage == "score":
        score(sys.argv[2] if len(sys.argv) > 2 else TEMPLATE)
