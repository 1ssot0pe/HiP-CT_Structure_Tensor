"""
gate_battery.py

Stress-test the lambda3 HOST-gate (refined_field.cheap_gate + run_i74_gated)
across MANY phantoms with KNOWN crossing/single/empty regions, to find where --
if anywhere -- the cheap gate mislabels tissue before it decides whether to run
expensive HOST.

WHY THIS BATTERY EXISTS
-----------------------
The gate fires HOST where lambda3 (the SMALLEST structure-tensor eigenvalue) is
high and skips it (cheap single-e3) where lambda3 is low, using a global
PERCENTILE threshold over the image (refined_field.run_i74_gated, pct=55).

lambda3 is the ABSOLUTE eigenvalue -- it scales with local contrast/texture
amplitude, NOT just with crossing-ness. The OLD crossing guard
(crossing_shape_3d) instead gated on the NORMALISED sphericity
c_s = 3*lambda3 / (lambda1+lambda2+lambda3), which is contrast-invariant by
construction. So the new cheap gate may have quietly traded away that
invariance. On a uniform-contrast phantom you cannot see the difference; with
spatially VARYING contrast you can.

The dangerous failure is NOT wasting HOST on a single fibre (merely slow) -- it
is a LOW-CONTRAST CROSSING whose lambda3 falls below the percentile and so gets
SKIPPED and silently reported as a single fibre (a MISSED crossing).

So for every patch we compute BOTH the new metric (lambda3) and the old metric
(c_s) against the known region, then score each gate at the SAME percentile rule
on: crossing recall, arm false-fire, and -- the one that matters -- missed
crossings. If lambda3 breaks where c_s holds, the fix is concrete: gate on the
normalised ratio, not the raw eigenvalue.

Fast by design: the gate itself is one gradient pass, so no full HOST is needed
to test LOCALISATION (we have ground-truth region masks). A light HOST confirm
on a few fired crossing patches closes the loop (fired => actually resolvable).

Reuses patch_gradients + analyse (host_fod), and the "+"-phantom idea from
crossing_trigger_test (make_perp_bundles) generalised here to arbitrary crossing
angle and a spatial contrast-gain field.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root (crossing_shape_3d, host_fod)
warnings.filterwarnings("ignore")

import numpy as np

from core.crossing_shape_3d import _tube_family, _coords
from core.host_fod import patch_gradients, analyse

EPS = 1e-12

SHAPE = (160, 160, 160)
CROSS_HALF = 40
HALF = 20                       # 41^3 patches (fits sigma=3 gradient margin)
Z = SHAPE[0] // 2
CENTERS = [30, 55, 80, 105, 130]
GATE_PCT = 55                   # the threshold rule run_i74_gated uses


# ---------------------------------------------------------------------------
# generalised "+" phantom: two slabs of Gaussian tubes, arbitrary crossing
# angle, optional spatial contrast-gain field. Region masks are exact.
# ---------------------------------------------------------------------------

def make_plus(angle_deg=90.0, spacing=10.0, spacing2=13.0, width=1.5,
              noise_sigma=0.2, seed=0, gain=None):
    """Horizontal slab (|y|<CROSS_HALF) filled with tubes along d1=x; vertical
    slab (|x|<CROSS_HALF) filled with tubes along d2 at `angle_deg` to d1. Their
    overlap (centre square) is a genuine per-voxel crossing; the arms are single
    fibre; the corners empty.

    gain: optional callable g(x, y) -> multiplicative contrast field applied
    BEFORE noise, so a low-gain region has a genuine crossing at LOW SNR (the
    contrast-mismatch stress). None => uniform contrast.
    """
    x, y, z = _coords(SHAPE)
    th = np.deg2rad(angle_deg)
    d1 = [1.0, 0.0, 0.0]
    d2 = [np.cos(th), np.sin(th), 0.0]
    b1 = _tube_family((x, y, z), d1, spacing, width)
    b2 = _tube_family((x, y, z), d2, spacing2, width)
    horiz = np.abs(y) < CROSS_HALF
    vert = np.abs(x) < CROSS_HALF
    vol = np.where(horiz, b1, 0.0) + np.where(vert, b2, 0.0)
    if gain is not None:
        vol = vol * gain(x, y)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        vol = vol + noise_sigma * rng.standard_normal(SHAPE).astype(np.float32)
    vol = (vol - vol.min()) / (np.ptp(vol) + EPS)
    masks = dict(crossing=horiz & vert, arm_h=horiz & ~vert,
                 arm_v=vert & ~horiz, empty=~horiz & ~vert)
    return vol.astype(np.float32), masks


def low_centre_gain(g_min=0.25, g_max=1.0, radius=70.0):
    """Contrast LOW at the centre (where the crossing is), rising to g_max at
    the edges -- so the genuine crossing has the WEAKEST signal in the image."""
    def g(x, y):
        r = np.sqrt(x * x + y * y)
        return g_min + (g_max - g_min) * np.clip(r / radius, 0.0, 1.0)
    return g


# ---------------------------------------------------------------------------
# per-patch cheap read: lambda3 (new gate) AND c_s (old, normalised) from ONE
# gradient pass -- exactly the pass cheap_gate uses (sigma_ref=3.0).
# ---------------------------------------------------------------------------

def patch_metrics(patch, sigma_ref=3.0):
    G = patch_gradients(patch, sigma_ref)
    if G.shape[0] < 40:
        return None
    S2 = G.T @ G
    lam = np.sort(np.linalg.eigvalsh(S2))           # ascending: lam[0]=lambda3
    lam3 = float(lam[0])
    c_s = float(3.0 * lam[0] / (lam.sum() + EPS))
    return dict(lam3=lam3, c_s=c_s)


def region_of(masks, cy, cx):
    for name in ("crossing", "arm_h", "arm_v", "empty"):
        if masks[name][Z, cy, cx]:
            return name
    return "empty"


def scan(vol, masks):
    """Every grid patch -> (region, lam3, c_s)."""
    rows = []
    for cy in CENTERS:
        for cx in CENTERS:
            patch = vol[Z - HALF:Z + HALF + 1, cy - HALF:cy + HALF + 1,
                        cx - HALF:cx + HALF + 1]
            m = patch_metrics(patch)
            if m is None:
                continue
            rows.append(dict(region=region_of(masks, cy, cx), cy=cy, cx=cx, **m))
    return rows


# ---------------------------------------------------------------------------
# score a gate metric on one phantom at the percentile rule
# ---------------------------------------------------------------------------

def score_gate(rows, key, pct=GATE_PCT):
    vals = np.array([r[key] for r in rows])
    thr = float(np.percentile(vals, pct))
    cross = [r for r in rows if r["region"] == "crossing"]
    arms = [r for r in rows if r["region"] in ("arm_h", "arm_v")]
    fired = lambda rs: np.mean([r[key] > thr for r in rs]) if rs else np.nan
    recall = fired(cross)                       # crossing patches that fire HOST
    arm_fire = fired(arms)                       # wasted HOST on single fibre
    missed = int(sum(1 for r in cross if r[key] <= thr))   # DANGER
    # separation: (mean_cross - mean_arm) / pooled std  -> a d'-like margin
    cv = np.array([r[key] for r in cross]); av = np.array([r[key] for r in arms])
    pooled = np.sqrt(0.5 * (cv.var() + av.var())) + EPS
    dprime = float((cv.mean() - av.mean()) / pooled)
    return dict(thr=thr, recall=recall, arm_fire=arm_fire, missed=missed,
                n_cross=len(cross), dprime=dprime)


# ---------------------------------------------------------------------------
# the battery
# ---------------------------------------------------------------------------

def build_battery():
    """List of (family, label, vol, masks). Many phantoms, known regions."""
    items = []
    # A. baseline "+" across noise
    for ns in (0.1, 0.2, 0.3, 0.5):
        for seed in (0, 1):
            v, m = make_plus(noise_sigma=ns, seed=seed)
            items.append(("A_noise", f"noise={ns} s{seed}", v, m))
    # B. fibre-width sweep (tube spacing)
    for sp in (6.0, 10.0, 14.0):
        v, m = make_plus(spacing=sp, spacing2=sp + 3, noise_sigma=0.2, seed=0)
        items.append(("B_width", f"spacing={sp:.0f}", v, m))
    # C. contrast-gradient "+": genuine crossing at LOW contrast (the killer)
    for gmin in (0.35, 0.2):
        v, m = make_plus(noise_sigma=0.2, seed=0, gain=low_centre_gain(g_min=gmin))
        items.append(("C_contrast", f"centre-gain={gmin}", v, m))
    # D. angled crossings (not just 90 deg)
    for ang in (60.0, 45.0):
        v, m = make_plus(angle_deg=ang, noise_sigma=0.2, seed=0)
        items.append(("D_angle", f"angle={ang:.0f}", v, m))
    return items


def run(save="gate_battery.png"):
    import time
    t0 = time.time()
    items = build_battery()
    print("=" * 92)
    print(f"lambda3 HOST-GATE BATTERY -- {len(items)} phantoms, "
          f"percentile-{GATE_PCT} gate rule (as in run_i74_gated)")
    print("  compares NEW gate (absolute lambda3) vs OLD metric (normalised c_s) "
          "on the SAME patches")
    print("=" * 92)
    hdr = (f"  {'phantom':>22} | {'metric':>5} | {'recall':>6} {'missed':>7} "
           f"{'armfire':>7} {'d-prime':>7}")
    results = []
    for fam, label, vol, masks in items:
        rows = scan(vol, masks)
        s_l = score_gate(rows, "lam3")
        s_c = score_gate(rows, "c_s")
        results.append(dict(fam=fam, label=label, rows=rows, lam3=s_l, c_s=s_c))
        print("-" * 92)
        print(hdr if fam.endswith("__never__") else f"  {label:>22} | "
              f"{'l3':>5} | {s_l['recall']:6.0%} {s_l['missed']:>3}/{s_l['n_cross']:<3} "
              f"{s_l['arm_fire']:7.0%} {s_l['dprime']:7.2f}")
        print(f"  {'':>22} | {'c_s':>5} | {s_c['recall']:6.0%} "
              f"{s_c['missed']:>3}/{s_c['n_cross']:<3} {s_c['arm_fire']:7.0%} "
              f"{s_c['dprime']:7.2f}")

    # ---- aggregate verdict per family ----
    print("=" * 92)
    print("VERDICT by family (a gate is SAFE only if it MISSES ZERO crossings):")
    fams = ["A_noise", "B_width", "C_contrast", "D_angle"]
    for fam in fams:
        rs = [r for r in results if r["fam"] == fam]
        for key in ("lam3", "c_s"):
            miss = sum(r[key]["missed"] for r in rs)
            ncr = sum(r[key]["n_cross"] for r in rs)
            rec = np.mean([r[key]["recall"] for r in rs])
            tag = "SAFE " if miss == 0 else f"MISSED {miss}"
            name = "lambda3(new)" if key == "lam3" else "c_s(old)   "
            print(f"  {fam:>12} {name} : recall={rec:5.0%}  "
                  f"missed={miss}/{ncr} crossings  -> {tag}")
    print("=" * 92)

    # ---- light HOST confirm: a fired crossing patch must actually resolve ----
    print("HOST confirm (a patch the gate FIRES must yield 2 peaks):")
    cphan = next(r for r in results if r["fam"] == "C_contrast")
    vol = next(v for f, l, v, m in items if f == "C_contrast")
    thr = cphan["lam3"]["thr"]
    checked = 0
    for r in cphan["rows"]:
        if r["region"] == "crossing" and r["lam3"] > thr and checked < 3:
            patch = vol[Z - HALF:Z + HALF + 1, r["cy"] - HALF:r["cy"] + HALF + 1,
                        r["cx"] - HALF:r["cx"] + HALF + 1]
            m = analyse(patch, sigma_grad=3.0, order=6)
            npk = 0 if m is None else m["n_peaks"]
            print(f"    fired crossing patch (cy={r['cy']},cx={r['cx']}) "
                  f"-> HOST n_peaks={npk}")
            checked += 1
    if checked == 0:
        print("    (gate fired on NO crossing patch here -- see missed count above)")

    _figure(results, save)
    print(f"\n({time.time()-t0:.0f}s)  Saved {save}")
    return results


# ---------------------------------------------------------------------------
# figure: (1) the money scatter lambda3 vs c_s on the contrast phantom, coloured
# by true region, with both percentile gates drawn; (2) missed-crossing % per
# family for each metric.
# ---------------------------------------------------------------------------

# project palette (consistent with refined_field): region identity colours
C_CROSS, C_SINGLE, C_EMPTY, INK = "#eb6834", "#2a78d6", "#9b9894", "#0b0b0b"
# two categorical hues for the metric comparison (fixed order: new, old)
C_L3, C_CS = "#6a4bc4", "#1f9e89"


def _figure(results, save):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))

    # --- panel 1: contrast phantom scatter, lambda3 (x) vs c_s (y) ---
    cph = next(r for r in results if r["fam"] == "C_contrast")
    reg_col = {"crossing": C_CROSS, "arm_h": C_SINGLE, "arm_v": C_SINGLE,
               "empty": C_EMPTY}
    for r in cph["rows"]:
        ax[0].scatter(r["lam3"], r["c_s"], c=reg_col[r["region"]], s=70,
                      edgecolors="white", linewidths=1.0, zorder=3)
    ax[0].axvline(cph["lam3"]["thr"], color=C_L3, ls="--", lw=2,
                  label=f"lambda3 gate (pct{GATE_PCT})")
    ax[0].axhline(cph["c_s"]["thr"], color=C_CS, ls="--", lw=2,
                  label=f"c_s gate (pct{GATE_PCT})")
    ax[0].set_xlabel("lambda3  (NEW gate -- absolute, contrast-dependent)")
    ax[0].set_ylabel("c_s  (OLD metric -- normalised)")
    ax[0].set_title("Contrast-gradient phantom: a crossing (orange) LEFT of the\n"
                    "lambda3 line is a MISSED crossing; the c_s line keeps it",
                    fontsize=10, color=INK)
    ax[0].legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=C_CROSS, mec="white",
               label="crossing (needs HOST)"),
        Line2D([], [], marker="o", ls="", mfc=C_SINGLE, mec="white",
               label="single-fibre arm"),
        Line2D([], [], marker="o", ls="", mfc=C_EMPTY, mec="white", label="empty"),
        Line2D([], [], color=C_L3, ls="--", label="lambda3 gate"),
        Line2D([], [], color=C_CS, ls="--", label="c_s gate")],
        fontsize=8, loc="upper left")

    # --- panel 2: missed-crossing count per family, lambda3 vs c_s ---
    fams = ["A_noise", "B_width", "C_contrast", "D_angle"]
    miss_l = [sum(r["lam3"]["missed"] for r in results if r["fam"] == f) for f in fams]
    miss_c = [sum(r["c_s"]["missed"] for r in results if r["fam"] == f) for f in fams]
    xpos = np.arange(len(fams)); w = 0.38
    ax[1].bar(xpos - w / 2, miss_l, w, color=C_L3, label="lambda3 (new gate)")
    ax[1].bar(xpos + w / 2, miss_c, w, color=C_CS, label="c_s (old metric)")
    for xp, ml, mc in zip(xpos, miss_l, miss_c):
        if ml:
            ax[1].text(xp - w / 2, ml + 0.05, str(ml), ha="center", va="bottom",
                       fontsize=9, color=INK)
        if mc:
            ax[1].text(xp + w / 2, mc + 0.05, str(mc), ha="center", va="bottom",
                       fontsize=9, color=INK)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels([f.split("_")[1] for f in fams])
    ax[1].set_ylabel("MISSED crossings (lower is safer; 0 = safe)")
    ax[1].set_title("Crossings the gate SKIPS (silently mislabels as single)\n"
                    "across the phantom battery", fontsize=10, color=INK)
    ax[1].legend(fontsize=9, frameon=False)
    ax[1].margins(y=0.15)

    fig.suptitle("lambda3 HOST-gate battery: does the cheap gate ever MISS a "
                 "crossing before HOST runs?", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save, dpi=130)


if __name__ == "__main__":
    run()
