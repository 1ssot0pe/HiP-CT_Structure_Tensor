"""
i74_metric_agreement.py -- do the two independent i74 crossing-demotion metrics
agree on the SAME voxels, not just on similar aggregate rates?

phase2_i74_demote.py (forward-support straightness) and
real_i74_mass_conservation.py (intensity mass-conservation ratio) each report
that most/all real HOST 2-peak calls are demoted -- but from separate random
samples. Similar aggregate numbers could still arise from two metrics that
are sensitive to different, uncorrelated things. This script runs BOTH
certifiers on the SAME set of real HOST crossing candidates and reports the
per-voxel agreement (confusion matrix + correlation), which is the real test
of whether they're detecting the same phenomenon.

No i74 ground truth (per the host-gate work): agreement between two
independently-motivated, phantom-validated metrics is the evidence available,
not a fabricated accuracy against a label.

Run:  .venv/Scripts/python -u i74_metric_agreement.py [--crop N] [--n-cross N]
"""

import os as _sr_os, sys as _sr_sys
_SR_ROOT = _sr_os.path.dirname(_sr_os.path.dirname(__file__))
_sr_sys.path.insert(0, _SR_ROOT)  # repo root (host_fod, crossing_shape_3d, kissing_phantom, ...)
_sr_sys.path.insert(0, _sr_os.path.join(_SR_ROOT, "global_tractography"))  # phase1/phase2 certifiers

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from reconstruction import load_volume, DEFAULT_VOLUME_DIR
from phase2_i74_demote import collect_crossings, peak_reach, HALF, STRAIGHT_MIN, REACH_MIN
from mass_conservation_validate import local_calibration, THRESH, EPS
from real_i74_mass_conservation import _random_shell_positions, N_CAL_TRIES


def run(crop=200, n_cross=60, save="i74_metric_agreement.png", seed=0):
    print("=" * 88)
    print(f"i74 METRIC AGREEMENT -- straightness certifier vs mass-conservation ratio, "
          f"SAME candidates ({DEFAULT_VOLUME_DIR.name})")
    print("=" * 88)
    c0 = (768 - crop) // 2
    vol = load_volume(crop=(c0, c0 + crop)).astype(np.float64)
    print(f"  volume {vol.shape}, mean={vol.mean():.1f}")
    cx = collect_crossings(vol, n_cross, seed=seed)
    print(f"  found {len(cx)} HOST-'crossing' candidates; scoring both metrics on each ...")

    rng = np.random.default_rng(seed + 1)
    rows = []
    for i, (pos, dirs) in enumerate(cx):
        print(f"    scoring {i+1}/{len(cx)}", end="\r", flush=True)

        # --- metric 1: forward-support straightness (phase2) ---
        rc0, s0 = peak_reach(vol, pos, dirs[0])
        rc1, s1 = peak_reach(vol, pos, dirs[1])
        weak_straight, min_reach = min(s0, s1), min(rc0, rc1)
        demoted_straight = (weak_straight < STRAIGHT_MIN) or (min_reach < REACH_MIN)

        # --- metric 2: mass-conservation ratio ---
        cal_positions = _random_shell_positions(pos, vol.shape, rng)
        calib = local_calibration(vol, HALF, cal_positions)
        if np.isnan(calib):
            continue
        val = float(vol[tuple(int(round(c)) for c in pos)])
        ratio = val / (calib + EPS)
        demoted_ratio = ratio <= THRESH

        rows.append(dict(weak_straight=weak_straight, ratio=ratio,
                          demoted_straight=demoted_straight, demoted_ratio=demoted_ratio))
    print("-" * 88)

    n = len(rows)
    if n == 0:
        print("  no candidates scored by both metrics -- nothing to report.")
        return
    ws = np.array([r["weak_straight"] for r in rows])
    ra = np.array([r["ratio"] for r in rows])
    ds = np.array([r["demoted_straight"] for r in rows])
    dr = np.array([r["demoted_ratio"] for r in rows])

    both_demoted = int(np.sum(ds & dr))
    only_straight = int(np.sum(ds & ~dr))
    only_ratio = int(np.sum(~ds & dr))
    neither = int(np.sum(~ds & ~dr))
    agree = (both_demoted + neither) / n

    print(f"  scored {n}/{len(cx)} candidates by both metrics")
    print(f"\n  confusion matrix (demoted = below each metric's own threshold):")
    print(f"                      ratio: DEMOTED   ratio: CONFIRMED")
    print(f"  straightness DEMOTED    {both_demoted:>10d}       {only_straight:>10d}")
    print(f"  straightness CONFIRMED  {only_ratio:>10d}       {neither:>10d}")
    print(f"\n  agreement (both demoted OR both confirmed) = {agree:.0%} of {n}")

    if n >= 3 and np.std(ws) > 0 and np.std(ra) > 0:
        pear_r, pear_p = pearsonr(ws, ra)
        spear_r, spear_p = spearmanr(ws, ra)
        print(f"  weaker-peak straightness vs ratio: Pearson r={pear_r:.2f} (p={pear_p:.3f}), "
              f"Spearman rho={spear_r:.2f} (p={spear_p:.3f})")
    print("=" * 88)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    colors = np.where(ds & dr, "tab:red", np.where(~ds & ~dr, "tab:green", "tab:orange"))
    ax[0].scatter(ws, ra, c=colors, s=32, edgecolor="0.3", lw=0.4)
    ax[0].axvline(STRAIGHT_MIN, color="k", ls="--", lw=1)
    ax[0].axhline(THRESH, color="k", ls="--", lw=1)
    ax[0].set_xlabel("weaker-peak straightness (phase2)")
    ax[0].set_ylabel("mass-conservation ratio")
    ax[0].set_title(f"per-voxel agreement: {agree:.0%}\nred=both demoted, green=both confirmed, "
                    f"orange=disagree")

    labels = ["both\ndemoted", "straight.\nonly", "ratio\nonly", "both\nconfirmed"]
    counts = [both_demoted, only_straight, only_ratio, neither]
    ax[1].bar(labels, counts, color=["tab:red", "tab:orange", "tab:orange", "tab:green"])
    ax[1].set_ylabel("count")
    ax[1].set_title(f"n={n}, agreement={agree:.0%}")

    fig.suptitle("Do the two independent i74 crossing-demotion metrics agree per-voxel?", fontsize=12)
    fig.tight_layout()
    fig.savefig(save, dpi=130)
    print(f"Saved {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=200)
    ap.add_argument("--n-cross", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(crop=args.crop, n_cross=args.n_cross, seed=args.seed)
