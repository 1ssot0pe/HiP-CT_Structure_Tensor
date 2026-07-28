"""
i74_global_density_calibration.py -- second fix attempt for the confirmed
selection bias in real_i74_mass_conservation.py.

i74_calibration_bias_check.py showed the problem is DENSITY, not distance:
unscored candidates sit in denser tissue with ~0% locally-"single" neighbour
patches. i74_directional_calibration.py tried following the fibre axis
outward and only recovered 10% of them -- dense regions mostly don't have
ANY single-fibre patch within a plausible walk.

This script instead builds a GLOBAL pool of "single"-classified patches from
across the whole volume (a coarse grid scan, not restricted to each
candidate's neighbourhood), tagged by their own local tissue density. Each
candidate is then calibrated from the pool patches with the CLOSEST matching
density, wherever in the volume they happen to sit -- density-matched, not
distance-matched. This directly targets the confound: if dense-tissue single
patches exist ANYWHERE in the volume (just not necessarily near a given
candidate), this recovers them.

Critical honesty check included: for each candidate, report the density GAP
between the candidate and its matched pool patches. If the pool has no
patches near a dense candidate's density, the "closest match" is still a
low-density patch pretending to calibrate a high-density point -- silently
reintroducing the same bias. We report this explicitly rather than trusting
the ratio blindly.

Run:  .venv/Scripts/python -u i74_global_density_calibration.py [--crop N] [--n-cross N]
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

from reconstruction import load_volume, DEFAULT_VOLUME_DIR
from host_fod import analyse, classify
from phase2_i74_demote import collect_crossings, HALF
from mass_conservation_validate import THRESH, EPS, _patch_at
from real_i74_mass_conservation import _random_shell_positions
from mass_conservation_validate import local_calibration
from i74_calibration_bias_check import neighbourhood_stats

GRID_STEP = 12          # global pool scan spacing (vox)
K_MATCH = 15            # nearest-by-density pool patches to calibrate from
DENSITY_GAP_WARN = 20.0  # flag candidates whose match is this far off in density


def build_global_pool(vol, step=GRID_STEP, half=HALF):
    shape = vol.shape
    lo = half + 2
    his = [shape[i] - half - 2 for i in range(3)]
    dens, peaks = [], []
    for z in range(lo, his[0], step):
        for y in range(lo, his[1], step):
            for x in range(lo, his[2], step):
                patch = _patch_at(vol, (z, y, x), half)
                if patch.shape != (2 * half + 1,) * 3:
                    continue
                if patch.mean() < 0.05 * vol.max():
                    continue
                m = analyse(patch)
                if m is None or classify(m) != "single":
                    continue
                dens.append(float(patch.mean()))
                peaks.append(float(patch.max()))
    return np.array(dens), np.array(peaks)


def density_matched_calibration(pool_dens, pool_peaks, density, k=K_MATCH):
    if len(pool_dens) == 0:
        return np.nan, np.nan
    d = np.abs(pool_dens - density)
    idx = np.argsort(d)[:k]
    gap = float(np.median(d[idx]))
    return float(np.median(pool_peaks[idx])), gap


def run(crop=200, n_cross=60, save="i74_global_density_calibration.png", seed=0):
    print("=" * 88)
    print(f"i74 GLOBAL DENSITY-MATCHED CALIBRATION ({DEFAULT_VOLUME_DIR.name})")
    print("=" * 88)
    c0 = (768 - crop) // 2
    vol = load_volume(crop=(c0, c0 + crop)).astype(np.float64)
    print(f"  volume {vol.shape}, mean={vol.mean():.1f}")

    print(f"  building global 'single'-patch pool (grid step={GRID_STEP}) ...")
    pool_dens, pool_peaks = build_global_pool(vol)
    print(f"  pool size: {len(pool_dens)} single-classified patches, "
          f"density range=[{pool_dens.min():.1f}, {pool_dens.max():.1f}]"
          if len(pool_dens) else "  pool EMPTY")

    cx = collect_crossings(vol, n_cross, seed=seed)
    print(f"  found {len(cx)} candidates")

    rng = np.random.default_rng(seed + 1)
    old_scored = []
    for pos, dirs in cx:
        cal_positions = _random_shell_positions(pos, vol.shape, rng)
        calib = local_calibration(vol, HALF, cal_positions)
        old_scored.append(not np.isnan(calib))

    ratios, gaps, was_old_scored, densities = [], [], [], []
    for i, ((pos, dirs), old_ok) in enumerate(zip(cx, old_scored)):
        print(f"    scoring {i+1}/{len(cx)}", end="\r", flush=True)
        dens, _frac_single, _n = neighbourhood_stats(vol, pos)
        if np.isnan(dens):
            continue
        calib, gap = density_matched_calibration(pool_dens, pool_peaks, dens)
        if np.isnan(calib):
            continue
        val = float(vol[tuple(int(round(c)) for c in pos)])
        ratios.append(val / (calib + EPS))
        gaps.append(gap)
        was_old_scored.append(old_ok)
        densities.append(dens)
    print("-" * 88)

    ratios = np.array(ratios)
    gaps = np.array(gaps)
    was_old_scored = np.array(was_old_scored)
    densities = np.array(densities)
    n = len(ratios)
    n_old_scored = int(np.sum(old_scored))

    print(f"  OLD (random shell)        coverage: {n_old_scored}/{len(cx)} = {n_old_scored/len(cx):.0%}")
    print(f"  NEW (global density-match) coverage: {n}/{len(cx)} = {n/len(cx):.0%}")

    if n:
        fire_frac = float(np.mean(ratios > THRESH))
        recovered = ~was_old_scored
        fire_frac_rec = float(np.mean(ratios[recovered] > THRESH)) if np.any(recovered) else float("nan")
        fire_frac_old = float(np.mean(ratios[was_old_scored] > THRESH)) if np.any(was_old_scored) else float("nan")
        print(f"\n  FULL sample (n={n}): ratio median={np.median(ratios):.2f}, "
              f"fire fraction = {fire_frac:.0%}")
        print(f"  previously-scored subset (n={int(np.sum(was_old_scored))}): fire fraction = {fire_frac_old:.0%}")
        print(f"  newly-recovered subset (n={int(np.sum(recovered))}, dense/complex ones): "
              f"fire fraction = {fire_frac_rec:.0%}")

        print(f"\n  --- honesty check: density-match quality ---")
        print(f"  median density gap (candidate vs matched pool patches): {np.median(gaps):.1f}")
        print(f"  median density gap, previously-scored subset : {np.median(gaps[was_old_scored]):.1f}"
              if np.any(was_old_scored) else "")
        print(f"  median density gap, newly-recovered subset   : {np.median(gaps[recovered]):.1f}"
              if np.any(recovered) else "")
        n_poor = int(np.sum(gaps > DENSITY_GAP_WARN))
        print(f"  candidates with density gap > {DENSITY_GAP_WARN} (poor match, distrust their ratio): "
              f"{n_poor}/{n} ({n_poor/n:.0%})")
        if np.any(recovered):
            n_poor_rec = int(np.sum(gaps[recovered] > DENSITY_GAP_WARN))
            print(f"  ... of which, in the newly-recovered subset: {n_poor_rec}/{int(np.sum(recovered))} "
                  f"({n_poor_rec/max(int(np.sum(recovered)),1):.0%})")
    print("=" * 88)

    if n:
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        bins = np.linspace(0, max(3.0, ratios.max() * 1.05), 30)
        ax[0].hist(ratios[was_old_scored], bins=bins, color="tab:blue", alpha=0.7, label="also old-scored")
        ax[0].hist(ratios[recovered], bins=bins, color="tab:red", alpha=0.7, label="recovered")
        ax[0].axvline(THRESH, color="k", ls="--")
        ax[0].set_xlabel("mass-conservation ratio"); ax[0].set_ylabel("count")
        ax[0].set_title(f"n={n}, fire fraction={fire_frac:.0%}")
        ax[0].legend(fontsize=8)

        ax[1].scatter(densities, gaps, c=np.where(recovered, "tab:red", "tab:blue"), s=28, edgecolor="0.3", lw=0.4)
        ax[1].axhline(DENSITY_GAP_WARN, color="k", ls="--")
        ax[1].set_xlabel("candidate local density"); ax[1].set_ylabel("density gap to matched pool")
        ax[1].set_title("match quality (red=recovered candidates)")

        ax[2].scatter(pool_dens, pool_peaks, s=6, color="0.6", alpha=0.5, label="global single-patch pool")
        ax[2].scatter(densities, densities * 0 + np.nan, s=0)  # keep axes consistent
        ax[2].set_xlabel("patch mean (density)"); ax[2].set_ylabel("patch max (peak intensity)")
        ax[2].set_title(f"global calibration pool, n={len(pool_dens)}")
        ax[2].legend(fontsize=8)

        fig.suptitle("Global density-matched calibration: coverage, ratios, and match honesty", fontsize=12)
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
