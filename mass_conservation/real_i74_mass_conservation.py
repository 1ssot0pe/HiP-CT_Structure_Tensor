"""
real_i74_mass_conservation.py -- apply the mass-conservation ratio metric
(mass_conservation_validate.py, gate-PASSED on phantoms: catch=100%/FP=0%) to
REAL HOST 2-peak crossing candidates in the i74 volume itself.

Phantom validation only ever asked "does the ratio separate a phantom crossing
from a phantom kissing pair". It never touched a real HOST-flagged ambiguous
voxel. This script closes that gap: find real voxels where HOST's per-voxel
classifier calls a 2-way crossing (phase2_i74_demote.collect_crossings), then
compute ratio = intensity-at-the-point / local "single"-fibre calibration
(mass_conservation_validate.local_calibration, same recipe used in the
phantom proof) and report the DISTRIBUTION of ratios and the fire-fraction
above THRESH=1.5.

No i74 ground truth exists (per the host-gate work), so this is descriptive,
not an accuracy claim: it tells us what fraction of real HOST crossings look
"superposed" (ratio>1.5, mass-conserved-consistent with a true crossing) vs
"single-branch-like" (ratio<=1.5, consistent with kissing/partial-volume),
using the exact metric and threshold the phantom sweep proved generalises.

Run:  .venv/Scripts/python -u real_i74_mass_conservation.py [--crop N] [--n-cross N]
"""

import os as _sr_os, sys as _sr_sys
_SR_ROOT = _sr_os.path.dirname(_sr_os.path.dirname(__file__))
_sr_sys.path.insert(0, _SR_ROOT)  # repo root (host_fod, reconstruction, ...)
_sr_sys.path.insert(0, _sr_os.path.join(_SR_ROOT, "3_global_tractography"))  # phase2 certifier

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reconstruction import load_volume, DEFAULT_VOLUME_DIR
from host_fod import analyse, classify
from phase2_i74_demote import collect_crossings, HALF
from mass_conservation_validate import local_calibration, local_ratio_threshold, THRESH, EPS

N_CAL_TRIES = 40          # random nearby offsets to try per candidate
CAL_RADIUS = (12, 30)     # shell (vox) to search for confidently-"single" neighbours


def _random_shell_positions(pos, shape, rng, n=N_CAL_TRIES, rmin=CAL_RADIUS[0], rmax=CAL_RADIUS[1]):
    out = []
    for _ in range(n):
        # random direction, random radius in [rmin, rmax]
        v = rng.standard_normal(3)
        v /= np.linalg.norm(v) + EPS
        r = rng.uniform(rmin, rmax)
        cand = pos + v * r
        if all(HALF + 2 <= cand[i] < shape[i] - HALF - 2 for i in range(3)):
            out.append(cand)
    return out


def run(crop=200, n_cross=60, save="real_i74_mass_conservation.png", seed=0):
    print("=" * 88)
    print(f"REAL i74 MASS-CONSERVATION -- ratio on actual HOST 2-peak crossing voxels "
          f"({DEFAULT_VOLUME_DIR.name})")
    print("=" * 88)
    c0 = (768 - crop) // 2
    vol = load_volume(crop=(c0, c0 + crop)).astype(np.float64)
    print(f"  volume {vol.shape}, mean={vol.mean():.1f}")
    print(f"  seeking {n_cross} HOST-'crossing' voxels ...")
    cx = collect_crossings(vol, n_cross, seed=seed)
    print(f"  found {len(cx)} candidates")

    rng = np.random.default_rng(seed + 1)
    ratios, seps, n_calib = [], [], []
    for i, (pos, dirs) in enumerate(cx):
        print(f"    scoring {i+1}/{len(cx)}", end="\r", flush=True)
        cal_positions = _random_shell_positions(pos, vol.shape, rng)
        calib = local_calibration(vol, HALF, cal_positions)
        if np.isnan(calib):
            continue
        val = float(vol[tuple(int(round(c)) for c in pos)])
        ratio = val / (calib + EPS)
        ratios.append(ratio)
        n_calib.append(sum(1 for p in cal_positions))  # positions tried, not all used
        seps.append(float(np.degrees(np.arccos(np.clip(abs(dirs[0] @ dirs[1]), 0, 1)))))
    print("-" * 88)

    ratios = np.array(ratios)
    n = len(ratios)
    if n == 0:
        print("  no candidates yielded a valid calibration -- nothing to report.")
        return

    thr, thr_reason = local_ratio_threshold(ratios)   # data-driven cut from THIS volume
    fire_frac = float(np.mean(ratios > thr))
    print(f"  scored {n}/{len(cx)} candidates (rest had no confidently-'single' "
          f"neighbour within radius {CAL_RADIUS})")
    print(f"  ratio: median={np.median(ratios):.2f}  mean={np.mean(ratios):.2f}  "
          f"IQR=[{np.percentile(ratios,25):.2f}, {np.percentile(ratios,75):.2f}]")
    print(f"  threshold = {thr:.2f}  [{thr_reason}]")
    print(f"  FIRE FRACTION (ratio > {thr:.2f}, mass-conserved / crossing-consistent) "
          f"= {fire_frac:.0%}")
    print(f"  (Descriptive only: no i74 ground truth. Phantom sweep gave catch=100% on "
          f"true crossings, FP=0% on true kissing -- so this fraction is the real-tissue "
          f"analogue of that catch rate, read against those two anchor points, not a "
          f"validated accuracy on real data.)")
    print("=" * 88)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.linspace(0, max(3.0, ratios.max() * 1.05), 30)
    ax[0].hist(ratios, bins=bins, color="tab:purple", alpha=0.8)
    ax[0].axvline(thr, color="k", ls="--", label=f"threshold={thr:.2f}")
    ax[0].set_xlabel("mass-conservation ratio (intensity / local single-fibre calib)")
    ax[0].set_ylabel("HOST crossing voxels")
    ax[0].set_title(f"{n} real i74 HOST crossings\nfire fraction (ratio>{thr:.2f}) = {fire_frac:.0%}")
    ax[0].legend(fontsize=8)

    ax[1].scatter(seps, ratios, s=28, c=(ratios > thr), cmap="RdYlGn", edgecolor="0.3", lw=0.4)
    ax[1].axhline(thr, color="k", ls="--")
    ax[1].set_xlabel("HOST crossing separation (deg)")
    ax[1].set_ylabel("mass-conservation ratio")
    ax[1].set_title("ratio vs crossing angle")

    fig.suptitle("Mass-conservation ratio on REAL i74 HOST-flagged crossings", fontsize=12)
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
