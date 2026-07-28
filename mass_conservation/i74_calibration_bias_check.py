"""
i74_calibration_bias_check.py -- does requiring a nearby confidently-"single"
calibration patch (real_i74_mass_conservation.py) systematically SELECT for
candidates sitting in sparser/simpler tissue, and exclude the ones embedded in
denser, more fibre-packed tissue (where real crossings are more likely to
occur)? If so, the 0% real-i74 fire-fraction result is biased low, not clean.

Method: reproduce the EXACT same scored/unscored split from
real_i74_mass_conservation.py (same seed, same candidate list, same RNG
sequence), then independently characterise each candidate's local
neighbourhood -- local mean intensity (tissue density) and the fraction of a
regular grid of nearby patches HOST classifies as "single" vs anything else
(crossing/dispersed/disorganised = local complexity) -- and compare the
SCORED group against the UNSCORED group with a two-sided Mann-Whitney U test.

If scored/unscored don't differ -> the calibration requirement isn't
selecting on density/complexity, and the 0% fire-fraction result stands.
If they do differ (unscored = denser/more complex) -> confirms the bias:
the requirement is throwing away exactly the candidates most likely to be
real crossings, and the 0% number needs the structure-aware fix before it
can be trusted.

Run:  .venv/Scripts/python -u i74_calibration_bias_check.py [--crop N] [--n-cross N]
"""

import os as _sr_os, sys as _sr_sys
_SR_ROOT = _sr_os.path.dirname(_sr_os.path.dirname(__file__))
_sr_sys.path.insert(0, _SR_ROOT)  # repo root (host_fod, crossing_shape_3d, kissing_phantom, ...)
_sr_sys.path.insert(0, _sr_os.path.join(_SR_ROOT, "global_tractography"))  # phase1/phase2 certifiers

import argparse
import numpy as np
from scipy.stats import mannwhitneyu

from reconstruction import load_volume, DEFAULT_VOLUME_DIR
from host_fod import analyse, classify
from phase2_i74_demote import collect_crossings, HALF
from mass_conservation_validate import local_calibration, EPS
from real_i74_mass_conservation import _random_shell_positions

NEIGH_RADIUS = 20         # cube half-width for the neighbourhood characterisation
NEIGH_STEP = 10


def _patch_at(vol, center_zyx, half):
    z, y, x = [int(round(c)) for c in center_zyx]
    z0, y0, x0 = max(z - half, 0), max(y - half, 0), max(x - half, 0)
    return vol[z0:z + half + 1, y0:y + half + 1, x0:x + half + 1]


def neighbourhood_stats(vol, pos, radius=NEIGH_RADIUS, step=NEIGH_STEP, half=HALF):
    """Regular grid of patches around pos (independent of the calibration RNG):
    local mean intensity, and fraction classified 'single' vs anything else."""
    shape = vol.shape
    offsets = range(-radius, radius + 1, step)
    means, labels = [], []
    for dz in offsets:
        for dy in offsets:
            for dx in offsets:
                c = pos + np.array([dz, dy, dx], float)
                if not all(half + 2 <= c[i] < shape[i] - half - 2 for i in range(3)):
                    continue
                patch = _patch_at(vol, c, half)
                if patch.shape != (2 * half + 1,) * 3:
                    continue
                if patch.mean() < 0.05 * vol.max():
                    continue
                means.append(float(patch.mean()))
                m = analyse(patch)
                labels.append(classify(m) if m is not None else "n/a")
    if not means:
        return np.nan, np.nan, 0
    labels = np.array(labels)
    frac_single = float(np.mean(labels == "single"))
    return float(np.mean(means)), frac_single, len(means)


def run(crop=200, n_cross=60, seed=0):
    print("=" * 88)
    print(f"i74 CALIBRATION BIAS CHECK -- scored vs unscored candidates, "
          f"same split as real_i74_mass_conservation.py ({DEFAULT_VOLUME_DIR.name})")
    print("=" * 88)
    c0 = (768 - crop) // 2
    vol = load_volume(crop=(c0, c0 + crop)).astype(np.float64)
    print(f"  volume {vol.shape}, mean={vol.mean():.1f}")
    cx = collect_crossings(vol, n_cross, seed=seed)
    print(f"  found {len(cx)} candidates; reproducing the scored/unscored split ...")

    rng = np.random.default_rng(seed + 1)   # identical RNG sequence/order to the original run
    scored, unscored = [], []
    for pos, dirs in cx:
        cal_positions = _random_shell_positions(pos, vol.shape, rng)
        calib = local_calibration(vol, HALF, cal_positions)
        dens, frac_single, n_patch = neighbourhood_stats(vol, pos)
        row = dict(pos=pos, density=dens, frac_single=frac_single, n_patch=n_patch)
        (scored if not np.isnan(calib) else unscored).append(row)

    print(f"  scored={len(scored)}  unscored={len(unscored)}")

    def _vals(rows, key):
        return np.array([r[key] for r in rows if not np.isnan(r[key])])

    print("\n  --- local tissue density (mean patch intensity in neighbourhood) ---")
    ds, du = _vals(scored, "density"), _vals(unscored, "density")
    print(f"  scored   n={len(ds)} median={np.median(ds):.1f}")
    print(f"  unscored n={len(du)} median={np.median(du):.1f}")
    if len(ds) >= 3 and len(du) >= 3:
        u, p = mannwhitneyu(ds, du, alternative="two-sided")
        print(f"  Mann-Whitney U={u:.1f}, p={p:.3f}")

    print("\n  --- local complexity (fraction of neighbourhood patches HOST calls 'single') ---")
    fs, fu = _vals(scored, "frac_single"), _vals(unscored, "frac_single")
    print(f"  scored   n={len(fs)} median={np.median(fs):.2f}  (higher = simpler/sparser neighbourhood)")
    print(f"  unscored n={len(fu)} median={np.median(fu):.2f}")
    if len(fs) >= 3 and len(fu) >= 3:
        u2, p2 = mannwhitneyu(fs, fu, alternative="two-sided")
        print(f"  Mann-Whitney U={u2:.1f}, p={p2:.3f}")

    print("\n" + "=" * 88)
    biased = False
    if len(ds) >= 3 and len(du) >= 3:
        _, pdens = mannwhitneyu(ds, du, alternative="two-sided")
        _, pfrac = mannwhitneyu(fs, fu, alternative="two-sided")
        biased = (pdens < 0.05 and np.median(du) > np.median(ds)) or \
                 (pfrac < 0.05 and np.median(fu) > np.median(fs))
    if biased:
        print("VERDICT: BIAS CONFIRMED -- unscored candidates sit in denser/more complex "
              "tissue than scored ones. The 0% fire-fraction result is likely biased low; "
              "needs the structure-aware calibration search before it can be trusted.")
    else:
        print("VERDICT: NO SIGNIFICANT BIAS DETECTED -- scored and unscored candidates look "
              "similar in local density/complexity. The 0% fire-fraction result is not an "
              "artifact of the calibration requirement.")
    print("=" * 88)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=200)
    ap.add_argument("--n-cross", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(crop=args.crop, n_cross=args.n_cross, seed=args.seed)
