"""
i74_directional_calibration.py -- fix for the confirmed selection bias in
real_i74_mass_conservation.py: a blind random-shell search for a "confidently
single" calibration neighbour excludes dense/complex-tissue candidates almost
entirely (i74_calibration_bias_check.py: unscored candidates have far higher
local density and ~0% locally-"single" neighbourhood patches, p<0.001) -- i.e.
it throws away exactly the candidates most likely to BE real crossings.

Fix: instead of random offsets, WALK outward along each of HOST's own two
recovered peak directions (both signs -- 4 walks per candidate), the same
continuity-following mechanism peak_straightness.py already uses to trace
fibres. A real fibre continues past the local ambiguity onto a single-family
segment; walking the actual fibre axis is far more likely to reach genuine
single-fibre tissue in dense regions than a random nearby voxel. At each step,
classify the patch; the first "single" hit along a walk is the calibration
point.

Reports: (a) overall coverage vs. the old random-shell method, (b) coverage
recovered specifically on the subset the old method COULDN'T calibrate
(the dense/complex ones), and (c) the resulting ratio distribution / fire
fraction on the FULL, now-less-biased sample.

Run:  .venv/Scripts/python -u i74_directional_calibration.py [--crop N] [--n-cross N]
"""

import os as _sr_os, sys as _sr_sys
_SR_ROOT = _sr_os.path.dirname(_sr_os.path.dirname(__file__))
_sr_sys.path.insert(0, _SR_ROOT)  # repo root (host_fod, crossing_shape_3d, kissing_phantom, ...)
_sr_sys.path.insert(0, _sr_os.path.join(_SR_ROOT, "3_global_tractography"))  # phase1/phase2 certifiers

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.reconstruction import load_volume, DEFAULT_VOLUME_DIR
from core.host_fod import analyse, classify
from fibre_continuity import local_fibreness, V
from peak_straightness import collect_crossings, HALF
from mass_conservation_validate import local_calibration, THRESH, EPS, _patch_at
from real_i74_mass_conservation import _random_shell_positions

MAX_STEPS = 30
CONE_DEG = 35.0
R_STOP = 0.55


def _walk_for_single(vol, pos0, dir_xyz, max_steps=MAX_STEPS, cone=CONE_DEG,
                      r_stop=R_STOP, half=HALF):
    """Walk from pos0 along dir_xyz, following the dominant local fibre
    direction (same stepping rule as phase2's continuity walk). Return the
    peak intensity of the first patch along the way HOST classifies "single",
    or None if the fibre is lost / walk exits the volume first."""
    pos = np.array(pos0, float)
    prev = np.asarray(dir_xyz, float)
    prev /= np.linalg.norm(prev) + EPS
    coscone = np.cos(np.deg2rad(cone))
    for _ in range(max_steps):
        if not all(half + 2 <= pos[i] < vol.shape[i] - half - 2 for i in range(3)):
            return None
        patch = _patch_at(vol, pos, half)
        if patch.shape == (2 * half + 1,) * 3 and patch.mean() >= 0.05 * vol.max():
            m = analyse(patch)
            if m is not None and classify(m) == "single":
                return float(patch.max())
        f = local_fibreness(vol, pos, half=half)
        if f is None:
            return None
        dots = V @ prev
        incone = np.abs(dots) >= coscone
        if not incone.any():
            return None
        idx = np.where(incone)[0]
        b = idx[np.argmax(f[idx])]
        if float(f[b]) / (float(f.max()) + EPS) < r_stop:
            return None
        u = V[b] * (1.0 if dots[b] >= 0 else -1.0)
        pos = pos + np.array([u[2], u[1], u[0]])
        prev = u
    return None


def directional_calibration(vol, pos, dirs):
    hits = []
    for d in (dirs[0], -dirs[0], dirs[1], -dirs[1]):
        v = _walk_for_single(vol, pos, d)
        if v is not None:
            hits.append(v)
    return float(np.median(hits)) if hits else np.nan


def run(crop=200, n_cross=60, save="i74_directional_calibration.png", seed=0):
    print("=" * 88)
    print(f"i74 DIRECTIONAL CALIBRATION -- walk-along-fibre fix for the random-shell "
          f"selection bias ({DEFAULT_VOLUME_DIR.name})")
    print("=" * 88)
    c0 = (768 - crop) // 2
    vol = load_volume(crop=(c0, c0 + crop)).astype(np.float64)
    print(f"  volume {vol.shape}, mean={vol.mean():.1f}")
    cx = collect_crossings(vol, n_cross, seed=seed)
    print(f"  found {len(cx)} candidates")

    rng = np.random.default_rng(seed + 1)   # reproduce the OLD method's split for comparison
    old_scored = []
    for pos, dirs in cx:
        cal_positions = _random_shell_positions(pos, vol.shape, rng)
        calib = local_calibration(vol, HALF, cal_positions)
        old_scored.append(not np.isnan(calib))

    ratios, was_old_scored = [], []
    n_new_scored = 0
    n_recovered = 0   # old couldn't score it, new can
    for i, ((pos, dirs), old_ok) in enumerate(zip(cx, old_scored)):
        print(f"    scoring {i+1}/{len(cx)}", end="\r", flush=True)
        calib = directional_calibration(vol, pos, dirs)
        if np.isnan(calib):
            continue
        n_new_scored += 1
        if not old_ok:
            n_recovered += 1
        val = float(vol[tuple(int(round(c)) for c in pos)])
        ratios.append(val / (calib + EPS))
        was_old_scored.append(old_ok)
    print("-" * 88)

    ratios = np.array(ratios)
    was_old_scored = np.array(was_old_scored)
    n = len(ratios)
    n_old_scored = int(np.sum(old_scored))
    n_old_unscored = len(cx) - n_old_scored

    print(f"  OLD (random shell)   coverage: {n_old_scored}/{len(cx)} = {n_old_scored/len(cx):.0%}")
    print(f"  NEW (directional)    coverage: {n_new_scored}/{len(cx)} = {n_new_scored/len(cx):.0%}")
    print(f"  of the {n_old_unscored} candidates OLD method COULDN'T calibrate, "
          f"NEW method recovered {n_recovered} ({n_recovered/max(n_old_unscored,1):.0%})")

    if n:
        fire_frac = float(np.mean(ratios > THRESH))
        fire_frac_recovered = float(np.mean(ratios[~was_old_scored] > THRESH)) if np.any(~was_old_scored) else float("nan")
        fire_frac_old = float(np.mean(ratios[was_old_scored] > THRESH)) if np.any(was_old_scored) else float("nan")
        print(f"\n  FULL sample (n={n}): ratio median={np.median(ratios):.2f}, "
              f"fire fraction (ratio>{THRESH}) = {fire_frac:.0%}")
        print(f"  previously-scored subset (n={int(np.sum(was_old_scored))}): "
              f"fire fraction = {fire_frac_old:.0%}")
        print(f"  newly-recovered subset (n={int(np.sum(~was_old_scored))}, the dense/complex "
              f"ones the old method dropped): fire fraction = {fire_frac_recovered:.0%}")
    print("=" * 88)

    if n:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        bins = np.linspace(0, max(3.0, ratios.max() * 1.05), 30)
        ax[0].hist(ratios[was_old_scored], bins=bins, color="tab:blue", alpha=0.7, label="also scored by old method")
        ax[0].hist(ratios[~was_old_scored], bins=bins, color="tab:red", alpha=0.7, label="recovered (old couldn't score)")
        ax[0].axvline(THRESH, color="k", ls="--")
        ax[0].set_xlabel("mass-conservation ratio"); ax[0].set_ylabel("count")
        ax[0].set_title(f"n={n}, fire fraction={fire_frac:.0%}")
        ax[0].legend(fontsize=8)

        labels = ["old\ncoverage", "new\ncoverage"]
        ax[1].bar(labels, [n_old_scored / len(cx), n_new_scored / len(cx)], color=["tab:gray", "tab:green"])
        ax[1].set_ylim(0, 1)
        ax[1].set_ylabel("fraction of candidates calibrated")
        ax[1].set_title(f"coverage: {n_old_scored}/{len(cx)} -> {n_new_scored}/{len(cx)}")
        fig.suptitle("Directional (walk-along-fibre) calibration vs. random-shell baseline", fontsize=12)
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
