"""
loss_surface.py -- map L over the whole (step, angle_deg) plane and read the
minimum off it.

Why a sweep and not a descent
-----------------------------
The search collapsed to TWO free parameters. `link_weight` left because the
fitted per-streamline weights subsume it exactly; `seed_density` left because
it is monotone -- more streamlines is more columns in Phi, hence a larger
feasible set for the fit, so the residual can only fall. It pinned to the grid
ceiling at 32 and again at 64, which is the behaviour of a budget, not of a
parameter with an optimum.

Two parameters is small enough to evaluate exhaustively, and that is strictly
better than descending on it:

  * the minimum found is GLOBAL. Coordinate descent has stopped after a single
    improving sweep on every run in this project, which is exactly the failure
    an exhaustive grid cannot have;
  * the SHAPE is visible -- flat, ridged, multimodal, or minimised on an edge.
    A descent returns a point and discards all of that;
  * nothing needs tuning: no step size, tolerance, restart count or line search.

Bounds
------
Deliberately wider than the old GRID. The last i74 run selected angle_deg=25.0
(the smallest offered) and step=1.5 (the largest), i.e. BOTH optima sat on a
boundary, which usually means the true optimum is outside the box.

Run
---
    python loss_surface.py                    # phantom, fine grid
    python loss_surface.py --i74 --n 8        # i74, coarse
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import self_optimising_tractography as S3                        # noqa: E402
from sigma_rho_gated_field import GatedField, build_gated_field  # noqa: E402

STEP_RANGE = (0.25, 3.0)
ANGLE_RANGE = (10.0, 80.0)


def sweep(vol, gated, n=12, seed_density=32.0, n_seeds=1, verbose=True):
    _sc, e3, lam = S3.fields_for(vol)
    field, junctions, I0 = S3.field_from_gated(gated, vol, e3=e3, host_only=True)
    if not field or not junctions:
        print("  no HOST junctions -- nothing to sweep")
        return None

    # Same decomposition as the field the streamlines follow.
    D_obs, tissue = S3.anisotropy_target(vol, lam=lam)
    support = tissue & S3.host_support(field, vol.shape,
                                       float(S3.THETA0["assoc_radius"]))
    D_obs = S3.renormalise(D_obs, support)

    steps = np.linspace(*STEP_RANGE, n)
    angles = np.linspace(*ANGLE_RANGE, n)
    L = np.full((n, n), np.nan)
    MSE = np.full((n, n), np.nan)
    PEN = np.full((n, n), np.nan)

    if verbose:
        print(f"  {len(field)} HOST patches, {len(junctions)} junctions, "
              f"support {int(support.sum())} voxels")
        print(f"  sweeping {n}x{n} over step {STEP_RANGE} x angle {ANGLE_RANGE}"
              f" at seed_density={seed_density:g}, averaging {n_seeds} seed(s)")

    t0 = time.time()
    for i, st in enumerate(steps):
        for j, an in enumerate(angles):
            th = dict(S3.THETA0)
            th.update(step=float(st), angle_deg=float(an),
                      seed_density=float(seed_density))
            # Average over RNG realisations: `optimise` reseeds identically for
            # every trial, so a single draw fits one noise realisation and the
            # surface inherits its speckle.
            ls, ms, ps = [], [], []
            for s in range(n_seeds):
                _L, info = S3.loss(vol, field, D_obs, support, th, junctions,
                                   I0, rng=np.random.default_rng(s))
                ls.append(_L); ms.append(info["mse"]); ps.append(info["penalty"])
            L[i, j] = float(np.mean(ls))
            MSE[i, j] = float(np.mean(ms))
            PEN[i, j] = float(np.mean(ps))
        if verbose:
            print(f"    step {st:5.2f}: L {np.nanmin(L[i]):.4f} .. "
                  f"{np.nanmax(L[i]):.4f}   ({time.time() - t0:.0f}s)")

    k = int(np.nanargmin(L))
    bi, bj = np.unravel_index(k, L.shape)
    best = dict(step=float(steps[bi]), angle_deg=float(angles[bj]),
                L=float(L[bi, bj]), mse=float(MSE[bi, bj]),
                penalty=float(PEN[bi, bj]))
    on_edge = bi in (0, n - 1) or bj in (0, n - 1)

    if verbose:
        print(f"\n  minimum  L={best['L']:.4f} at step={best['step']:.3f}, "
              f"angle={best['angle_deg']:.1f} deg "
              f"(mse {best['mse']:.4f} + pen {best['penalty']:.4f})")
        print(f"  surface  L range [{np.nanmin(L):.4f}, {np.nanmax(L):.4f}], "
              f"relief {np.nanmax(L) - np.nanmin(L):.4f}")
        if on_edge:
            print("  WARNING: the minimum sits on a boundary -- widen the "
                  "range, the optimum is probably outside the box")
        else:
            print("  the minimum is INTERIOR -- the box contains the optimum")
    return dict(steps=steps.tolist(), angles=angles.tolist(), L=L.tolist(),
                mse=MSE.tolist(), penalty=PEN.tolist(), best=best,
                on_edge=bool(on_edge), seed_density=float(seed_density))


def plot(res, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = np.array(res["L"])
    steps, angles = np.array(res["steps"]), np.array(res["angles"])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.pcolormesh(angles, steps, L, shading="nearest", cmap="viridis")
    cs = ax.contour(angles, steps, L, 12, colors="w", linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    b = res["best"]
    ax.plot(b["angle_deg"], b["step"], "r*", ms=16, mec="k", mew=0.8,
            label=f"min L={b['L']:.4f}\nstep={b['step']:.2f}, "
                  f"angle={b['angle_deg']:.0f}")
    ax.set_xlabel("angle_deg  (widest turn allowed per step)")
    ax.set_ylabel("step  (voxels advanced per iteration)")
    ax.set_title(title or "Stage 3 loss surface")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.colorbar(im, ax=ax, label="L = MSE + verdict penalty")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(i74=False, n=12, radius=48, stride=6, seed_density=32.0, n_seeds=1,
         outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results")
    print("=" * 88)
    print("LOSS SURFACE -- exhaustive minimum over (step, angle_deg)")
    print("=" * 88)

    if i74:
        from reconstruction import load_volume
        lo = 384 - radius
        vol = load_volume(crop=(lo, 384 + radius),
                          dtype=np.uint8).astype(np.float32)
        cache = os.path.join(outdir, "unified_field_i74.npz")
        gated = (GatedField.load(cache) if os.path.exists(cache)
                 else build_gated_field(vol, block=32, stride=stride))
        tag, title = "i74", f"i74 {vol.shape}"
    else:
        from localized_phantoms import (raw_localized_crossing, CROSS_SHAPE,
                                        CROSS_R1, CROSS_TAPER)
        X = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5,
                                   CROSS_R1, CROSS_TAPER)
        r = 24
        vol = X[tuple(slice(c - r, c + r)
                      for c in (s // 2 for s in CROSS_SHAPE))].astype(np.float32)
        gated = build_gated_field(vol, block=32, stride=stride, verbose=False)
        tag, title = "phantom", f"crossing phantom {vol.shape}"

    res = sweep(vol, gated, n=n, seed_density=seed_density, n_seeds=n_seeds)
    if res is None:
        return None
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"loss_surface_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    p = plot(res, os.path.join(outdir, f"loss_surface_{tag}.png"),
             title=f"Stage 3 loss surface -- {title}")
    print(f"\nwrote {p}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="exhaustive loss surface")
    ap.add_argument("--i74", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--radius", type=int, default=48)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--seed-density", type=float, default=32.0)
    ap.add_argument("--n-seeds", type=int, default=1)
    a = ap.parse_args()
    main(i74=a.i74, n=a.n, radius=a.radius, stride=a.stride,
         seed_density=a.seed_density, n_seeds=a.n_seeds)
