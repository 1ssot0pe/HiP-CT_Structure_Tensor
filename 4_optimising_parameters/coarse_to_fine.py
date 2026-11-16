r"""
coarse_to_fine.py -- locate the best (sigma, rho) by refining the grid, not by
making one grid finer everywhere.

Why not a single fine grid
--------------------------
The error surface measured so far is a broad U in rho and nearly flat in sigma.
A single n x n sweep spends n^2 tensor evaluations resolving regions that are
obviously wrong: at n = 7 over the standard range, 40 of the 49 points sit more
than 5 deg above the minimum and none of them change the answer. Doubling n to
halve the spacing costs 4x for a result that only improves near one point.

Refining instead: sweep coarsely, take the best cell, rebuild a grid spanning
its immediate neighbours, repeat. Each level costs the same n^2 but the spacing
shrinks geometrically, so k levels reach the resolution of a single n^k grid for
k*n^2 evaluations instead of n^(2k).

WHAT THIS ASSUMES, AND HOW IT IS CHECKED
----------------------------------------
Refining around one cell assumes the surface is unimodal -- if there were a
second, better basin elsewhere, zooming into the first would never find it. That
assumption is TESTED rather than trusted: the sign changes in the coarse
level's rho profile are counted, and more than one interior extremum makes the
result suspect and says so. Measured on two chunks and two metrics, every rho
profile had exactly one sign change, so the assumption holds on this data -- but
it is re-checked per run rather than inherited.

The bracket is also checked. If the best cell sits on the edge of a level, the
zoom is EXTRAPOLATING, and the run reports that instead of quietly returning a
boundary value as if it were an optimum.

Run
---
    python coarse_to_fine.py --run ../results/acc_r12 --levels 3 --n 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification"),
           os.path.join(_ROOT, "3_global_tractography")):
    sys.path.insert(0, _p)

from core.tensor import tensor_field                        # noqa: E402
from sigma_rho_match import angular_error, scale_range  # noqa: E402

EPS = 1e-12


def _grid(lo, hi, n):
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))


def _zoom(vals, i):
    """The bracket around index i: its immediate neighbours, or the edge pair.

    Returns (lo, hi, on_edge). On an edge the bracket is the outermost pair,
    which means the next level cannot move past it -- reported, not hidden.
    """
    if len(vals) < 2:
        return float(vals[0]), float(vals[0]), True
    if i == 0:
        return float(vals[0]), float(vals[1]), True
    if i == len(vals) - 1:
        return float(vals[-2]), float(vals[-1]), True
    return float(vals[i - 1]), float(vals[i + 1]), False


def search(vol, target, mask, s_lo, s_hi, r_lo, r_hi, n=5, levels=3,
           stat="mean", verbose=True):
    """Coarse-to-fine (sigma, rho) search. Returns (best, log)."""
    log = []
    best = None
    for lv in range(levels):
        sg, rh = _grid(s_lo, s_hi, n), _grid(r_lo, r_hi, n)
        M = np.full((n, n), np.nan)
        t0 = time.time()
        for i, s in enumerate(sg):
            for j, r in enumerate(rh):
                e3 = tensor_field(vol, float(s), float(r), invariants=False)["e3"]
                med, mean, p95, cnt = angular_error(e3, target, mask)
                M[i, j] = mean if stat == "mean" else med
                if best is None or M[i, j] < best["value"]:
                    best = dict(sigma=float(s), rho=float(r),
                                value=float(M[i, j]), median=med, mean=mean,
                                p95=p95, n=int(cnt), level=lv)
        k = np.unravel_index(int(np.nanargmin(M)), M.shape)

        # Is the level's rho profile unimodal? More than one interior extremum
        # means zooming into one basin may be discarding another.
        prof = M[k[0]]
        d = np.sign(np.diff(prof))
        n_sign = int((np.diff(d[d != 0]) != 0).sum()) if (d != 0).any() else 0

        s2 = _zoom(sg, k[0])
        r2 = _zoom(rh, k[1])
        row = dict(level=lv, n=n, sigma=[float(v) for v in sg],
                   rho=[float(v) for v in rh], values=M.tolist(),
                   best=dict(sigma=float(sg[k[0]]), rho=float(rh[k[1]]),
                             value=float(M[k])),
                   sigma_on_edge=bool(s2[2]), rho_on_edge=bool(r2[2]),
                   rho_sign_changes=n_sign, seconds=time.time() - t0)
        log.append(row)
        if verbose:
            print(f"  level {lv}: {n}x{n} over sigma [{sg[0]:.3f},{sg[-1]:.3f}] "
                  f"rho [{rh[0]:.3f},{rh[-1]:.3f}]  ->  best "
                  f"sigma={sg[k[0]]:.4f} rho={rh[k[1]]:.4f} "
                  f"{stat}={M[k]:.3f} deg  ({row['seconds']:.0f}s)")
            flags = []
            if s2[2]:
                flags.append("sigma ON EDGE")
            if r2[2]:
                flags.append("rho ON EDGE")
            if n_sign > 1:
                flags.append(f"rho profile has {n_sign} interior extrema "
                             f"-- NOT unimodal, the zoom may discard a basin")
            if flags:
                print(f"           {' | '.join(flags)}")
        s_lo, s_hi = s2[0], s2[1]
        r_lo, r_hi = r2[0], r2[1]
        if s_hi - s_lo < 1e-6 and r_hi - r_lo < 1e-6:
            break
    return best, log


def main(run_dir, variant="vec1", n=5, levels=3, stat="mean", mode="adaptive",
         outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results", "coarse_to_fine")
    os.makedirs(outdir, exist_ok=True)

    import descent_loop as DL
    from cone_descent import stage2

    with open(os.path.join(run_dir, "descent_loop.json")) as f:
        run = json.load(f)
    d = run["runs"][variant]

    vol, origin, label = DL.load_setup(run["dataset"], run["centre"],
                                       run["radius"], verbose=False)
    dense, _D, tissue, _FA, scales = stage2(vol, mode=mode, verbose=False)
    mask = np.asarray(dense.n_dirs) >= 1

    npz = os.path.join(run_dir, f"refined_field_{variant.replace('+', '_')}.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"no refined field at {npz}; run stage 3 with "
                         f"--refine-field")
    target = np.load(npz)["vec1"]

    s_lo, s_hi, r_lo, r_hi, prov = scale_range(scales)
    print("=" * 78)
    print(f"COARSE-TO-FINE (sigma, rho) -- {label} {vol.shape}, "
          f"{levels} levels of {n}x{n} = {levels * n * n} evaluations")
    print(f"  vs a single grid of the same final spacing: "
          f"{(n ** levels) ** 2} evaluations")
    print(f"  start range from {prov}")
    print("=" * 78)
    best, log = search(vol, target, mask, s_lo, s_hi, r_lo, r_hi, n=n,
                       levels=levels, stat=stat)
    print(f"\n  BEST  sigma={best['sigma']:.4f}  rho={best['rho']:.4f}  "
          f"{stat}={best['value']:.3f} deg  "
          f"(median {best['median']:.3f}, p95 {best['p95']:.3f}) "
          f"found at level {best['level']}")

    out = dict(run_dir=run_dir, variant=variant, label=label, stat=stat,
               n=n, levels=levels, evaluations=levels * n * n,
               equivalent_single_grid=(n ** levels) ** 2,
               field_updates=len(d.get("field_moved", [])),
               metric=run.get("metric"), best=best, levels_log=log)
    p = os.path.join(outdir, f"coarse_to_fine_{os.path.basename(run_dir)}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote {p}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", default="vec1")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--stat", choices=("mean", "median"), default="mean")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    main(a.run, variant=a.variant, n=a.n, levels=a.levels, stat=a.stat,
         outdir=a.outdir)
