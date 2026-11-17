r"""
rho_sweep.py -- Stage 4 along one axis: error against rho (or sigma) at a fixed
value of the other, for one or more Stage 3 runs.

Why a line, not a grid
----------------------
`sigma_rho_match.py` sweeps an n x n (sigma, rho) grid. On i74 the error surface
is nearly flat in sigma and a broad U in rho, so once sigma is pinned the
question that matters is one-dimensional: WHERE along rho is the minimum, and
does it move with the seed count or the Stage 3 objective? A line answers that
at 14 tensor evaluations per run instead of 196, which is what makes comparing
many runs affordable.

What is compared -- identical to sigma_rho_match / coarse_to_fine
-----------------------------------------------------------------
For each run directory written by `descent_loop.py --refine-field`:
  * the volume chunk is rebuilt with `descent_loop.load_setup`,
  * Stage 2 is re-run (`cone_descent.stage2`) only to recover the mask of
    oriented voxels (`n_dirs >= 1`), the voxels the error is scored on,
  * the target is the SAVED accumulated field `refined_field_<variant>.npz`,
  * each point is one plain `tensor_field(vol, sigma, rho)` scored by
    `sigma_rho_match.angular_error` (|cos|, so e3 and -e3 agree).
Runs that share a chunk (same dataset, centre, radius) share one volume load and
one Stage 2 pass.

Output schema (what docs/figures/make_stage4_rho_sweep.py reads)
-----------------------------------------------------------------
    --axis rho    {"rhos": [...],   "sigma": s, "res": {label: {"median": [...],
                                                  "mean": [...], ...}}}
    --axis sigma  {"sigmas": [...], "rho": r,   "res": {label: {...}}}

Reproducing the recorded results (README "Key result")
------------------------------------------------------
    # results/seed_rho_curves.json -- rho at sigma = 1.00, five seed counts (MSE)
    python 4_optimising_parameters/rho_sweep.py \
        --runs results/seed_1000 results/seed_2000 results/seed_3000 \
               results/seed_4000 results/seed_5000 \
        --labels 1000 2000 3000 4000 5000 \
        --out results/seed_rho_curves_rerun.json --check results/seed_rho_curves.json

    # results/sigma_sweep_r24.json -- sigma at rho = 3.792, MSE vs MAE
    python 4_optimising_parameters/rho_sweep.py --axis sigma --fixed 3.792 \
        --lo 0.4 --hi 3.0 --n 11 \
        --runs results/seed_5000 results/seed_5000_mae --labels MSE MAE \
        --out results/sigma_sweep_r24_rerun.json --check results/sigma_sweep_r24.json

`--check` compares the new numbers with a recorded file and exits non-zero if
any differs by more than `--tol` degrees, so a reproduction is a test, not a
judgement by eye. The run directories hold `refined_field_vec1.npz`; they are
not in git, but are in the v1.0.0 release archive.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification"),
           os.path.join(_ROOT, "3_global_tractography")):
    sys.path.insert(0, _p)

from sigma_rho_match import sweep                   # noqa: E402

AXES = {"rho": ("rhos", "sigma"), "sigma": ("sigmas", "rho")}


def log_grid(lo, hi, n):
    """Log-spaced grid, computed exactly as the recorded sweeps were."""
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))


def load_run(run_dir, variant="vec1"):
    with open(os.path.join(run_dir, "descent_loop.json")) as f:
        run = json.load(f)
    npz = os.path.join(run_dir, f"refined_field_{variant.replace('+', '_')}.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"no refined field at {npz} -- run Stage 3 with "
                         f"--refine-field (or unzip the v1.0.0 release archive)")
    meta = dict(dataset=run["dataset"], centre=run["centre"],
                radius=run["radius"], metric=run.get("metric"),
                turn=run.get("turn"), n_seeds=run.get("n_seeds"),
                n_updates=run["runs"][variant].get(
                    "field_updates", len(run["runs"][variant].get("field_moved", []))))
    return np.load(npz)["vec1"], meta


def chunk(meta, mode, cache):
    """(vol, mask) for a run's chunk, computed once per distinct chunk."""
    key = (meta["dataset"], json.dumps(meta["centre"]), meta["radius"], mode)
    if key not in cache:
        import descent_loop as DL
        from cone_descent import stage2
        vol, _origin, label = DL.load_setup(meta["dataset"], meta["centre"],
                                            meta["radius"], verbose=False)
        dense, *_ = stage2(vol, mode=mode, verbose=False)
        cache[key] = (vol, np.asarray(dense.n_dirs) >= 1)
        print(f"  [chunk] {label} {vol.shape} at centre {meta['centre']}: "
              f"{int(cache[key][1].sum())} oriented voxels")
    return cache[key]


def run_sweep(runs, labels, axis, fixed, grid, variant="vec1", mode="adaptive"):
    cache, res = {}, {}
    for run_dir, label in zip(runs, labels):
        target, meta = load_run(run_dir, variant)
        vol, mask = chunk(meta, mode, cache)
        print(f"\n  {label}: {run_dir}  ({meta['metric']}, "
              f"{meta['n_seeds']} seeds, {meta['n_updates']} field updates)")
        if axis == "rho":
            _s, _r, M, E, _b = sweep(vol, target, mask, [fixed], grid)
            med, mean = M[0], E[0]
        else:
            _s, _r, M, E, _b = sweep(vol, target, mask, grid, [fixed])
            med, mean = M[:, 0], E[:, 0]
        res[label] = dict(median=[float(v) for v in med],
                          mean=[float(v) for v in mean],
                          n_updates=meta["n_updates"], metric=meta["metric"],
                          n_seeds=meta["n_seeds"], run_dir=run_dir)
    return res


def summarise(res, grid, axis):
    print("\n" + "=" * 78)
    for label, r in res.items():
        i = int(np.nanargmin(r["median"]))
        edge = "  ON THE GRID EDGE -- the minimum may lie outside" \
            if i in (0, len(grid) - 1) else ""
        print(f"  {label:>8}: {axis}* = {grid[i]:.4f}   median "
              f"{r['median'][i]:.3f} deg   mean {r['mean'][i]:.3f} deg{edge}")
    print("=" * 78)


def check(out, ref, path, tol):
    """Max |difference| against a recorded result, per label and statistic."""
    gk = next(k for k in ("rhos", "sigmas") if k in out)
    worst = float(np.max(np.abs(np.asarray(out[gk]) - np.asarray(ref[gk]))))
    print(f"\n  CHECK against {path}")
    print(f"    grid {gk}: max |diff| {worst:.2e}")
    bad = worst > 1e-9
    for label, r in out["res"].items():
        if label not in ref["res"]:
            print(f"    {label}: not in the recorded file")
            bad = True
            continue
        for stat in ("median", "mean"):
            d = float(np.max(np.abs(np.asarray(r[stat]) -
                                    np.asarray(ref["res"][label][stat]))))
            ok = d <= tol
            bad |= not ok
            print(f"    {label:>8} {stat:6}: max |diff| {d:.4f} deg  "
                  f"{'OK' if ok else '*** DIFFERS ***'}")
    print(f"  {'REPRODUCED' if not bad else 'NOT REPRODUCED'} "
          f"(tolerance {tol} deg)")
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="descent_loop output directories (with --refine-field)")
    ap.add_argument("--labels", nargs="+",
                    help="one key per run in the output (default: directory name)")
    ap.add_argument("--axis", choices=tuple(AXES), default="rho")
    ap.add_argument("--fixed", type=float, default=1.0,
                    help="value of the OTHER scale (sigma for --axis rho)")
    ap.add_argument("--lo", type=float, default=0.7071)
    ap.add_argument("--hi", type=float, default=8.0)
    ap.add_argument("--n", type=int, default=14, help="grid points (log-spaced)")
    ap.add_argument("--variant", default="vec1")
    ap.add_argument("--mode", choices=("adaptive", "global"), default="adaptive")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--check", help="recorded JSON to compare against")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="--check tolerance in degrees")
    a = ap.parse_args()

    labels = a.labels or [os.path.basename(os.path.normpath(r)) for r in a.runs]
    if len(labels) != len(a.runs):
        ap.error("--labels needs one entry per --runs directory")
    grid = log_grid(a.lo, a.hi, a.n)
    gk, fk = AXES[a.axis]
    # Read the reference BEFORE anything is written: --out and --check may name
    # the same file, and checking a file against itself proves nothing.
    ref = None
    if a.check:
        with open(a.check) as f:
            ref = json.load(f)

    print("=" * 78)
    print(f"STAGE 4 LINE SWEEP -- {a.axis} {a.lo:g}..{a.hi:g} ({a.n} log-spaced) "
          f"at {fk} = {a.fixed:g}, {len(a.runs)} run(s)")
    print("=" * 78)
    res = run_sweep(a.runs, labels, a.axis, a.fixed, grid, a.variant, a.mode)
    summarise(res, grid, a.axis)

    out = {gk: [float(v) for v in grid], fk: a.fixed, "res": res}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {a.out}")
    if ref is not None and not check(out, ref, a.check, a.tol):
        sys.exit(1)


if __name__ == "__main__":
    main()
