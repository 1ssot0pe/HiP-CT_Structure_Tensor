"""
descent_curve.py -- the descent's convergence, and what the OLD objective did
at the same time.

The gradient descent minimises the ALIGN loss, not mse(TDI, D_obs). Plotting a
convergence curve labelled "MSE" would therefore be wrong, and wrong in the one
way this whole line of work has been about. So both are drawn, on the same
iterations and -- importantly -- on the SAME tractogram:

    ALIGN   1 - <(d . v)^2>, survival-weighted. What the descent is minimising.
    MSE     mse(TDI, D_obs) with one fitted scalar. The objective Stage 3 used
            to descend on, evaluated at the parameters the align descent chose.

Computing the MSE from the same smooth tractogram, rather than re-tracking with
the hard tracker, is what makes the comparison a statement about the two
OBJECTIVES. Re-tracking would change the tracts at the same moment as the score
and the difference could then be read either way.

Two panels, never a second y-axis: the whole question is whether two quantities
move together, and a dual axis can manufacture agreement or disagreement by
choice of scale.

Run
---
    python descent_curve.py
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE = os.path.dirname(_HERE)             # 3_global_tractography/
_ROOT = os.path.dirname(_STAGE)             # repository root
for _p in (_ROOT, _STAGE, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import smooth_tracker as ST                                  # noqa: E402
import cone_objective as CO                                  # noqa: E402
from cone_descent import stage2, seed, FLOOR_FRAC            # noqa: E402
from gradient_optimise import Problem                        # noqa: E402
from param_volume import load_chunk                          # noqa: E402

ALIGN_C = "#2a78d6"      # categorical slot 1
MSE_C = "#eb6834"        # slot 2
START_C = "#1baf7a"      # slot 3


def evaluate(run, centre=384, radius=32, n_seeds=100, seeding="fa",
             verbose=True):
    """Re-read both objectives at every point on the recorded descent path."""
    vol, _o = load_chunk(centre, radius, verbose=verbose)
    dense, D_obs, tissue, FA, sc = stage2(vol, verbose=False)
    s, sd = seed(dense, tissue, FA, seeding, n=n_seeds)
    floor = FLOOR_FRAC * float(vol.max())
    prob = Problem(vol, dense, s, sd, floor)
    ctx = CO.Context(vol, dense, D_obs, tissue, rho=float(sc["rho_global"]))

    tr = run["descent"]["trace"]
    rows = []
    for i, step in enumerate(tr):
        pos, dirs, w = prob.track(step["L"], step["theta"], step["alpha"])
        # Each STEP as its own two-point segment: that is the geometry the
        # align loss is defined on, so both objectives read the same tracts.
        paths = ST.as_paths(pos, dirs, w, n_keep=0.5)
        rows.append(dict(
            iter=int(step["iter"]), L=float(step["L"]),
            theta_deg=float(np.degrees(step["theta"])),
            alpha_deg=float(np.degrees(step["alpha"])),
            align=float(ST.align_loss(pos, dirs, w, dense)),
            align_recorded=float(step["loss"]),
            mse=float(CO.data_mse(paths, ctx)),
            n_segments=len(paths)))
        if verbose:
            r = rows[-1]
            print(f"    it {r['iter']:3d}  align {r['align']:.6f}  "
                  f"mse {r['mse']:.6f}   L={r['L']:.2f} "
                  f"theta={r['theta_deg']:.1f} alpha={r['alpha_deg']:.1f}",
                  flush=True)

    # The recomputed align must reproduce the recorded one; the tracker is
    # deterministic, so any drift means the seeds or the chunk differ from the
    # run being explained and the MSE column would be about a different problem.
    d = max(abs(r["align"] - r["align_recorded"]) for r in rows)
    if verbose:
        print(f"  reproducibility: max |align - recorded| = {d:.2e} "
              + ("OK" if d < 1e-9 else "*** MISMATCH -- different setup ***"))
    return rows, float(d)


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it = np.arange(len(rows))
    al = np.array([r["align"] for r in rows])
    ms = np.array([r["mse"] for r in rows])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 8.0), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Gradient descent convergence on i74   "
                 "(same tractogram scored two ways)", fontsize=12)

    for ax, y, col, name in ((a1, al, ALIGN_C, "align loss"),
                             (a2, ms, MSE_C, "mse(TDI, D_obs)")):
        ax.plot(it, y, lw=2.2, marker="o", ms=6, color=col)
        ax.scatter([it[0]], [y[0]], s=170, color=START_C, zorder=6,
                   edgecolors="white", linewidths=1.4)
        ax.scatter([it[-1]], [y[-1]], s=220, marker="*", color=col, zorder=6,
                   edgecolors="white", linewidths=1.0)
        ax.annotate(f"START  {y[0]:.5f}", (it[0], y[0]), xytext=(10, 10),
                    textcoords="offset points", fontsize=9, color=START_C)
        ax.annotate(f"END  {y[-1]:.5f}", (it[-1], y[-1]), xytext=(-10, -18),
                    textcoords="offset points", fontsize=9, color=col,
                    ha="right")
        pct = 100.0 * (y[-1] - y[0]) / abs(y[0])
        ax.set_ylabel(name)
        ax.set_title(f"{name}:  {y[0]:.5f} -> {y[-1]:.5f}   ({pct:+.1f}%)",
                     fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#00000010")
        ax.set_facecolor("white")
        # Headroom, so the START/END callouts sit inside the axes instead of
        # being clipped by the frame or colliding with the title.
        ax.margins(y=0.22)

    # Placed in AXES coordinates, not data coordinates: a note anchored to a
    # data point moves with the curve and lands on the title whenever the curve
    # happens to be high there.
    a1.text(0.98, 0.06, "this is what the descent minimises",
            transform=a1.transAxes, fontsize=8.5, color="#555555", ha="right")
    a2.text(0.98, 0.06,
            "this is the objective Stage 3 used to descend on,\n"
            "read at the parameters the align descent chose",
            transform=a2.transAxes, fontsize=8.5, color="#555555", ha="right")
    a2.set_xlabel("descent iteration   (-1 = start, before any step)")
    a2.set_xticks(it)
    a2.set_xticklabels([str(r["iter"]) for r in rows])

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def main(centre=384, radius=32, n_seeds=100, outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results")
    with open(os.path.join(outdir, "param_volume_run.json")) as f:
        run = json.load(f)
    print("=" * 78)
    print("DESCENT CURVE -- align loss and MSE at every iteration")
    print("=" * 78)
    rows, drift = evaluate(run, centre=centre, radius=radius, n_seeds=n_seeds)

    al = np.array([r["align"] for r in rows])
    ms = np.array([r["mse"] for r in rows])
    print(f"\n  align  {al[0]:.6f} -> {al[-1]:.6f}  "
          f"({100 * (al[-1] - al[0]) / al[0]:+.1f}%)")
    print(f"  mse    {ms[0]:.6f} -> {ms[-1]:.6f}  "
          f"({100 * (ms[-1] - ms[0]) / ms[0]:+.1f}%)")
    same = np.sign(al[-1] - al[0]) == np.sign(ms[-1] - ms[0])
    print(f"  -> the two objectives moved "
          + ("in the SAME direction over the descent"
             if same else "in OPPOSITE directions over the descent"))
    if len(al) > 2:
        r = float(np.corrcoef(al, ms)[0, 1])
        print(f"  -> correlation across iterations: {r:+.3f}")

    p = plot(rows, os.path.join(outdir, "descent_curve.png"))
    print(f"  [figure] {p}")
    with open(os.path.join(outdir, "descent_curve.json"), "w") as f:
        json.dump(dict(rows=rows, reproducibility=drift), f, indent=2,
                  default=float)
    print("=" * 78)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="descent convergence, both losses")
    ap.add_argument("--centre", type=int, default=384)
    ap.add_argument("--radius", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    main(centre=a.centre, radius=a.radius, n_seeds=a.seeds, outdir=a.outdir)
