"""
gradient_surface.py -- the sheet the gradient descent actually walks on.

`gradient_optimise` reports where it stopped. That is a fact about the PATH, not
about the surface, and the two questions a reader has -- is there a basin, and
did the descent find it -- can only be answered by drawing the surface itself.

So this evaluates the same smooth loss on a grid over (L, theta) at the alpha
the descent converged to, and draws the recorded path on top of it. The path is
read from `results/gradient_optimise.json`, not recomputed: re-running the
descent would produce a second trajectory that is not the one being explained.

Why a slice and not the whole thing
-----------------------------------
The loss is a scalar over THREE parameters, so its graph is four-dimensional and
cannot be drawn. Holding alpha at the converged value and drawing the (L, theta)
sheet is a slice through that graph -- an honest one, provided it is labelled as
such. The descent moved in alpha too, so its path does not lie exactly in this
plane; it is PROJECTED onto it, and the projection is why a path point can sit
slightly off the surface it is drawn against.

Run
---
    python gradient_surface.py                 # 12 x 12, ~7 min
    python gradient_surface.py --n 8           # coarser, faster
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE = os.path.dirname(_HERE)             # 3_global_tractography/
_ROOT = os.path.dirname(_STAGE)             # repository root
for _p in (_ROOT, _STAGE, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import smooth_tracker as ST                              # noqa: E402
from cone_descent import stage2, seed, FLOOR_FRAC        # noqa: E402
from gradient_optimise import Problem, _bar              # noqa: E402
from theta_validation import CurvedTruth                 # noqa: E402

# Sequential single hue for the loss (magnitude), one categorical hue for the
# path (identity). Never a rainbow for a magnitude.
PATH = "#eb6834"
START = "#1baf7a"


def build(n_seeds=150, curv=0.028, radius=2.0, ymax=24.0, noise=0.0):
    shape = (20, int(2 * ymax + 16), 40)
    truth = CurvedTruth(shape, curv=curv, radius=radius, ymax=ymax)
    vol = truth.volume(noise=noise)
    dense, D_obs, tissue, FA, _sc = stage2(vol, verbose=False)
    s, sd = seed(dense, tissue, FA, "fa", n=n_seeds)
    floor = FLOOR_FRAC * float(vol.max())
    return Problem(vol, dense, s, sd, floor)


def sheet(prob, alpha, n=12, L_range=(2.0, 11.0), theta_range=(0.03, 0.50),
          verbose=True):
    """Loss on a log-spaced (L, theta) grid at fixed alpha."""
    Ls = np.exp(np.linspace(*np.log(L_range), n))
    ths = np.exp(np.linspace(*np.log(theta_range), n))
    Z = np.zeros((n, n))
    t0, tot = time.time(), n * n
    for i, th in enumerate(ths):
        for j, L in enumerate(Ls):
            Z[i, j] = prob.f(np.log([L, th, alpha]))
            k = i * n + j + 1
            if verbose and (k % max(n // 2, 1) == 0 or k == tot):
                el = time.time() - t0
                print(f"    {_bar(k, tot)} {k}/{tot}  "
                      f"elapsed {el/60:.1f}m  eta {(el/k)*(tot-k)/60:.1f}m",
                      flush=True)
    return Ls, ths, Z


def plot(Ls, ths, Z, trace, alpha, path_png):
    """3D sheet for the shape, 2D contour for reading the path.

    Both, because they answer different questions. A 3D surface shows whether
    there is a basin at all; it is a poor thing to trace a trajectory on,
    because depth is ambiguous and the path disappears behind the ridge. The
    contour beneath it is where the path is actually legible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401

    pL = np.array([t["L"] for t in trace])
    pT = np.array([t["theta"] for t in trace])
    pZ = np.array([t["loss"] for t in trace])

    fig = plt.figure(figsize=(14.4, 6.2))
    fig.patch.set_facecolor("white")
    fig.suptitle("The sheet the gradient descent walks on   "
                 f"(slice at alpha = {alpha:.4f}; the descent also moved in "
                 f"alpha, so its path is projected)", fontsize=12)

    LL, TT = np.meshgrid(Ls, ths)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(LL, TT, Z, cmap="Blues", alpha=0.82, linewidth=0,
                    antialiased=True, rstride=1, cstride=1)
    ax.plot(pL, pT, pZ, color=PATH, lw=2.4, marker="o", ms=3.5,
            label="descent path", zorder=10)
    ax.scatter([pL[0]], [pT[0]], [pZ[0]], s=90, color=START, depthshade=False,
               label=f"start  loss {pZ[0]:.4f}")
    ax.scatter([pL[-1]], [pT[-1]], [pZ[-1]], s=140, marker="*", color=PATH,
               depthshade=False, label=f"converged  loss {pZ[-1]:.4f}")
    k = np.unravel_index(np.argmin(Z), Z.shape)
    ax.scatter([Ls[k[1]]], [ths[k[0]]], [Z[k]], s=110, marker="D",
               color="#1A1A1A", depthshade=False,
               label=f"grid min  loss {Z[k]:.4f}")
    ax.set_xlabel("L  (lookahead, vox)")
    ax.set_ylabel("theta  (rad)")
    ax.set_zlabel("align loss")
    ax.set_title("the surface -- is there a basin?", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
    ax.view_init(elev=28, azim=-58)

    ax2 = fig.add_subplot(1, 2, 2)
    cf = ax2.contourf(LL, TT, Z, levels=18, cmap="Blues")
    ax2.contour(LL, TT, Z, levels=18, colors="#00000018", linewidths=0.5)
    ax2.plot(pL, pT, color=PATH, lw=2.2, marker="o", ms=4,
             label="descent path")
    ax2.scatter([pL[0]], [pT[0]], s=110, color=START, zorder=6,
                edgecolors="white", linewidths=1.2, label="start")
    ax2.scatter([pL[-1]], [pT[-1]], s=190, marker="*", color=PATH, zorder=6,
                edgecolors="white", linewidths=1.0, label="converged")
    ax2.scatter([Ls[k[1]]], [ths[k[0]]], s=120, marker="D", color="#1A1A1A",
                zorder=6, edgecolors="white", linewidths=1.0, label="grid min")
    # Where the cone stops being a cone. Below theta ~ the quadrature step the
    # angular prior is narrower than the candidates it weights, so the tracker
    # is effectively locked to straight ahead.
    ax2.axhline(ST.TILT_STEP, color="#B3261E", ls="--", lw=1.6)
    ax2.annotate(f"theta = quadrature step ({ST.TILT_STEP:g})",
                 (Ls[-1], ST.TILT_STEP), xytext=(-6, 5),
                 textcoords="offset points", fontsize=8, color="#B3261E",
                 ha="right",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                           alpha=0.85))
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("L  (lookahead, vox)")
    ax2.set_ylabel("theta  (rad)")
    ax2.set_title("the same sheet, with the path legible", fontsize=10,
                  loc="left")
    ax2.legend(fontsize=8, framealpha=0.9)
    cb = fig.colorbar(cf, ax=ax2, shrink=0.85)
    cb.set_label("align loss  (DARKER = worse)", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path_png, dpi=150, facecolor="white")
    plt.close(fig)
    return path_png


def main(n=12, n_seeds=150, outdir=None, replot=False):
    outdir = outdir or os.path.join(_ROOT, "results")
    with open(os.path.join(outdir, "gradient_optimise.json")) as f:
        res = json.load(f)
    alpha = float(res["alpha"])
    trace = res["trace"]

    # The grid costs ~11 minutes and the figure gets iterated on; redrawing must
    # not recompute it, or the picture would be of a second sample of the same
    # surface rather than of the one already on disk.
    if replot:
        d = np.load(os.path.join(outdir, "gradient_surface.npz"))
        png = plot(d["L"], d["theta"], d["Z"], trace, float(d["alpha"]),
                   os.path.join(outdir, "gradient_surface.png"))
        print(f"  [figure] {png}")
        return
    print("=" * 78)
    print(f"GRADIENT SURFACE -- {n}x{n} sheet over (L, theta) at "
          f"alpha={alpha:.4f}")
    print(f"  descent to redraw: {len(trace)} recorded points, "
          f"loss {trace[0]['loss']:.5f} -> {trace[-1]['loss']:.5f}")
    print("=" * 78)

    prob = build(n_seeds=n_seeds)
    Ls, ths, Z = sheet(prob, alpha, n=n)

    k = np.unravel_index(np.argmin(Z), Z.shape)
    print(f"\n  grid minimum {Z[k]:.6f} at L={Ls[k[1]]:.3f} "
          f"theta={ths[k[0]]:.4f}")
    print(f"  descent ended {trace[-1]['loss']:.6f} at "
          f"L={trace[-1]['L']:.3f} theta={trace[-1]['theta']:.4f}")
    gap = trace[-1]["loss"] - float(Z[k])
    print(f"  descent is {gap:+.6f} vs the grid minimum on this slice "
          + ("(the descent found a BETTER point than the grid -- expected, it "
             "is not restricted to grid nodes)" if gap < 0 else
             "(the grid found a better point: the descent stopped short)"))

    png = plot(Ls, ths, Z, trace, alpha,
               os.path.join(outdir, "gradient_surface.png"))
    print(f"  [figure] {png}")
    np.savez_compressed(os.path.join(outdir, "gradient_surface.npz"),
                        L=Ls, theta=ths, Z=Z, alpha=alpha)
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="draw the gradient descent's sheet")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the saved grid instead of recomputing")
    a = ap.parse_args()
    main(n=a.n, n_seeds=a.seeds, outdir=a.outdir, replot=a.replot)
