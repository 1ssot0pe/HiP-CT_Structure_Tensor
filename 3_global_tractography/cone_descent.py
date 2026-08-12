"""
cone_descent.py -- take Stage 2's HOST field and ask, directly, whether the cone
tracker's three parameters have a minimum.

    L      LOOKAHEAD     (voxels)   how far ahead evidence is gathered
    alpha  ANGULAR STEP  (radians)  how finely the cone is sampled
    theta  CONE WIDTH    (radians)  the widest turn allowed in one step

Everything upstream is unchanged: Stage 1 certifies (sigma, rho), Stage 2 routes
patches to HOST, and `DenseField` paints one direction per voxel with HOST's
second direction over the top. This file starts where that finishes.

What is deliberately simple here
--------------------------------
* ONE score. mse(TDI, D_obs) over the tissue mask, with a single scalar fitted
  by least squares so the tracker is not penalised for producing a different
  NUMBER of tracts than the target's normalisation implies. No verdict penalty,
  no L1 weight fit, no per-streamline weights -- the question is whether the
  parameters alone move the fit, and every extra term is a place for the answer
  to come from something other than the parameters.
* alpha IS searched. `dense_optimise` fixes it on the argument that finer is
  always at least as good, so a descent walks to the edge of its ladder. That
  argument is about cost, and it is a prediction: the MSE should fall
  monotonically in 1/alpha and flatten. It is cheap to check rather than assume,
  and the sweep below is what checks it -- if alpha instead shows an interior
  minimum, the argument is wrong.
* NO iteration cap. The descent runs until no single-coordinate move on the
  ladder lowers the MSE. It is on a finite grid and the score strictly decreases,
  so it terminates on its own; a cap would only stop it early and call that
  convergence.

Two seedings, because seeding is the other thing that decides a tractogram:
    random  uniform over oriented tissue -- no assumption about where fibres are
    fa      only where FA > 0.7 -- seeds only in confidently anisotropic tissue

Run
---
    python cone_descent.py                    # crossing phantom, both seedings
    python cone_descent.py --quick
    python cone_descent.py --i74              # real data
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import self_optimising_tractography as S3               # noqa: E402
import dense_tracker as DT                              # noqa: E402
from dense_field import DenseField                      # noqa: E402
from tract_density import rasterise                     # noqa: E402
from tensor import field_for                            # noqa: E402
from reconstruction import tissue_mask, VOXEL_SIZE_UM   # noqa: E402
from select_scales import select_scales                 # noqa: E402
from sigma_rho_gated_field import build_gated_field, DEFAULT_STRIDE  # noqa: E402
from localized_phantoms import (raw_localized_crossing,  # noqa: E402
                                CROSS_SHAPE, CROSS_R1, CROSS_TAPER)

EPS = 1e-12
FLOOR_FRAC = 0.05        # support below this fraction of vol.max() ends a walk
FA_THRESHOLD = 0.7
N_SEEDS = 5000           # a compute budget, held fixed across every evaluation

# The three ladders the descent moves on. alpha's lower end is set by runtime:
# cost scales as (theta / alpha)^2.
LADDER = dict(
    L=[2.0, 3.0, 4.0, 6.0, 8.0, 10.0],
    alpha=[0.40, 0.30, 0.20, 0.15, 0.10],
    theta=[0.10, 0.15, 0.20, 0.30, 0.45, 0.60],
)
LADDER_QUICK = dict(L=[2.0, 4.0, 8.0], alpha=[0.40, 0.20],
                    theta=[0.15, 0.30, 0.60])


# ---------------------------------------------------------------------------
# stage 1 + 2, then stop
# ---------------------------------------------------------------------------

def stage2(vol, mode="adaptive", block=32, stride=DEFAULT_STRIDE, verbose=True):
    """Run the pipeline up to and including HOST, and hand back what Stage 3 eats.

    Returns (dense, D_obs, tissue, FA, scales). D_obs is the SAME
    eigen-decomposition's anisotropy that produced e3 -- not a second tensor at
    a second scale. `scales` comes back too because the integration scale rho is
    the scale D_obs was measured at, and anything comparing a tractogram against
    it has to know that -- see `cone_objective.data_mse_scalematched`.
    """
    tissue = tissue_mask(vol)
    if verbose:
        print(f"  [stage 1] tissue {int(tissue.sum())}/{tissue.size} "
              f"({tissue.mean():.1%}); certifying scales ...")
    scales = select_scales(vol, mode=mode, block=block, verbose=False)
    t = field_for(vol, scales, invariants=True)
    e3, lam, FA = t["e3"], t["lam"], t["FA"]

    if verbose:
        print("  [stage 2] routing patches to HOST ...")
    gated = build_gated_field(vol, scales=scales, e3_xyz=e3[::-1], stride=stride,
                              verbose=False)
    dense = DenseField.build(gated, e3, tissue=tissue, host_only=True)

    D_obs, _ = S3.anisotropy_target(vol, lam=lam, tissue=tissue)
    D_obs = S3.renormalise(D_obs, tissue)
    if verbose:
        print(f"  [stage 2] {int(gated.routing_mask().sum())}/{gated.n} patches "
              f"routed; dense field: {dense.summary()}")
        print(f"  [stage 2] FA over tissue: median {np.median(FA[tissue]):.3f}, "
              f"{(FA[tissue] > FA_THRESHOLD).mean():.1%} above {FA_THRESHOLD:g}")
    return dense, D_obs, tissue, FA, scales


# ---------------------------------------------------------------------------
# the two seedings
# ---------------------------------------------------------------------------

def _dirs_at(dense, idx):
    """Seeds along vec1 at every voxel, plus a second along vec2 where Stage 2
    resolved one. Without the second family a crossing is only ever entered
    along one of its fibres and its density can never reach two."""
    z, y, x = idx[:, 0], idx[:, 1], idx[:, 2]
    pos = [idx.astype(float)]
    dirs = [dense.vec1[:, z, y, x].T]
    if dense.n_multi:
        m = dense.n_dirs[z, y, x] >= 2
        if m.any():
            pos.append(idx[m].astype(float))
            dirs.append(dense.vec2[:, z[m], y[m], x[m]].T)
    return np.concatenate(pos), np.concatenate(dirs)


def seed(dense, tissue, FA, how, n=N_SEEDS, fa_threshold=FA_THRESHOLD, seed_rng=0):
    """(positions, directions) for one seeding strategy.

    random : uniform over oriented tissue. Assumes nothing about where fibres
             are, so it also samples the weakly-oriented tissue an FA gate
             throws away -- which is either honest coverage or wasted tracts,
             and that is the comparison.
    fa     : only voxels with FA > threshold. Fewer, better-conditioned starts.

    Both are drawn ONCE and held fixed for the whole sweep. Redrawing per
    evaluation would make two reads of the same parameters differ, and a descent
    cannot tell that noise from an improvement.
    """
    oriented = tissue & (dense.n_dirs >= 1)
    if how == "fa":
        oriented = oriented & (np.asarray(FA) > float(fa_threshold))
    idx = np.argwhere(oriented)
    if not len(idx):
        return np.zeros((0, 3)), np.zeros((0, 3))
    if len(idx) > n:
        rng = np.random.default_rng(seed_rng)
        idx = idx[np.sort(rng.choice(len(idx), size=n, replace=False))]
    return _dirs_at(dense, idx)


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------

def rings(alpha, theta):
    """Tilt rings the cone actually carries at (alpha, theta).

    `dense_tracker.cone_weights` builds round(theta / alpha) rings, so theta is
    QUANTISED in units of alpha and the two parameters cannot be read
    independently. Two settings with the same ring count are not merely similar,
    they are the identical cone and produce bit-identical tracts -- at
    alpha = 0.2 that makes theta 0.15 and theta 0.30 the same experiment.
    Reporting this alongside the MSE is what stops that degeneracy being read as
    a flat region of the loss surface.
    """
    return int(round(float(theta) / float(alpha)))


def mse_tdi(vol, dense, D_obs, support, seeds, sdirs, L, alpha, theta,
            floor=None):
    """mse(s * TDI, D_obs) over `support`, minimised over the single scalar s.

    Why s is fitted rather than fixed
    ---------------------------------
    D_obs is normalised to unit mean over the support; a TDI is in fibres per
    voxel. The two have no common unit, so an unfitted comparison scores the
    seed budget -- doubling the seeds doubles the TDI and changes the MSE
    without changing a single tract's SHAPE. One least-squares scalar removes
    exactly that and nothing else, which is what makes the remaining differences
    attributable to (L, alpha, theta).
    """
    floor = FLOOR_FRAC * float(vol.max()) if floor is None else float(floor)
    t0 = time.time()
    paths = DT.track(vol, dense, seeds, sdirs, alpha=float(alpha),
                     lookahead=float(L), theta=float(theta), floor=floor)
    D = rasterise(paths, vol.shape, step=DT.STEP)
    d, o = D[support], D_obs[support]
    dd = float(d @ d)
    s = float(d @ o) / dd if dd > EPS else 0.0
    return dict(mse=float(np.mean((s * d - o) ** 2)), scale=s,
                n_tracts=len(paths), rings=rings(alpha, theta),
                mean_len=float(np.mean([len(p) for p in paths])) if paths else 0.0,
                coverage=float((d > 0).mean()), seconds=time.time() - t0)


# ---------------------------------------------------------------------------
# the sweep and the descent
# ---------------------------------------------------------------------------

def sweep(vol, dense, D_obs, support, seeds, sdirs, ladder, verbose=True):
    """Every (L, alpha, theta) on the ladder. This is what the 3D plot draws.

    A descent alone cannot answer "is there a minimum" -- it reports where it
    stopped, which is a fact about the path, not the surface. The full grid is
    what distinguishes a single minimum from a flat basin from a monotone slope
    running off the edge of the ladder.
    """
    combos = list(itertools.product(ladder["L"], ladder["alpha"],
                                    ladder["theta"]))
    if verbose:
        print(f"\n  sweeping {len(combos)} settings "
              f"({len(seeds)} seeds, {int(support.sum())} scored voxels)")
        print(f"    {'L':>5} {'alpha':>6} {'theta':>6} {'rings':>6} | "
              f"{'MSE':>10} {'tracts':>7} {'cover':>7} {'s':>6}")
    rows = []
    for L, a, th in combos:
        r = mse_tdi(vol, dense, D_obs, support, seeds, sdirs, L, a, th)
        rows.append(dict(L=float(L), alpha=float(a), theta=float(th), **r))
        if verbose:
            print(f"    {L:5.1f} {a:6.2f} {th:6.2f} {r['rings']:6d} | "
                  f"{r['mse']:10.5f} {r['n_tracts']:7d} {r['coverage']:6.1%} "
                  f"{r['seconds']:6.1f}")
    if verbose:
        distinct = {(x["L"], x["alpha"], x["rings"]) for x in rows}
        print(f"    -> {len(distinct)} DISTINCT cones in {len(rows)} settings "
              f"(theta is quantised in units of alpha -- see `rings`)")
    return rows


def descend(vol, dense, D_obs, support, seeds, sdirs, ladder, start=None,
            verbose=True):
    """Coordinate descent on (L, alpha, theta), run to exhaustion.

    Coordinate descent, not a gradient: the MSE is piecewise constant in these
    parameters -- the tracker takes an argmax over a discrete cone and stops on
    a hard floor -- so a change too small to alter any of those decisions changes
    nothing, and one that does alter them changes the MSE in a jump. There is no
    derivative to descend.

    No iteration limit. Each accepted move strictly lowers the MSE and the grid
    is finite, so the loop cannot cycle and stops only when the point is a
    coordinate-wise minimum.
    """
    p = dict(start or {k: v[len(v) // 2] for k, v in ladder.items()})
    best = mse_tdi(vol, dense, D_obs, support, seeds, sdirs, **p)["mse"]
    path = [dict(p, mse=best, round=-1)]
    n_eval = 1
    if verbose:
        print(f"\n  descent from L={p['L']:g} alpha={p['alpha']:g} "
              f"theta={p['theta']:g}: MSE {best:.5f}")
    rnd = 0
    while True:
        improved = False
        for key, values in ladder.items():
            for v in values:
                if v == p[key]:
                    continue
                trial = dict(p, **{key: v})
                m = mse_tdi(vol, dense, D_obs, support, seeds, sdirs,
                            **trial)["mse"]
                n_eval += 1
                if np.isfinite(m) and m < best - 1e-12:
                    best, p, improved = m, trial, True
        path.append(dict(p, mse=best, round=rnd))
        if verbose:
            print(f"    round {rnd}: MSE {best:.5f} at L={p['L']:g} "
                  f"alpha={p['alpha']:g} theta={p['theta']:g}")
        if not improved:
            if verbose:
                print(f"    stopped -- no coordinate move lowers the MSE "
                      f"({n_eval} evaluations)")
            break
        rnd += 1
    return p, best, path, n_eval


def verdict(rows, ladder):
    """Does the swept surface have an INTERIOR minimum, or does it run off an edge?

    A descent reports where it stopped; that is a fact about the path. The grid
    is what can distinguish the three things it might have stopped on:

      * an interior optimum -- the minimum is off every face of the ladder, so
        the score genuinely selects a cone;
      * a boundary -- the minimum sits on a face, meaning the score is still
        improving in that direction and the ladder, not the data, chose the
        value. Widening the ladder would move the answer;
      * a DEGENERATE optimum -- the minimum has zero rings, i.e. no cone at all.
        This is the case worth naming separately, because a score minimised by
        the tracker not turning is not selecting a tractography; it is
        preferring not to have one, and no widening of the ladder fixes that.
    """
    best = min(rows, key=lambda x: x["mse"])
    detail, edges = [], []
    for key in ("L", "alpha", "theta"):
        vals = sorted(ladder[key])
        if best[key] <= vals[0]:
            edges.append(f"{key} at its LOWEST value ({vals[0]:g})")
        elif best[key] >= vals[-1]:
            edges.append(f"{key} at its HIGHEST value ({vals[-1]:g})")

    turning = [x for x in rows if x["rings"] >= 1]
    if best["rings"] == 0:
        bt = min(turning, key=lambda x: x["mse"]) if turning else None
        summary = ("NO usable minimum -- the best score is a DEGENERATE cone "
                   "(0 rings: the tracker cannot turn at all)")
        detail.append(f"minimum MSE {best['mse']:.5f} at L={best['L']:g} "
                      f"alpha={best['alpha']:g} theta={best['theta']:g}")
        if bt:
            detail.append(
                f"best cone that CAN turn: MSE {bt['mse']:.5f} at L={bt['L']:g} "
                f"alpha={bt['alpha']:g} theta={bt['theta']:g} "
                f"({bt['rings']} rings) -- {bt['mse'] - best['mse']:+.5f} worse")
            deg = [x["mse"] for x in rows if x["rings"] == 0]
            detail.append(
                f"every 0-ring setting scores {min(deg):.4f}-{max(deg):.4f}; "
                f"every turning cone scores "
                f"{min(x['mse'] for x in turning):.4f}-"
                f"{max(x['mse'] for x in turning):.4f}")
        detail.append("=> MSE(TDI, D_obs) alone does not select a cone. It is "
                      "monotone in theta towards zero, so the tracker is scored "
                      "best when it is switched off.")
    elif edges:
        summary = ("minimum is ON THE LADDER BOUNDARY -- " + "; ".join(edges))
        detail.append(f"minimum MSE {best['mse']:.5f} at L={best['L']:g} "
                      f"alpha={best['alpha']:g} theta={best['theta']:g}")
        detail.append("=> the ladder chose this value, not the data. Widen it "
                      "before quoting the number.")
    else:
        summary = "INTERIOR minimum -- the score selects a cone"
        detail.append(f"minimum MSE {best['mse']:.5f} at L={best['L']:g} "
                      f"alpha={best['alpha']:g} theta={best['theta']:g} "
                      f"({best['rings']} rings)")
    return dict(summary=summary, detail=detail, on_boundary=edges,
                degenerate=bool(best["rings"] == 0), best=best)


# ---------------------------------------------------------------------------
# the picture
# ---------------------------------------------------------------------------

def plot(results, out_png, title, basin_frac=0.05):
    """Two 3D views per seeding: the whole surface, then only its basin.

    LEFT   every swept point, MSE on a single-hue sequential ramp (dark = worse),
           with the descent's path drawn through it and its endpoint ringed.
    RIGHT  only the points within `basin_frac` of the global minimum. This is
           the panel that answers the question: one isolated dot means a single
           sharp optimum, a connected cloud means a basin whose exact location is
           not determined by the data, and a cloud pinned against a ladder face
           means the ladder is what stopped the descent, not the surface.

    DEGENERATE cones -- theta below alpha/2, so round(theta/alpha) = 0 and the
    cone collapses to its straight-ahead ray -- are ringed in red in BOTH panels.
    They are not a setting of the tracker, they are the tracker switched off:
    every streamline is a straight line from its seed. Drawing them as ordinary
    grid points would let "the minimum is at low theta" be read as a tuning
    result when it is the score declining to have a tractography at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    modes = list(results)
    fig = plt.figure(figsize=(6.2 * 3, 5.4 * len(modes)))
    fig.suptitle(title, fontsize=13)
    ncol = 3

    for r, mode in enumerate(modes):
        res = results[mode]
        rows = res["sweep"]
        L = np.array([x["L"] for x in rows])
        A = np.array([x["alpha"] for x in rows])
        T = np.array([x["theta"] for x in rows])
        M = np.array([x["mse"] for x in rows])
        R = np.array([x["rings"] for x in rows])
        deg = R == 0
        k = int(np.argmin(M))
        lo, hi = float(M.min()), float(M.max())
        cut = lo + basin_frac * (hi - lo)
        inb = M <= cut
        # ONE normalisation for both panels. Letting the basin panel scale to
        # its own range would paint the best points at the ramp's dark end --
        # the same colour the left panel gives the WORST settings -- so the two
        # halves of the same figure would disagree about which way is better.
        norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)

        ax = fig.add_subplot(len(modes), ncol, ncol * r + 1, projection="3d")
        # Sequential, one hue, light -> dark with magnitude. Size carries the
        # same number inverted so the good points are also the big ones -- a
        # redundant encoding, because the light end of a sequential ramp is
        # exactly where a colour difference is hardest to read.
        sc = ax.scatter(L, A, T, c=M, cmap="Blues", norm=norm,
                        s=18 + 90 * (hi - M) / max(hi - lo, EPS),
                        edgecolors="#00000022", linewidths=0.4, depthshade=False)
        pth = res["path"]
        ax.plot([q["L"] for q in pth], [q["alpha"] for q in pth],
                [q["theta"] for q in pth], color="#C44E00", lw=2, marker="o",
                ms=4, label="descent", zorder=5)
        b = res["best"]
        ax.scatter([b["L"]], [b["alpha"]], [b["theta"]], s=190,
                   facecolors="none", edgecolors="#C44E00", linewidths=2.2,
                   label=f"descent end  MSE {res['best_mse']:.4f}")
        if deg.any():
            ax.scatter(L[deg], A[deg], T[deg], s=120, facecolors="none",
                       edgecolors="#B3261E", linewidths=1.1, depthshade=False,
                       label=f"degenerate: 0 rings ({int(deg.sum())})")
        ax.scatter([L[k]], [A[k]], [T[k]], marker="D", s=70, color="#1A1A1A",
                   label=f"grid min  MSE {lo:.4f}")
        ax.set_xlabel("L  (lookahead, vox)")
        ax.set_ylabel("alpha  (rad)")
        ax.set_zlabel("theta  (rad)")
        ax.set_title(f"{mode} seeding -- MSE(TDI, D_obs) over the grid",
                     fontsize=10)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.10)
        cb.set_label("MSE  (darker = worse)", fontsize=8)

        ax2 = fig.add_subplot(len(modes), ncol, ncol * r + 2, projection="3d")
        ax2.scatter(L, A, T, s=8, color="#D8DEE6", depthshade=False)
        ax2.scatter(L[inb], A[inb], T[inb], s=95, c=M[inb], cmap="Blues",
                    norm=norm, edgecolors="#00000033", linewidths=0.5,
                    depthshade=False)
        bd = inb & deg
        if bd.any():
            ax2.scatter(L[bd], A[bd], T[bd], s=150, facecolors="none",
                        edgecolors="#B3261E", linewidths=1.4, depthshade=False)
        ax2.scatter([L[k]], [A[k]], [T[k]], marker="D", s=70, color="#1A1A1A")
        ax2.set_xlabel("L  (lookahead, vox)")
        ax2.set_ylabel("alpha  (rad)")
        ax2.set_zlabel("theta  (rad)")
        ax2.set_title(f"{mode} -- basin: {int(inb.sum())}/{len(M)} settings within "
                      f"{basin_frac:.0%} of the range above the minimum;\n"
                      f"{int(bd.sum())} of them are DEGENERATE (red: 0 rings, "
                      f"the tracker cannot turn)", fontsize=10)

        # A 3D scatter cannot say WHICH theta a point sits at -- the projection
        # is ambiguous in depth, and the whole finding is a statement about
        # theta. So the same numbers again against theta alone, one line per
        # lookahead at the finest alpha. This is the panel that shows the
        # surface is monotone rather than merely low somewhere.
        ax3 = fig.add_subplot(len(modes), ncol, ncol * r + 3)
        a_fine = min(A)
        cmap = plt.get_cmap("Blues")
        Ls = sorted(set(L))
        for i, Lv in enumerate(Ls):
            m = (A == a_fine) & (L == Lv)
            o = np.argsort(T[m])
            ax3.plot(T[m][o], M[m][o], lw=2, marker="o", ms=5,
                     color=cmap(0.30 + 0.65 * i / max(len(Ls) - 1, 1)),
                     label=f"L = {Lv:g}")
        d0 = deg & (A == a_fine)
        if d0.any():
            ax3.axvspan(0, float(T[d0].max()) + 1e-3, color="#B3261E", alpha=0.08)
            ax3.text(float(T[d0].max()) + 0.01, float(M.max()),
                     "0 rings:\nno cone", fontsize=8, color="#B3261E",
                     va="top", ha="left")
        ax3.set_xlabel("theta  (rad)")
        ax3.set_ylabel("MSE(TDI, D_obs)")
        ax3.set_title(f"{mode} -- MSE vs theta at alpha = {a_fine:g}",
                      fontsize=10)
        ax3.legend(fontsize=8, frameon=False, ncols=2)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.grid(axis="y", color="#00000012")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------

def _crop(vol, r):
    return vol[tuple(slice(c - r, c + r)
                     for c in (s // 2 for s in vol.shape))].astype(np.float32)


def replot(tag, outdir=None):
    """Redraw from the saved sweep. The sweep is the expensive part and the
    figure is the part that gets iterated on; re-running one to change the other
    would also re-seed and re-certify, so the picture would no longer be OF the
    numbers already on disk."""
    outdir = outdir or os.path.join(_ROOT, "results")
    p = os.path.join(outdir, f"cone_descent_{tag}.json")
    with open(p) as f:
        d = json.load(f)
    for mode, res in d["results"].items():
        res["verdict"] = verdict(res["sweep"], d["ladder"])
        print(f"  {mode}: {res['verdict']['summary']}")
        for line in res["verdict"]["detail"]:
            print(f"      {line}")
    png = plot(d["results"], os.path.join(outdir, f"cone_descent_{tag}.png"),
               f"Cone parameters vs MSE(TDI, D_obs) -- "
               f"{tag} {tuple(d['volume'])}")
    print(f"  [figure] {png}")
    return png


def main(i74=False, radius=24, centre=(384, 384, 384), quick=False,
         stride=DEFAULT_STRIDE, block=32, n_seeds=N_SEEDS,
         fa_threshold=FA_THRESHOLD, outdir=None, modes=("random", "fa")):
    outdir = outdir or os.path.join(_ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    ladder = LADDER_QUICK if quick else LADDER

    if i74:
        from reconstruction import load_volume
        lo = centre[0] - radius
        vol = load_volume(crop=(lo, centre[0] + radius),
                          dtype=np.uint8).astype(np.float32)
        tag, title = "i74", f"i74 {vol.shape} at {tuple(centre)}"
    else:
        X = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5, CROSS_R1,
                                   CROSS_TAPER)
        vol = _crop(X, radius)
        tag, title = "crossing", f"crossing phantom {vol.shape}"

    print("=" * 78)
    print(f"CONE DESCENT -- (L, alpha, theta) against MSE(TDI, D_obs)")
    print(f"{title}, voxel {VOXEL_SIZE_UM:g} um")
    print("=" * 78)

    dense, D_obs, tissue, FA, _scales = stage2(vol, block=block, stride=stride)
    support = tissue

    results = {}
    for mode in modes:
        s, sd = seed(dense, tissue, FA, mode, n=n_seeds,
                     fa_threshold=fa_threshold)
        print(f"\n{'-' * 78}\n{mode.upper()} SEEDING -- {len(s)} seeds\n{'-' * 78}")
        if not len(s):
            print("  no seeds -- skipping")
            continue
        rows = sweep(vol, dense, D_obs, support, s, sd, ladder)
        p, m, path, n_eval = descend(vol, dense, D_obs, support, s, sd, ladder)
        gmin = min(rows, key=lambda x: x["mse"])
        # Compared on the CONE the parameters produce, not on the parameters:
        # two thetas that quantise to the same ring count are one setting, and
        # calling them a local minimum would be an artifact of the ladder. At
        # ZERO rings the cone is the bare straight-ahead ray, so alpha -- which
        # only ever sets the spacing of rings that do not exist -- drops out of
        # the comparison too.
        nr = rings(p["alpha"], p["theta"])
        same = (abs(m - gmin["mse"]) < 1e-12 and p["L"] == gmin["L"]
                and nr == gmin["rings"]
                and (nr == 0 or p["alpha"] == gmin["alpha"]))
        print(f"\n  grid minimum:  MSE {gmin['mse']:.5f} at L={gmin['L']:g} "
              f"alpha={gmin['alpha']:g} theta={gmin['theta']:g} "
              f"({gmin['rings']} rings)")
        print(f"  descent found: MSE {m:.5f} at L={p['L']:g} "
              f"alpha={p['alpha']:g} theta={p['theta']:g} "
              f"({rings(p['alpha'], p['theta'])} rings)"
              f"   {'-- the same cone' if same else '-- a DIFFERENT cone: the descent stopped in a local minimum'}")
        results[mode] = dict(n_seeds=int(len(s)), sweep=rows, best=p,
                             best_mse=float(m), path=path,
                             n_evaluations=int(n_eval), grid_min=gmin)

    for mode, res in results.items():
        res["verdict"] = verdict(res["sweep"], ladder)

    if results:
        print(f"\n{'=' * 78}\nIS THERE A MINIMUM?\n{'=' * 78}")
        for mode, res in results.items():
            v = res["verdict"]
            print(f"  {mode}: {v['summary']}")
            for line in v["detail"]:
                print(f"      {line}")
        png = plot(results, os.path.join(outdir, f"cone_descent_{tag}.png"),
                   f"Cone parameters vs MSE(TDI, D_obs) -- {title}")
        print(f"\n  [figure] {png}")

    path_json = os.path.join(outdir, f"cone_descent_{tag}.json")
    with open(path_json, "w") as f:
        json.dump(dict(volume=list(vol.shape), tag=tag, ladder=ladder,
                       fa_threshold=fa_threshold, n_seeds_requested=n_seeds,
                       results=results), f, indent=2, default=float)
    print(f"  wrote {path_json}")
    print("=" * 78)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="cone (L, alpha, theta) vs TDI MSE, from Stage 2 onward")
    ap.add_argument("--i74", action="store_true")
    ap.add_argument("--centre", nargs=3, type=int, default=[384, 384, 384])
    ap.add_argument("--radius", type=int, default=24)
    ap.add_argument("--quick", action="store_true", help="a coarser ladder")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--fa", type=float, default=FA_THRESHOLD)
    ap.add_argument("--modes", nargs="+", default=["random", "fa"],
                    choices=["random", "fa"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--replot", metavar="TAG", default=None,
                    help="redraw the figure from a saved cone_descent_TAG.json "
                         "instead of sweeping again")
    a = ap.parse_args()
    if a.replot:
        replot(a.replot, outdir=a.outdir)
        sys.exit(0)
    main(i74=a.i74, radius=a.radius, centre=tuple(a.centre), quick=a.quick,
         stride=a.stride, block=a.block, n_seeds=a.seeds, fa_threshold=a.fa,
         outdir=a.outdir, modes=tuple(a.modes))