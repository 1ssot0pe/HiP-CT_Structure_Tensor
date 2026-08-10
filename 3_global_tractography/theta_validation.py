"""
theta_validation.py -- validate the TDI proxy loss against KNOWN geometry.

`cone_descent.py` found that mse(TDI, D_obs) is monotone in theta: the score is
best when the cone collapses to its straight-ahead ray and the tracker cannot
turn at all. That measurement was made on the STRAIGHT crossing phantom, where
turning buys nothing by construction -- so it cannot distinguish "the loss is
broken" from "there was nothing for the loss to reward".

This file makes the distinction, on a phantom that is nothing but curvature: a
single parabolic bundle whose tangent is known analytically at every point. No
crossing, no kiss, no ambiguity about what the right answer is. Then it sweeps
theta and measures THREE things at once:

    MSE          mse(TDI, D_obs)              the proxy loss, as before
    TRACT ERR    angle(tract step, truth)     what the loss is standing in for
    FIELD ERR    angle(e3, truth)             the tracker's own input

and the three together separate the outcomes:

  * TRACT ERR has a minimum at a sensible cone width while MSE is monotone
    -> the proxy is confirmed broken, and the gap between the two optima is the
       measurement of how broken. The loss must not be used unaccompanied.

  * TRACT ERR is ALSO monotone toward no-turning
    -> the problem is upstream. Either the field never supported turning, or the
       splatting cannot see it. FIELD ERR is what says which: if e3 is already as
       wrong as the tracts, no cone width could have helped and the loss was
       never the thing at fault.

FIELD ERR is the control, and it is constant in theta by construction -- it is a
property of Stage 1's output, measured before any tracking. Plotted as a floor,
because no tracker can beat its own input by turning: a tract error at the floor
means the cone is doing everything the field permits.

Angles are NEMATIC -- angle(d, t) uses |d . t|, because e3 and -e3 are the same
fibre and a tract walked backwards is the same tract.

Run
---
    python theta_validation.py                  # curved bundle, no noise
    python theta_validation.py --noise 0.05
    python theta_validation.py --curv 0.05      # a tighter curve
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import dense_tracker as DT                              # noqa: E402
from cone_descent import stage2, seed, mse_tdi, rings, FLOOR_FRAC  # noqa: E402
from kissing_phantom import curved_bundle                # noqa: E402
from crossing_shape_3d import _coords                    # noqa: E402

EPS = 1e-12

# Three categorical series (one per lookahead), taken in fixed slot order from
# the validated palette -- not invented, and not cycled.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

# alpha is held FIXED and fine. theta is quantised in units of alpha
# (round(theta / alpha) rings), so sweeping theta at a coarse alpha would sample
# three or four distinct cones and call the result a curve.
ALPHA = 0.10
THETAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]
LOOKAHEADS = [2.0, 4.0, 6.0]


# ---------------------------------------------------------------------------
# the phantom, and its truth
# ---------------------------------------------------------------------------

class CurvedTruth:
    """A single parabolic bundle: x = curv * (s^2 - ymax^2 / 2), y = s, z = 0.

    The x-offset centres the parabola's SPAN in the volume rather than its
    vertex, so the bundle uses the crop symmetrically instead of hugging one
    face -- a tract near a boundary dies on the volume edge rather than on the
    evidence, and that would show up as a change in tract length that has
    nothing to do with theta.

    Tangent is analytic: d/ds (x, y, z) = (2 * curv * s, 1, 0), which in the
    tracker's (z, y, x) component order is (0, 1, 2 * curv * s).
    """

    def __init__(self, shape, curv=0.028, radius=2.0, ymax=24.0, n=4000):
        self.shape, self.curv, self.radius, self.ymax = shape, curv, radius, ymax
        s = np.linspace(-ymax, ymax, n)
        x = curv * (s ** 2 - ymax ** 2 / 2.0)
        self.pts_xyz = np.stack([x, s, np.zeros_like(s)], axis=1)
        # centreline in the tracker's frame: (z, y, x) VOXEL indices
        c = np.array([shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0])
        self.pts_zyx = np.stack([np.zeros_like(s), s, x], axis=1) + c
        t = np.stack([np.zeros_like(s), np.ones_like(s), 2.0 * curv * s], axis=1)
        self.tan_zyx = t / np.linalg.norm(t, axis=1, keepdims=True)
        self._tree = cKDTree(self.pts_zyx)

    def volume(self, peak=100.0, noise=0.0, rng=0):
        vol = curved_bundle(_coords(self.shape), self.pts_xyz,
                            self.radius) * peak
        if noise > 0:
            vol = vol + np.random.default_rng(rng).normal(
                scale=noise * peak, size=vol.shape)
        return np.clip(vol, 0.0, None).astype(np.float32)

    def nearest(self, pts_zyx):
        """(tangent, distance) at each query point, from the nearest centreline
        sample. Distance is returned so a caller can tell a tract that is ON the
        fibre from one that has wandered off and is being scored against a
        tangent it has no business near."""
        d, i = self._tree.query(np.asarray(pts_zyx, float), k=1)
        return self.tan_zyx[i], d

    def turn_angle_deg(self):
        """The largest turn the truth itself makes over one voxel of arc.

        This is the number theta has to be able to afford. A cone narrower than
        this CANNOT follow the fibre, so it locates where an honest optimum
        would have to lie -- and makes "the minimum is at zero" a claim that can
        be checked against geometry rather than against intuition.
        """
        d = np.diff(self.pts_zyx, axis=0)
        seg = np.linalg.norm(d, axis=1)
        u = d / np.maximum(seg, EPS)[:, None]
        cos = np.clip(np.einsum("ij,ij->i", u[:-1], u[1:]), -1.0, 1.0)
        per_vox = np.arccos(cos) / np.maximum(seg[:-1], EPS)
        return float(np.degrees(per_vox.max()))


def nematic_angle_deg(a, b):
    """Angle between two director sets, in degrees, folded to [0, 90]."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), EPS)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), EPS)
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(-1)), 0.0, 1.0)))


# ---------------------------------------------------------------------------
# the two angular errors
# ---------------------------------------------------------------------------

def field_error(e3, truth, core=2.0):
    """angle(e3, true tangent) over voxels on the fibre. Independent of theta.

    `core` is in units of the bundle radius: only voxels genuinely inside the
    tube are scored, because e3 in empty space is not a claim about anything.
    """
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in truth.shape], indexing="ij")
    P = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], 1).astype(float)
    tan, dist = truth.nearest(P)
    on = dist <= core * truth.radius
    v = e3[:, zz.ravel()[on], yy.ravel()[on], xx.ravel()[on]].T
    ang = nematic_angle_deg(v, tan[on])
    ang = ang[np.isfinite(ang)]
    return dict(median=float(np.median(ang)), p90=float(np.percentile(ang, 90)),
                n=int(ang.size))


def tract_error(paths, truth, core=2.0):
    """angle(tract step, true tangent), over every step of every tract.

    NOTHING IS EXCLUDED from `median`. A step is scored against the tangent at
    the nearest centreline point however far away it is, because the alternative
    -- dropping steps that have strayed -- rewards a tracker for wandering
    somewhere the truth cannot follow it. `median_on_fibre` and `stray_frac`
    report the same walk with the strayed steps separated out, so a good score
    that was bought by leaving the bundle is visible rather than hidden.
    """
    if not paths:
        return dict(median=np.nan, median_on_fibre=np.nan, stray_frac=np.nan,
                    n=0)
    D, P = [], []
    for p in paths:
        if len(p) < 2:
            continue
        d = np.diff(p, axis=0)
        n = np.linalg.norm(d, axis=1)
        ok = n > EPS
        D.append(d[ok] / n[ok, None])
        P.append(0.5 * (p[:-1] + p[1:])[ok])
    if not D:
        return dict(median=np.nan, median_on_fibre=np.nan, stray_frac=np.nan,
                    n=0)
    D, P = np.concatenate(D), np.concatenate(P)
    tan, dist = truth.nearest(P)
    ang = nematic_angle_deg(D, tan)
    on = dist <= core * truth.radius
    return dict(median=float(np.median(ang)),
                median_on_fibre=(float(np.median(ang[on])) if on.any()
                                 else float("nan")),
                stray_frac=float(1.0 - on.mean()), n=int(ang.size))


# ---------------------------------------------------------------------------

def run(curv=0.028, radius=2.0, ymax=24.0, noise=0.0, n_seeds=800,
        seeding="fa", outdir=None, thetas=None, lookaheads=None, alpha=ALPHA):
    outdir = outdir or os.path.join(_ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    thetas = list(thetas or THETAS)
    lookaheads = list(lookaheads or LOOKAHEADS)

    shape = (20, int(2 * ymax + 16), 40)
    truth = CurvedTruth(shape, curv=curv, radius=radius, ymax=ymax)
    vol = truth.volume(noise=noise)

    print("=" * 78)
    print("THETA VALIDATION -- proxy loss vs angular error, known geometry")
    print(f"  single curved bundle {shape}, curv={curv:g}, radius={radius:g}, "
          f"noise={noise:g}")
    print(f"  truth turns up to {truth.turn_angle_deg():.1f} deg per voxel "
          f"= {np.radians(truth.turn_angle_deg()):.3f} rad")
    print("=" * 78)

    dense, D_obs, tissue, FA, _scales = stage2(vol)
    e3 = dense.vec1
    fe = field_error(e3, truth)
    print(f"\n  FIELD ERROR (e3 vs truth, {fe['n']} voxels on the fibre): "
          f"median {fe['median']:.2f} deg, p90 {fe['p90']:.2f} deg")
    print("  -- this is the floor. No cone width can beat the field it follows.")

    s, sd = seed(dense, tissue, FA, seeding, n=n_seeds)
    print(f"  {seeding} seeding: {len(s)} seeds\n")
    print(f"    {'L':>4} {'theta':>6} {'rings':>6} | {'MSE':>9} | "
          f"{'TRACT ERR':>10} {'on-fibre':>9} {'stray':>7} {'tracts':>7}")

    rows = []
    floor = FLOOR_FRAC * float(vol.max())
    for L in lookaheads:
        for th in thetas:
            m = mse_tdi(vol, dense, D_obs, tissue, s, sd, L, alpha, th)
            paths = DT.track(vol, dense, s, sd, alpha=alpha, lookahead=L,
                             theta=th, floor=floor)
            te = tract_error(paths, truth)
            rows.append(dict(L=float(L), alpha=float(alpha), theta=float(th),
                             rings=rings(alpha, th), mse=m["mse"],
                             n_tracts=m["n_tracts"], **{f"ang_{k}": v
                                                        for k, v in te.items()}))
            print(f"    {L:4g} {th:6.2f} {rings(alpha, th):6d} | "
                  f"{m['mse']:9.5f} | {te['median']:9.2f}d "
                  f"{te['median_on_fibre']:8.2f}d {te['stray_frac']:6.1%} "
                  f"{m['n_tracts']:7d}")

    out = dict(shape=list(shape), curv=curv, radius=radius, noise=noise,
               alpha=alpha, seeding=seeding, n_seeds=int(len(s)),
               truth_turn_deg=truth.turn_angle_deg(), field_error=fe, rows=rows)
    out["verdict"] = interpret(out)
    print("\n" + "=" * 78)
    for line in out["verdict"]["lines"]:
        print("  " + line)
    print("=" * 78)

    png = plot(out, os.path.join(outdir, "theta_validation.png"))
    print(f"  [figure] {png}")
    p = os.path.join(outdir, "theta_validation.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote {p}")
    return out


def interpret(out):
    """Which of the two outcomes the numbers show. Decided per lookahead, then
    reported as one verdict, because a proxy that fails at one L and not another
    has not been validated -- it has been found to depend on a setting the loss
    was supposed to choose."""
    rows, lines = out["rows"], []
    fe = out["field_error"]["median"]
    per_L, broken, upstream = {}, [], []
    for L in sorted({r["L"] for r in rows}):
        rs = sorted([r for r in rows if r["L"] == L], key=lambda r: r["theta"])
        best_mse = min(rs, key=lambda r: r["mse"])
        best_ang = min(rs, key=lambda r: r["ang_median"])
        mse_mono = best_mse["theta"] == min(r["theta"] for r in rs)
        ang_mono = best_ang["theta"] == min(r["theta"] for r in rs)
        gain = rs[0]["ang_median"] - best_ang["ang_median"]
        # The dominant structure is a CLIFF, not a smooth optimum: the whole
        # angular improvement is bought crossing from 0 rings to 1, after which
        # the curve is nearly flat. Reported separately, because a shallow
        # interior minimum sitting on the ladder's top rung is a much weaker
        # claim than "turning at all is worth 8 degrees" -- and the second is
        # the one the numbers actually support.
        turning = [r for r in rs if r["rings"] >= 1]
        no_turn = [r for r in rs if r["rings"] == 0]
        cliff = (float(np.mean([r["ang_median"] for r in no_turn]))
                 - float(min(r["ang_median"] for r in turning))
                 if no_turn and turning else float("nan"))
        spread = (float(max(r["ang_median"] for r in turning)
                        - min(r["ang_median"] for r in turning))
                  if turning else float("nan"))
        at_edge = best_ang["theta"] == max(r["theta"] for r in rs)
        # What choosing the truth-optimal cone costs in the proxy: the proxy is
        # not merely uninformative here, it actively penalises the right answer.
        mse_penalty = best_ang["mse"] - best_mse["mse"]
        per_L[L] = dict(mse_theta=best_mse["theta"], mse_rings=best_mse["rings"],
                        ang_theta=best_ang["theta"], ang_rings=best_ang["rings"],
                        ang_best=best_ang["ang_median"],
                        ang_at_no_turn=rs[0]["ang_median"],
                        ang_gain_deg=float(gain), cliff_deg=cliff,
                        spread_among_turning_deg=spread,
                        ang_opt_at_ladder_edge=bool(at_edge),
                        mse_penalty_for_truth=float(mse_penalty),
                        mse_monotone=bool(mse_mono), ang_monotone=bool(ang_mono))
        (upstream if ang_mono else broken).append(L)
        lines.append(
            f"L={L:g}: MSE best at theta={best_mse['theta']:g} "
            f"({best_mse['rings']} rings) | angular error best at "
            f"theta={best_ang['theta']:g} ({best_ang['rings']} rings), "
            f"{best_ang['ang_median']:.2f} deg "
            f"(vs {rs[0]['ang_median']:.2f} deg at no-turn: {gain:+.2f} deg)"
            f"{'  [at the ladder edge]' if at_edge else ''}")
        lines.append(
            f"      cliff 0->1 ring {cliff:+.2f} deg; spread across ALL turning "
            f"cones only {spread:.2f} deg; picking the truth-optimal cone costs "
            f"{mse_penalty:+.4f} MSE")

    lines.append(f"field error (the floor): {fe:.2f} deg; truth turns up to "
                 f"{out['truth_turn_deg']:.2f} deg/voxel")
    if not broken:
        verdict = "UPSTREAM"
        lines.append("VERDICT: angular error is ALSO monotone toward no-turning "
                     "at every L. The loss is not the thing at fault -- the "
                     "field or the splatting never supported turning.")
    elif not upstream:
        verdict = "PROXY_BROKEN"
        cl = [v["cliff_deg"] for v in per_L.values()]
        sp = [v["spread_among_turning_deg"] for v in per_L.values()]
        pen = [v["mse_penalty_for_truth"] for v in per_L.values()]
        lines.append(
            "VERDICT: the two disagree in DIRECTION. MSE falls toward theta=0; "
            "angular error rises there, at every L. The proxy is confirmed "
            "broken -- not merely uninformative but ANTI-correlated across the "
            "one transition that matters.")
        lines.append(
            f"  the transition is a cliff, not a smooth optimum: turning at all "
            f"buys {min(cl):.1f}-{max(cl):.1f} deg, while the choice of cone "
            f"width among turning cones is worth only {min(sp):.1f}-{max(sp):.1f} "
            f"deg. So the proxy gets the one decision that matters backwards, "
            f"and the rest of theta barely matters.")
        lines.append(
            f"  and it PENALISES the right answer: adopting the truth-optimal "
            f"cone raises MSE by {min(pen):+.4f} to {max(pen):+.4f}.")
        if all(v["ang_opt_at_ladder_edge"] for v in per_L.values()):
            lines.append("  caveat: the angular optimum sits on the theta "
                         "ladder's top rung at every L, so its exact location "
                         "is not pinned -- only its direction.")
    else:
        verdict = "MIXED"
        lines.append(f"VERDICT: MIXED -- angular error turns at L={broken}, is "
                     f"monotone at L={upstream}. The proxy's failure depends on "
                     f"a parameter it was meant to choose.")
    return dict(verdict=verdict, per_L=per_L, lines=lines)


def plot(out, path):
    """MSE and angular error against theta, as two panels sharing one x-axis.

    Two panels and not one with two y-axes. A dual-axis chart lets the two
    curves be slid over each other until they appear to agree or disagree by
    choosing the scales, and the entire question here is where two optima sit
    relative to each other -- exactly the claim a second axis can manufacture.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["rows"]
    Ls = sorted({r["L"] for r in rows})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 8.6), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Does the TDI proxy agree with the truth?  "
                 f"single curved bundle, curv={out['curv']:g}, "
                 f"noise={out['noise']:g}", fontsize=12)

    for i, L in enumerate(Ls):
        rs = sorted([r for r in rows if r["L"] == L], key=lambda r: r["theta"])
        t = [r["theta"] for r in rs]
        col = SERIES[i % len(SERIES)]
        a1.plot(t, [r["mse"] for r in rs], lw=2, marker="o", ms=6, color=col,
                label=f"L = {L:g}")
        a2.plot(t, [r["ang_median"] for r in rs], lw=2, marker="o", ms=6,
                color=col, label=f"L = {L:g}")
        # Direct labels as well as the legend, so identity is never colour alone.
        a1.annotate(f"L={L:g}", (t[-1], rs[-1]["mse"]), color=col, fontsize=9,
                    xytext=(5, 0), textcoords="offset points", va="center")
        a2.annotate(f"L={L:g}", (t[-1], rs[-1]["ang_median"]), color=col,
                    fontsize=9, xytext=(5, 0), textcoords="offset points",
                    va="center")
        for ax, key in ((a1, "mse"), (a2, "ang_median")):
            k = int(np.argmin([r[key] for r in rs]))
            ax.scatter([t[k]], [rs[k][key]], s=150, facecolors="none",
                       edgecolors=col, linewidths=2.0, zorder=5)

    # The region where the cone carries no rings at all: not a narrow tracker,
    # no tracker. Shaded in both panels so an optimum sitting inside it is
    # obvious rather than needing to be worked out from theta and alpha.
    deg_t = [r["theta"] for r in rows if r["rings"] == 0]
    for ax in (a1, a2):
        if deg_t:
            ax.axvspan(0, max(deg_t) + 1e-3, color="#B3261E", alpha=0.07)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#00000010")
        ax.set_facecolor("white")
    if deg_t:
        a1.text(max(deg_t) + 0.006, a1.get_ylim()[1], "0 rings:\nno turning",
                fontsize=8, color="#B3261E", va="top")

    a1.set_ylabel("MSE(TDI, D_obs)   -- the proxy")
    a1.set_title("the loss being optimised", fontsize=10, loc="left")
    a1.legend(frameon=False, fontsize=9)

    fe = out["field_error"]["median"]
    a2.axhline(fe, color="#333333", ls="--", lw=1.5)
    a2.annotate(f"field error floor: e3 vs truth = {fe:.2f} deg",
                (0.0, fe), xytext=(4, 5), textcoords="offset points",
                fontsize=9, color="#333333")
    turn = np.radians(out["truth_turn_deg"])
    if turn <= max(r["theta"] for r in rows):
        a2.axvline(turn, color="#666666", ls=":", lw=1.4)
        a2.annotate(f"turn the truth demands\n({turn:.3f} rad)", (turn, 0),
                    xytext=(5, 12), textcoords="offset points", fontsize=8,
                    color="#666666")
    a2.set_ylabel("median angle to true tangent  (deg)")
    a2.set_xlabel("theta   (cone half-width, rad)")
    a2.set_title("what the proxy is standing in for", fontsize=10, loc="left")
    a2.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="validate the TDI loss against "
                                             "known phantom geometry")
    ap.add_argument("--curv", type=float, default=0.028)
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--ymax", type=float, default=24.0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Gaussian sigma as a fraction of peak intensity")
    ap.add_argument("--seeds", type=int, default=800)
    ap.add_argument("--seeding", choices=("fa", "random"), default="fa")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--reinterpret", action="store_true",
                    help="re-read the saved sweep, re-run the verdict and "
                         "redraw, without tracking again")
    a = ap.parse_args()
    if a.reinterpret:
        d = a.outdir or os.path.join(_ROOT, "results")
        with open(os.path.join(d, "theta_validation.json")) as f:
            out = json.load(f)
        out["verdict"] = interpret(out)
        for line in out["verdict"]["lines"]:
            print("  " + line)
        print(f"  [figure] {plot(out, os.path.join(d, 'theta_validation.png'))}")
        with open(os.path.join(d, "theta_validation.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)
        sys.exit(0)
    run(curv=a.curv, radius=a.radius, ymax=a.ymax, noise=a.noise,
        n_seeds=a.seeds, seeding=a.seeding, alpha=a.alpha, outdir=a.outdir)
