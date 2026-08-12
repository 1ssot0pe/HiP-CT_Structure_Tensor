"""
objective_search.py -- which objective, if you descend on it, lands you at the
right cone?

The discipline this follows
---------------------------
A proxy loss is not validated by being reasonable. It is validated by being
descended on, and then checking the TRUE angular error of the tracks you get.
So every candidate in `cone_objective` is scored the only way that answers the
question:

    take the objective's argmin over the (L, theta) grid,
    then report the angular error the tracts ACTUALLY have there.

Call that the objective's ANGULAR ERROR AT ITS OWN OPTIMUM. Against it sit three
reference points, all measured on the same grid:

    ORACLE     the lowest angular error any grid setting achieves. What a
               perfect objective would find.
    INCUMBENT  what mse_raw -- today's objective -- delivers.
    FIELD      angle(e3, truth). Nothing can beat this by turning, so it is the
               floor and not a target.

An objective's REGRET is its angular error minus the oracle's. Regret near zero
means descending on it is as good as knowing the answer.

Everything is scored on ONE set of tracks per (L, theta). Two objectives
compared on separately-tracked path sets would differ by the tracking as much as
by the objective.

The correlation panel is the diagnostic, not the verdict
--------------------------------------------------------
Spearman rho between an objective and the true angular error across the grid
says whether the objective is pointed the right way overall. It is reported
because it is informative, but the bar chart is what decides: an objective can
correlate well on average and still put its MINIMUM in the wrong place, and the
minimum is the only point a descent ever visits.

Run
---
    python objective_search.py
    python objective_search.py --noise 0.05
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import dense_tracker as DT                                      # noqa: E402
import cone_objective as CO                                     # noqa: E402
from cone_descent import stage2, seed, rings, FLOOR_FRAC        # noqa: E402
from theta_validation import CurvedTruth, field_error, tract_error  # noqa: E402

INCUMBENT = "mse_raw"
THETAS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80]
LOOKAHEADS = [2.0, 4.0]
ALPHA = 0.10

# Categorical slots in fixed order from the validated palette.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def run(curv=0.028, radius=2.0, ymax=24.0, noise=0.0, n_seeds=800,
        seeding="fa", alpha=ALPHA, thetas=None, lookaheads=None, outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    thetas = list(thetas or THETAS)
    lookaheads = list(lookaheads or LOOKAHEADS)

    shape = (20, int(2 * ymax + 16), 40)
    truth = CurvedTruth(shape, curv=curv, radius=radius, ymax=ymax)
    vol = truth.volume(noise=noise)

    print("=" * 78)
    print("OBJECTIVE SEARCH -- which loss lands on the right cone?")
    print(f"  curved bundle {shape}, curv={curv:g}, radius={radius:g}, "
          f"noise={noise:g}")
    print("=" * 78)

    dense, D_obs, tissue, FA, scales = stage2(vol)
    rho = float(scales["rho_global"])
    fe = field_error(dense.vec1, truth)
    print(f"\n  FIELD floor (e3 vs truth): {fe['median']:.2f} deg")
    print(f"  scale-matching will smooth the TDI by rho = {rho:g}")

    ctx = CO.Context(vol, dense, D_obs, tissue, rho=rho)
    s, sd = seed(dense, tissue, FA, seeding, n=n_seeds)
    floor = FLOOR_FRAC * float(vol.max())
    print(f"  {seeding} seeding: {len(s)} seeds; grid "
          f"{len(lookaheads)}x{len(thetas)} = "
          f"{len(lookaheads) * len(thetas)} tracked settings\n")

    rows = []
    print(f"    {'L':>4} {'theta':>6} {'rings':>6} | {'TRUE ang':>9} | "
          + " ".join(f"{n[:14]:>14}" for n in CO.names()))
    for L in lookaheads:
        for th in thetas:
            paths = DT.track(vol, dense, s, sd, alpha=alpha, lookahead=L,
                             theta=th, floor=floor)
            sc = CO.score_all(paths, ctx)
            te = tract_error(paths, truth)
            rows.append(dict(L=float(L), theta=float(th),
                             rings=rings(alpha, th), n_tracts=len(paths),
                             ang=te["median"], stray=te["stray_frac"],
                             **{f"obj_{k}": v for k, v in sc.items()}))
            print(f"    {L:4g} {th:6.2f} {rings(alpha, th):6d} | "
                  f"{te['median']:8.2f}d | "
                  + " ".join(f"{sc[n]:14.5f}" for n in CO.names()))

    out = dict(shape=list(shape), curv=curv, noise=noise, alpha=alpha,
               rho=rho, seeding=seeding, n_seeds=int(len(s)),
               field_error=fe, rows=rows)
    out["ranking"] = rank(out)
    report(out)
    png = plot(out, os.path.join(outdir, "objective_search.png"))
    print(f"  [figure] {png}")
    p = os.path.join(outdir, "objective_search.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote {p}")
    return out


def rank(out):
    """Each objective's angular error at its own optimum, plus its regret."""
    rows = out["rows"]
    ang = np.array([r["ang"] for r in rows], float)
    oracle = float(np.nanmin(ang))
    oracle_row = rows[int(np.nanargmin(ang))]
    no_turn = [r for r in rows if r["rings"] == 0]
    res = []
    for name in CO.names():
        v = np.array([r[f"obj_{name}"] for r in rows], float)
        k = int(np.nanargmin(v))
        good = np.isfinite(v) & np.isfinite(ang)
        rho_s = (float(spearmanr(v[good], ang[good]).statistic)
                 if good.sum() > 2 else float("nan"))
        res.append(dict(
            name=name, blurb=CO.BLURB.get(name, ""),
            opt_L=rows[k]["L"], opt_theta=rows[k]["theta"],
            opt_rings=rows[k]["rings"], ang_at_opt=float(ang[k]),
            regret=float(ang[k] - oracle), spearman=rho_s,
            picks_turning=bool(rows[k]["rings"] >= 1)))
    res.sort(key=lambda r: (r["ang_at_opt"], -r["spearman"]))
    return dict(oracle_ang=oracle,
                oracle_setting=dict(L=oracle_row["L"], theta=oracle_row["theta"],
                                    rings=oracle_row["rings"]),
                no_turn_ang=(float(np.median([r["ang"] for r in no_turn]))
                             if no_turn else float("nan")),
                field_floor=out["field_error"]["median"], objectives=res)


def report(out):
    R = out["ranking"]
    print("\n" + "=" * 78)
    print("ANGULAR ERROR AT EACH OBJECTIVE'S OWN OPTIMUM  (lower is better)")
    print("=" * 78)
    print(f"  ORACLE (best any setting achieves): {R['oracle_ang']:.2f} deg "
          f"at L={R['oracle_setting']['L']:g} "
          f"theta={R['oracle_setting']['theta']:g}")
    print(f"  no-turn settings:                   {R['no_turn_ang']:.2f} deg")
    print(f"  FIELD floor (e3 vs truth):          {R['field_floor']:.2f} deg\n")
    print(f"    {'objective':>30} {'ang@opt':>9} {'regret':>8} {'rho':>7} "
          f"{'picks':>7}  optimum")
    for r in R["objectives"]:
        mark = "  <-- today" if r["name"] == INCUMBENT else ""
        print(f"    {r['name']:>30} {r['ang_at_opt']:8.2f}d "
              f"{r['regret']:+8.2f} {r['spearman']:7.2f} "
              f"{'TURN' if r['picks_turning'] else 'no-turn':>7}  "
              f"L={r['opt_L']:g} theta={r['opt_theta']:g} "
              f"({r['opt_rings']} rings){mark}")

    best = R["objectives"][0]
    inc = next(r for r in R["objectives"] if r["name"] == INCUMBENT)
    print()
    if best["name"] == INCUMBENT:
        print("  VERDICT: nothing tried beats the current objective.")
    elif not best["picks_turning"]:
        print("  VERDICT: even the best candidate still picks a no-turn cone. "
              "None of these repairs the objective.")
    else:
        print(f"  VERDICT: '{best['name']}' ({best['blurb']}) picks a TURNING "
              f"cone and lands at {best['ang_at_opt']:.2f} deg, against "
              f"{inc['ang_at_opt']:.2f} deg for the current objective "
              f"-- {inc['ang_at_opt'] - best['ang_at_opt']:.2f} deg better, "
              f"regret {best['regret']:+.2f} deg from the oracle.")
        fixed = [r["name"] for r in R["objectives"] if r["picks_turning"]]
        print(f"  {len(fixed)}/{len(R['objectives'])} candidates pick a turning "
              f"cone: {', '.join(fixed)}")
    print("=" * 78)


def plot(out, path):
    """The decision on top, the diagnostic underneath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R, rows = out["ranking"], out["rows"]
    objs = R["objectives"]
    n = len(objs)
    fig = plt.figure(figsize=(12.6, 5.0 + 2.6 * int(np.ceil(n / 4))))
    fig.patch.set_facecolor("white")
    fig.suptitle("Which objective lands on the right cone?   "
                 f"curved bundle, curv={out['curv']:g}, noise={out['noise']:g}",
                 fontsize=13)
    gs = fig.add_gridspec(1 + int(np.ceil(n / 4)), 4,
                          height_ratios=[2.1] + [1.0] * int(np.ceil(n / 4)))

    # ---- the decision -----------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    y = np.arange(n)[::-1]
    names = [o["name"] for o in objs]
    vals = [o["ang_at_opt"] for o in objs]
    cols = [ORANGE if o["name"] == INCUMBENT else
            (BLUE if o["picks_turning"] else "#9AA5B1") for o in objs]
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, o in zip(y, objs):
        ax.text(o["ang_at_opt"] + 0.12, yi,
                f"{o['ang_at_opt']:.2f}d  (regret {o['regret']:+.2f})",
                va="center", fontsize=9, color="#1A1A1A")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{nm}{'  [today]' if nm == INCUMBENT else ''}"
                        for nm in names], fontsize=9)
    ax.axvline(R["oracle_ang"], color="#1A1A1A", ls="--", lw=1.5)
    ax.annotate(f"oracle {R['oracle_ang']:.2f}d", (R["oracle_ang"], y[0] + 0.6),
                fontsize=9, color="#1A1A1A", xytext=(4, 0),
                textcoords="offset points")
    ax.axvline(R["field_floor"], color="#666666", ls=":", lw=1.4)
    ax.annotate(f"field floor {R['field_floor']:.2f}d",
                (R["field_floor"], y[0] + 0.6), fontsize=9, color="#666666",
                xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel("TRUE median angle to the tangent, at that objective's own "
                  "optimum  (deg)")
    ax.set_title("descend on each objective, then measure the truth "
                 "(blue = picks a turning cone, grey = picks no-turn)",
                 fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#00000010")
    ax.set_xlim(0, max(vals) * 1.30)

    # ---- the diagnostic ---------------------------------------------------
    ang = np.array([r["ang"] for r in rows], float)
    for i, o in enumerate(objs):
        a = fig.add_subplot(gs[1 + i // 4, i % 4])
        v = np.array([r[f"obj_{o['name']}"] for r in rows], float)
        deg = np.array([r["rings"] == 0 for r in rows], bool)
        a.scatter(v[~deg], ang[~deg], s=26, color=BLUE, label="turns")
        a.scatter(v[deg], ang[deg], s=32, facecolors="none",
                  edgecolors="#B3261E", linewidths=1.3, label="0 rings")
        k = int(np.nanargmin(v))
        a.scatter([v[k]], [ang[k]], s=110, facecolors="none",
                  edgecolors="#1A1A1A", linewidths=1.8)
        a.set_title(f"{o['name']}\nrho={o['spearman']:+.2f}", fontsize=8)
        a.tick_params(labelsize=7)
        a.spines[["top", "right"]].set_visible(False)
        if i % 4 == 0:
            a.set_ylabel("true ang (deg)", fontsize=8)
        a.set_xlabel("objective", fontsize=8)
        if i == 0:
            a.legend(fontsize=7, frameon=False, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="rank candidate cone objectives "
                                             "against phantom angular error")
    ap.add_argument("--curv", type=float, default=0.028)
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--ymax", type=float, default=24.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=800)
    ap.add_argument("--seeding", choices=("fa", "random"), default="fa")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    run(curv=a.curv, radius=a.radius, ymax=a.ymax, noise=a.noise,
        n_seeds=a.seeds, seeding=a.seeding, alpha=a.alpha, outdir=a.outdir)
