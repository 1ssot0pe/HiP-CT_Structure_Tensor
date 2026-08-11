"""
descent_mse.py -- give the OLD objective the wheel.

`descent_curve.py` read mse(TDI, D_obs) along a path the ALIGN loss chose. That
says what the MSE thought of someone else's route; it does not say where the MSE
would have gone. This runs the identical gradient descent -- same i74 chunk, same
seeds, same start, same iteration cap, same smooth tracker -- with the MSE as the
objective being minimised, and reports both descents against both objectives.

The comparison that matters is not which descent reaches a lower value of its own
loss (each wins that trivially, by construction). It is where each one ENDS in
parameter space, and what the other objective says about that point.

Run
---
    python descent_mse.py
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import cone_objective as CO                                  # noqa: E402
from cone_descent import stage2, seed, FLOOR_FRAC            # noqa: E402
from gradient_optimise import Problem, descend               # noqa: E402
from param_volume import load_chunk                          # noqa: E402

ALIGN_C = "#2a78d6"
MSE_C = "#eb6834"
START_C = "#1baf7a"


def setup(centre=384, radius=32, n_seeds=100, seeding="fa", verbose=True):
    vol, _o = load_chunk(centre, radius, verbose=verbose)
    dense, D_obs, tissue, FA, sc = stage2(vol, verbose=False)
    s, sd = seed(dense, tissue, FA, seeding, n=n_seeds)
    floor = FLOOR_FRAC * float(vol.max())
    ctx = CO.Context(vol, dense, D_obs, tissue, rho=float(sc["rho_global"]))
    return vol, dense, s, sd, floor, ctx


def main(centre=384, radius=32, n_seeds=100, L0=4.0, theta0=0.15, alpha0=0.15,
         max_iters=10, outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results")
    print("=" * 78)
    print("MSE-DRIVEN GRADIENT DESCENT  (same setup as the align-driven run)")
    print("=" * 78)
    vol, dense, s, sd, floor, ctx = setup(centre, radius, n_seeds)
    print(f"  {len(s)} seeds; start L={L0:g} vox, "
          f"theta={np.degrees(theta0):.1f} deg, "
          f"alpha={np.degrees(alpha0):.1f} deg; cap {max_iters} iters\n")

    prob = Problem(vol, dense, s, sd, floor, objective="mse", ctx=ctx)
    L, th, al, trace = descend(prob, L0=L0, theta0=theta0, alpha0=alpha0,
                               max_iters=max_iters)

    # Score BOTH descents by BOTH objectives, on this same setup.
    ref = Problem(vol, dense, s, sd, floor, objective="align", ctx=ctx)
    end_mse = np.log([L, th, al])
    a_at_mse, m_at_mse = ref.both(end_mse)

    with open(os.path.join(outdir, "param_volume_run.json")) as f:
        prev = json.load(f)["descent"]
    end_align = np.log([prev["L"], prev["theta"], prev["alpha"]])
    a_at_align, m_at_align = ref.both(end_align)
    start = np.log([L0, theta0, alpha0])
    a0, m0 = ref.both(start)

    rows = [
        ("start", L0, theta0, alpha0, a0, m0),
        ("align-driven end", prev["L"], prev["theta"], prev["alpha"],
         a_at_align, m_at_align),
        ("MSE-driven end", L, th, al, a_at_mse, m_at_mse),
    ]
    print("\n" + "=" * 78)
    print(f"  {'point':>18} {'L':>7} {'theta':>9} {'alpha':>9} "
          f"{'align':>10} {'mse':>10}")
    for nm, Lv, tv, av, a_, m_ in rows:
        print(f"  {nm:>18} {Lv:7.2f} {np.degrees(tv):8.1f}d "
              f"{np.degrees(av):8.1f}d {a_:10.6f} {m_:10.6f}")

    print(f"\n  MSE-driven descent moved theta "
          f"{np.degrees(theta0):.1f} -> {np.degrees(th):.1f} deg, "
          f"L {L0:.2f} -> {L:.2f} vox")
    print(f"  align-driven descent moved theta "
          f"{np.degrees(theta0):.1f} -> {np.degrees(prev['theta']):.1f} deg, "
          f"L {L0:.2f} -> {prev['L']:.2f} vox")
    if a_at_mse > a_at_align:
        print(f"  -> the MSE's chosen point is WORSE by the align loss "
              f"({a_at_mse:.6f} vs {a_at_align:.6f}, "
              f"{100 * (a_at_mse - a_at_align) / a_at_align:+.1f}%)")
    else:
        print(f"  -> the MSE's chosen point is BETTER by the align loss "
              f"({a_at_mse:.6f} vs {a_at_align:.6f})")
    print("=" * 78)

    out = dict(centre=centre, radius=radius, n_seeds=int(len(s)),
               start=dict(L=L0, theta=theta0, alpha=alpha0, align=a0, mse=m0),
               mse_driven=dict(L=L, theta=th, alpha=al, align=a_at_mse,
                               mse=m_at_mse, trace=trace,
                               n_passes=prob.n_eval),
               align_driven=dict(L=prev["L"], theta=prev["theta"],
                                 alpha=prev["alpha"], align=a_at_align,
                                 mse=m_at_align, trace=prev["trace"]))
    p = plot(out, os.path.join(outdir, "descent_mse.png"))
    print(f"  [figure] {p}")
    with open(os.path.join(outdir, "descent_mse.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def plot(out, path):
    """Where each descent went, in the two parameters that carry the argument."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    fig.patch.set_facecolor("white")
    fig.suptitle("Who drives? the same descent run on the align loss and on "
                 "the MSE   (i74, identical seeds and start)", fontsize=12)

    tr_a = out["align_driven"]["trace"]
    tr_m = out["mse_driven"]["trace"]

    # --- each descent's OWN loss, as % change from its own start ------------
    # Plotted as a percentage, on ONE axis, deliberately. Drawn as raw values
    # on separate axes, matplotlib offsets the MSE panel by +1.37 and a change
    # of 0.0008 acquires the same visual shape as a change of 0.029 -- the two
    # descents then look equally successful. The fair comparison is relative,
    # and relative is dimensionless, so it can honestly share an axis.
    ax = axes[0]
    for tr, col, nm in ((tr_a, ALIGN_C, "align-driven, on the align loss"),
                        (tr_m, MSE_C, "MSE-driven, on the MSE")):
        y = np.array([t["loss"] for t in tr], float)
        pct = 100.0 * (y - y[0]) / abs(y[0])
        ax.plot(range(len(pct)), pct, lw=2.2, marker="o", ms=5, color=col,
                label=f"{nm}  ({pct[-1]:+.2f}%)")
        ax.scatter([len(pct) - 1], [pct[-1]], s=190, marker="*", color=col,
                   zorder=5, edgecolors="white", linewidths=0.9)
    ax.scatter([0], [0.0], s=150, color=START_C, zorder=6,
               edgecolors="white", linewidths=1.2, label="shared start")
    ax.axhline(0.0, color="#00000030", lw=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("change in that descent's OWN loss  (%)")
    ax.set_title("each descent, measured on the loss it is minimising",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#00000010")

    # --- where they went, in (L, theta) ------------------------------------
    a = axes[1]
    for tr, col, nm in ((tr_a, ALIGN_C, "align-driven"),
                        (tr_m, MSE_C, "MSE-driven")):
        L = [t["L"] for t in tr]
        T = np.degrees([t["theta"] for t in tr])
        a.plot(L, T, lw=2.2, marker="o", ms=4.5, color=col, label=nm)
        a.scatter([L[-1]], [T[-1]], s=200, marker="*", color=col, zorder=6,
                  edgecolors="white", linewidths=0.9)
    st = out["start"]
    a.scatter([st["L"]], [np.degrees(st["theta"])], s=170, color=START_C,
              zorder=7, edgecolors="white", linewidths=1.3, label="shared start")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("L  lookahead (vox)")
    a.set_ylabel("theta  cone half-width (deg)")
    a.set_title("where each objective takes the cone", fontsize=10, loc="left")
    a.legend(fontsize=8.5, frameon=False)
    a.spines[["top", "right"]].set_visible(False)
    # Limits set from the DATA, not left to autoscale: the MSE descent barely
    # moves, and an autoscaled box around its few points would zoom in until a
    # 1% wander filled the panel and looked like a journey.
    allL = [t["L"] for t in tr_a] + [t["L"] for t in tr_m]
    allT = list(np.degrees([t["theta"] for t in tr_a])) + \
        list(np.degrees([t["theta"] for t in tr_m]))
    a.set_xlim(min(allL) * 0.80, max(allL) * 1.25)
    a.set_ylim(min(allT) * 0.80, max(allT) * 1.25)
    a.annotate(f"MSE moved {100 * abs(tr_m[-1]['L'] - tr_m[0]['L']) / tr_m[0]['L']:.0f}% in L; "
               f"align moved {100 * abs(tr_a[-1]['L'] - tr_a[0]['L']) / tr_a[0]['L']:.0f}%",
               (0.02, 0.04), xycoords="axes fraction", fontsize=8,
               color="#555555")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="run the descent on the MSE")
    ap.add_argument("--centre", type=int, default=384)
    ap.add_argument("--radius", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--L0", type=float, default=4.0)
    ap.add_argument("--theta0", type=float, default=0.15)
    ap.add_argument("--alpha0", type=float, default=0.15)
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    main(centre=a.centre, radius=a.radius, n_seeds=a.seeds, L0=a.L0,
         theta0=a.theta0, alpha0=a.alpha0, max_iters=a.max_iters,
         outdir=a.outdir)
