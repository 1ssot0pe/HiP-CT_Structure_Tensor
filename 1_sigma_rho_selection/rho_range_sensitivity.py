"""
rho_range_sensitivity.py

Step 1 of the rho question: does the fibre field actually depend on WHICH rho you
pick inside the credible range?

adaptive_sigma_benefit.py showed rho has huge leverage -- switching rho 0.85 ->
3.46 moves e3 by a median 39.8 deg, more than any sigma change tested. But 0.85
is not a candidate value; it is obviously under-integrated. The surviving
uncertainty in SIGMA_RHO.md is rho in [2.0, 3.5] (the Lindeberg drift plateau
said 3.5, the since-rejected FSC said 2.0). Leverage over a range nobody
proposes is not the same as uncertainty over the range that is actually open.

So the decisive number is not 0.85 -> 3.46. It is

        angle( e3(sigma=0.85, rho=2.0) , e3(sigma=0.85, rho=3.5) )

with sigma held fixed, over certified tissue. Read against the same scale bar
the sigma question used: how far e3 moves when sigma goes 0.85 -> 0.90, the two
dense-box picks the project already treats as interchangeable.

  if the endpoint spread is <= the sigma scale bar  -> rho is effectively pinned
                                                       too; scale selection is
                                                       CLOSED, stop here.
  if it is much larger                              -> rho genuinely needs
                                                       pinning (step 2: the
                                                       reproducibility metric on
                                                       the lung half-maps).

The settling ladder
-------------------
The endpoint number alone cannot say WHY. The ladder of consecutive differences
from rho = 0.85 up to 5.0 shows where the field stops changing: if the curve has
flattened well before 2.0, the credible range sits on a plateau and the endpoint
number is small for a reason, not by luck. Consecutive steps are reported per
unit log(rho) as well as raw, because the ladder is geometric-ish and a raw step
difference would make the coarse end look busier than it is purely from spacing.

Reuses the block loader, e3, angle and summary helpers from
adaptive_sigma_benefit so this measures the same voxels the same way.

Run
---
    python rho_range_sensitivity.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from core.reconstruction import VOXEL_SIZE_UM, DEFAULT_VOLUME_DIR
from lindeberg_feature_size import (SIGMAS, RHO_RATIO, TRUNCATE, RESP_SMOOTH,
                                    AIR_FRAC, scale_select, kernel_halo)
# NOTE: adaptive_sigma_benefit.py was never committed and has been lost, so this
# script cannot currently run. Its recorded output is
# rho_range_sensitivity_results.json, kept as the record of the finding.
from adaptive_sigma_benefit import (load_block, e3_constant, angle_deg, crop,
                                    summarise, SIGMA_FLOOR)

# The credible range from SIGMA_RHO.md, and context either side of it.
RHO_LADDER = [0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
RHO_LO, RHO_HI = 2.0, 3.5          # the range that is actually open
SIGMA_BAR = 0.90                   # the other dense-box pick, for the scale bar

INK = "#0b0b0b"
MUTED = "#6b6b6b"
GRID = "#d8d8d6"
C_RHO = "#6a4bc4"
C_BAR = "#1f9e89"
C_BAND = "#b9b9b7"


def run(centre=(384, 384, 384), core=96, out="rho_range_sensitivity",
        folder=DEFAULT_VOLUME_DIR):
    halo = kernel_halo(SIGMAS, RHO_RATIO, TRUNCATE, RESP_SMOOTH)
    h2 = core // 2
    rng = [(c - h2 - halo, c + h2 + halo) for c in centre]

    print("=" * 84)
    print("STEP 1 -- does e3 depend on WHICH rho inside the credible range?")
    print(f"  core {core}^3 at {centre}, sigma fixed at {SIGMA_FLOOR}, "
          f"credible rho range [{RHO_LO}, {RHO_HI}]")
    print("=" * 84)

    blk = load_block(folder, *rng)

    # certified-tissue mask, identical definition to the sigma experiment
    sel = scale_select(blk, SIGMAS, RHO_RATIO, TRUNCATE, RESP_SMOOTH, verbose=False)
    core_vol = crop(blk, halo)
    tissue = core_vol > AIR_FRAC * 255.0
    certified = tissue & ~crop(sel["rail_lo"], halo) & ~crop(sel["rail_hi"], halo)
    del sel
    print(f"  certified tissue voxels: {int(certified.sum()):,} "
          f"({certified.mean():.0%} of core)")

    print(f"\n[1] e3 at sigma={SIGMA_FLOOR} for each rho on the ladder")
    fields = {}
    for r in RHO_LADDER:
        fields[r] = crop(e3_constant(blk, SIGMA_FLOOR, r), halo).copy()
        print(f"    rho={r:5.2f}  done", flush=True)

    # ---- THE decisive number ---------------------------------------------
    dec = summarise(angle_deg(fields[RHO_LO], fields[RHO_HI]), certified)

    # ---- the sigma scale bar, at both ends of the credible range ----------
    print(f"\n[2] scale bar: sigma {SIGMA_FLOOR} -> {SIGMA_BAR} at fixed rho")
    bars = {}
    for r in (RHO_LO, RHO_HI):
        e_alt = crop(e3_constant(blk, SIGMA_BAR, r), halo)
        bars[r] = summarise(angle_deg(fields[r], e_alt), certified)
        print(f"    at rho={r:4.2f}: median {bars[r]['median']:.2f} deg")
        del e_alt

    # ---- the settling ladder ---------------------------------------------
    print("\n[3] settling ladder: consecutive differences along rho")
    ladder = []
    for a, b in zip(RHO_LADDER[:-1], RHO_LADDER[1:]):
        s = summarise(angle_deg(fields[a], fields[b]), certified)
        dlog = np.log(b / a)
        ladder.append(dict(rho_lo=a, rho_hi=b, per_octave=s["median"] / dlog * np.log(2),
                           **s))
        print(f"    rho {a:4.2f} -> {b:4.2f} : median {s['median']:6.2f} deg "
              f"({ladder[-1]['per_octave']:6.2f} deg per octave of rho)"
              + ("   <-- inside the credible range"
                 if a >= RHO_LO and b <= RHO_HI else ""))
    del blk, fields

    # ---- verdict ----------------------------------------------------------
    bar = max(bars[RHO_LO]["median"], bars[RHO_HI]["median"])
    pinned = dec["median"] <= bar
    print("\n" + "=" * 84)
    print(f"DECISIVE: e3(rho={RHO_LO}) vs e3(rho={RHO_HI}), sigma fixed")
    print(f"  median {dec['median']:.2f} deg, p95 {dec['p95']:.2f}, "
          f">5deg {dec['frac_gt_5']:.1%}")
    print(f"  sigma scale bar (0.85 -> 0.90 at the same rho): {bar:.2f} deg")
    print(f"  ratio: the rho range moves e3 {dec['median'] / bar:.1f}x "
          f"the sigma pick difference")
    print("  VERDICT: " + (
        "rho is effectively PINNED as well -- the open range costs no more than a "
        "sigma choice already treated as noise. Scale selection is CLOSED."
        if pinned else
        f"rho genuinely needs pinning -- the open range moves e3 "
        f"{dec['median'] / bar:.1f}x more than the sigma pick difference. Proceed to step 2."))
    print("=" * 84)

    res = dict(centre=list(centre), core=core, sigma=SIGMA_FLOOR,
               rho_ladder=RHO_LADDER, credible_range=[RHO_LO, RHO_HI],
               n_certified=int(certified.sum()),
               decisive=dec, scale_bar=bars, ladder=ladder,
               scale_bar_max=bar, ratio=dec["median"] / bar, pinned=bool(pinned))
    with open(f"{out}_results.json", "w") as f:
        json.dump(res, f, indent=2)
    _figure(res, f"{out}.png")
    print(f"wrote {out}_results.json")
    return res


def _figure(res, save):
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # (1) settling curve: consecutive change per octave vs rho
    x = [0.5 * (l["rho_lo"] + l["rho_hi"]) for l in res["ladder"]]
    y = [l["per_octave"] for l in res["ladder"]]
    ax[0].axvspan(*res["credible_range"], color=C_RHO, alpha=0.12, lw=0, zorder=0)
    ax[0].plot(x, y, "-o", color=C_RHO, lw=2.0, ms=7, zorder=3)
    ax[0].axhline(res["scale_bar_max"], color=C_BAR, lw=2.0, ls="--", zorder=2)
    ax[0].text(4.9, res["scale_bar_max"] * 1.15,
               f"sigma 0.85 -> 0.90 scale bar ({res['scale_bar_max']:.2f} deg)",
               fontsize=9, color=C_BAR, ha="right", va="bottom")
    ax[0].text(np.mean(res["credible_range"]), max(y) * 0.62,
               f"credible range\n[{res['credible_range'][0]}, {res['credible_range'][1]}]",
               fontsize=9.5, color=C_RHO, ha="center", va="top", fontweight="bold")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("rho  [voxels, log]", fontsize=10)
    ax[0].set_ylabel("change in e3 per octave of rho  [deg, log]", fontsize=10)
    ax[0].set_title("Where the field stops moving with rho\n"
                    "(sigma fixed at 0.85)", fontsize=11, color=INK)
    ax[0].set_xticks([1, 2, 3, 5])
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # (2) the decisive comparison
    labels = [f"rho {res['credible_range'][0]} vs {res['credible_range'][1]}\n"
              f"(the open range)",
              f"sigma 0.85 vs 0.90\n(pick difference, rho={res['credible_range'][1]})"]
    vals = [res["decisive"]["median"], res["scale_bar_max"]]
    # scale_bar is keyed by float in memory and by string after a JSON round
    # trip, so accept either rather than depending on how we got here.
    sb = res["scale_bar"]
    hi = res["credible_range"][1]
    sb_hi = sb.get(hi, sb.get(str(hi)))
    p95s = [res["decisive"]["p95"], sb_hi["p95"]]
    cols = [C_RHO, C_BAR]
    yp = np.arange(2)
    for i in range(2):
        ax[1].plot([vals[i], p95s[i]], [i, i], "-", color=cols[i], lw=2.2,
                   alpha=0.55, solid_capstyle="round", zorder=2)
        ax[1].plot([p95s[i]], [i], "|", color=cols[i], ms=10, mew=2.0, zorder=3)
        ax[1].plot([vals[i]], [i], "o", color=cols[i], ms=13, zorder=4)
        ax[1].text(vals[i], i - 0.26, f"{vals[i]:.2f} deg", fontsize=11,
                   color=INK, ha="center", va="bottom", fontweight="bold")
    ax[1].set_yticks(yp); ax[1].set_yticklabels(labels, fontsize=9.5)
    ax[1].set_ylim(-0.6, 1.6); ax[1].invert_yaxis()
    ax[1].set_xscale("log"); ax[1].set_xlim(0.3, 220)
    ax[1].set_xlabel("angular change in e3  [deg, log]   "
                     "dot = median over certified tissue, bar to p95", fontsize=10)
    ax[1].set_title("The decisive comparison\n"
                    + ("rho range costs LESS than the sigma pick -> pinned"
                       if res["pinned"] else
                       "rho range costs MORE than the sigma pick -> needs pinning"),
                    fontsize=11, color=INK)

    for a in ax:
        a.grid(True, color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=INK, labelsize=9)

    fig.suptitle(
        f"Step 1: does e3 depend on WHICH rho in the credible range?  "
        f"i74 {res['core']}^3 core, {res['n_certified']:,} certified tissue voxels",
        fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(save, dpi=135)
    print(f"Saved {save}")


if __name__ == "__main__":
    # --figure-only re-renders from the saved JSON without recomputing the
    # ladder (12 structure-tensor passes), so plot edits are free.
    if "--figure-only" in sys.argv:
        with open("rho_range_sensitivity_results.json") as f:
            _figure(json.load(f), "rho_range_sensitivity.png")
    else:
        run()
