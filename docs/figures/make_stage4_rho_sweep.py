"""Figure: Stage 4 integration-scale sweep across seed counts.

Plots results/seed_rho_curves.json exactly as recorded (no smoothing, no
interpolation between the 14 measured rho values) and writes
docs/figures/stage4_rho_sweep.png (README) and .pdf (vector, for reports).

    python docs/figures/make_stage4_rho_sweep.py

Data: i74, 48^3 chunk at centre 384, sigma = 1.00, MSE objective, relax turn
mode, six field updates per run; median and mean angular error between the
Stage 3 refined field and a plain structure tensor at each rho.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "results", "seed_rho_curves.json")
OUT = os.path.join(HERE, "stage4_rho_sweep")

RHO_STAGE1 = 2 ** 0.5        # Stage 1's rho on this chunk (stage4_sigma_rho.json)
CHUNK_HALF = 24              # 48^3 chunk; the Gaussian window reaches ~4 rho

# Colour: seed count is an ordered quantity, so one hue light -> dark
# (validated ordinal blue ramp, steps 250-650). Ink and chrome are neutral.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SHADE = "#e1e0d9", "#c3c2b7", "#f0efec"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.linewidth": 0.8,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def panel_label(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=INK, va="bottom", ha="left")


def main():
    with open(DATA) as f:
        d = json.load(f)
    rhos = d["rhos"]
    seeds = sorted(d["res"], key=int)
    med = {s: d["res"][s]["median"] for s in seeds}
    mean = {s: d["res"][s]["mean"] for s in seeds}
    i_min = {s: min(range(len(rhos)), key=med[s].__getitem__) for s in seeds}
    rho_star = {rhos[i_min[s]] for s in seeds}
    assert len(rho_star) == 1, "minima differ between seed counts"
    rho_star = rho_star.pop()

    fig, (a, b) = plt.subplots(
        1, 2, figsize=(7.2, 3.1),
        gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.38))

    # (a) error against rho, one line per seed count ----------------------
    a.axvspan(CHUNK_HALF / 4, rhos[-1] * 1.08, color=SHADE, lw=0, zorder=0)
    a.text((CHUNK_HALF / 4 * rhos[-1] * 1.08) ** 0.5, 34,
           "window 4ρ exceeds\nchunk half-width", color=MUTED, fontsize=6.5,
           rotation=90, ha="center", va="center", linespacing=1.1)
    a.axvline(RHO_STAGE1, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=1)
    a.text(RHO_STAGE1 * 1.04, 41.5, "Stage 1\nρ = 1.41", color=INK2,
           fontsize=7, va="top")
    a.axvline(rho_star, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
    a.text(rho_star * 1.04, 41.5, f"ρ* = {rho_star:.2f}", color=INK,
           fontsize=7.5, fontweight="bold", va="top")
    for s, c in zip(seeds, RAMP):
        a.plot(rhos, med[s], color=c, lw=1.5, marker="o", ms=3.2,
               mec="white", mew=0.5, label=f"{int(s):,}", zorder=3)
        j = i_min[s]
        a.plot(rhos[j], med[s][j], "o", ms=7, mfc="none", mec=c, mew=1.1,
               zorder=4)
    a.set_xscale("log", base=2)
    ticks = [0.71, 1, 1.41, 2, 2.83, 4, 5.66, 8]
    a.xaxis.set_major_locator(FixedLocator(ticks))
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xticklabels([f"{t:g}" for t in ticks])
    a.set_xlim(rhos[0] / 1.08, rhos[-1] * 1.08)
    a.set_ylim(17.5, 43.5)
    a.set_xlabel("Integration scale ρ (voxels)")
    a.set_ylabel("Median angular error (°)")
    leg = a.legend(title="Seed count", frameon=False, loc="lower left",
                   handlelength=1.8, borderaxespad=0.4, labelcolor=INK2)
    leg.get_title().set_color(INK2)
    leg.get_title().set_fontsize(8)
    panel_label(a, "a")

    # (b) error at the optimum against seed count -------------------------
    x = [int(s) for s in seeds]
    ym = [med[s][i_min[s]] for s in seeds]
    ya = [mean[s][i_min[s]] for s in seeds]
    b.plot(x, ya, color=INK2, lw=1.2, ls=(0, (4, 2)), zorder=2)
    b.plot(x, ym, color=INK, lw=1.2, zorder=2)
    for xi, m1, m2, c in zip(x, ym, ya, RAMP):
        b.plot(xi, m2, "s", ms=5.5, mfc=c, mec="white", mew=0.7, zorder=3)
        b.plot(xi, m1, "o", ms=6, mfc=c, mec="white", mew=0.7, zorder=3)
    b.text(x[-1] + 230, ya[-1], "mean", color=INK2, fontsize=8, va="center")
    b.text(x[-1] + 230, ym[-1], "median", color=INK, fontsize=8,
           va="center")
    b.text(0.02, 0.98, f"ρ* = {rho_star:.2f} for every seed count",
           transform=b.transAxes, color=INK2, fontsize=7.5, va="top")
    b.set_xticks(x)
    b.set_xticklabels([f"{v:,}" for v in x])
    b.set_xlim(700, 6500)
    b.set_ylim(17.5, 29.5)
    b.set_xlabel("Seed count")
    b.set_ylabel("Angular error at ρ* (°)")
    panel_label(b, "b")

    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", pad_inches=0.04, **kw)
    print(f"wrote {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
