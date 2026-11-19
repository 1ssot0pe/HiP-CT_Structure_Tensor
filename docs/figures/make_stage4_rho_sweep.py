"""Figure: Stage 4 integration-scale sweep (the README's key result).

One panel: median angular error between the Stage 3 refined field and a plain
structure tensor, against the integration scale rho. The line is the
5,000-seed run; the band spans all five seed counts (1,000-5,000), which
overlap almost exactly. Plots results/seed_rho_curves.json as recorded (no
smoothing, no interpolation between the 14 measured rho values).

Writes, next to this script:
    stage4_rho_sweep.png       light theme (README, GitHub light mode)
    stage4_rho_sweep_dark.png  dark theme  (README, GitHub dark mode)
    stage4_rho_sweep.pdf       light theme, vector (reports)

    python docs/figures/make_stage4_rho_sweep.py

Data: i74, 48^3 chunk at centre 384, sigma = 1.00, MSE objective, relax turn
mode, six field updates per run.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.patches import Patch                     # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "results", "seed_rho_curves.json")
OUT = os.path.join(HERE, "stage4_rho_sweep")

RHO_STAGE1 = 2 ** 0.5    # Stage 1's rho on this chunk (stage4_sigma_rho.json)
LINE_SEEDS = "5000"      # the run drawn as the line; the band covers all runs

# One series, so one hue. Surfaces match GitHub's page backgrounds so the
# figure sits flush in both themes. Line colours validated (>= 3:1 contrast,
# lightness band, chroma floor) with the dataviz palette validator.
THEMES = {
    "light": dict(bg="#ffffff", line="#2a78d6", band="#2a78d6", band_a=0.18,
                  ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e8e7e1", axis="#c3c2b7"),
    "dark": dict(bg="#0d1117", line="#3987e5", band="#3987e5", band_a=0.28,
                 ink="#f0efec", ink2="#c3c2b7", muted="#8b8a84",
                 grid="#21262d", axis="#3d444d"),
}


def load():
    with open(DATA) as f:
        d = json.load(f)
    rhos = d["rhos"]
    seeds = sorted(d["res"], key=int)
    med = {s: d["res"][s]["median"] for s in seeds}
    i_min = {s: min(range(len(rhos)), key=med[s].__getitem__) for s in seeds}
    stars = {i_min[s] for s in seeds}
    assert len(stars) == 1, "minima differ between seed counts"
    lo = [min(med[s][i] for s in seeds) for i in range(len(rhos))]
    hi = [max(med[s][i] for s in seeds) for i in range(len(rhos))]
    return rhos, med[LINE_SEEDS], lo, hi, stars.pop(), seeds


def draw(theme, rhos, line, lo, hi, j, seeds):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink"],
        "axes.linewidth": 0.8,
        "xtick.color": t["axis"], "ytick.color": t["axis"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "xtick.major.size": 3, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": t["grid"], "grid.linewidth": 0.7,
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"],
        "savefig.facecolor": t["bg"], "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.15)

    # Title states the finding; subtitle states the data.
    fig.text(0.09, 0.955, "The refined field is best matched at ρ = 3.79",
             fontsize=11.5, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.09, 0.885,
             "i74 brain, 48³-voxel chunk, σ = 1.00  ·  same optimum for "
             f"every seed count ({int(seeds[0]):,}–{int(seeds[-1]):,})",
             fontsize=8.5, color=t["ink2"], va="top")

    # Stage 1's data-driven pick, for comparison.
    ax.axvline(RHO_STAGE1, color=t["muted"], lw=0.9, ls=(0, (1.5, 2)),
               zorder=1)
    ax.text(RHO_STAGE1 * 1.05, 44.2, "Stage 1 pick\nρ = 1.41",
            color=t["ink2"], fontsize=8, va="top", linespacing=1.15)

    # Data: band over seed counts, line for one run.
    ax.fill_between(rhos, lo, hi, color=t["band"], alpha=t["band_a"], lw=0,
                    zorder=2)
    ax.plot(rhos, line, color=t["line"], lw=2, marker="o", ms=4,
            mec=t["bg"], mew=0.8, zorder=3)

    # The optimum.
    x0, y0 = rhos[j], line[j]
    ax.plot(x0, y0, "o", ms=9, mfc=t["line"], mec=t["bg"], mew=1.5,
            zorder=4)
    ax.annotate(f"ρ* = {x0:.2f}\nmedian error {y0:.1f}°",
                xy=(x0, y0), xytext=(x0, y0 - 6.2), ha="center", va="top",
                fontsize=8.5, color=t["ink"], linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.8,
                                shrinkA=0, shrinkB=6))

    ticks = [0.71, 1, 1.41, 2, 2.83, 4, 5.66, 8]
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_xlim(rhos[0] / 1.1, rhos[-1] * 1.1)
    ax.set_ylim(10, 45)
    ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels(["10°", "20°", "30°", "40°"])
    ax.set_xlabel("Integration scale ρ (voxels, log scale)")
    ax.set_ylabel("Median angular error")

    handles = [
        Line2D([], [], color=t["line"], lw=2, marker="o", ms=4,
               mec=t["bg"], mew=0.8, label=f"{int(LINE_SEEDS):,} seeds"),
        Patch(color=t["band"], alpha=t["band_a"], lw=0,
              label=f"range, {int(seeds[0]):,}–{int(seeds[-1]):,} seeds"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right",
              fontsize=8, handlelength=1.8, labelcolor=t["ink2"])
    return fig


def main():
    rhos, line, lo, hi, j, seeds = load()
    for theme in ("light", "dark"):
        fig = draw(theme, rhos, line, lo, hi, j, seeds)
        suffix = "" if theme == "light" else "_dark"
        fig.savefig(f"{OUT}{suffix}.png", dpi=300)
        if theme == "light":
            fig.savefig(f"{OUT}.pdf")
        plt.close(fig)
    print(f"wrote {OUT}.png, {OUT}_dark.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
