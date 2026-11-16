"""
render_adaptive_benefit.py -- figure for adaptive_sigma_benefit.py.

    python adaptive_sigma_benefit.py        # produces the _results.json / _fields.npz
    python render_adaptive_benefit.py       # this figure

Three panels, in the order the argument runs:
  (1) where sigma* sits relative to the discretisation floor -- the premise;
  (2) how far each arm moves the fibre field, against the method-noise floor
      and the scale bar -- the measurement;
  (3) where the movement is, spatially -- the sanity check that a median is not
      hiding a structured effect in some sub-region.

Colour carries ONE distinction: adaptive-sigma arms vs constant-sigma scale-bar
arms (two hues, checked against the CVD/lightness/chroma/contrast validator --
the pink/teal pair that reads fine to me collapses to dE 4.6 under deuteranopia
and was rejected). Deployability is carried by fill, not hue: arms that use a
sigma below the 0.85 floor are drawn hollow, so the "you are not allowed to do
this" distinction survives greyscale and colour-blind viewing.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

INK = "#0b0b0b"
MUTED = "#6b6b6b"
GRID = "#d8d8d6"
C_ADAPT = "#6a4bc4"      # adaptive-sigma arms
C_CONST = "#1f9e89"      # constant-sigma scale bar
C_NOISE = "#b9b9b7"      # method-noise band

# (key, short label, colour, deployable?)  -- ordered as plotted, low to high
ROWS = [
    ("F_adaptive_floored_rho346", "adaptive sigma*, floored\nrho = 3.46", C_ADAPT, True),
    ("B_adaptive_floored",        "adaptive sigma*, floored\nrho = 1 x sigma", C_ADAPT, True),
    ("H_sigma090_rho346",         "sigma 0.85 -> 0.90\nrho = 3.46", C_CONST, True),
    ("C_sigma090",                "sigma 0.85 -> 0.90\nrho = 1 x sigma", C_CONST, True),
    ("G_adaptive_raw_rho346",     "adaptive sigma*, UNCLAMPED\nrho = 3.46", C_ADAPT, False),
    ("A_adaptive_raw",            "adaptive sigma*, UNCLAMPED\nrho = 1 x sigma", C_ADAPT, False),
    ("D_sigma115",                "sigma 0.85 -> 1.15\nrho = 1 x sigma", C_CONST, True),
    ("E_rho346",                  "rho 0.85 -> 3.46\n(sigma unchanged)", C_CONST, True),
]


def render(prefix="adaptive_sigma_benefit", save="adaptive_sigma_benefit.png"):
    with open(f"{prefix}_results.json") as f:
        R = json.load(f)
    F = np.load(f"{prefix}_fields.npz")
    arms, ss = R["arms"], R["sigma_star"]
    floor = R["sigma_floor"]
    noise = R["method_control"]["interpolation_at_midpoint"]["median"]

    fig = plt.figure(figsize=(18, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.5, 0.85],
                          height_ratios=[1, 1], wspace=0.30, hspace=0.38)
    ax_hist = fig.add_subplot(gs[:, 0])
    ax_arms = fig.add_subplot(gs[:, 1])
    ax_map1 = fig.add_subplot(gs[0, 2])
    ax_map2 = fig.add_subplot(gs[1, 2])

    # ---- (1) the premise: sigma* vs the floor -----------------------------
    s = F["sigma_star_certified"]
    n, _, _ = ax_hist.hist(s, bins=np.linspace(0.5, 1.3, 90), color=C_ADAPT,
                           alpha=0.85, edgecolor="none")
    top = n.max() * 1.28
    ax_hist.set_ylim(0, top)
    ax_hist.axvspan(0.5, floor, color=C_NOISE, alpha=0.35, zorder=0, lw=0)
    ax_hist.axvline(floor, color=INK, lw=1.8, ls="--")
    ax_hist.axvline(ss["sigma_median"], color=C_ADAPT, lw=1.8)
    ax_hist.annotate(f"discretisation floor {floor}\n(picks below are rejected)",
                     xy=(floor, top * 0.70), xytext=(1.02, top * 0.80),
                     fontsize=9, color=INK, va="center", ha="left",
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax_hist.annotate(f"median {ss['sigma_median']:.2f} vox ({ss['um_median']:.0f} um)",
                     xy=(ss["sigma_median"], top * 0.84),
                     xytext=(0.90, top * 0.92), fontsize=9.5, color=C_ADAPT,
                     va="center", ha="left", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=C_ADAPT, lw=1.2))
    ax_hist.text(0.515, top * 0.50,
                 f"{ss['frac_below_floor']:.1%}\nof certified\ntissue voxels\n"
                 f"sit BELOW\nthe floor",
                 fontsize=10.5, color=INK, va="top", ha="left")
    ax_hist.set_xlabel("selected scale sigma*(x)  [voxels]", fontsize=10)
    ax_hist.set_ylabel("certified tissue voxels", fontsize=10)
    ax_hist.set_xlim(0.5, 1.3)
    ax_hist.set_title("1. The premise\nthe map's own picks are below the usable floor",
                      fontsize=11, color=INK)

    # ---- (2) the measurement: how far each arm moves the field ------------
    ypos = np.arange(len(ROWS))
    for i, (key, lab, col, deploy) in enumerate(ROWS):
        a = arms[key]
        ax_arms.plot([a["median"], a["p95"]], [i, i], "-", color=col, lw=2.0,
                     solid_capstyle="round", alpha=0.55, zorder=2)
        ax_arms.plot([a["p95"]], [i], "|", color=col, ms=9, mew=2.0, zorder=3)
        ax_arms.plot([a["median"]], [i], "o", ms=11, zorder=4,
                     color=col if deploy else "white",
                     markeredgecolor=col, markeredgewidth=2.0)
        ax_arms.text(a["median"] * 0.88, i - 0.30, f"{a['median']:.2f}",
                     fontsize=10, color=INK, ha="right", va="center",
                     fontweight="bold")
    ax_arms.axvspan(1e-3, noise, color=C_NOISE, alpha=0.45, zorder=0, lw=0)
    ax_arms.axvline(noise, color=MUTED, lw=1.3, ls=":")
    ax_arms.text(noise * 1.25, len(ROWS) - 0.62,
                 f"method noise floor {noise:.2f} deg  "
                 f"(what the scale-blend itself invents)",
                 fontsize=8.5, color=MUTED, va="center", ha="left")
    ax_arms.set_yticks(ypos)
    ax_arms.set_yticklabels([r[1] for r in ROWS], fontsize=9)
    ax_arms.set_ylim(-1.1, len(ROWS) - 0.3)
    ax_arms.invert_yaxis()
    ax_arms.set_xscale("log")
    ax_arms.set_xlim(0.02, 300)
    ax_arms.set_xlabel("angular change in the fibre direction e3  [degrees, log]\n"
                       "dot = median over certified tissue, bar to p95", fontsize=10)
    ax_arms.set_title("2. The measurement\nthe deployable adaptive field moves e3 LESS than "
                      "switching sigma 0.85 -> 0.90", fontsize=11, color=INK)
    ax_arms.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=C_ADAPT, ms=10,
               label="adaptive sigma*(x)"),
        Line2D([], [], marker="o", ls="none", color=C_CONST, ms=10,
               label="constant-sigma / rho scale bar"),
        Line2D([], [], marker="o", ls="none", color="white", ms=10,
               markeredgecolor=INK, markeredgewidth=1.8,
               label=f"hollow = uses sigma < {floor}, NOT deployable"),
    ], fontsize=8.5, loc="upper right", framealpha=0.95)

    # ---- (3) where the movement is ----------------------------------------
    mid_ok = F["certified_mid"]
    cmap = matplotlib.colormaps["magma"].with_extremes(bad="#d8d8d6")
    for ax, key, ttl in ((ax_map1, "F_adaptive_floored_rho346",
                          "3. Where the movement is  (mid slice)\n"
                          "deployable arm F: floored sigma*, rho = 3.46"),
                         (ax_map2, "A_adaptive_raw",
                          "unclamped arm A: sigma*(x), rho = 1 x sigma\n"
                          "(structureless -- sub-floor instability, not signal)")):
        a = np.where(mid_ok, F[f"ang_{key}_mid"], np.nan)
        im = ax.imshow(a, cmap=cmap, vmin=0, vmax=20)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=9, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("change in e3 [deg]", fontsize=8)
        cb.ax.tick_params(labelsize=8)

    for ax in (ax_hist, ax_arms):
        ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK, labelsize=9)

    ratio = (arms["H_sigma090_rho346"]["median"]
             / arms["F_adaptive_floored_rho346"]["median"])
    fig.suptitle(
        f"Does per-voxel adaptive sigma change the fibre field?\n"
        f"No -- floored to sigma >= {floor} it moves e3 by "
        f"{arms['F_adaptive_floored_rho346']['median']:.2f} deg (median), {ratio:.0f}x less "
        f"than the 0.85 -> 0.90 pick difference and {arms['E_rho346']['median'] / arms['F_adaptive_floored_rho346']['median']:.0f}x "
        f"less than the rho ambiguity   "
        f"[i74 {R['core']}^3 core at {tuple(R['centre'])}, {R['voxel_um']:g} um/voxel]",
        fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(save, dpi=135)
    print(f"Saved {save}")


if __name__ == "__main__":
    render()
