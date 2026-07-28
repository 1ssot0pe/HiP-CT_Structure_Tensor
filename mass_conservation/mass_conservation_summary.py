"""
mass_conservation_summary.py -- cheap plot of the mass_conservation_validate.py
results. Deliberately does NOT re-run the sweep (that costs several minutes to
tens of minutes); it just renders the already-computed summary numbers printed
in mc_validate.log, grouped exactly as that log reports them.
"""

import os as _sr_os, sys as _sr_sys
_SR_ROOT = _sr_os.path.dirname(_sr_os.path.dirname(__file__))
_sr_sys.path.insert(0, _SR_ROOT)  # repo root (host_fod, crossing_shape_3d, kissing_phantom, ...)
_sr_sys.path.insert(0, _sr_os.path.join(_SR_ROOT, "global_tractography"))  # phase1/phase2 certifiers

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESH = 1.5

# (label, mean_ratio, n, is_crossing) -- taken directly from mc_validate.log
BY_NOISE = [
    ("crossing\nnoise=0.00", 2.00, 6, True),
    ("crossing\nnoise=0.07", 1.71, 5, True),
    ("kissing\nnoise=0.00", 0.21, 15, False),
    ("kissing\nnoise=0.07", 0.18, 7, False),
    ("kissing\nnoise=0.15", 0.03, 1, False),
]

BY_GAP = [
    ("gap=2", 0.01, 4),
    ("gap=4\n(bridges in\ntractography)", 0.13, 5),
    ("gap=6\n(diff-size\ntrigger)", 0.37, 10),
    ("gap=8", 0.00, 3),
    ("gap=20", 0.03, 1),
]

CROSS_OVERALL = (1.87, 11, 100)   # mean, n, catch%
KISS_SAME = (0.03, 18, 0)
KISS_DIFF = (0.76, 5, 0)


def bar_panel(ax, labels, values, ns, colors, title):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="k", linewidth=0.6)
    ax.axhline(THRESH, color="0.3", ls="--", lw=1.3, label=f"threshold = {THRESH}")
    for xi, v, n in zip(x, values, ns):
        ax.text(xi, v + 0.06, f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("mass-conservation ratio\n(intensity / local single-fibre calibration)")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=8, loc="upper right")


fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))

labels = [l for l, *_ in BY_NOISE]
values = [v for _, v, *_ in BY_NOISE]
ns = [n for *_, n, _ in BY_NOISE]
colors = ["#2a78d6" if is_x else "#eb6834" for *_, is_x in BY_NOISE]
bar_panel(ax[0], labels, values, ns, colors,
          "By noise level: crossing stays >=1.7x, kissing stays <=0.2x")

labels2 = [l for l, *_ in BY_GAP]
values2 = [v for _, v, _ in BY_GAP]
ns2 = [n for *_, n in BY_GAP]
colors2 = ["#eb6834"] * len(BY_GAP)
bar_panel(ax[1], labels2, values2, ns2, colors2,
          "Kissing only, by gap: ratio stays far below threshold\neven at the tightest gaps")

from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor="#2a78d6", edgecolor="k", label="true crossing"),
                    Patch(facecolor="#eb6834", edgecolor="k", label="kissing (spurious local read)")],
          loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(f"Mass-conservation metric: catch={CROSS_OVERALL[2]:.0f}% (n={CROSS_OVERALL[1]} crossing), "
             f"FP=0% (n={KISS_SAME[1]+KISS_DIFF[1]} kissing, same-size {KISS_SAME[0]:.2f} / "
             f"diff-size {KISS_DIFF[0]:.2f})", fontsize=12)
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
fig.savefig("mass_conservation_summary.png", dpi=140)
print("Saved mass_conservation_summary.png")
