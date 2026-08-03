"""
brain_phantom_arrows.py -- visualise WHAT the pipeline recovered: overlay the HOST
fibre directions as arrows on each all-features phantom block, coloured by the
classification. Reuses the phantom + half-maps from brain_phantom_pipeline.

  single    (blue)   one direction
  crossing  (orange) two directions (both drawn)
  kissing   (blue)   one direction per half -> two blue fields meeting at the seam
  complex   (grey)   messy / out-of-plane (short) glyphs

Arrows are the IN-PLANE projection with length scaled by in-plane magnitude, so a
fibre running through the slice is honestly a short stub, not a fake in-plane arrow.
"""

import os as _sr_os, sys as _sr_sys
_sr_sys.path.insert(0, _sr_os.path.dirname(_sr_os.path.dirname(__file__)))  # repo root (shared kernels)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from host_fod import analyse, classify
import brain_phantom_pipeline as BP

C_SINGLE, C_CROSS, C_DISP = "#2a78d6", "#eb6834", "#8c8a84"
HALF, SEG, STEP = BP.HALF, 6.5, 9


def draw_dir(ax, cx, cy, d, color, lw=2.1):
    dx, dy = float(d[0]), float(d[1])
    ip = np.hypot(dx, dy)
    if ip < 1e-3:
        ax.plot(cx, cy, ".", color=color, ms=3); return
    ux, uy, L = dx / ip, dy / ip, SEG * ip
    ax.annotate("", xy=(cx + L * ux, cy + L * uy), xytext=(cx - L * ux, cy - L * uy),
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=lw, mutation_scale=8))


def overlay(ax, avg, title):
    z = avg.shape[0] // 2
    ax.imshow(avg[z], cmap="gray", origin="upper")
    for cy in range(HALF + 2, avg.shape[1] - HALF - 2, STEP):
        for cx in range(HALF + 2, avg.shape[2] - HALF - 2, STEP):
            patch = avg[z-HALF:z+HALF+1, cy-HALF:cy+HALF+1, cx-HALF:cx+HALF+1]
            m = analyse(patch)
            if m is None or m["dirs"] is None or not len(m["dirs"]):
                continue
            lab = classify(m)
            if lab == "single":
                draw_dir(ax, cx, cy, m["dirs"][0], C_SINGLE)
            elif lab == "crossing":
                draw_dir(ax, cx, cy, m["dirs"][0], C_CROSS)
                if len(m["dirs"]) > 1:
                    draw_dir(ax, cx, cy, m["dirs"][1], C_CROSS)
            else:                                   # complex / dispersed / disorganised
                for d in m["dirs"][:2]:
                    draw_dir(ax, cx, cy, d, C_DISP, lw=1.3)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])


def run(save="brain_phantom_arrows.png"):
    feats = {k: BP.build(k, seed=1) for k in ("single", "crossing", "kissing", "complex")}
    avg = {k: 0.5 * (a + b) for k, (a, b) in
           ((k, BP.halfmaps(S, seed=i)) for i, (k, S) in enumerate(feats.items()))}
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.7))
    for a, k in zip(ax, feats):
        overlay(a, avg[k], k)
        print(f"  drew {k}", flush=True)
    fig.legend(handles=[
        Line2D([], [], color=C_SINGLE, lw=3, label="single (1 direction)"),
        Line2D([], [], color=C_CROSS, lw=3, label="crossing (2 directions)"),
        Line2D([], [], color=C_DISP, lw=2, label="complex / unresolved"),
    ], loc="lower center", ncol=3, fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Recovered fibre directions per region (HOST orientations on the "
                 "averaged half-maps)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(save, dpi=140); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    run()
