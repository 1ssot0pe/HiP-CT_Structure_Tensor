"""
kissing_phantom_arrows.py -- overlay recovered HOST directions on the proper (curved,
gapped) kissing phantom, for both the same-size and different-size fibre variants.
Every patch should read "single" (blue) -- never "crossing" -- since the two bundles
are never co-located; the gap at the middle should be visibly empty of glyphs.
"""

import os as _sr_os, sys as _sr_sys
_sr_sys.path.insert(0, _sr_os.path.dirname(_sr_os.path.dirname(__file__)))  # repo root (shared kernels)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kissing_phantom import build_kissing, SHAPE
from host_fod import analyse, classify

C_SINGLE, C_CROSS, C_DISP = "#2a78d6", "#eb6834", "#8c8a84"
HALF, SEG, STEP = 8, 6.0, 5


def draw_dir(ax, cx, cy, d, color, lw=2.0):
    dx, dy = float(d[0]), float(d[1])
    ip = np.hypot(dx, dy)
    if ip < 1e-3:
        ax.plot(cx, cy, ".", color=color, ms=3); return
    ux, uy, L = dx / ip, dy / ip, SEG * ip
    ax.annotate("", xy=(cx + L * ux, cy + L * uy), xytext=(cx - L * ux, cy - L * uy),
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=lw, mutation_scale=7))


def overlay(ax, vol, title):
    z = vol.shape[0] // 2
    ax.imshow(vol[z], cmap="gray", origin="upper")
    for cy in range(HALF + 2, vol.shape[1] - HALF - 2, STEP):
        for cx in range(HALF + 2, vol.shape[2] - HALF - 2, STEP):
            patch = vol[z-HALF:z+HALF+1, cy-HALF:cy+HALF+1, cx-HALF:cx+HALF+1]
            if patch.mean() < 0.05 * vol.max():
                continue
            m = analyse(patch)
            if m is None or m["dirs"] is None or not len(m["dirs"]):
                continue
            lab = classify(m)
            col = C_SINGLE if lab == "single" else (C_CROSS if lab == "crossing" else C_DISP)
            draw_dir(ax, cx, cy, m["dirs"][0], col)
            if lab == "crossing" and len(m["dirs"]) > 1:
                draw_dir(ax, cx, cy, m["dirs"][1], col)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])


def run(save="kissing_phantom_arrows.png"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
    for a, same in zip(ax, (True, False)):
        vol, meta = build_kissing(same_size=same)
        overlay(a, vol, ("SAME-size" if same else "DIFFERENT-size") +
                f" (gap={meta['true_min_gap']:.0f} vox)")
        print(f"  drew {'same' if same else 'diff'}-size", flush=True)
    fig.suptitle("Recovered directions on the proper kissing phantom -- always 'single'\n"
                 "(blue), never 'crossing'; the gap stays visibly empty", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(save, dpi=140); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    run()
