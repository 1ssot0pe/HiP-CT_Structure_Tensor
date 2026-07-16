"""
visualize_i74_orientations.py

Overlay the recovered HOST-FOD orientations on the real i74 volume. Runs the
scale-adaptive + triage pipeline (adaptive_triage.triage_patch_v2) on a grid of
patches across one slice of the interior tissue box and draws, at each patch
centre, a glyph coloured by triage TIER:

  single    (blue)   -- one confident in-plane direction segment
  crossing  (orange) -- two direction segments (the resolved fibres)
  dispersed/unresolvable (grey dot) -- no trustworthy single direction

Direction basis: analyse() returns unit vectors (x, y, z) = (axis2, axis1,
axis0) -- the same basis the phantom validation matched truths in. The slice is
taken at fixed axis0 (z), so imshow shows (axis1=y=rows, axis2=x=cols); each
segment's horizontal component is dx and vertical is dy. Segment length is
scaled by the IN-PLANE magnitude sqrt(dx^2+dy^2), so a fibre running through the
slice (large dz) is honestly foreshortened to a short stub / dot rather than
faking an in-plane direction.

No ground truth exists, so this is a qualitative map of WHERE the pipeline is
confident (single), WHERE it resolves crossings, and WHERE it gives up
(dispersed) -- laid over the anatomy. Reuses load_interior (selfcalib_threshold)
and triage_patch_v2 (adaptive_triage).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from selfcalib_threshold import load_interior
from adaptive_triage import triage_patch_v2

# dataviz reference palette (light)
C_SINGLE = "#2a78d6"     # blue
C_CROSS = "#eb6834"      # orange
C_DISP = "#898781"       # muted grey
INK = "#0b0b0b"

SIGMAS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
HALF = 20
Z_SLICE = 100            # axis0 index within the 200^3 interior box
STEP = 28                # patch-centre spacing across the slice
SEG = 12.0               # max half-length of a direction segment (voxels)


def _draw_dir(ax, cx, cy, d, color, lw=2.2):
    """d = (dx, dy, dz) unit vector; draw its in-plane projection, length scaled
    by in-plane magnitude (out-of-plane fibres -> short stub)."""
    dx, dy = float(d[0]), float(d[1])
    inplane = np.hypot(dx, dy)
    if inplane < 1e-3:
        ax.plot(cx, cy, ".", color=color, ms=4)     # ~perpendicular to slice
        return
    ux, uy = dx / inplane, dy / inplane
    L = SEG * inplane
    ax.plot([cx - L * ux, cx + L * ux], [cy - L * uy, cy + L * uy],
            "-", color=color, lw=lw, solid_capstyle="round")


if __name__ == "__main__":
    import time

    vol = load_interior()
    sl = vol[Z_SLICE]
    centers = list(range(HALF + 10, vol.shape[1] - HALF - 5, STEP))
    grid = [(cy, cx) for cy in centers for cx in centers]
    print(f"i74 interior {vol.shape}, slice z={Z_SLICE}; {len(grid)} patches "
          f"(half={HALF}, step={STEP})")

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(sl, cmap="gray", origin="upper")
    ax.set_xticks([]); ax.set_yticks([])

    counts = {"single": 0, "crossing": 0, "dispersed": 0, "other": 0}
    t0 = time.time()
    for i, (cy, cx) in enumerate(grid):
        patch = vol[Z_SLICE - HALF:Z_SLICE + HALF + 1,
                    cy - HALF:cy + HALF + 1, cx - HALF:cx + HALF + 1]
        r = triage_patch_v2(patch, sigmas_lindeberg=SIGMAS, sigma_sweep=SIGMAS)
        m, label = r["m"], r["label"]
        if label == "single" and m is not None and len(m["dirs"]):
            _draw_dir(ax, cx, cy, m["dirs"][0], C_SINGLE)
            counts["single"] += 1
        elif label == "crossing" and m is not None and len(m["dirs"]) >= 2:
            _draw_dir(ax, cx, cy, m["dirs"][0], C_CROSS)
            _draw_dir(ax, cx, cy, m["dirs"][1], C_CROSS)
            counts["crossing"] += 1
        elif label == "dispersed":
            ax.plot(cx, cy, "o", color=C_DISP, ms=4, alpha=0.7)
            counts["dispersed"] += 1
        else:
            counts["other"] += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(grid)} ... ({time.time()-t0:.0f}s)")

    n = len(grid)
    ax.set_title(
        f"i74 recovered orientations (slice z={Z_SLICE}) -- "
        f"single {counts['single']}/{n}, crossing {counts['crossing']}/{n}, "
        f"unresolvable {counts['dispersed']}/{n}",
        fontsize=12, color=INK)
    ax.legend(handles=[
        Line2D([], [], color=C_SINGLE, lw=3, label="single fibre (1 direction)"),
        Line2D([], [], color=C_CROSS, lw=3, label="crossing (2 directions)"),
        Line2D([], [], color=C_DISP, marker="o", ls="", label="unresolvable / dispersed"),
    ], loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig("i74_orientations.png", dpi=140)
    print(f"\n({time.time()-t0:.0f}s total)  counts={counts}")
    print("Saved i74_orientations.png")
