"""
kissing_gap_sweep_arrows.py -- image demonstration of the gap sweep result: one panel
per gap (same fibre width), showing the phantom slice, recovered HOST directions
(blue=single, orange=crossing), and whether global tractography BRIDGED the gap --
annotated with the valley-fusion diagnostic that explains WHY the smallest gaps break
(image-level fusion, not an algorithm bug).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from kissing_phantom import build_kissing
from host_fod import analyse, classify
from test_kissing_phantom import tractography_gap_test
from kissing_gap_sweep import GAPS, RADIUS, valley_ratio

C_SINGLE, C_CROSS, C_DISP = "#2a78d6", "#eb6834", "#8c8a84"
HALF, SEG, STEP = 8, 5.0, 5


def draw_dir(ax, cx, cy, d, color, lw=1.8):
    dx, dy = float(d[0]), float(d[1])
    ip = np.hypot(dx, dy)
    if ip < 1e-3:
        ax.plot(cx, cy, ".", color=color, ms=3); return
    ux, uy, L = dx / ip, dy / ip, SEG * ip
    ax.annotate("", xy=(cx + L * ux, cy + L * uy), xytext=(cx - L * ux, cy - L * uy),
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=lw, mutation_scale=6))


def overlay(ax, vol):
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


def run(save="kissing_gap_sweep_arrows.png"):
    fig, ax = plt.subplots(1, len(GAPS), figsize=(4.2 * len(GAPS), 5.2))
    for a, gap in zip(ax, GAPS):
        vol, meta = build_kissing(same_size=True, gap=gap)
        vr = valley_ratio(vol)
        _npos, _ntr, _big, spanning = tractography_gap_test(vol, meta["gap"])
        overlay(a, vol)
        bridged = spanning > 0
        status = f"BRIDGED ({spanning})" if bridged else "separate (0)"
        col = "#c62828" if bridged else "#2e7d32"
        a.set_title(f"gap={gap:.0f} vox ({gap/RADIUS:.1f}x radius)\n"
                    f"valley={vr:.0%}  |  tracts: {status}", fontsize=10.5, color=col)
        a.set_xticks([]); a.set_yticks([])
        print(f"  gap={gap:.0f} done", flush=True)
    fig.legend(handles=[
        Line2D([], [], color=C_SINGLE, lw=2.5, label="single (1 direction)"),
        Line2D([], [], color=C_CROSS, lw=2.5, label="crossing (2 directions)"),
        Line2D([], [], color=C_DISP, lw=2, label="unresolved"),
    ], loc="lower center", ncol=3, fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Kissing gap sweep -- same fibre width (radius {RADIUS}), gap = "
                 f"{'/'.join(f'{g:.0f}' for g in GAPS)} vox\n"
                 "red title = tractography bridged the gap (valley%=image fusion, "
                 "not an algorithm failure); green = held clean", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    fig.savefig(save, dpi=140); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    run()
