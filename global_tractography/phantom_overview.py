"""
phantom_overview.py -- a top-down ('from above') demonstration of the kissing vs
crossing 3D phantoms used to validate the global tractography pipeline.

Both are made of two straight fibre families (d1 at 0 deg = +x, d2 at 70 deg). The
fibres lie in-plane (z-component 0), so a z-projection ('from above') shows the full
geometry:
  * CROSSING : both families summed EVERYWHERE -> a fibre of each direction runs
               through every point; the two genuinely cross.
  * KISSING  : d1 only on the left of the x=0 contact, d2 only on the right -> the
               fibres TOUCH at the seam but no fibre crosses it.
That single difference is what the pipeline must (and does) tell apart.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from host_geometry_test import make_true_crossing_same_angles, make_kissing_phantom

SHAPE = (96, 96, 96)


def topdown(vol):
    """z-projection (mean over axis0) -> a top-down [axis1(y), axis2(x)] image."""
    return vol.mean(axis=0)


def run(save="phantom_overview.png"):
    cross, mc = make_true_crossing_same_angles(shape=SHAPE)
    kiss, mk = make_kissing_phantom(shape=SHAPE)
    cx = SHAPE[2] / 2.0
    d1, d2 = mk["d1"], mk["d2"]                       # (x, y, z), z=0
    ang2 = int(round(np.degrees(np.arctan2(d2[1], d2[0]))))

    fig, ax = plt.subplots(1, 2, figsize=(13, 6.6))

    def arrow(a, x0, y0, d, color, label=None):
        L = 14.0
        # draw a double-headed segment along the fibre axis (a fibre has no sign)
        a.annotate("", xy=(x0 + d[0] * L, y0 + d[1] * L),
                   xytext=(x0 - d[0] * L, y0 - d[1] * L),
                   arrowprops=dict(arrowstyle="<|-|>", color=color, lw=2.6))
        if label:                                    # label offset PERPENDICULAR
            px, py = -d[1], d[0]
            a.text(x0 + px * 12, y0 + py * 12, label, color=color, fontsize=11.5,
                   fontweight="bold", ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))

    # -- crossing --
    ax[0].imshow(topdown(cross), cmap="gray", origin="upper")
    arrow(ax[0], 24, 30, d1, "#3aa0ff", "d1 (0°)")
    arrow(ax[0], 24, 66, d2, "#ff5a3a", f"d2 ({ang2}°)")
    ax[0].set_title("CROSSING\nboth families everywhere — fibres truly cross",
                    fontsize=12)

    # -- kissing --
    ax[1].imshow(topdown(kiss), cmap="gray", origin="upper")
    ax[1].axvline(cx, color="#33e0e0", ls="--", lw=2.0)
    ax[1].text(cx, 4, "x = 0 contact", color="#33e0e0", fontsize=10, ha="center",
               va="top", fontweight="bold")
    arrow(ax[1], 22, 48, d1, "#3aa0ff", "d1 only")
    arrow(ax[1], 60, 40, d2, "#ff5a3a", "d2 only")
    ax[1].set_title("KISSING\nd1 left, d2 right — fibres touch, none crosses",
                    fontsize=12)

    for a in ax:
        a.set_xlabel("x"); a.set_ylabel("y")
        a.set_xticks([]); a.set_yticks([])

    fig.suptitle("Validation phantoms from above (z-projection): the ONLY difference "
                 "is whether\nfibres continue through the contact — what the global "
                 "tractography pipeline must resolve", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save, dpi=140); plt.close(fig)
    print(f"Saved {save}  (crossing d1={np.round(d1,2)}, d2={np.round(d2,2)})")


if __name__ == "__main__":
    run()
