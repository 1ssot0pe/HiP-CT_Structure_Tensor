"""
crossing_trigger_test.py

Can a CHEAP 2nd-order map localise the crossing region -- so HOST-FOD is run
only there, not everywhere? Our current trigger metric is sphericity
c_s = 3*lambda3/(lambda1+lambda2+lambda3) from the plain structure tensor, gated
at c_s > 0.15. This tests it on a clean phantom with a KNOWN crossing region.

Phantom: a "+" of two perpendicular Gaussian-tube bundles.
  - horizontal bundle (tubes along x) fills the slab |y| < HALF
  - vertical  bundle (tubes along y) fills the slab |x| < HALF
  -> central square (|x|<HALF & |y|<HALF)  = OVERLAP  = CROSSING  (needs HOST)
  -> horizontal arm (|y|<HALF, |x|>HALF)   = single fibre        (e3 is fine)
  -> vertical arm   (|x|<HALF, |y|>HALF)   = single fibre        (e3 is fine)

We compute the c_s field (one cheap structure-tensor pass) and ask: is c_s
systematically HIGHER in the crossing than in the single arms, and does the 0.15
gate separate them? Also compare planarity c_p (single fibre is planar) as an
alternative cheap trigger. Reuses _tube_family/_coords (crossing_shape_3d) and
the same structure_tensor_3d/eig_special_3d used everywhere.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from structure_tensor import structure_tensor_3d, eig_special_3d

from crossing_shape_3d import _tube_family, _coords

EPS = 1e-12


def make_perp_bundles(shape=(96, 96, 96), width=1.5, spacing=10.0, spacing2=13.0,
                      half=22, noise_sigma=0.2, seed=0):
    x, y, z = _coords(shape)                     # centred coords, (x,y,z)
    b1 = _tube_family((x, y, z), [1, 0, 0], spacing, width)    # tubes along x
    b2 = _tube_family((x, y, z), [0, 1, 0], spacing2, width)   # tubes along y
    horiz = np.abs(y) < half                     # where bundle 1 lives
    vert = np.abs(x) < half                      # where bundle 2 lives
    vol = np.where(horiz, b1, 0.0) + np.where(vert, b2, 0.0)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        vol = vol + noise_sigma * rng.standard_normal(shape).astype(np.float32)
    vol = (vol - vol.min()) / (np.ptp(vol) + EPS)
    masks = dict(crossing=horiz & vert, arm_h=horiz & ~vert,
                 arm_v=vert & ~horiz, empty=~horiz & ~vert)
    return vol.astype(np.float32), masks


def shape_fields(vol, sigma=1.0, rho=4.0):
    """Per-voxel c_s (sphericity) and c_p (planarity) from the 2nd-order tensor."""
    S = structure_tensor_3d(vol.astype(np.float32), sigma, rho)
    val, _ = eig_special_3d(S)                   # val desc: val[0]>=val[1]>=val[2]
    l1, l2, l3 = val[0], val[1], val[2]
    s = l1 + l2 + l3 + EPS
    c_s = 3.0 * l3 / s
    c_p = 2.0 * (l2 - l3) / s
    return c_s, c_p


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol, masks = make_perp_bundles()
    c_s, c_p = shape_fields(vol)
    h = vol.shape[0] // 2

    print("=" * 66)
    print("CROSSING-TRIGGER TEST: can c_s localise the crossing?")
    print(f"  {'region':>12} | {'mean c_s':>9} | {'mean c_p':>9} | {'frac c_s>0.15':>13}")
    for name in ("crossing", "arm_h", "arm_v", "empty"):
        m = masks[name]
        print(f"  {name:>12} | {c_s[m].mean():9.3f} | {c_p[m].mean():9.3f} | "
              f"{(c_s[m] > 0.15).mean():13.1%}")
    cross_cs = c_s[masks["crossing"]]
    arm_cs = c_s[masks["arm_h"] | masks["arm_v"]]
    # can the 0.15 gate separate crossing voxels from single-arm voxels?
    tp = (cross_cs > 0.15).mean()          # crossing correctly flagged
    fp = (arm_cs > 0.15).mean()            # single arm wrongly flagged
    print("-" * 66)
    print(f"  gate c_s>0.15:  crossing flagged (recall)={tp:.1%},  "
          f"single-arm falsely flagged (FPR)={fp:.1%}")
    sep = cross_cs.mean() - arm_cs.mean()
    print(f"  mean c_s gap (crossing - single arm) = {sep:+.3f}")
    print("=" * 66)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    ax[0].imshow(vol[h], cmap="gray"); ax[0].set_title("phantom: perpendicular '+' bundles")
    im1 = ax[1].imshow(c_s[h], cmap="magma", vmin=0, vmax=0.6)
    ax[1].set_title("sphericity c_s (cheap trigger map)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    trig = (c_s[h] > 0.15).astype(float)
    ax[2].imshow(vol[h], cmap="gray")
    ax[2].imshow(np.ma.masked_where(trig < 0.5, trig), cmap="autumn", alpha=0.55)
    ax[2].set_title("c_s>0.15 trigger (overlay) vs true centre crossing")
    # outline the true crossing square
    xs = np.where(masks["crossing"][h].any(axis=0))[0]
    ys = np.where(masks["crossing"][h].any(axis=1))[0]
    if xs.size and ys.size:
        ax[2].add_patch(plt.Rectangle((xs[0], ys[0]), xs[-1]-xs[0], ys[-1]-ys[0],
                                      fill=False, edgecolor="cyan", lw=2))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Does the cheap c_s map flag the crossing region (cyan box) that needs HOST?",
                 fontsize=13)
    fig.tight_layout(); fig.savefig("crossing_trigger_test.png", dpi=130)
    print("Saved crossing_trigger_test.png")
