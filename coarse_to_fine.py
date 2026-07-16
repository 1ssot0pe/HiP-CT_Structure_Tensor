"""
coarse_to_fine.py -- step 3 (C1) of the gate fix plan.

Steps 1+2 showed no cheap PER-PATCH scalar reliably localises crossings: a
single fibre's texture looks spherical enough that any voxel/patch threshold
either misses crossings or fires on half the single fibre. The failure is that
each patch is judged in ISOLATION.

C1 uses SPATIAL COHERENCE instead of a per-patch decision. A real crossing
region is spatially EXTENDED; a noise-driven high-sphericity patch is ISOLATED.
So:

  1. compute the cheap per-voxel sphericity field c_s over the WHOLE volume
     (one structure_tensor_3d pass -- vectorised, no per-patch gradient loops),
  2. threshold it and take CONNECTED COMPONENTS, keeping only spatially extended
     clusters (min_size) as candidate crossing REGIONS,
  3. run expensive HOST only on patches whose centre falls in a candidate region.

The connected-component step is what a per-patch threshold cannot do: it drops
isolated false fires (raising precision) while keeping the extended true crossing
(keeping recall). We validate that on a phantom with a KNOWN crossing region,
then apply it to i74 for the HOST-saved count.

Reuses make_perp_bundles + shape_fields (crossing_trigger_test),
load_interior (selfcalib_threshold), scipy.ndimage connected components.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from crossing_trigger_test import make_perp_bundles, shape_fields

EPS = 1e-12


def candidate_regions(cs_field, thr=None, min_frac=0.002):
    """Threshold the sphericity field, then keep only connected components larger
    than min_frac of the volume. Returns (region_mask, thr, n_kept, n_raw)."""
    if thr is None:
        thr = float(threshold_otsu(cs_field))
    raw = cs_field > thr
    lbl, n_raw = ndimage.label(raw)
    if n_raw == 0:
        return np.zeros_like(raw), thr, 0, 0
    sizes = ndimage.sum(np.ones_like(lbl, float), lbl, index=np.arange(1, n_raw + 1))
    min_size = max(1, int(min_frac * cs_field.size))
    keep_ids = np.where(sizes >= min_size)[0] + 1
    region = np.isin(lbl, keep_ids)
    return region, thr, len(keep_ids), n_raw


def _iou_recall_prec(pred, truth):
    inter = float((pred & truth).sum())
    rec = inter / (float(truth.sum()) + EPS)
    prec = inter / (float(pred.sum()) + EPS)
    iou = inter / (float((pred | truth).sum()) + EPS)
    return rec, prec, iou


def phantom_validate():
    print("=" * 80)
    print("C1 PHANTOM validation: does spatial connected-components localise the")
    print("crossing region better than a per-voxel threshold alone?")
    vol, masks = make_perp_bundles(shape=(120, 120, 120), half=28,
                                   spacing=10.0, spacing2=13.0, noise_sigma=0.3)
    cs, _cp = shape_fields(vol)
    truth = masks["crossing"]
    thr = float(threshold_otsu(cs))

    # (a) per-voxel threshold only (no spatial reasoning)
    pv = cs > thr
    r_pv, p_pv, i_pv = _iou_recall_prec(pv, truth)
    # (b) coarse-to-fine: same threshold + connected components (min size)
    region, thr, n_keep, n_raw = candidate_regions(cs, thr=thr)
    r_cc, p_cc, i_cc = _iou_recall_prec(region, truth)

    print(f"  otsu thr on c_s = {thr:.3f};  raw components={n_raw}, kept (extended)={n_keep}")
    print(f"  {'method':>26} | {'recall':>6} {'precision':>9} {'IoU':>5} | {'vol flagged':>11}")
    print(f"  {'per-voxel threshold':>26} | {r_pv:6.0%} {p_pv:9.0%} {i_pv:5.2f} | "
          f"{pv.mean():11.0%}")
    print(f"  {'coarse-to-fine (CC)':>26} | {r_cc:6.0%} {p_cc:9.0%} {i_cc:5.2f} | "
          f"{region.mean():11.0%}")
    gain = p_cc - p_pv
    print(f"  -> connected-components changes precision by {gain:+.0%} at "
          f"{r_cc - r_pv:+.0%} recall (HOST runs on {region.mean():.0%} of tissue)")
    print("=" * 80)
    return vol, masks, cs, region, thr


def i74_apply(z_slice=100, half=20, step=14, save="coarse_to_fine.png"):
    import time
    from selfcalib_threshold import load_interior
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    vol = load_interior()
    Z = z_slice
    sub = vol[Z - half:Z + half + 1].astype(np.float32)
    sub = (sub - sub.min()) / (np.ptp(sub) + EPS)
    cs, _cp = shape_fields(sub)                      # per-voxel sphericity field
    region, thr, n_keep, n_raw = candidate_regions(cs, min_frac=0.003)

    # HOST-saved: fraction of a patch grid whose centres are OUTSIDE candidate
    centers = list(range(half + 8, vol.shape[1] - half - 5, step))
    grid = [(cy, cx) for cy in centers for cx in centers]
    in_region = [region[half, cy, cx] for cy, cx in grid]
    frac_fire = float(np.mean(in_region))
    print("=" * 80)
    print(f"C1 on i74 slice z={Z}: otsu c_s thr={thr:.3f}, raw comps={n_raw}, "
          f"kept={n_keep}")
    print(f"  candidate crossing tissue = {region.mean():.0%} of voxels")
    print(f"  patch grid ({len(grid)} centres): HOST fires on {frac_fire:.0%}, "
          f"skips {1 - frac_fire:.0%}")
    print("=" * 80)

    sl = sub[half]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.6))
    ax[0].imshow(sl, cmap="gray"); ax[0].set_title("i74 slice", fontsize=10)
    ax[1].imshow(sl, cmap="gray")
    ax[1].imshow(np.ma.masked_where(~region[half], region[half]), cmap="autumn",
                 alpha=0.5)
    for cy, cx in grid:
        fired = region[half, cy, cx]
        ax[1].plot(cx, cy, "s", ms=8, mfc="none",
                   mec=("#eb6834" if fired else "#2a78d6"), mew=1.8)
    ax[1].set_title(f"candidate crossing regions (spatially coherent)\n"
                    f"HOST fires on {frac_fire:.0%} of patches (orange)", fontsize=10)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("C1 coarse-to-fine: run HOST only in spatially-coherent "
                 "high-sphericity regions", fontsize=13, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save, dpi=130)
    print(f"({time.time()-t0:.0f}s)  Saved {save}")
    return region


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "phantom"):
        phantom_validate()
    if stage in ("all", "i74"):
        i74_apply()
