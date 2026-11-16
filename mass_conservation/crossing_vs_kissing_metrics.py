"""
crossing_vs_kissing_metrics.py -- cheap test of two candidate metrics for telling a
REAL crossing (two fibres physically superposed) apart from a KISSING pair (two
separate fibres locally unresolvable) at the point where ordinary HOST classification
already can't do it (the gap=4 kissing case from kissing_gap_sweep.py, valley~88%,
which HOST spuriously calls "crossing" at the pinch point).

Candidate A -- mass conservation (raw, un-normalised intensity):
    A real crossing has both tube centrelines passing exactly through the
    intersection point, so raw intensity there ~= 2x a single tube's peak.
    A kissing pair's closest-approach point has NEITHER centreline through it,
    so raw intensity there stays < 1x a single tube's peak, however small the
    gap. This uses UN-normalised amplitudes (the phantom builders normalise by
    the volume max, which invalidates any absolute-intensity comparison).

Candidate B -- scale covariance of the ODF peak separation:
    Sweep the gradient-smoothing scale (sigma_grad) and track HOST's recovered
    peak separation angle. A real crossing's separation should stay roughly
    stable as sigma grows (two real orientations coexist). A kissing spurious
    "crossing" is really one tangent direction per bundle smeared together, so
    its apparent separation should be unstable / collapse as sigma grows.
"""

import os as _sr_os, sys as _sr_sys
_SR_ROOT = _sr_os.path.dirname(_sr_os.path.dirname(__file__))
_sr_sys.path.insert(0, _SR_ROOT)  # repo root (host_fod, crossing_shape_3d, kissing_phantom, ...)
_sr_sys.path.insert(0, _sr_os.path.join(_SR_ROOT, "3_global_tractography"))  # phase1/phase2 certifiers

import numpy as np

from core.crossing_shape_3d import _tube_family, _coords as _coords_cross, make_crossing_phantom_3d
from core.kissing_phantom import _curve_pts, curved_bundle, SHAPE as KSHAPE
from core.host_fod import analyse, classify

WIDTH = 1.5   # matches crossing_shape_3d's default tube width; kissing_phantom uses 1.6


# ---------------------------------------------------------------------------
# raw (UN-normalised) volumes, so absolute intensity is physically meaningful
# ---------------------------------------------------------------------------

def raw_crossing(shape=(64, 64, 64), sep_deg=40.0, spacing=12.0, spacing2=17.0, width=WIDTH):
    half = np.deg2rad(sep_deg / 2.0)
    d1 = np.array([np.cos(half), np.sin(half), 0.0])
    d2 = np.array([np.cos(half), -np.sin(half), 0.0])
    coords = _coords_cross(shape)
    vol = (_tube_family(coords, d1, spacing, width) + _tube_family(coords, d2, spacing2, width))
    return vol.astype(np.float32)


def raw_kissing(gap=4.0, curv=0.028, radius=1.6):
    from core.crossing_shape_3d import _coords as _c
    C = _c(KSHAPE)
    ptsA, ptsB = _curve_pts(-1, gap, curv), _curve_pts(+1, gap, curv)
    A = curved_bundle(C, ptsA, radius)
    B = curved_bundle(C, ptsB, radius)
    return (A + B).astype(np.float32)


def find_intersection_voxel(shape, sep_deg=40.0, spacing=12.0, spacing2=17.0, width=WIDTH):
    """Raw crossing volume's global max voxel -- the point both tube families
    truly co-occupy (superposition)."""
    vol = raw_crossing(shape, sep_deg, spacing, spacing2, width)
    idx = np.unravel_index(np.argmax(vol), vol.shape)
    return vol, idx


def find_kissing_pinch(gap=4.0, curv=0.028, radius=1.6):
    """Centre-of-volume voxel -- the closest-approach point between the two
    curved bundles (by construction, both curves pass through y=0 at their
    smallest |x| separation there)."""
    vol = raw_kissing(gap, curv, radius)
    z, y, x = [s // 2 for s in vol.shape]
    return vol, (z, y, x)


# ---------------------------------------------------------------------------
# Candidate A -- mass conservation
# ---------------------------------------------------------------------------

def metric_a():
    print("=" * 78)

    print("CANDIDATE A -- mass conservation (raw intensity / single-fibre peak)")
    print("  single isolated tube peaks at 1.0 by construction (Gaussian, d=0)")

    vol_x, idx_x = find_intersection_voxel(shape=(64, 64, 64), sep_deg=40.0)
    ratio_x = float(vol_x[idx_x])
    print(f"  TRUE CROSSING (sep=40 deg) at intersection voxel: raw intensity = {ratio_x:.3f}"
          f"  ({'>= ~2x single-fibre peak: SUPERPOSED' if ratio_x > 1.5 else 'below 2x'})")

    vol_k, idx_k = find_kissing_pinch(gap=4.0)
    ratio_k = float(vol_k[idx_k])
    print(f"  KISSING (gap=4, fused-looking) at pinch voxel:    raw intensity = {ratio_k:.3f}"
          f"  ({'stays below 1x single-fibre peak: NOT superposed' if ratio_k < 1.0 else 'unexpectedly >= 1x'})")

    print("-" * 78)
    
    separates = ratio_x > 1.5 and ratio_k < 1.0
    print(f"  VERDICT: mass-conservation {'SEPARATES' if separates else 'does NOT separate'} "
          f"crossing from kissing at the fused point")
    
    print("=" * 78)
    return ratio_x, ratio_k, separates


# ---------------------------------------------------------------------------
# Candidate B -- scale covariance of ODF peak separation
# ---------------------------------------------------------------------------

SIGMAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
HALF_PATCH = 14


def _patch_at(vol, idx, half=HALF_PATCH):
    z, y, x = idx
    z0, y0, x0 = max(z - half, 0), max(y - half, 0), max(x - half, 0)
    return vol[z0:z + half + 1, y0:y + half + 1, x0:x + half + 1]


def metric_b():
    print("=" * 78)
    print("CANDIDATE B -- scale covariance of the ODF peak separation angle")
    print(f"  sweeping sigma_grad over {SIGMAS}")

    # true crossing: centred patch at the intersection voxel (normalise like the
    # rest of the pipeline does -- HOST classification only cares about shape)
    vol_x, idx_x = find_intersection_voxel(shape=(80, 80, 80), sep_deg=40.0, spacing=14.0, spacing2=19.0)
    vol_x = vol_x / (vol_x.max() + 1e-12)
    patch_x = _patch_at(vol_x, idx_x)

    # kissing: patch at the pinch voxel
    vol_k, idx_k = find_kissing_pinch(gap=4.0)
    vol_k = vol_k / (vol_k.max() + 1e-12)
    patch_k = _patch_at(vol_k, idx_k)

    def sweep(patch, label):
        seps = []
        for sg in SIGMAS:
            m = analyse(patch, sigma_grad=sg)
            lab = classify(m) if m else "n/a"
            sep = m["sep"] if (m and not np.isnan(m.get("sep", np.nan))) else np.nan
            seps.append(sep)
            print(f"    {label:>16} sigma={sg:4.1f} | n_peaks={m['n_peaks'] if m else '-'} "
                  f"sep={sep if not np.isnan(sep) else float('nan'):5.1f} deg | {lab}")
        return np.array(seps, dtype=float)

    seps_x = sweep(patch_x, "TRUE CROSSING")
    seps_k = sweep(patch_k, "KISSING")

    def stability(seps):
        v = seps[~np.isnan(seps)]
        return float(np.std(v)) if len(v) > 1 else float("nan"), int(len(v))

    std_x, n_x = stability(seps_x)
    std_k, n_k = stability(seps_k)
    print("-" * 78)
    print(f"  TRUE CROSSING: separation std across sigma = {std_x:.2f} deg "
          f"(over {n_x}/{len(SIGMAS)} scales with 2 peaks)")
    print(f"  KISSING:       separation std across sigma = {std_k:.2f} deg "
          f"(over {n_k}/{len(SIGMAS)} scales with 2 peaks)")
    # A crossing that reliably shows 2 peaks at MOST scales with a stable angle,
    # vs kissing showing 2 peaks at FEW scales and/or an unstable angle, is the
    # separating signature.
    separates = (n_x >= n_k) and (np.isnan(std_k) or std_x <= std_k) and n_x >= len(SIGMAS) - 2
    print(f"  VERDICT: scale-covariance {'SEPARATES' if separates else 'does NOT cleanly separate'} "
          f"crossing from kissing here")
    print("=" * 78)
    return seps_x, seps_k, separates


if __name__ == "__main__":
    metric_a()
    metric_b()
