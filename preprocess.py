"""
preprocess.py -- image preprocessing applied right after load_volume(), before
any structure-tensor analysis. Kept as a thin, explicit stage so every downstream
folder can opt in with one call and the exact steps are visible in one place.

Current steps:
  1. clip_percentile -- clip intensities to the volume's own [p_lo, p_hi]
     percentiles (default 1-99). Removes bright/dark outliers (dust, hot voxels,
     reconstruction spikes) that would otherwise dominate the gradient magnitude
     the structure tensor is built from. Data-local: the bounds come from THIS
     volume's own histogram, no external intensity constants.
  2. normalise_01 -- rescale the clipped range onto [0, 1], so volumes of
     different dtypes / exposures arrive at the analysis on one common scale.
     This is a single global affine rescale, so it leaves every ratio-based
     structure-tensor measure (coherence, FA, sphericity, eigenvector
     directions) unchanged; only absolute intensity-unit quantities move. NB:
     anything measured in raw intensity units (e.g. a noise floor N) must be
     measured on the SAME scale as the volume it is subtracted from.

The noise-model (noise_floor) de-biasing is deliberately NOT an image step here:
the measured N corrects the STRUCTURE TENSOR (E[S]-N via
noise_floor.structure_tensor_field), i.e. at analysis time, not the image. See
the module note below / the README for why image-domain noise removal is the
wrong place for this project's correlated + anisotropic noise.
"""

import numpy as np

# clipping raw volume intensities to 1 - 99 percentile, simple and local way to remove outliers
def clip_percentile(vol, p_lo=1.0, p_hi=99.0, mask_air=False, air_below=None):
    """Clip intensities to the volume's own [p_lo, p_hi] percentiles.

    vol       : 3D array (any dtype); returned as float64.
    p_lo,p_hi : percentiles for the clip bounds, from THIS volume's histogram.
    mask_air  : if True, compute the percentiles over FOREGROUND only (voxels
                >= air_below), so a large air peak at 0 doesn't drag p_lo down to
                a no-op. air_below defaults to 5% of the volume max.
    Returns (clipped_float64, (lo_value, hi_value)).
    """
    v = np.asarray(vol, dtype=np.float64)
    if mask_air:
        thr = air_below if air_below is not None else 0.05 * float(v.max())
        sample = v[v >= thr]
        if sample.size == 0:
            sample = v.ravel()
    else:
        sample = v.ravel()
    lo, hi = np.percentile(sample, [p_lo, p_hi])
    return np.clip(v, lo, hi), (float(lo), float(hi))

def normalise_01(vol, lo=None, hi=None):
    """Rescale intensities onto [0, 1] as (v - lo) / (hi - lo).

    lo, hi : the range to map onto [0, 1]; default to the volume's own
             min/max, which for an already-clipped volume are the clip bounds.
    A degenerate range (hi <= lo, i.e. a constant volume) maps to all zeros
    rather than dividing by zero.
    Returns (normalised_float64, (lo_value, hi_value)).
    """
    v = np.asarray(vol, dtype=np.float64)
    lo = float(v.min()) if lo is None else float(lo)
    hi = float(v.max()) if hi is None else float(hi)
    if hi <= lo:
        return np.zeros_like(v), (lo, hi)
    return (v - lo) / (hi - lo), (lo, hi)

def preprocess_volume(vol, clip_pct=(1.0, 99.0), mask_air=False, normalise=True,
                      verbose=False):
    """Run the preprocessing stage. Returns the float64 preprocessed volume.

    clip_pct  : (p_lo, p_hi) for percentile clipping, or None to skip.
    mask_air  : compute clip percentiles over foreground only (see clip_percentile).
    normalise : rescale the clipped range onto [0, 1] (see normalise_01).
    """
    v = np.asarray(vol, dtype=np.float64)
    lo = hi = None
    if clip_pct is not None:
        v, (lo, hi) = clip_percentile(v, clip_pct[0], clip_pct[1], mask_air=mask_air)
        if verbose:
            print(f"[preprocess] clip {clip_pct[0]:g}-{clip_pct[1]:g}% -> [{lo:.1f}, {hi:.1f}]"
                  f"  ({(vol < lo).mean():.2%} below, {(vol > hi).mean():.2%} above clipped)")
    if normalise:
        v, (lo, hi) = normalise_01(v, lo, hi)
        if verbose:
            print(f"[preprocess] normalise [{lo:.1f}, {hi:.1f}] -> [0, 1]")
    return v


if __name__ == "__main__":
    from reconstruction import load_volume, DEFAULT_VOLUME_DIR
    if DEFAULT_VOLUME_DIR.exists():
        c0 = (768 - 200) // 2
        vol = load_volume(crop=(c0, c0 + 200), dtype=np.uint8)
        print(f"raw i74 crop {vol.shape}: min={vol.min()}, max={vol.max()}, mean={vol.mean():.1f}")
        pv = preprocess_volume(vol, clip_pct=(1, 99), mask_air=True, verbose=True)
        print(f"preprocessed: min={pv.min():.3f}, max={pv.max():.3f}, mean={pv.mean():.3f}")
    else:
        print(f"[skipped: {DEFAULT_VOLUME_DIR} not found]")
