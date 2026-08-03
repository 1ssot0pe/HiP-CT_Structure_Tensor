"""
selfcalib_threshold.py

Make the sphericity guard's threshold SELF-CALIBRATING -- derived from a
volume's own c_s distribution shape, never from ground-truth angular error 

Layer 0: does the REAL i74 chunk's per-voxel
sphericity histogram actually have a low-c_s bulk + separable high-c_s tail with
a valley, or is it one smooth unimodal smear? If no separable structure, the
whole self-calibration idea is void and we stop.

Reuses: reconstruction.load_volume (real data); crossing_shape_3d.onset_sigma
(coherence-onset selector) and its structure_tensor_3d/eig_special_3d + crop
primitives; nothing rebuilt.
"""

import sys

import numpy as np
from structure_tensor import structure_tensor_3d, eig_special_3d

from orthogonality_residual import crop_margin, _crop3
from crossing_shape_3d import onset_sigma, SIGMAS_STRESS

EPS = 1e-12

# Validated tissue-only interior box (see lindeberg_real_test.py): <1% air, so
# the histogram reflects tissue structure, not a sample/air boundary. A raw
# whole-cube crop would straddle air, and air (pure noise) is highly spherical
# -> a FALSE high-c_s tail unrelated to crossings.
INTERIOR = (slice(60, 260), slice(240, 440), slice(240, 440))


def sphericity_field(vol, sigma, rho, truncate=4.0, subtract_N=None):
    """Per-voxel sphericity c_s = 3*l3/(l1+l2+l3), cropped to the valid border.
    Same formula as crossing_shape_3d.shape_measures, but the field not the mean.

    subtract_N (3x3 axis-basis noise structure tensor from noise_model.debias):
    if given, c_s is computed from the NOISE-CORRECTED tensor E[S]-N. OFF by
    default -- on i74 the noise level can't be soundly measured (no half-maps),
    so this is only sound for data with half-maps; see noise_model/debias.py."""
    if subtract_N is not None:
        import noise_floor as _nf
        S = _nf.structure_tensor_field(vol, sigma, rho, subtract_N=subtract_N, truncate=truncate)
        l1, l2, l3, _e3, _clip = _nf.eig_fields(S, psd_clip=True, want_vec=False)
        cs = 3.0 * l3 / (l1 + l2 + l3 + EPS)
        return _crop3(cs, crop_margin(sigma, rho, truncate))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = structure_tensor_3d(vol.astype(np.float32), sigma, rho, truncate=truncate)
        val, _vec = eig_special_3d(S)
    l1, l2, l3 = val[0], val[1], val[2]
    cs = 3.0 * l3 / (l1 + l2 + l3 + EPS)
    return _crop3(cs, crop_margin(sigma, rho, truncate))


def load_interior(dtype=np.float32):
    from reconstruction import load_volume
    full = load_volume(crop=None, dtype=np.uint8)
    vol = full[INTERIOR].astype(dtype)
    del full
    return vol


def layer0(save="selfcal_layer0.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("LAYER 0 -- real i74 sphericity histogram: separable tail or smear?")
    vol = load_interior()
    print(f"  interior tissue box {vol.shape}, range [{vol.min():.0f},{vol.max():.0f}], "
          f"air-frac(<25) = {(vol < 25).mean():.3f}")

    # coherence-onset selector picks the sigma (its own procedure, no truth)
    s_on, sigs, C3, shapes = onset_sigma(vol, sigmas=SIGMAS_STRESS)
    print(f"  coherence-onset sigma = {s_on:.2f} (rho = {2*s_on:.2f})")

    cs = sphericity_field(vol, s_on, 2 * s_on).ravel()
    cs = cs[np.isfinite(cs)]
    q = np.percentile(cs, [5, 25, 50, 75, 90, 95, 99])
    print(f"  c_s percentiles  5/25/50/75/90/95/99 = "
          + " ".join(f"{v:.3f}" for v in q))
    print(f"  c_s mean={cs.mean():.3f} sd={cs.std():.3f}  "
          f"(single-fibre synthetic ~0.07-0.15; crossings high tail)")

    # crude modality read: smoothed histogram, count interior local maxima and
    # whether a valley (antimode) sits between a low bulk and a high tail.
    counts, edges = np.histogram(cs, bins=80, range=(0, 1), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    sm = np.convolve(counts, k, mode="same")
    maxima = [i for i in range(1, len(sm) - 1) if sm[i - 1] < sm[i] >= sm[i + 1]]
    minima = [i for i in range(1, len(sm) - 1) if sm[i - 1] > sm[i] <= sm[i + 1]]
    print(f"  smoothed histogram: {len(maxima)} local max at c_s="
          f"{[round(centres[i],2) for i in maxima]}, "
          f"{len(minima)} valley(s) at c_s={[round(centres[i],2) for i in minima]}")
    verdict_bimodal = len(maxima) >= 2 and len(minima) >= 1
    print(f"  READ: {'BIMODAL (bulk + tail with a valley)' if verdict_bimodal else 'looks UNIMODAL (smooth smear, no clear valley)'}")
    print("=" * 70)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].hist(cs, bins=80, range=(0, 1), color="steelblue", edgecolor="white")
    ax[0].plot(centres, sm * len(cs) * (edges[1] - edges[0]), "r-", lw=1.5, label="smoothed")
    ax[0].set_xlabel("sphericity c_s"); ax[0].set_ylabel("voxel count")
    ax[0].set_title(f"real i74 c_s @ onset sigma={s_on:.2f} (linear)")
    ax[0].legend(fontsize=8)
    ax[1].hist(cs, bins=80, range=(0, 1), color="steelblue", edgecolor="white")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("sphericity c_s"); ax[1].set_ylabel("voxel count (log)")
    ax[1].set_title("same, log count (tail visibility)")
    fig.suptitle("Layer 0: is there a separable high-c_s tail on REAL data?")
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return cs, s_on


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "0"
    if stage == "0":
        layer0()
