"""
debias.py -- the noise_model's ANALYSIS-TIME de-biasing layer.

The measured noise adds a bias N to the structure tensor: E[S_observed] = S_true + N.
noise_floor.structure_tensor_field can return the corrected tensor E[S] - N. This
module (a) provides the measured N for the pipeline and (b) exposes the de-biased
eigenvalue quantities (c_s, FA, aniso) that the noise most distorts.

Why de-bias eigenVALUES (not the direction): c_s/FA/aniso are functions of the
three eigenvalues only, so they are invariant to component ordering -- we can take
them straight from noise_floor's own (l1,l2,l3) with no convention-matching risk.
The fibre DIRECTION would need careful basis matching and is left untouched.

Honest caveats (carried from the noise-floor work):
  * i74 has NO half-maps, so N is measured from LUNG half-maps n=(A-B)/sqrt2 -- a
    cross-organ proxy. Its isotropic spectral shape transfers; its anisotropy may
    not (see BAYES_TRACTOGRAPHY.md). So this de-bias is only as good as that proxy.
  * On i74 a white-noise floor barely moved c_s; the value here is that the lung N
    is CORRELATED + ANISOTROPIC, so it can correct DIRECTIONAL noise bias a white
    floor cannot. The __main__ demo quantifies how much it actually changes c_s.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root
import noise_floor as nf

EPS = 1e-30
_N_CACHE = {}


def _repo_root():
    return os.path.dirname(os.path.dirname(__file__))


def _hf_noise_power(vol, band=(0.30, 0.48)):
    """Mean radial power in a high-frequency (noise-dominated) band, cycles/voxel.
    On a signal volume this is noise + unresolved structure (an UPPER bound); on
    pure noise it is the noise power."""
    P, wmag, _ = nf.power_spectrum_3d(np.asarray(vol, np.float64))
    k = (wmag / (2.0 * np.pi)).ravel()
    m = (k >= band[0]) & (k <= band[1])
    return float(P.ravel()[m].mean()) if m.any() else np.nan


def measured_N(sigma, slab_dir=None, truncate=4.0, crop=(81, 256, 256), scale_to_vol=None):
    """Full-recon-equivalent noise structure tensor N (3x3, axis basis) at scale
    `sigma`, from the lung half-maps n=(A-B)/sqrt2. N measured on n is the HALF-MAP
    floor; the full-recon-equivalent is N/2 (each half carries ~2x the variance).

    scale_to_vol: if given (the target volume, e.g. i74), rescale N's AMPLITUDE to
    that volume's own noise level -- since the lung proxy only supplies the noise
    SHAPE (anisotropy), not its level, and i74 has no half-maps to measure the
    level directly. The scale is the ratio of high-frequency spectral power
    (target / lung n); the noise structure tensor is linear in the noise PSD, so a
    PSD scale factor scales N by the same factor. The target's HF tail includes
    unresolved structure, so this is an UPPER bound (the de-bias errs strong).
    Cached per (sigma, scale key). Raises FileNotFoundError if lung_A/B absent."""
    key = (round(float(sigma), 4), None if scale_to_vol is None else round(_hf_noise_power(scale_to_vol), 12))
    if key in _N_CACHE:
        return _N_CACHE[key]
    root = slab_dir or _repo_root()
    A = np.load(os.path.join(root, "lung_A.npy")).astype(np.float64)
    B = np.load(os.path.join(root, "lung_B.npy")).astype(np.float64)
    n = (A - B) / np.sqrt(2.0)
    if crop is not None:
        cz, cy, cx = [min(c, s) for c, s in zip(crop, n.shape)]
        z0 = (n.shape[0] - cz) // 2; y0 = (n.shape[1] - cy) // 2; x0 = (n.shape[2] - cx) // 2
        n = n[z0:z0 + cz, y0:y0 + cy, x0:x0 + cx]
    N = nf.N_matrix(n, sigma, truncate=truncate) / 2.0
    if scale_to_vol is not None:
        ratio = _hf_noise_power(scale_to_vol) / (_hf_noise_power(n) + 1e-300)
        N = N * ratio
    _N_CACHE[key] = N
    return N


def debiased_eigfields(vol, sigma, rho, N, truncate=4.0, psd_clip=True):
    """Per-voxel eigenvalues (l1>=l2>=l3) of the NOISE-CORRECTED tensor E[S]-N.
    Returns (l1, l2, l3, clip_frac)."""
    S = nf.structure_tensor_field(np.asarray(vol, np.float64), sigma, rho,
                                  subtract_N=N, truncate=truncate)
    l1, l2, l3, _e3, clip = nf.eig_fields(S, psd_clip=psd_clip, want_vec=False)
    return l1, l2, l3, clip


def debiased_shape_fields(vol, sigma, rho, N, truncate=4.0):
    """Noise-corrected c_s, FA and aniso fields (all from the de-biased
    eigenvalues, so no component-ordering assumptions). Returns dict of arrays."""
    l1, l2, l3, clip = debiased_eigfields(vol, sigma, rho, N, truncate)
    ssum = l1 + l2 + l3 + EPS
    c_s = 3.0 * l3 / ssum
    fa = np.sqrt(0.5) * np.sqrt((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) \
        / (np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2) + EPS)
    aniso = (l1 - l3) / (l1 + EPS)
    return dict(c_s=c_s, FA=fa, aniso=aniso, l1=l1, l2=l2, l3=l3, clip_frac=clip)


if __name__ == "__main__":
    # Demo: how much does subtracting the measured (lung) N change i74's c_s?
    from reconstruction import load_volume, DEFAULT_VOLUME_DIR
    SIGMA, RHO = 4.0, 8.0
    root = _repo_root()
    if not (DEFAULT_VOLUME_DIR.exists()
            and os.path.exists(os.path.join(root, "lung_A.npy"))):
        print("[skipped: need the i74 volume + lung_A/B.npy]"); sys.exit(0)

    c0 = (768 - 200) // 2
    vol = load_volume(crop=(c0, c0 + 200), dtype=np.float64)
    print(f"i74 crop {vol.shape}; measuring N at sigma={SIGMA} from lung half-maps ...")
    N = measured_N(SIGMA, scale_to_vol=vol)   # amplitude-matched to i74's own noise level
    w = np.linalg.eigvalsh(N)
    print(f"  N (amplitude-matched to i74) eigenvalues={np.array2string(w, precision=1)}  "
          f"anisotropy l1/l3={w[-1] / (w[0] + 1e-30):.2f}")

    # raw c_s (no correction) vs de-biased c_s
    raw = nf.structure_tensor_field(vol, SIGMA, RHO, subtract_N=None)
    rl1, rl2, rl3, _, _ = nf.eig_fields(raw, psd_clip=True, want_vec=False)
    cs_raw = (3.0 * rl3 / (rl1 + rl2 + rl3 + EPS)).ravel()
    deb = debiased_shape_fields(vol, SIGMA, RHO, N)
    cs_deb = deb["c_s"].ravel()
    print(f"  c_s median: raw={np.median(cs_raw):.3f} -> de-biased={np.median(cs_deb):.3f} "
          f"(shift {np.median(cs_deb) - np.median(cs_raw):+.3f})")
    print(f"  c_s mean  : raw={cs_raw.mean():.3f} -> de-biased={cs_deb.mean():.3f}")
    print(f"  PSD clip fraction after subtract-N = {deb['clip_frac']:.1%}")
    print("\n  FINDING: subtract-N is not soundly calibratable on i74. With no i74"
          "\n  half-maps its noise LEVEL is unmeasurable; the raw lung N is on the"
          "\n  wrong intensity scale (no-op), and matching the amplitude to i74's own"
          "\n  high-freq tail over-counts (tail = noise + unresolved structure) ->"
          "\n  massive over-subtraction (see clip fraction). The mechanism is correct"
          "\n  and ready for any dataset WITH half-maps; on brain i74 it needs half-map"
          "\n  acquisition. Left OFF by default (subtract_N=None) in the pipeline.")
