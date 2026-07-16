"""
quality_metric.py -- a ground-truth-free output-quality metric that (unlike FSC)
does NOT drift toward over-smoothing, so it can serve as an INDEPENDENT certifier
of the (sigma, rho) pick.

Why FSC failed (resolution_sweep trust gate): FSC of two noise-LIC reconstructions
scores FIELD COHERENCE. Large rho -> smooth field -> both noise recons smear along
near-identical streamlines -> correlate to high frequency -> high score even when
the directions are WRONG. Any metric that rewards smoothness over-smooths.

The fix -- CROSS-SCALE ORTHOGONALITY RESIDUAL. Score e3 against FINE-SCALE real
structure, not against the smoothed field's own coherence:

    R(sigma, rho) = < (grad_fine . e3)^2 >  /  < |grad_fine|^2 >

  * e3 = fibre_direction(vol, sigma, rho)          (the estimate under test)
  * grad_fine = gradient at a FIXED small sigma_ref (carries the real local
    orientation, independent of the swept smoothing)

A correct e3 is perpendicular to the local structure, so grad_fine . e3 ~ 0 and R
is LOW. Over-smoothing bends e3 away from the fine structure at curves/crossings
-> grad_fine . e3 grows -> R RISES (penalised). Too-small sigma -> e3 noisy ->
also misaligned -> R rises. So R has an interior MINIMUM at the right scale, and
LOWER = better (same polarity as supervised angular error), which is exactly what
the trust gate needs.

validate() reuses the fanning phantom + supervised angular sweep from
resolution_sweep, and PASSES iff argmin(R) coincides with argmin(angular error)
and the two surfaces are POSITIVELY correlated (both lower=better).
"""

import numpy as np

from noise_floor import _derivs, crop_border
from orthogonality_residual import fibre_direction, crop_margin

EPS = 1e-12
SIGMA_REF = 1.0          # fine-scale gradient reference (fixed, near the floor)


def cross_scale_residual(vol, sigma, rho, sigma_ref=SIGMA_REF, truncate=4.0,
                         mag_frac=0.1):
    """R(sigma,rho) = <(grad_fine . e3)^2> / <|grad_fine|^2> over high-gradient,
    interior voxels. LOWER = e3 better explains the fine structure = better."""
    vol = np.asarray(vol, np.float64)
    d0, d1, d2 = _derivs(vol, sigma_ref, truncate)          # fine gradient (axis basis)
    e3 = fibre_direction(vol, sigma, rho, truncate)          # (3,...) axis basis
    e3 = np.where(np.isfinite(e3), e3, 0.0)
    dot = d0 * e3[0] + d1 * e3[1] + d2 * e3[2]
    mag2 = d0 * d0 + d1 * d1 + d2 * d2
    m = crop_margin(max(sigma, sigma_ref), rho, truncate)
    dot = crop_border(dot, m); mag2 = crop_border(mag2, m)
    keep = mag2 > mag_frac * (mag2.max() + EPS)              # ignore flat voxels
    if not np.any(keep):
        return np.nan
    return float(np.sum(dot[keep] ** 2) / (np.sum(mag2[keep]) + EPS))


def tensor_reproducibility(vol, sigma, rho, noise_rel=0.2, truncate=4.0, nbins=40):
    """Frequency-resolved reproducibility of the ORIENTATION under noise
    resampling. Perturb vol with independent noise twice, build the sign-free
    structure-tensor (e3 outer-product) field for each, and FSC the two 6-channel
    fields. cutoff freq = the finest scale at which the orientation field is
    REPRODUCIBLE. Over-smoothing -> no reproducible high-freq detail -> low
    cutoff; noise -> low-freq disagreement -> low cutoff. HIGHER = better."""
    from resolution_sweep import fsc, resolution_frequency
    vol = np.asarray(vol, np.float64)
    sd = noise_rel * vol.std()
    m = crop_margin(sigma, rho, truncate)

    def tensor6(seed):
        v = vol + np.random.default_rng(seed).standard_normal(vol.shape) * sd
        e3 = fibre_direction(v, sigma, rho, truncate)
        e3 = np.where(np.isfinite(e3), e3, 0.0)
        # sign-free outer product components [00,11,22,01,02,12]
        comps = [e3[0]*e3[0], e3[1]*e3[1], e3[2]*e3[2],
                 e3[0]*e3[1], e3[0]*e3[2], e3[1]*e3[2]]
        return [crop_border(c, m) for c in comps]

    Ta, Tb = tensor6(1), tensor6(2)
    # FSC summed over the 6 channels (single cutoff for the whole tensor field)
    num = da = db = None
    from resolution_sweep import _shell_index
    for ca, cb in zip(Ta, Tb):
        Fa = np.fft.fftn(ca).ravel(); Fb = np.fft.fftn(cb).ravel()
        idx, freqs = _shell_index(ca.shape, nbins)
        keep = (idx >= 0) & (idx < nbins); idx = idx[keep]
        n = np.bincount(idx, weights=np.real(Fa*np.conj(Fb))[keep], minlength=nbins)
        a = np.bincount(idx, weights=(np.abs(Fa)**2)[keep], minlength=nbins)
        b = np.bincount(idx, weights=(np.abs(Fb)**2)[keep], minlength=nbins)
        num = n if num is None else num + n
        da = a if da is None else da + a
        db = b if db is None else db + b
    curve = num / (np.sqrt(da*db) + EPS)
    return resolution_frequency(freqs, curve, threshold=0.143)


def sweep_reproducibility(vol, sigmas, rhos, verbose=True):
    from reconstruction import iter_progress, grid_cells
    sigmas = np.asarray(sigmas, float); rhos = np.asarray(rhos, float)
    Q = np.full((sigmas.size, rhos.size), np.nan)
    for i, j in iter_progress(grid_cells(sigmas.size, rhos.size),
                              desc="reproducibility", enable=verbose):
        Q[i, j] = tensor_reproducibility(vol, sigmas[i], rhos[j])
    i, j = np.unravel_index(np.nanargmax(Q), Q.shape)          # MAX: finest reproducible
    best = dict(sigma=float(sigmas[i]), rho=float(rhos[j]), i=int(i), j=int(j),
                value=float(Q[i, j]))
    if verbose:
        print(f"[repro] argmax cutoff = {best['value']:.4f} at "
              f"sigma={best['sigma']:.3g}, rho={best['rho']:.3g}")
    return Q, best


def sweep_residual(vol, sigmas, rhos, sigma_ref=SIGMA_REF, verbose=True):
    from reconstruction import iter_progress, grid_cells
    sigmas = np.asarray(sigmas, float); rhos = np.asarray(rhos, float)
    R = np.full((sigmas.size, rhos.size), np.nan)
    for i, j in iter_progress(grid_cells(sigmas.size, rhos.size),
                              desc="xscale residual", enable=verbose):
        R[i, j] = cross_scale_residual(vol, sigmas[i], rhos[j], sigma_ref)
    i, j = np.unravel_index(np.nanargmin(R), R.shape)         # MIN: best fidelity
    best = dict(sigma=float(sigmas[i]), rho=float(rhos[j]), i=int(i), j=int(j),
                value=float(R[i, j]))
    if verbose:
        print(f"[xscale] argmin residual = {best['value']:.4f} at "
              f"sigma={best['sigma']:.3g}, rho={best['rho']:.3g}")
    return R, best


# ---------------------------------------------------------------------------
# trust gate: does R track supervised angular error on the fanning phantom?
# ---------------------------------------------------------------------------

def validate(sigmas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
             rhos=(1.0, 2.0, 3.0, 4.0), save="quality_metric_validate.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from orthogonality_residual import make_fanning_phantom
    from resolution_sweep import sweep_angular_error

    sigmas = np.asarray(sigmas, float); rhos = np.asarray(rhos, float)
    vol, truth = make_fanning_phantom(shape=(96, 96, 96))
    print(f"fanning phantom {vol.shape}")
    R, bestR = sweep_residual(vol, sigmas, rhos)
    A, bestA = sweep_angular_error(vol, truth, sigmas, rhos)

    d = int(max(abs(bestR["i"] - bestA["i"]), abs(bestR["j"] - bestA["j"])))
    ok = np.isfinite(R) & np.isfinite(A)
    corr = float(np.corrcoef(R[ok], A[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    # both LOWER=better -> want argmins to coincide AND POSITIVE correlation
    passed = (d <= 1) and (corr > 0.3)
    print("-" * 70)
    print(f"cross-scale residual argmin -> sigma={bestR['sigma']:.3g}, rho={bestR['rho']:.3g}")
    print(f"supervised ang     argmin -> sigma={bestA['sigma']:.3g}, rho={bestA['rho']:.3g}")
    print(f"separation: {d} cell(s);  corr(R, ang_err) = {corr:+.3f} (want strongly POSITIVE)")
    print(f"TRUST GATE: {'PASS' if passed else 'FAIL'} -- cross-scale residual "
          f"{'TRACKS' if passed else 'does NOT track'} truth")
    print("-" * 70)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    extent = [rhos[0], rhos[-1], sigmas[-1], sigmas[0]]
    for a, M, best, ttl in ((ax[0], R, bestR, "cross-scale residual (lower=better)"),
                            (ax[1], A, bestA, "supervised angular error (deg)")):
        im = a.imshow(M, origin="upper", aspect="auto", extent=extent, cmap="viridis")
        a.plot(best["rho"], best["sigma"], "r*", ms=16, mec="white")
        a.set_xlabel("rho"); a.set_ylabel("sigma"); a.set_title(ttl)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(f"Cross-scale residual vs truth  corr={corr:+.2f}  "
                 + ("TRACKS" if passed else "does NOT track"))
    fig.tight_layout(); fig.savefig(save, dpi=120); plt.close(fig)
    print(f"Saved {save}")
    return dict(R=R, A=A, bestR=bestR, bestA=bestA, corr=corr, passed=passed, d=d)


if __name__ == "__main__":
    validate()
