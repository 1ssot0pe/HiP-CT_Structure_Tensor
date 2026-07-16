"""
resolution_sweep.py

Completes the reconstruction pipeline: turns a LIC reconstruction into a single
SHARPNESS NUMBER, then sweeps (sigma, rho) to pick the pair that gives the
sharpest -- i.e. best-resolved -- reconstruction. Ground-truth-free.

Pipeline position:
    input -> structure tensor(sigma, rho) -> principal eigenvector field
          -> LIC reconstruction (reconstruction.py)
          -> [THIS: FSC resolution score + (sigma, rho) sweep]

Why FSC (Fourier Shell Correlation)
-----------------------------------
We need one number per (sigma, rho) so the sweep can compare them. "Sharpness"
is really "up to how fine a spatial frequency is the texture REAL and not noise".
FSC measures exactly that, ground-truth-free, by a two-noise trick:

    1. Reconstruct the SAME direction field TWICE with two independent noise
       seeds (method="lic", seed=1 and seed=2).
    2. Both reconstructions carry the SAME anisotropic structure (imposed by the
       shared directions) but INDEPENDENT noise.
    3. Correlate them in Fourier shells of increasing frequency. Where the
       directions are coherent the imposed structure survives to high frequency
       and the two agree; where the directions are junk the two are just
       independent noise and decorrelate early.
    4. The frequency at which FSC falls through a threshold (0.143, the cryo-EM
       two-half-map standard) IS the resolution. Higher cutoff = finer = better.

So the score to MAXIMISE is the FSC cutoff frequency.

The honest caveat (same trap that sank the orthogonality residual)
------------------------------------------------------------------
"Sharper = better" is NOT automatically true. A large rho makes the direction
field very smooth, which makes both noise reconstructions smear along the same
near-parallel streamlines -> they correlate to high frequency -> high score,
even though the directions no longer follow the real fibres. So this metric can
ALSO drift toward over-smoothing. That is why this module ships with a phantom
validation (validate()) that checks the resolution argmax against the supervised
angular-error argmin, exactly as orthogonality_residual.py does -- do not trust
the sweep on real data until the phantom says the two argmins coincide.

Everything reuses reconstruction.py's primitives; the sweep/validation layout
mirrors orthogonality_residual.py so the two metrics are directly comparable.
"""

import numpy as np

from reconstruction import (
    reconstruct_from_eigenvector,
    _demo_principal_direction,
    suggested_margin,
    crop_region,
    load_volume,
    DEFAULT_VOLUME_DIR,
    iter_progress,
    grid_cells,
)


# ---------------------------------------------------------------------------
# Fourier Shell Correlation
# ---------------------------------------------------------------------------

def _shell_index(shape, nbins):
    """Radial-frequency shell index (0..nbins-1) per voxel, and shell centres.

    Frequencies are in cycles/voxel via fftfreq; we only bin 0..0.5 (Nyquist).
    Voxels beyond 0.5 (cube corners) fall outside and are dropped.
    """
    ks = [np.fft.fftfreq(n) for n in shape]
    KZ, KY, KX = np.meshgrid(*ks, indexing="ij")
    kr = np.sqrt(KZ ** 2 + KY ** 2 + KX ** 2).ravel()
    edges = np.linspace(0.0, 0.5, nbins + 1)
    idx = np.digitize(kr, edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    return idx, centres


def fsc(vol_a, vol_b, nbins=50, eps=1e-12):
    """Fourier Shell Correlation curve between two equal-shape volumes.

    Returns (freqs, curve): freqs in cycles/voxel (shell centres), curve in
    [-1, 1] (per shell, real part of the normalised cross-power).
    """
    a = np.asarray(vol_a, float)
    b = np.asarray(vol_b, float)
    if a.shape != b.shape:
        raise ValueError("fsc: volumes must have the same shape")
    Fa = np.fft.fftn(a).ravel()
    Fb = np.fft.fftn(b).ravel()
    idx, freqs = _shell_index(a.shape, nbins)

    cross = np.real(Fa * np.conj(Fb))
    pa = np.abs(Fa) ** 2
    pb = np.abs(Fb) ** 2

    keep = (idx >= 0) & (idx < nbins)
    idx = idx[keep]
    num = np.bincount(idx, weights=cross[keep], minlength=nbins)
    da = np.bincount(idx, weights=pa[keep], minlength=nbins)
    db = np.bincount(idx, weights=pb[keep], minlength=nbins)

    curve = num / (np.sqrt(da * db) + eps)
    return freqs, curve


def resolution_frequency(freqs, curve, threshold=0.143):
    """First frequency (cycles/voxel) where the FSC curve crosses `threshold`
    downward, linearly interpolated. Higher = finer resolution = better.

    If the curve never drops below the threshold, resolution is Nyquist-limited
    -> return the highest shell frequency.
    """
    below = np.where(curve < threshold)[0]
    if below.size == 0:
        return float(freqs[-1])
    i = int(below[0])
    if i == 0:
        return float(freqs[0])
    f0, f1 = freqs[i - 1], freqs[i]
    c0, c1 = curve[i - 1], curve[i]
    t = (threshold - c0) / ((c1 - c0) if (c1 - c0) != 0 else 1e-12)
    return float(f0 + t * (f1 - f0))


# ---------------------------------------------------------------------------
# The score (scalar) for one (sigma, rho)
# ---------------------------------------------------------------------------

def reconstruction_resolution(volume, sigma, rho,
                              length=8, step_size=0.5, kernel="hanning",
                              nbins=50, threshold=0.143):
    """FSC cutoff frequency of the LIC reconstruction at (sigma, rho).

    Two independent-noise reconstructions of the SAME eigenvector field, cropped
    identically, correlated in Fourier shells. Returns the cutoff frequency
    (cycles/voxel); HIGHER = finer resolution = better.
    """
    field = _demo_principal_direction(volume, sigma, rho)
    a = reconstruct_from_eigenvector(field, length=length, step_size=step_size,
                                     kernel=kernel, method="lic", seed=1)
    b = reconstruct_from_eigenvector(field, length=length, step_size=step_size,
                                     kernel=kernel, method="lic", seed=2)
    m = suggested_margin(sigma, rho, length, step_size)
    a = crop_region(a, m)
    b = crop_region(b, m)
    freqs, curve = fsc(a, b, nbins=nbins)
    return resolution_frequency(freqs, curve, threshold)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_resolution(volume, sigmas, rhos,
                     length=8, step_size=0.5, kernel="hanning",
                     nbins=50, threshold=0.143, verbose=True):
    """Grid-evaluate the FSC resolution over sigmas x rhos.

    Returns
    -------
    F : (len(sigmas), len(rhos)) array of cutoff frequencies (higher = better)
    best : dict(sigma, rho, i, j, value)  -- the ARGMAX (finest resolution)
    """
    sigmas = np.asarray(sigmas, float)
    rhos = np.asarray(rhos, float)
    F = np.full((sigmas.size, rhos.size), np.nan)
    cells = grid_cells(sigmas.size, rhos.size)
    for i, j in iter_progress(cells, desc="resolution sweep", enable=verbose):
        F[i, j] = reconstruction_resolution(volume, sigmas[i], rhos[j], length,
                                            step_size, kernel, nbins, threshold)
    i, j = np.unravel_index(np.nanargmax(F), F.shape)   # MAX: finest resolution
    best = dict(sigma=float(sigmas[i]), rho=float(rhos[j]), i=int(i), j=int(j),
                value=float(F[i, j]))
    if verbose:
        print(f"[resolution] argmax cutoff = {best['value']:.4f} cyc/vox "
              f"at sigma={best['sigma']:.3g}, rho={best['rho']:.3g} "
              f"(grid cell [{i},{j}])")
    return F, best


# ---------------------------------------------------------------------------
# Supervised cross-check (phantom only) -- angular error vs known ground truth
# ---------------------------------------------------------------------------

def _mean_angular_error(field, truth, margin):
    """Mean angular error (deg) between the demo eigenvector field and truth.

    Both are (Nx,Ny,Nz,3) in the SAME [axis0,axis1,axis2] index basis
    (_demo_principal_direction builds S from np.gradient in that order, and
    make_fanning_phantom's truth stores [axis0,axis1,axis2] components), so the
    dot product is a direct, reversal-free comparison. Sign-agnostic (v ~ -v).
    """
    f = crop_region(field, margin)
    t = crop_region(truth, margin)
    valid = np.isfinite(f).all(-1) & np.isfinite(t).all(-1)
    fn = f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-12)
    cos = np.clip(np.abs(np.sum(fn * t, axis=-1)), 0.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return float(np.mean(ang[valid])) if valid.any() else float("nan")


def sweep_angular_error(volume, truth, sigmas, rhos,
                        length=8, step_size=0.5, verbose=True):
    """Grid-evaluate supervised mean angular error on the SAME grid / same field
    as sweep_resolution (so the two surfaces are directly comparable)."""
    sigmas = np.asarray(sigmas, float)
    rhos = np.asarray(rhos, float)
    A = np.full((sigmas.size, rhos.size), np.nan)
    cells = grid_cells(sigmas.size, rhos.size)
    for i, j in iter_progress(cells, desc="angular    sweep", enable=verbose):
        s, r = sigmas[i], rhos[j]
        field = _demo_principal_direction(volume, s, r)
        A[i, j] = _mean_angular_error(field, truth,
                                      suggested_margin(s, r, length, step_size))
    i, j = np.unravel_index(np.nanargmin(A), A.shape)
    best = dict(sigma=float(sigmas[i]), rho=float(rhos[j]), i=int(i), j=int(j),
                value=float(A[i, j]))
    if verbose:
        print(f"[angular   ] argmin err = {best['value']:.3f} deg "
              f"at sigma={best['sigma']:.3g}, rho={best['rho']:.3g} "
              f"(grid cell [{i},{j}])")
    return A, best


# ---------------------------------------------------------------------------
# Validation: does the resolution ARGMAX land on the supervised ARGMIN?
# ---------------------------------------------------------------------------

def validate(shape=(96, 96, 96),
             sigmas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
             rhos=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
             length=8, step_size=0.5,
             save_path="resolution_validation.png", show=False, **phantom_kw):
    """Fanning-phantom validation of the FSC resolution sweep.

    Sweeps FSC resolution and supervised angular error on the same grid, prints
    both extrema, their separation and surface correlation, and a verdict, then
    writes a side-by-side heatmap. Because resolution is HIGHER=better and error
    is LOWER=better, a WORKING metric shows the two ANTI-correlated (corr < 0)
    and their argmax/argmin in the same grid cell.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from orthogonality_residual import make_fanning_phantom

    sigmas = np.asarray(sigmas, float)
    rhos = np.asarray(rhos, float)
    vol, truth = make_fanning_phantom(shape=shape, **phantom_kw)
    print(f"fanning phantom {vol.shape}  "
          f"truth-defined voxels={int(np.isfinite(truth).all(-1).sum())}")

    F, bestF = sweep_resolution(vol, sigmas, rhos, length, step_size)
    A, bestA = sweep_angular_error(vol, truth, sigmas, rhos, length, step_size)

    d_cells = int(max(abs(bestF["i"] - bestA["i"]), abs(bestF["j"] - bestA["j"])))
    ok = np.isfinite(F) & np.isfinite(A)
    corr = float(np.corrcoef(F[ok], A[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    # Working = argmax(resolution) coincides with argmin(error) AND the surfaces
    # are anti-correlated (finer resolution goes with lower error).
    trustworthy = (d_cells <= 1) and (corr < -0.3)

    print("-" * 70)
    print(f"FSC resolution argmax -> sigma={bestF['sigma']:.3g}, rho={bestF['rho']:.3g}")
    print(f"supervised ang argmin -> sigma={bestA['sigma']:.3g}, rho={bestA['rho']:.3g}")
    print(f"separation: {d_cells} grid cell(s)")
    print(f"surface correlation corr(resolution, angular_error) = {corr:+.3f} "
          f"(want strongly NEGATIVE)")
    if trustworthy:
        print("VERDICT: FSC resolution TRACKS angular error (argmax==argmin + "
              "anti-correlation). Sweep looks trustworthy on this phantom.")
    else:
        print("VERDICT: FSC resolution does NOT track angular error -> NOT "
              "trustworthy as-is. Likely the over-smoothing drift noted in the "
              "module header (large rho inflates FSC). Do NOT use the sweep's "
              "argmax on real data until this passes.")
    print("-" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    extent = [rhos[0], rhos[-1], sigmas[-1], sigmas[0]]
    for ax, M, best, title, cbl in (
        (axes[0], F, bestF, "FSC resolution (cyc/vox, higher=better)", "cutoff freq"),
        (axes[1], A, bestA, "Supervised mean angular error (deg)", "degrees (lower=better)"),
    ):
        im = ax.imshow(M, origin="upper", aspect="auto", extent=extent, cmap="viridis")
        ax.set_xlabel("rho"); ax.set_ylabel("sigma"); ax.set_title(title)
        ax.plot(best["rho"], best["sigma"], "r*", ms=16, mec="white",
                label=f"({best['sigma']:.2g}, {best['rho']:.2g})")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, label=cbl, fraction=0.046, pad=0.04)
    fig.suptitle(f"FSC reconstruction resolution vs angular error  corr={corr:+.2f}  "
                 + ("TRACKS" if trustworthy else "does NOT track"))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"heatmaps written to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return dict(F=F, A=A, bestF=bestF, bestA=bestA, sigmas=sigmas, rhos=rhos,
                d_cells=d_cells, corr=corr, trustworthy=trustworthy)


if __name__ == "__main__":
    # Fair test first: does the FSC resolution sweep pick the right (sigma, rho)?
    validate()

    # Real HiP-CT volume -- runs only if the TIFF folder is present. No ground
    # truth here, so this is the resolution surface + its argmax alone.
    if DEFAULT_VOLUME_DIR.exists():
        print("\n=== real volume (i74 sensitivity chunk): resolution sweep ===")
        volume = load_volume(crop=(200, 328))   # 128^3; widen for the full cube
        F, best = sweep_resolution(
            volume,
            sigmas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
            rhos=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
        )
    else:
        print(f"\n[real volume skipped: {DEFAULT_VOLUME_DIR} not found]")
