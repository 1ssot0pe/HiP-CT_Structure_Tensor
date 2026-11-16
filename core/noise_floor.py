"""
noise_floor.py -- noise-floor characterisation of the structure tensor.

Tests the falsifiable claim that the structure tensor on real i74 carries an
additive, (near)isotropic noise term N(sigma) that does NOT average away with
rho, and that a large N drives the Westin sphericity c_s -> 1 for every voxel --
making "pervasively high c_s / 80% multi-orientation" a NOISE-FLOOR ARTIFACT
rather than a tissue property.

Theory (implemented, not re-derived -- see the brief):
    S(sigma,rho) = G_rho * (grad L_sigma grad L_sigma^T),   grad L = grad(G_sigma * I)
    E[S] = S_sig + N(sigma),   N(sigma) = E[ grad(G_sigma * n) grad(G_sigma * n)^T ]
  (i)  N is a MEAN of a squared quantity -> rho reduces its fluctuation, never its
       expectation. No rho removes N. (T2 proves this empirically.)
  (ii) an isotropic floor N ~ eta*I sends c_s = 3(l3+eta)/(trS_sig+3eta) -> 1.

  White noise, 3D, closed form (Parseval on the Gaussian derivative):
       tr N(sigma) = 3 sigma_n^2 / (16 pi^{3/2} sigma^5)  ~ 0.033678 sigma_n^2 sigma^-5
  => log-log slope exactly -5, N isotropic (all eigenvalues equal). Hard checks.

Conventions: sigma, rho in VOXEL units; float64 everywhere in the floor
arithmetic (at sigma=8 the floor is ~1e-6 of the sigma=1 value; float32 destroys
the slope). Component order matches the `structure_tensor` package so a measured
N can be subtracted from its S: S = [d2 d2, d1 d1, d0 d0, d2 d1, d2 d0, d1 d0]
with dk = d/d(axis k).
"""

import numpy as np
from scipy.ndimage import gaussian_filter

# analytic white-noise trace constant: tr N = C_WHITE * sigma_n^2 * sigma^-5
C_WHITE = 3.0 / (16.0 * np.pi ** 1.5)          # ~0.0336785
C_WHITE_EIG = 1.0 / (16.0 * np.pi ** 1.5)      # per equal eigenvalue

_TRUNC = 4.0


# ---------------------------------------------------------------------------
# raw (pre-rho) structure tensor and the N estimate
# ---------------------------------------------------------------------------

def _derivs(I, sigma, truncate=_TRUNC):
    """Gaussian first derivatives d0,d1,d2 = d/d(axis k) of (G_sigma * I). f64."""
    I = np.asarray(I, dtype=np.float64)
    d0 = gaussian_filter(I, sigma, order=(1, 0, 0), mode="reflect", truncate=truncate)
    d1 = gaussian_filter(I, sigma, order=(0, 1, 0), mode="reflect", truncate=truncate)
    d2 = gaussian_filter(I, sigma, order=(0, 0, 1), mode="reflect", truncate=truncate)
    return d0, d1, d2


def structure_tensor_raw(I, sigma, truncate=_TRUNC):
    """Per-voxel outer product grad L grad L^T, NO rho smoothing. Returns the 6
    unique fields in `structure_tensor` package order [xx,yy,zz,xy,xz,yz] with
    x=axis2, y=axis1, z=axis0 (i.e. [d2^2, d1^2, d0^2, d2 d1, d2 d0, d1 d0])."""
    d0, d1, d2 = _derivs(I, sigma, truncate)
    return np.stack([d2 * d2, d1 * d1, d0 * d0, d2 * d1, d2 * d0, d1 * d0])


def crop_border(a, m):
    """Crop m voxels from every spatial face (last 3 axes)."""
    m = int(m)
    if m <= 0:
        return a
    sl = (Ellipsis, slice(m, -m), slice(m, -m), slice(m, -m))
    return a[sl]


def N_matrix(I, sigma, truncate=_TRUNC, crop=None):
    """3x3 N(sigma) = mean over interior voxels of the raw outer product.
    Interior = crop ceil(truncate*sigma) from every face (boundary contamination
    corrupts the floor at large sigma). Returns N in AXIS-INDEX basis N[i,j] with
    i,j in {0,1,2} = axis0,axis1,axis2."""
    J = structure_tensor_raw(I, sigma, truncate)          # (6, ...)
    m = int(np.ceil(truncate * sigma)) if crop is None else int(crop)
    xx, yy, zz, xy, xz, yz = (crop_border(J[c], m).mean() for c in range(6))
    # J is [d2^2, d1^2, d0^2, d2d1, d2d0, d1d0] -> axis basis:
    N = np.array([[zz, yz, xz],       # [d0d0, d0d1, d0d2]
                  [yz, yy, xy],       # [d0d1, d1d1, d1d2]
                  [xz, xy, xx]], dtype=np.float64)
    return N


# ---------------------------------------------------------------------------
# tensor shape helpers (Westin) on a 3x3 or on eigenvalues
# ---------------------------------------------------------------------------

def westin(M, eps=1e-30):
    """(l1>=l2>=l3, cl, cp, cs, fa) for a single 3x3 symmetric matrix."""
    w = np.linalg.eigvalsh(np.asarray(M, dtype=np.float64))   # ascending
    l3, l2, l1 = w
    s = l1 + l2 + l3 + eps
    cl = (l1 - l2) / s
    cp = 2.0 * (l2 - l3) / s
    cs = 3.0 * l3 / s
    md = (l1 + l2 + l3) / 3.0
    fa = np.sqrt(1.5 * ((l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2)
                 / (l1 * l1 + l2 * l2 + l3 * l3 + eps))
    return dict(l1=l1, l2=l2, l3=l3, cl=cl, cp=cp, cs=cs, fa=fa,
                aniso_ratio=l1 / (l3 + eps))


def principal_eigvec(M):
    """Eigenvector of the LARGEST eigenvalue of 3x3 M (axis-index basis)."""
    w, v = np.linalg.eigh(np.asarray(M, dtype=np.float64))
    return v[:, -1]


# ---------------------------------------------------------------------------
# rho-smoothed structure tensor FIELD (for c_s maps), optional N subtraction
# ---------------------------------------------------------------------------

def _N_to_package(N):
    """3x3 axis-basis N -> 6-vector in package component order [xx,yy,zz,xy,xz,yz]
    with x=axis2,y=axis1,z=axis0."""
    return np.array([N[2, 2], N[1, 1], N[0, 0], N[1, 2], N[0, 2], N[0, 1]])


def structure_tensor_field(I, sigma, rho, subtract_N=None, truncate=_TRUNC):
    """S(sigma,rho) per voxel as a (6, ...) f64 array in package order. If
    subtract_N (3x3 axis-basis) is given, E[S]-N is returned (the noise-corrected
    tensor). rho smoothing uses reflect."""
    d0, d1, d2 = _derivs(I, sigma, truncate)
    comps = [d2 * d2, d1 * d1, d0 * d0, d2 * d1, d2 * d0, d1 * d0]
    S = np.stack([gaussian_filter(c, rho, mode="reflect", truncate=truncate)
                  for c in comps])
    if subtract_N is not None:
        nv = _N_to_package(subtract_N)
        for c in range(6):
            S[c] = S[c] - nv[c]
    return S


def eig_fields(S, psd_clip=False, want_vec=False, chunk=24):
    """Per-voxel eigenvalues (descending l1,l2,l3) and optionally e3 (smallest-eig
    vector) from a (6,...) package-order S. Robust chunked numpy eigh (the
    package's closed-form solver NaNs on near-degenerate voxels). Returns
    (l1,l2,l3, e3 or None, clip_frac). psd_clip clips negatives to 0; clip_frac =
    fraction of voxels with l3<0 pre-clip."""
    shp = S.shape[1:]
    l1 = np.empty(shp); l2 = np.empty(shp); l3 = np.empty(shp)
    e3 = np.empty((3,) + shp) if want_vec else None
    n_neg = 0; n_tot = 0
    for a in range(0, shp[0], chunk):
        b = slice(a, min(a + chunk, shp[0]))
        s = S[:, b]                                          # (6, c, ...)
        cs = s.shape[1:]
        M = np.empty(cs + (3, 3))
        # package [xx,yy,zz,xy,xz,yz], x=axis2 y=axis1 z=axis0 -> axis basis
        M[..., 0, 0] = s[2]; M[..., 1, 1] = s[1]; M[..., 2, 2] = s[0]
        M[..., 0, 1] = M[..., 1, 0] = s[5]
        M[..., 0, 2] = M[..., 2, 0] = s[4]
        M[..., 1, 2] = M[..., 2, 1] = s[3]
        w, v = np.linalg.eigh(M)                             # ascending
        n_neg += int(np.sum(w[..., 0] < 0)); n_tot += w[..., 0].size
        wl3, wl2, wl1 = w[..., 0], w[..., 1], w[..., 2]
        if psd_clip:
            wl1 = np.clip(wl1, 0, None); wl2 = np.clip(wl2, 0, None)
            wl3 = np.clip(wl3, 0, None)
        l1[b] = wl1; l2[b] = wl2; l3[b] = wl3
        if want_vec:
            e3[:, b] = np.moveaxis(v[..., 0], -1, 0)         # smallest-eig vector
    return l1, l2, l3, e3, float(n_neg / max(n_tot, 1))


def cs_field(S, psd_clip=False, eps=1e-30):
    l1, l2, l3, _vec, clip = eig_fields(S, psd_clip=psd_clip)
    cs = 3.0 * l3 / (l1 + l2 + l3 + eps)
    return cs, clip


# ---------------------------------------------------------------------------
# power spectrum of a noise region (T4) + analytic N from measured P_n
# ---------------------------------------------------------------------------

def power_spectrum_3d(block):
    """|FFT|^2 of a mean-subtracted, windowed block (float64), and the radial
    frequency grid |omega| in rad/voxel."""
    b = np.asarray(block, dtype=np.float64)
    b = b - b.mean()
    # separable Hann window to cut spectral leakage
    win = [np.hanning(n) for n in b.shape]
    w = win[0][:, None, None] * win[1][None, :, None] * win[2][None, None, :]
    F = np.fft.fftn(b * w)
    P = np.abs(F) ** 2 / (w.sum() ** 2 + 1e-30)
    freqs = [2.0 * np.pi * np.fft.fftfreq(n) for n in b.shape]   # rad/voxel
    wx, wy, wz = np.meshgrid(freqs[0], freqs[1], freqs[2], indexing="ij")
    wmag = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
    return P, wmag, (wx, wy, wz)


def N_trace_from_spectrum(P, wcomp, sigma):
    """Predict tr N(sigma) = (2pi)^-3 int |omega|^2 exp(-sigma^2|omega|^2) P(omega).
    Discrete sum over the FFT grid (each cell dV = (2pi/N)^3 folds in via the
    1/(2pi)^3 * cell volume). Returns a scalar."""
    wx, wy, wz = wcomp
    w2 = wx ** 2 + wy ** 2 + wz ** 2
    integrand = w2 * np.exp(-sigma ** 2 * w2) * P
    # discrete int f dw ~ sum f * prod(dw_k); dw_k = 2pi/N_k
    dV = np.prod([2.0 * np.pi / n for n in P.shape])
    return float(integrand.sum() * dV / (2.0 * np.pi) ** 3)


# ---------------------------------------------------------------------------
# broken power-law fit for tr N vs sigma (T3): flat -> sigma^-p, knee at ell
# ---------------------------------------------------------------------------

def fit_broken_power_law(sigma, trN):
    """Fit log tr N = a  (sigma<=ell)  and  = b - p log sigma  (sigma>ell),
    continuous at ell. Grid-search the knee over the sampled sigmas. Returns
    dict(ell, p, plateau, sse)."""
    s = np.log(np.asarray(sigma, float)); y = np.log(np.asarray(trN, float))
    best = None
    for ki in range(1, len(s) - 1):                 # knee between sampled points
        ell = s[ki]
        lo = s <= ell; hi = s > ell
        if lo.sum() < 1 or hi.sum() < 2:
            continue
        plateau = y[lo].mean()                       # flat level
        # high side: fit slope with continuity y=plateau - p*(s-ell)
        ss = s[hi] - ell
        p = -np.sum(ss * (y[hi] - plateau)) / (np.sum(ss * ss) + 1e-30)
        pred_hi = plateau - p * ss
        sse = np.sum((y[lo] - plateau) ** 2) + np.sum((y[hi] - pred_hi) ** 2)
        if best is None or sse < best["sse"]:
            best = dict(ell=float(np.exp(ell)), p=float(p),
                        plateau=float(np.exp(plateau)), sse=float(sse))
    return best


def loglog_slope(sigma, trN):
    """Overall least-squares slope of log tr N vs log sigma."""
    s = np.log(np.asarray(sigma, float)); y = np.log(np.asarray(trN, float))
    A = np.vstack([s, np.ones_like(s)]).T
    m, _c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)

if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    from .io import load_nifti, save_nifti

    parser = argparse.ArgumentParser(description="Compute structure tensor noise floor N(sigma) from a noise region.")
    parser.add_argument("input", type=Path, help="Input NIfTI file (noise region).")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[1, 2, 4, 8], help="List of sigma values to compute N(sigma).")
    parser.add_argument("--truncate", type=float, default=_TRUNC, help="Truncate Gaussian kernel at this many sigmas.")
    parser.add_argument("--crop", type=int, default=None, help="Crop this many voxels from each face before averaging.")
    parser.add_argument("--output", type=Path, default=None, help="Output N(sigma) as a text file.")
    
    args = parser.parse_args()

    # Load the noise region
    I = load_nifti(args.input)

    # Compute N(sigma) for each sigma
    results = []
    for sigma in args.sigmas:
        N = N_matrix(I, sigma, truncate=args.truncate, crop=args.crop)
        trN = np.trace(N)

        results.append((sigma, trN))

        