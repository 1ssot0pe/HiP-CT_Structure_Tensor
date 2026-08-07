"""
dense_host.py -- HOST's higher-order ODF as a dense field, plus the lambda3
routing gate.

Two changes to how HOST is evaluated, and only one of them is about speed.

1. THE ODF IS BUILT BY CONVOLUTION, NOT PER PATCH
-------------------------------------------------
`host_fod.analyse` takes a (2*half+1)^3 crop, differentiates it in isolation,
and sums grad (x) grad over that box. The directional energy it forms is

    E_2n(u) = sum_i w_i (ghat_i . u)^2n / sum_i w_i ,      w_i = |g_i|^2

and the only thing that expression needs from the neighbourhood is a WEIGHTED
SUM over it. Expanding the power multinomially,

    (g . u)^2n = sum_{a+b+c=2n} (2n)!/(a!b!c!) gx^a gy^b gz^c ux^a uy^b uz^c

so the whole neighbourhood dependence collapses into 28 scalar fields (for
2n = 6), each a product of gradient components, each summed over the
neighbourhood. A weighted sum over a neighbourhood is a CONVOLUTION, so those
28 fields can be Gaussian-integrated in one separable pass each and the ODF is
then available at EVERY voxel:

    E(u) proportional to  sum_{abc} M_abc * ux^a uy^b uz^c

Neighbourhood integration is untouched -- this is the same average over the same
kind of neighbourhood. What changes is that the window SLIDES instead of being
tiled, and that it is a smooth window over real data instead of a box with an
artificial boundary.

That second point is the one that matters. At sigma_grad = 5 scipy's derivative
kernel has radius 4*sigma = 20, wider than the half = 14 patch, so in the patch
formulation every gradient is computed substantially from `mode="nearest"` edge
replication. Measured on 120 i74 centres, holding box, cutoff and sigma fixed
and changing ONLY whether the gradients saw the surrounding tissue: mean c_s
0.291 -> 0.149, and the routing gate went from firing 11/120 to 0/120. The patch
was not a neutral container.

Normalisation note: the 1/sum(w) denominator, and the per-voxel weight field
generally, CANCEL. Peak extraction uses f = max(E) - E normalised by its own
maximum, and `aniso` is (max-min)/mean -- both invariant to a positive per-voxel
scale. So only the 28 numerator fields are built.

2. THE GATE IS lambda3/lambda2, RANKED
--------------------------------------
The c_s > FA rule it replaces does not survive contact with either the phantom
or i74. Measured per voxel at the selected scale, it fires on 0% of phantom
tissue (including the true crossing) and 51% of i74 tissue -- the same test, the
same scale, routing nothing on the validation object and half the volume on the
data. Its discrimination is barely better than chance:

    statistic            phantom AUC (crossing zone vs single-family, same volume)
    c_s - FA                 0.709
    c_s - c_p (Westin)       0.926
    lambda3/lambda1          0.998
    lambda3/lambda2          0.998

The physics is that a single fibre has constant intensity ALONG itself, so its
gradient tensor is planar -- lambda1 ~ lambda2 >> lambda3. A second fibre family
lifts lambda3. c_s and FA are each normalised over all three eigenvalues, so
each only half-measures this and their difference mixes in lambda1;
lambda3/lambda2 asks the question directly.

Ranked, not thresholded: absolute levels do not transfer (i74's median
lambda3/lambda2 is 0.610 against the phantom crossing zone's 0.099 -- real
tissue at this scale is far less planar than a noise-free phantom). The fire
fraction is a COMPUTE BUDGET, not a physical threshold.
"""

import os
import sys

import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection")):
    sys.path.insert(0, _p)

from host_fod import SPHERE, EPS                                  # noqa: E402
from reconstruction import tissue_mask                            # noqa: E402

_V = SPHERE.vertices

# Peak-extraction settings are host_fod.analyse's, unchanged and deliberately
# so: they concern how a peak is read off an ODF, not what scale the ODF is
# built at, and they are already validated across separation and noise.
REL_PEAK = 0.30
MIN_SEP = 30.0
SHARPEN = 3.0
DEFAULT_ORDER = 6
# Fire fraction. 14% reproduces the current HOST budget almost exactly (211
# stride-6 centres against today's 189) and sits at the knee of the phantom
# recall curve: 96.5% of crossing-zone voxels at 1.30% false fire in
# single-family tissue. Below it recall falls away (77% at 10%); above it recall
# is already saturated and only the false fires grow (6.6% at 20%, 12% at 25%).
DEFAULT_FIRE_FRACTION = 0.14


# ---------------------------------------------------------------------------
# the moment basis
# ---------------------------------------------------------------------------

def moment_indices(order=DEFAULT_ORDER):
    """[(a, b, c, multinomial coefficient)] for every a+b+c == order."""
    from math import factorial
    out = []
    for a in range(order + 1):
        for b in range(order + 1 - a):
            c = order - a - b
            coef = factorial(order) // (factorial(a) * factorial(b) * factorial(c))
            out.append((a, b, c, coef))
    return out


def sphere_basis(order=DEFAULT_ORDER, vertices=None):
    """(n_vertices, n_terms) matrix of ux^a uy^b uz^c for the sphere."""
    V = _V if vertices is None else np.asarray(vertices, float)
    idx = moment_indices(order)
    B = np.empty((V.shape[0], len(idx)), np.float64)
    for k, (a, b, c, _coef) in enumerate(idx):
        B[:, k] = (V[:, 0] ** a) * (V[:, 1] ** b) * (V[:, 2] ** c)
    return B


def moment_field(vol, sigma_grad, rho, order=DEFAULT_ORDER, mag_frac=0.1,
                 truncate=4.0, verbose=False):
    """Dense (n_terms, Z, Y, X) numerator moments of the fibre-ness ODF.

    sigma_grad : differentiation scale for the gradient
    rho        : the NEIGHBOURHOOD -- Gaussian integration width. This is the
                 slid counterpart of the patch's box half-width, and it is what
                 makes an ODF at a voxel a statement about a region rather than
                 about one gradient vector (a single gradient has a degenerate
                 ODF; the neighbourhood is not optional).
    mag_frac   : voxels whose |grad| is below this fraction of the volume max
                 are dropped, as `patch_gradients` drops them per patch. Applied
                 globally here, because a per-patch relative cutoff is not
                 separable -- a voxel's inclusion would depend on which patch
                 asked.

    Component order matches `host_fod`: (x, y, z) = (axis2, axis1, axis0).
    """
    v = np.asarray(vol, np.float32)
    gz = ndimage.gaussian_filter(v, sigma_grad, order=(1, 0, 0), mode="nearest",
                                 truncate=truncate)
    gy = ndimage.gaussian_filter(v, sigma_grad, order=(0, 1, 0), mode="nearest",
                                 truncate=truncate)
    gx = ndimage.gaussian_filter(v, sigma_grad, order=(0, 0, 1), mode="nearest",
                                 truncate=truncate)
    mag2 = gx * gx + gy * gy + gz * gz
    mag = np.sqrt(mag2)
    if mag_frac and mag_frac > 0:
        keep = mag > float(mag_frac) * float(mag.max())
    else:
        keep = np.ones(mag.shape, bool)

    # w_i (ghat_i . u)^order = (g_i . u)^order / |g_i|^(order - 2)
    denom = np.power(mag + EPS, order - 2)
    scale = np.where(keep, 1.0 / denom, 0.0).astype(np.float32)

    idx = moment_indices(order)
    M = np.empty((len(idx),) + v.shape, np.float32)
    for k, (a, b, c, coef) in enumerate(idx):
        term = np.ones_like(v)
        if a:
            term = term * np.power(gx, a)
        if b:
            term = term * np.power(gy, b)
        if c:
            term = term * np.power(gz, c)
        term *= coef * scale
        # THE neighbourhood integration -- a weighted sum over a region, done as
        # a separable convolution instead of a per-patch loop.
        M[k] = ndimage.gaussian_filter(term, rho, mode="nearest",
                                       truncate=truncate)
        if verbose:
            print(f"    moment {k + 1}/{len(idx)}  (a,b,c)=({a},{b},{c})")
    return M


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def lambda_ratio(lam):
    """lambda3 / lambda2 -- the routing statistic. `lam` is DESCENDING."""
    lam = np.asarray(lam, float)
    return lam[2] / (lam[1] + EPS)


def gate_mask(lam, tissue, fire_fraction=DEFAULT_FIRE_FRACTION):
    """Top `fire_fraction` of tissue voxels by lambda3/lambda2.

    Returns (mask, threshold). Ranked rather than thresholded at a fixed value
    because the absolute level does not transfer between phantom and tissue --
    see the module docstring.
    """
    r = lambda_ratio(lam)
    tissue = np.asarray(tissue, bool)
    v = r[tissue]
    v = v[np.isfinite(v)]
    if not v.size:
        return np.zeros(r.shape, bool), np.inf
    thr = float(np.quantile(v, 1.0 - float(fire_fraction)))
    return (r > thr) & tissue & np.isfinite(r), thr


# ---------------------------------------------------------------------------
# ODF evaluation and peak extraction, at selected voxels only
# ---------------------------------------------------------------------------

def odf_at(M, voxels, basis=None, order=DEFAULT_ORDER, chunk=20000):
    """E(u) over the sphere at the given voxels. (n_voxels, n_vertices).

    A matrix multiply, not a loop: the moments at N voxels are (N, 28) and the
    basis is (28, 724), so the whole ODF evaluation is one GEMM. Chunked because
    the output is the big array (N x 724 floats), not the input.
    """
    B = sphere_basis(order) if basis is None else basis
    vox = np.asarray(voxels, int)
    out = np.empty((vox.shape[0], B.shape[0]), np.float32)
    for i in range(0, vox.shape[0], chunk):
        s = vox[i:i + chunk]
        m = M[:, s[:, 0], s[:, 1], s[:, 2]].T          # (chunk, n_terms)
        out[i:i + chunk] = m @ B.T
    return out


def peaks_from_odf(E, rel=REL_PEAK, min_sep=MIN_SEP, sharpen=SHARPEN):
    """Fibre directions for one voxel's ODF, exactly as `host_fod.analyse` does.

    A fibre is a direction of LOW gradient energy (intensity is constant along
    it), so fibre-ness is max(E) - E and the fibres are its peaks.
    """
    from dipy.direction import peak_directions
    f = E.max() - E
    fn = (f / (f.max() + EPS)) ** sharpen
    dirs, vals, _ = peak_directions(fn.astype(np.float64), SPHERE,
                                    relative_peak_threshold=rel,
                                    min_separation_angle=min_sep)
    aniso = float((E.max() - E.min()) / (abs(E.mean()) + EPS))
    sep = amp2 = np.nan
    if len(vals) >= 2:
        sep = float(np.degrees(np.arccos(np.clip(abs(dirs[0] @ dirs[1]), 0, 1))))
        amp2 = float(vals[1] / (vals[0] + EPS))
    return dict(n_peaks=len(vals), dirs=dirs, vals=vals, sep=sep, amp2=amp2,
                aniso=aniso)


def host_at_voxels(M, voxels, order=DEFAULT_ORDER, basis=None, verbose=True,
                   progress_every=20000):
    """Run peak extraction at the gated voxels. Returns a list of dicts.

    This is the only per-voxel Python loop that survives, and it is confined to
    the voxels the gate fired on -- which is the point of gating at all.
    """
    B = sphere_basis(order) if basis is None else basis
    E = odf_at(M, voxels, basis=B, order=order)
    out = []
    for i in range(E.shape[0]):
        if verbose and progress_every and i and i % progress_every == 0:
            print(f"    peaks {i}/{E.shape[0]}")
        out.append(peaks_from_odf(E[i]))
    return out
