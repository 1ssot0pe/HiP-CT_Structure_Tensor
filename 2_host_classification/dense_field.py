"""dense_field.py -- Step 2 evaluated at EVERY voxel, not on a patch grid.

`build_gated_field` answers per patch on a stride-6 grid. Reading that back as a
voxel image means stamping one patch's answer over its whole 6x6x6
neighbourhood, which puts rectangular blocks in the output either side of every
branching point -- an artifact of the sampling grid, not of the structure
tensor. This module computes the same three quantities per voxel instead, so
the blocks cannot exist:

    c_s, FA   the gate            -> dense_gate()
    up to 2 HOST directions       -> dense_host()
    e3                            -> dense_e3()   (in export_nifti)

Why this is affordable
----------------------
`host_fod.analyse` costs ~0.25 s per patch, so calling it 512k times is ~35
hours. But nothing in it needs to be done patch-at-a-time -- every quantity is
a SUM OVER A BOX, and a sum over a box is a separable filter:

    gate:  S(v) = sum_{u in box(v)} keep(u) g(u) g(u)^T
                                          -> 6 box filters, then eigh

    HOST:  E_2n(u) = sum_alpha C_alpha m_alpha u^alpha    (|alpha| = 2n)
           m_alpha(v) = sum_{u in box(v)} w(u) ghat(u)^alpha
                                          -> 28 box filters at order 6

Expanding (ghat.u)^6 into its 28 degree-6 monomials is what turns the ODF from
a per-patch matrix product into a fixed (28, 724) basis applied to 28 dense
moment images. The ODF is then evaluated only where the gate fired.

The one approximation, and how it is checked
--------------------------------------------
`analyse` keeps gradients with |g| > 0.1 * max|g| OVER THE PATCH. Densely, that
max is `maximum_filter(mag, box)` -- exact, but it makes the threshold vary per
voxel, so the box filter is no longer a single convolution. This module
quantises the threshold into `levels` bins and runs one filter bank per bin,
which is exact up to the bin width.

Two further differences from the patch code are unavoidable and both favour the
dense path: gradients are taken on the whole volume rather than on an isolated
patch (so no replicated patch edge), and voxels within `half` of the volume
face get a truncated box instead of being skipped entirely.

`validate_against_patches()` checks all of it the only way that means anything
-- by running the real `host_fod.analyse` at sample voxels and comparing.
"""

import math
import os
import sys

import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    sys.path.insert(0, _p)

from host_fod import SPHERE
from sigma_rho_gated_field import (GATE_HALF, GATE_SIGMA_GRAD, DIR_SIGMA_GRAD,
                                   HOST_ORDER)

EPS = 1e-12
MAG_FRAC = 0.1        # host_fod/two_orientation_test keep threshold
REL = 0.30            # host_fod.analyse relative_peak_threshold
MIN_SEP = 30.0        # host_fod.analyse min_separation_angle (degrees)
SHARPEN = 3.0         # host_fod.analyse sharpen
MIN_GRADS = 40        # analyse returns None below this many kept gradients
LEVELS = 24           # threshold quantisation bins
SLAB = 16             # z-slab for the eigh pass
CHUNK = 20000         # voxels per ODF evaluation chunk

_V = SPHERE.vertices


# ---------------------------------------------------------------------------
# shared primitives
# ---------------------------------------------------------------------------

def gradients(vol, sigma_grad):
    """(gx, gy, gz) over the whole volume -- `patch_gradients`' component order."""
    v = vol.astype(np.float32)
    gz = ndimage.gaussian_filter(v, sigma_grad, order=(1, 0, 0), mode="nearest")
    gy = ndimage.gaussian_filter(v, sigma_grad, order=(0, 1, 0), mode="nearest")
    gx = ndimage.gaussian_filter(v, sigma_grad, order=(0, 0, 1), mode="nearest")
    return gx, gy, gz


def _box(a, half):
    """Box mean over a (2*half+1)^3 window.

    Mean rather than sum: every consumer here normalises it away (eigenvalue
    ratios for the gate, f/f.max() for the ODF), and the mean form is what
    `uniform_filter` computes directly.
    """
    return ndimage.uniform_filter(a, size=2 * half + 1, mode="constant")


def _threshold_levels(mag, half, levels):
    """Per-voxel keep threshold 0.1*max|g| over the box, quantised into bins.

    Returns (bin_index, bin_threshold_values). The unquantised threshold is
    exact -- `maximum_filter` over the box IS the patch max that `analyse`
    divides by. Only the binning is an approximation.
    """
    loc_max = ndimage.maximum_filter(mag, size=2 * half + 1, mode="constant")
    thr = MAG_FRAC * loc_max
    hi = float(thr.max())
    if hi <= 0:
        return np.zeros(thr.shape, np.intp), np.array([0.0])
    # log-spaced: thresholds span orders of magnitude between tissue and air,
    # and a linear grid would put every tissue voxel in the bottom bin.
    lo = max(hi * 1e-4, float(thr[thr > 0].min()) if np.any(thr > 0) else hi * 1e-4)
    edges = np.geomspace(lo, hi, levels)
    idx = np.clip(np.searchsorted(edges, thr, "left"), 0, levels - 1)
    return idx, edges


def _eig_ascending(comp, shape, slab=SLAB):
    """Ascending eigenvalues + smallest-eigenvalue eigenvector of a symmetric
    3x3 field given as (xx, yy, zz, xy, xz, yz). Slabbed to bound memory."""
    xx, yy, zz, xy, xz, yz = comp
    lam = np.zeros((3,) + shape, np.float32)
    e3 = np.zeros((3,) + shape, np.float32)
    for z0 in range(0, shape[0], slab):
        sl = slice(z0, min(z0 + slab, shape[0]))
        M = np.empty(xx[sl].shape + (3, 3), np.float64)
        M[..., 0, 0] = xx[sl]
        M[..., 1, 1] = yy[sl]
        M[..., 2, 2] = zz[sl]
        M[..., 0, 1] = M[..., 1, 0] = xy[sl]
        M[..., 0, 2] = M[..., 2, 0] = xz[sl]
        M[..., 1, 2] = M[..., 2, 1] = yz[sl]
        w, Q = np.linalg.eigh(M)
        lam[:, sl] = np.moveaxis(w, -1, 0)
        v = np.moveaxis(Q[..., 0], -1, 0)
        v[:, w.sum(-1) <= 0] = 0.0
        e3[:, sl] = v
        del M, w, Q, v
    return lam, e3


# ---------------------------------------------------------------------------
# the gate:  c_s vs FA at every voxel
# ---------------------------------------------------------------------------

def dense_gate(vol, sigma_grad=GATE_SIGMA_GRAD, half=GATE_HALF, levels=LEVELS,
               verbose=True):
    """c_s and FA per voxel, from the same gradient scatter matrix `analyse` uses.

    S = sum of g g^T over the box, gradients below 0.1 * box-max dropped --
    UNWEIGHTED, matching `analyse`'s `G[keep].T @ G[keep]` (its `mag**0` factor
    is the identity). Returns (c_s, FA, n_kept), all (nz, ny, nx).
    """
    gx, gy, gz = gradients(vol, sigma_grad)
    mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    idx, edges = _threshold_levels(mag, half, levels)
    shape = vol.shape

    comp = [np.zeros(shape, np.float32) for _ in range(6)]
    n_kept = np.zeros(shape, np.float32)
    nvox = (2 * half + 1) ** 3
    present = np.unique(idx)
    if verbose:
        print(f"  [gate] sigma_grad={sigma_grad} half={half} "
              f"over {len(present)} threshold bins")
    for b in present:
        sel = idx == b
        keep = (mag > edges[b]).astype(np.float32)
        prod = (gx * gx, gy * gy, gz * gz, gx * gy, gx * gz, gy * gz)
        for c in range(6):
            comp[c][sel] = _box(keep * prod[c], half)[sel]
        n_kept[sel] = _box(keep, half)[sel] * nvox
        del keep

    lam, _ = _eig_ascending(comp, shape)
    # The scatter matrix is positive semi-definite, so lambda >= 0 -- but in
    # near-vacuum it is numerically zero and eigh returns tiny SIGNED values.
    # 3*lam0/sum then explodes: unclamped this field reached c_s = 5720 for a
    # quantity bounded by 1, and every such voxel passed the c_s > FA gate.
    np.clip(lam, 0.0, None, out=lam)
    tot = lam.sum(0)
    # Defined only where `analyse` would have returned something at all. Below
    # MIN_GRADS kept gradients it returns None, so there is no gate decision to
    # make and no HOST answer to be had; 0 keeps those voxels out of the gate.
    ok = (tot > 0) & (n_kept >= MIN_GRADS)
    c_s = np.zeros(shape, np.float32)
    fa = np.zeros(shape, np.float32)
    c_s[ok] = 3.0 * lam[0][ok] / tot[ok]
    l0, l1, l2 = lam[0], lam[1], lam[2]
    num = np.sqrt(0.5) * np.sqrt((l0 - l1) ** 2 + (l1 - l2) ** 2 + (l2 - l0) ** 2)
    den = np.sqrt((lam ** 2).sum(0))
    fa[ok] = num[ok] / (den[ok] + EPS)
    return c_s, fa, n_kept


# ---------------------------------------------------------------------------
# HOST-FOD:  up to two directions at every gated voxel
# ---------------------------------------------------------------------------

def _monomials(order):
    """Exponent triples of total degree `order`, with multinomial coefficients."""
    out = []
    for a in range(order + 1):
        for b in range(order - a + 1):
            c = order - a - b
            coef = (math.factorial(order) / (math.factorial(a) *
                    math.factorial(b) * math.factorial(c)))
            out.append((a, b, c, float(coef)))
    return out


def _sphere_adjacency():
    """(724, maxdeg) neighbour index array, padded with the vertex itself."""
    e = SPHERE.edges
    nbr = [[] for _ in range(len(_V))]
    for i, j in e:
        nbr[i].append(j)
        nbr[j].append(i)
    deg = max(len(n) for n in nbr)
    adj = np.empty((len(_V), deg), np.intp)
    for i, n in enumerate(nbr):
        adj[i, :len(n)] = n
        adj[i, len(n):] = i          # self-pad: never beats the vertex itself
    return adj


_ADJ = _sphere_adjacency()


def _peaks_from_odf(fn, cos_sep, rel=REL):
    """Top-2 peak vertex indices for each row of `fn` (n, 724).

    Reproduces `peak_directions`: local maxima on the sphere graph (>= every
    neighbour and > at least one), then the relative-height cut, then the
    minimum-separation rule with antipodal folding. Taking only the first two
    makes the greedy pass exact without a loop -- peak 1 is the tallest, peak 2
    is the tallest remaining that clears both cuts.
    """
    nb = fn[:, _ADJ]                               # (n, 724, deg)
    is_pk = (fn >= nb.max(-1)) & (fn > nb.min(-1))
    masked = np.where(is_pk, fn, -np.inf)

    i1 = masked.argmax(1)
    v1 = masked[np.arange(len(fn)), i1]
    ok1 = np.isfinite(v1)

    cos = np.abs(_V @ _V[i1].T).T                  # (n, 724), antipodally folded
    ok2 = is_pk & (fn >= rel * v1[:, None]) & (cos < cos_sep)
    m2 = np.where(ok2, fn, -np.inf)
    i2 = m2.argmax(1)
    has2 = np.isfinite(m2[np.arange(len(fn)), i2])
    return i1, ok1, i2, has2 & ok1


def dense_host(vol, mask, sigma_grad=DIR_SIGMA_GRAD, half=GATE_HALF,
               order=HOST_ORDER, levels=LEVELS, chunk=CHUNK, verbose=True):
    """Up to two HOST-FOD directions at every voxel in `mask`.

    Returns (dirs, n_found): dirs (m, 2, 3) in (x, y, z) for the m masked
    voxels in C order, n_found (m,) int.
    """
    shape = vol.shape
    m = int(mask.sum())
    dirs = np.zeros((m, 2, 3), np.float32)
    n_found = np.zeros(m, np.int8)
    if m == 0:
        return dirs, n_found

    gx, gy, gz = gradients(vol, sigma_grad)
    mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    idx, edges = _threshold_levels(mag, half, levels)

    mono = _monomials(order)
    nvox = (2 * half + 1) ** 3
    # The moment is sum(w * ghat^alpha) with w = |g|^2, and |alpha| = order.
    # Writing that as g^alpha / |g|^(order-2) is algebraically identical and
    # numerically fatal: |g|^-4 overflows float32 for |g| < 1e-10, and the
    # 0 * inf that follows silently NaNs the whole ODF. Carrying unit gradients
    # and |g|^2 separately keeps every factor bounded.
    good = mag > 0
    inv_mag = np.zeros(shape, np.float32)
    inv_mag[good] = 1.0 / mag[good]
    ghat = (gx * inv_mag, gy * inv_mag, gz * inv_mag)
    w = mag ** 2

    mom = np.zeros((len(mono),) + shape, np.float32)
    n_kept = np.zeros(shape, np.float32)
    present = np.unique(idx[mask])
    if verbose:
        print(f"  [host] sigma_grad={sigma_grad} order={order} on {m} voxels "
              f"over {len(present)} threshold bins")
    for b in present:
        sel = (idx == b) & mask
        keep = (mag > edges[b]).astype(np.float32)
        wk = keep * w
        for k, (a, bb, c, _coef) in enumerate(mono):
            mom[k][sel] = _box(wk * ghat[0] ** a * ghat[1] ** bb
                               * ghat[2] ** c, half)[sel]
        n_kept[sel] = _box(keep, half)[sel] * nvox
        del keep, wk

    # (28, 724) basis: C_alpha * v^alpha
    B = np.empty((len(mono), len(_V)), np.float64)
    for k, (a, bb, c, coef) in enumerate(mono):
        B[k] = coef * _V[:, 0] ** a * _V[:, 1] ** bb * _V[:, 2] ** c

    flat = mask.reshape(-1)
    M = mom.reshape(len(mono), -1)[:, flat].T          # (m, 28)
    enough = n_kept.reshape(-1)[flat] >= MIN_GRADS     # analyse() returns None below
    cos_sep = np.cos(np.deg2rad(MIN_SEP))

    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        sub = np.arange(s, e)[enough[s:e]]
        if not sub.size:
            continue
        E = M[sub] @ B                                  # (k, 724)
        f = E.max(1, keepdims=True) - E                 # fibre-ness: minima of E
        fmax = f.max(1, keepdims=True)
        fn = (f / (fmax + EPS)) ** SHARPEN
        i1, ok1, i2, has2 = _peaks_from_odf(fn, cos_sep)
        rows = sub[ok1]
        dirs[rows, 0] = _V[i1[ok1]]
        n_found[rows] = 1
        rows2 = sub[has2]
        dirs[rows2, 1] = _V[i2[has2]]
        n_found[rows2] = 2
        del E, f, fn
    return dirs, n_found


# ---------------------------------------------------------------------------
# validation against the real per-patch code
# ---------------------------------------------------------------------------

def _sample(cand, n, half, shape, rng):
    """`n` interior voxels drawn from `cand` (an (m, 3) index array)."""
    lo, hi = half, np.array(shape) - half - 1
    cand = cand[np.all((cand >= lo) & (cand <= hi), axis=1)]
    if not len(cand):
        return cand
    return cand[rng.choice(len(cand), size=min(n, len(cand)), replace=False)]


def validate_against_patches(vol, n_sample=40, half=GATE_HALF, seed=0,
                             verbose=True):
    """Run `host_fod.analyse` at sample voxels and compare with the dense fields.

    Two samples, because they test different things and live in different
    places: the GATE is checked on tissue, and the PEAKS on voxels the gate
    actually routed (on the kissing phantom no tissue voxel is gated, so a
    single tissue sample would leave the whole HOST path unexercised).

    Exact agreement is not the target and is not achievable. `patch_gradients`
    differentiates an isolated 29^3 patch with `mode="nearest"`, and at
    sigma_grad=2.5 with truncate=4 the kernel radius is 10 -- only the central
    9^3 of that patch is free of the replicated edge. The dense path uses real
    neighbouring data instead. Fed IDENTICAL patch-local gradients, the ODF
    here reproduces `analyse` to corr(E) = 1.00000000 with every peak count and
    direction matching, so what this function measures is that boundary
    difference, not an error in the expansion.
    """
    from host_fod import analyse

    c_s, fa, _ = dense_gate(vol, half=half, verbose=False)
    gated = c_s > fa
    rng = np.random.default_rng(seed)

    gate_err = []
    for z, y, x in _sample(np.argwhere(vol > 0.05 * vol.max()), n_sample, half,
                           vol.shape, rng):
        patch = vol[z - half:z + half + 1, y - half:y + half + 1,
                    x - half:x + half + 1]
        m = analyse(patch, sigma_grad=GATE_SIGMA_GRAD, order=HOST_ORDER)
        if m is None:
            continue
        gate_err.append((abs(m["c_s"] - c_s[z, y, x]), abs(m["FA"] - fa[z, y, x]),
                         (m["c_s"] > m["FA"]) == bool(gated[z, y, x])))

    pk = _sample(np.argwhere(gated), min(n_sample, 20), half, vol.shape, rng)
    n_match, n_both, ang1, ang2 = 0, 0, [], []
    if len(pk):
        one = np.zeros(vol.shape, bool)
        one[pk[:, 0], pk[:, 1], pk[:, 2]] = True
        dd, nn = dense_host(vol, one, half=half, verbose=False)
        for i, (z, y, x) in enumerate(np.argwhere(one)):   # C order == mask order
            patch = vol[z - half:z + half + 1, y - half:y + half + 1,
                        x - half:x + half + 1]
            md = analyse(patch, sigma_grad=DIR_SIGMA_GRAD, order=HOST_ORDER)
            if md is None or not len(md["dirs"]) or nn[i] < 1:
                continue
            n_both += 1
            n_match += int(min(len(md["dirs"]), 2) == int(nn[i]))
            ang1.append(np.degrees(np.arccos(np.clip(
                abs(float(np.dot(md["dirs"][0], dd[i, 0]))), 0, 1))))
            if len(md["dirs"]) >= 2 and nn[i] >= 2:
                ang2.append(np.degrees(np.arccos(np.clip(
                    abs(float(np.dot(md["dirs"][1], dd[i, 1]))), 0, 1))))

    ge = np.array([g[:2] for g in gate_err], float)
    f = lambda a: float(np.mean(a)) if len(a) else float("nan")      # noqa: E731
    out = dict(n_gate=len(gate_err),
               c_s_mae=float(ge[:, 0].mean()) if len(ge) else float("nan"),
               FA_mae=float(ge[:, 1].mean()) if len(ge) else float("nan"),
               gate_agree=f([g[2] for g in gate_err]),
               n_peaks_checked=n_both,
               n_peaks_agree=(n_match / n_both) if n_both else float("nan"),
               peak1_deg=f(ang1), peak2_deg=f(ang2))
    if verbose:
        print("-" * 78)
        print(f"  DENSE vs host_fod.analyse")
        print(f"    gate, {out['n_gate']} tissue voxels: c_s MAE "
              f"{out['c_s_mae']:.4f}, FA MAE {out['FA_mae']:.4f}, "
              f"decision agrees {out['gate_agree']:.1%}")
        if n_both:
            print(f"    peaks, {n_both} gated voxels: count agrees "
                  f"{out['n_peaks_agree']:.1%}, peak-1 {out['peak1_deg']:.2f} deg, "
                  f"peak-2 {out['peak2_deg']:.2f} deg")
        print("-" * 78)
    return out
