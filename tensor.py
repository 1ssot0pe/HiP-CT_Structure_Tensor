"""
tensor.py -- the structure tensor, computed in one place.

Why this exists
---------------
Three modules computed their own dense structure tensor, at three different
hardcoded scales, and none of them consulted the scale Stage 1 had certified:

    sigma_rho_gated_field.tensor_fields          Stage 1's adaptive (sigma, rho)
    self_optimising_tractography.orientation_field       sigma=1.0, rho=2.0
    self_optimising_tractography.anisotropy_target       sigma=1.0, rho=3.0

So the field a streamline followed, the target density it was scored against,
and the directions the gate stored were three different measurements of the same
volume. The last two were not choices anyone had defended -- they were defaults
that nothing overrode. Everything now goes through `tensor_field`, which takes
its scale as a required argument, and the eigenvalues come back with it so
`lambda1 - lambda3` is a read of the same decomposition rather than a fourth.

Component order, decided once
-----------------------------
Skielex's `eig_special_3d` returns eigenvector components as [x, y, z] =
[axis2, axis1, axis0], while every position in this pipeline is indexed
(z, y, x). Reconciling that at each call site is how a direction gets silently
mirrored -- the failure mode `3_global_tractography/test_frame_convention.py`
exists to catch, and which reads as "the structure curves away" rather than as a
bug. `tensor_field` therefore returns e3 in (z, y, x) components, ONCE, at the
source. Callers that need the (x, y, z) convention (`GatedField.patches[i]
["dirs"]`, anything handed to `host_fod`) reverse on the way out and say so.

c_s and FA here are NOT host_fod's c_s and FA
---------------------------------------------
Both files compute "sphericity" and "fractional anisotropy" from three
eigenvalues, but of DIFFERENT matrices:

    host_fod._scatter_eigvals   G^T G over a patch -- an unweighted sum over a
                                BOX window, gradients taken at sigma_grad
    tensor_field                the structure tensor -- a GAUSSIAN-weighted
                                outer product at integration scale rho

They answer the same question and agree in the limit where rho approximates the
box, but they are not interchangeable and the routing decision is currently made
with the first. The dense versions are provided here for the per-voxel gate
(pipeline Step 3), which must be checked against the per-patch decision before
it replaces it -- see the note in `2_host_classification/README.md`. Nothing
routes on these yet.

Memory
------
One call allocates the 6-component tensor plus a 3-component eigenvalue field
and a 3-component eigenvector field, ~12x the volume in float32. That is the
reason `distinct_scales` groups patches by scale: in adaptive mode the cost is
per DISTINCT (sigma, rho), not per patch, and in global mode it is one call.
"""

import os
import sys

import numpy as np
from structure_tensor import structure_tensor_3d, eig_special_3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orthogonality_residual import crop_margin          # noqa: E402

EPS = 1e-12

__all__ = ["tensor_field", "tensor_field_adaptive", "sphericity",
           "fractional_anisotropy", "anisotropy", "e3_to_xyz"]


# ---------------------------------------------------------------------------
# scalar invariants of the eigenvalue triple
# ---------------------------------------------------------------------------

def sphericity(lam):
    """c_s = 3 * lambda_min / trace -- how much the voxel has NO dominant plane.

    `lam` is (3, ...) in DESCENDING order, as Skielex returns it, so the minimum
    is lam[2]. Passing an ascending triple silently computes 3*lambda_max/trace,
    which is a different number that still lies in [0, 1] and still looks
    plausible -- hence the explicit ordering contract.
    """
    lam = np.asarray(lam, float)
    return 3.0 * lam[2] / (lam.sum(axis=0) + EPS)


def fractional_anisotropy(lam):
    """FA from the same three eigenvalues. Symmetric, so ordering is irrelevant."""
    lam = np.asarray(lam, float)
    l1, l2, l3 = lam[0], lam[1], lam[2]
    num = np.sqrt(0.5) * np.sqrt((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2)
    return num / (np.sqrt((lam ** 2).sum(axis=0)) + EPS)


def anisotropy(lam):
    """lambda1 - lambda3, the magnitude Stage 3 uses as its density target."""
    lam = np.asarray(lam, float)
    return lam[0] - lam[2]


def e3_to_xyz(e3_zyx):
    """(z,y,x) components -> (x,y,z), for handing a direction to host_fod/NIfTI."""
    return np.asarray(e3_zyx)[::-1]


# ---------------------------------------------------------------------------
# the dense field
# ---------------------------------------------------------------------------

def tensor_field(vol, sigma, rho, truncate=4.0, invariants=True):
    """Structure tensor eigen-decomposition at ONE (sigma, rho).

    sigma, rho are REQUIRED -- there is deliberately no default. A default here
    is how three modules ended up measuring the same volume at three scales
    nobody had chosen; a missing scale should be a TypeError, not a silent 1.0.

    Returns a dict:
        lam   (3, Z, Y, X)  eigenvalues, DESCENDING
        e3    (3, Z, Y, X)  smallest-eigenvalue eigenvector -- the fibre
                            direction -- in (z, y, x) COMPONENTS
        c_s   (Z, Y, X)     sphericity      (only if invariants)
        FA    (Z, Y, X)     frac. anisotropy(only if invariants)
        sigma, rho          echoed back, so a field carries its own provenance
    """
    vol = np.asarray(vol, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Skielex normalises the eigenvector by a norm that is 0 at isotropic
        # voxels -> a benign divide-by-zero, masked by callers. Silence it here.
        S = structure_tensor_3d(vol, float(sigma), float(rho), truncate=truncate)
        val, vec = eig_special_3d(S)
    del S
    out = dict(lam=np.asarray(val), e3=np.asarray(vec)[::-1],
               sigma=float(sigma), rho=float(rho))
    if invariants:
        out["c_s"] = sphericity(val)
        out["FA"] = fractional_anisotropy(val)
    return out


def tensor_field_adaptive(vol, scales, truncate=4.0, invariants=True,
                          verbose=False):
    """The same, with each block computed at its OWN certified (sigma, rho).

    Mirrors `sigma_rho_adaptive.fibre_direction_adaptive`, which does this for e3
    alone; the eigenvalues are needed too (for c_s, FA and the density target),
    and recomputing the tensor a second time to get them would defeat the point
    of this module.

    Each block is computed with an EXTRA context margin pulled from its
    neighbours, then cropped back to its own interior before being stitched in,
    so the Gaussian kernels have real support at block edges instead of a
    zero-padded boundary. The result is piecewise-constant in scale across block
    boundaries, not smoothly blended -- see that function's note for why that is
    an intentional simplification rather than an oversight.

    Blocks with no certified scale fall back to the volume CONSENSUS pair, not
    to the discretisation floor: the floor is a lower bound on what is
    measurable, not an estimate of this volume's fibre scale.
    """
    vol = np.asarray(vol, dtype=np.float32)
    if scales.get("sigma_map") is None:
        return tensor_field(vol, scales["sigma_global"], scales["rho_global"],
                            truncate=truncate, invariants=invariants)

    sig_map = np.asarray(scales["sigma_map"], float)
    rho_map = np.asarray(scales["rho_map"], float)
    zs, ys, xs, block = scales["zs"], scales["ys"], scales["xs"], scales["block"]
    s_fallback = float(scales["sigma_global"])
    r_fallback = float(scales["rho_global"])

    lam = np.full((3,) + vol.shape, np.nan, np.float32)
    e3 = np.full((3,) + vol.shape, np.nan, np.float32)

    for bi, z0 in enumerate(zs):
        for bj, y0 in enumerate(ys):
            for bk, x0 in enumerate(xs):
                s, r = sig_map[bi, bj, bk], rho_map[bi, bj, bk]
                if not np.isfinite(s):
                    s, r = s_fallback, r_fallback
                if not np.isfinite(r):
                    r = s
                pad = crop_margin(s, r, truncate)
                z1, y1, x1 = (min(z0 + block, vol.shape[0]),
                              min(y0 + block, vol.shape[1]),
                              min(x0 + block, vol.shape[2]))
                z0p, z1p = max(z0 - pad, 0), min(z1 + pad, vol.shape[0])
                y0p, y1p = max(y0 - pad, 0), min(y1 + pad, vol.shape[1])
                x0p, x1p = max(x0 - pad, 0), min(x1 + pad, vol.shape[2])
                ctx = vol[z0p:z1p, y0p:y1p, x0p:x1p]
                if min(ctx.shape) < 3:
                    continue
                if verbose:
                    print(f"    block ({bi},{bj},{bk}) sigma={s:.2f} rho={r:.2f} "
                          f"context {ctx.shape}")
                t = tensor_field(ctx, s, r, truncate=truncate, invariants=False)
                sl = (slice(None),
                      slice(z0 - z0p, z0 - z0p + (z1 - z0)),
                      slice(y0 - y0p, y0 - y0p + (y1 - y0)),
                      slice(x0 - x0p, x0 - x0p + (x1 - x0)))
                lam[:, z0:z1, y0:y1, x0:x1] = t["lam"][sl]
                e3[:, z0:z1, y0:y1, x0:x1] = t["e3"][sl]

    out = dict(lam=lam, e3=e3, sigma=s_fallback, rho=r_fallback, adaptive=True)
    if invariants:
        out["c_s"] = sphericity(lam)
        out["FA"] = fractional_anisotropy(lam)
    return out


def field_for(vol, scales, **kw):
    """Dispatch on the run's mode -- global gets one call, adaptive gets blocks."""
    if scales["mode"] == "global":
        return tensor_field(vol, scales["sigma_global"], scales["rho_global"], **kw)
    return tensor_field_adaptive(vol, scales, **kw)
