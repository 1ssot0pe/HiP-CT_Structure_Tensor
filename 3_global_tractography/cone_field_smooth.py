r"""
cone_field_smooth.py -- apply the search cone to the FIELD, not to a walker.

Stage 3 optimised (L, theta, alpha) and left the field untouched: after a
descent, `vec1` and `vec2` are byte-identical to what Stage 2 handed over
(verified by hash; max |change| exactly 0.0). The cone was a stepping rule, so
its only product was a tractogram. This module makes the cone a FIELD OPERATOR,
so the product becomes a refined orientation field and the parameters become a
by-product.

The operation, per voxel
------------------------
Identical geometry to `dense_tracker.track(turn="relax")`, with the voxel
playing the part of the walker:

    1. take the voxel's current direction v as the cone axis
    2. sample the field inside a cone of length L and half-angle theta
    3. average those orientations as OUTER PRODUCTS and take the principal
       eigenvector -- the mean director m
    4. rotate v toward m by fraction alpha, clamped to theta

So the same three parameters that decided how a tract turns now decide how far
each voxel's orientation is pulled toward its neighbourhood. `_mean_director`
and `_slerp` are reused unchanged rather than reimplemented, which is what
guarantees the field operator and the tracker mean the same thing by (L, theta,
alpha).

FROM THE ORIGINAL FIELD, EVERY TIME
-----------------------------------
`smooth_field(field_0, L, theta, alpha)` always starts from `field_0`. It is NOT
applied to its own output. That is the whole design and it is worth being
explicit about why:

  * ACCUMULATED smoothing would make the result depend on the descent's PATH --
    twenty small steps and one large step would give different fields for the
    same final parameters, so "the optimal field" would not be a function of the
    optimal parameters at all.
  * It would also compound: each iteration re-smooths an already-smoothed field,
    so a long descent drives every orientation toward the volume mean regardless
    of what the loss wanted. The loss would then be minimised by iterating for
    longer, which is not a property of the data.

With one application from the original, `params -> field` is a pure function.
The optimum is a point in parameter space, and the refined field is exactly
`smooth_field(field_0, *params_star)` -- reproducible from the JSON alone.

Cost
----
A pass touches every oriented voxel with K cone directions x n samples, which is
the same shape of work as one tracking pass but over ~85k voxels rather than
~5k walkers. Voxels are processed in chunks because the sample array is
(n_vox, K, n, 3) and materialising that whole is several GB at full volume size.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    sys.path.insert(0, _p)

import dense_tracker as DT                    # noqa: E402

EPS = 1e-12
CHUNK = 20000            # voxels per batch; see the Cost note
N_SAMPLE = 4             # samples along each cone ray
RINGS = 2                # tilt rings used to sample the cone


def smooth_directions(v1, v2, x, v, L, theta, alpha, n=N_SAMPLE, rings=RINGS,
                      clamp_to_theta=False):
    """One cone-relaxation of the directions `v` at positions `x`.

    v1, v2 : (6, Z, Y, X) DYADIC fields to read the neighbourhood from
    x      : (M, 3) positions, (z, y, x)
    v      : (M, 3) current unit directions -- the cone axes
    Returns (M, 3) updated unit directions.

    THETA IS THE APERTURE, NOT A ROTATION CAP -- and this is the one place the
    field operator must NOT copy the tracker.

    In `dense_tracker.track` theta caps the turn taken in ONE STEP, which is
    right: a walker takes ~90 steps and the turns accumulate along the tract. A
    field voxel is corrected ONCE, so the same cap becomes a ceiling on the
    total correction it can ever receive -- a voxel 30 deg from its
    neighbourhood mean could only be moved 3.7 deg.

    Measured, with the cap on, at the stage-3 converged cone: the move was
    EXACTLY alpha*theta at the median AND the p95, for every alpha tried
    (0.7035, 1.8426, 3.6852 deg). The clamp was saturated at essentially every
    voxel, because with L ~ 10 voxels the neighbourhood mean is almost always
    further than a 3.7 deg theta. The operator had degenerated into "rotate
    every voxel by a fixed angle", which carries no information about how
    misaligned any particular voxel was.

    With the cap off, the move is alpha * angle(v, mean), which varies per voxel
    and is large exactly where the field disagrees with its neighbourhood --
    which is what smoothing is supposed to mean. theta still selects WHICH
    neighbours are averaged, which is its real job here.
    """
    if not len(x):
        return v.copy()
    wv, wa, wb = DT.cone_sample_weights(float(theta), rings)
    out = np.empty_like(v)
    for i in range(0, len(x), CHUNK):
        xs, vs = x[i:i + CHUNK], v[i:i + CHUNK]
        a, b = DT._basis(vs)
        dirs = DT._unit(wv[None, :, None] * vs[:, None, :]
                        + wa[None, :, None] * a[:, None, :]
                        + wb[None, :, None] * b[:, None, :], axis=2)
        m = DT._mean_director(v1, v2, xs, dirs, float(L), int(n), vs, True)
        out[i:i + CHUNK] = DT._slerp(
            vs, m, float(alpha),
            max_angle=float(theta) if clamp_to_theta else None)
    return out


def smooth_field(dense, L, theta, alpha, n=N_SAMPLE, rings=RINGS,
                 second=True, clamp_to_theta=False):
    """A cone-relaxed COPY of `dense`. The input is never modified.

    Returns a new DenseField whose `vec1` (and `vec2`, if `second`) have each
    been pulled one step of fraction alpha toward their cone mean. `n_dirs`,
    `labels` and `patch_index` are carried through unchanged -- this operation
    moves orientations, it does not create or destroy directions, and a voxel
    that had no second direction does not acquire one by being smoothed.
    """
    out = dense.copy()
    T1 = DT.dyadic(np.asarray(dense.vec1, float))
    T2 = (DT.dyadic(np.asarray(dense.vec2, float))
          if getattr(dense, "n_multi", 0) else None)

    idx = np.argwhere(np.asarray(dense.n_dirs) >= 1).astype(float)
    if len(idx):
        z, y, xg = (idx[:, 0].astype(int), idx[:, 1].astype(int),
                    idx[:, 2].astype(int))
        v = np.asarray(dense.vec1, float)[:, z, y, xg].T
        nv = smooth_directions(T1, T2, idx, v, L, theta, alpha, n, rings,
                               clamp_to_theta=clamp_to_theta)
        out.vec1[:, z, y, xg] = nv.T.astype(out.vec1.dtype)

    if second and getattr(dense, "n_multi", 0):
        idx2 = np.argwhere(np.asarray(dense.n_dirs) >= 2).astype(float)
        if len(idx2):
            z, y, xg = (idx2[:, 0].astype(int), idx2[:, 1].astype(int),
                        idx2[:, 2].astype(int))
            v = np.asarray(dense.vec2, float)[:, z, y, xg].T
            # vec2 is relaxed against the SAME neighbourhood. `_mean_director`
            # picks whichever family each sample is better aligned with, so a
            # second direction is pulled toward other second directions rather
            # than collapsing onto the dominant one.
            nv = smooth_directions(T1, T2, idx2, v, L, theta, alpha, n,
                                   rings, clamp_to_theta=clamp_to_theta)
            out.vec2[:, z, y, xg] = nv.T.astype(out.vec2.dtype)
    return out


def angular_change(a, b, mask=None, degrees=True):
    """Median and mean angle between two director fields, (3, Z, Y, X) each.

    DIRECTORS, so the angle is taken on |cos| -- v and -v are the same
    orientation and a sign flip is not a rotation. Without the absolute value a
    field that merely re-signed itself would report a median change near 90 deg.
    """
    A = DT._unit(np.asarray(a, float), axis=0)
    B = DT._unit(np.asarray(b, float), axis=0)
    c = np.abs((A * B).sum(axis=0))
    ok = np.isfinite(c)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    if not ok.any():
        return dict(median=float("nan"), mean=float("nan"), p95=float("nan"),
                    n=0)
    ang = np.arccos(np.clip(c[ok], 0.0, 1.0))
    if degrees:
        ang = np.degrees(ang)
    return dict(median=float(np.median(ang)), mean=float(ang.mean()),
                p95=float(np.percentile(ang, 95)), n=int(ok.sum()))


def _selftest():
    """Three properties the operator must have, on a synthetic field."""
    print("cone_field_smooth self-test")
    sys.path.insert(0, os.path.join(_ROOT, "1_sigma_rho_selection"))
    sys.path.insert(0, os.path.join(_ROOT, "2_host_classification"))
    from dense_field import DenseField

    shape = (16, 16, 16)
    rng = np.random.default_rng(0)
    # a coherent field along x, with noise added
    base = np.zeros((3,) + shape)
    base[2] = 1.0
    noisy = base + 0.35 * rng.normal(size=base.shape)
    f = DenseField.from_e3(noisy, tissue=np.ones(shape, bool))

    a0 = angular_change(f.vec1, base)
    out_half = smooth_field(f, L=3.0, theta=0.5, alpha=0.5)
    out_zero = smooth_field(f, L=3.0, theta=0.5, alpha=0.0)
    a_half = angular_change(out_half.vec1, base)
    a_zero = angular_change(out_zero.vec1, base)
    moved = angular_change(out_half.vec1, f.vec1)

    print(f"  noisy field vs truth       median {a0['median']:.2f} deg")
    print(f"  alpha=0.5 smoothed vs truth median {a_half['median']:.2f} deg  "
          f"(should be LOWER)")
    print(f"  alpha=0.0 smoothed vs truth median {a_zero['median']:.2f} deg  "
          f"(should be UNCHANGED)")
    print(f"  alpha=0.5 moved the field   median {moved['median']:.2f} deg")
    print(f"  input untouched: {np.abs(np.asarray(f.vec1) - np.asarray(DenseField.from_e3(noisy, tissue=np.ones(shape, bool)).vec1)).max():.3e}")

    ok = (a_half["median"] < a0["median"]           # smoothing helps
          and abs(a_zero["median"] - a0["median"]) < 1e-6   # alpha=0 is identity
          and moved["median"] > 0.5)                # alpha=0.5 actually moved it
    print("  " + ("ALL PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    _selftest()
