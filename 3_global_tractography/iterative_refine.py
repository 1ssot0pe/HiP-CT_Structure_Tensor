r"""
iterative_refine.py -- refine a nematic orientation field by ANNEALED CONE SEARCH.

Not a tracker. This sits at each voxel, searches a cone of candidate directions
around that voxel's current estimate, picks the one best supported by the
spatial neighbourhood, and writes it back. Sweeps repeat over the whole volume
until the field stops moving; the cone then narrows and the process repeats.

Why this rather than the relaxation in `cone_field_smooth`
---------------------------------------------------------
That operator moves each voxel a FRACTION alpha toward its neighbourhood mean,
and alpha was a free parameter handed to a gradient descent. Measured, the
descent drove alpha -> 0.018 and L -> 0.138 -- two independent routes to "change
nothing" -- because the objective it was scored on (either tract density vs
image, or field vs image gradient) is already at its optimum for the UNSMOOTHED
field. e3 is defined as the minimiser of sum (grad I . v)^2 over the rho
neighbourhood, so no smoothing can improve it and the optimiser correctly found
that out.

The argmax formulation has no such escape. A voxel MUST adopt one of its
candidates. It can adopt the zero-tilt candidate -- its own current direction --
but only if that genuinely scores highest against the neighbourhood, which is a
statement about the data rather than a free parameter collapsing. theta and
alpha are set by a fixed annealing SCHEDULE, not optimised, so nothing can tune
them to zero.

The field is updated every sweep and carried forward, so this is a fixed-point
iteration: it runs until the change falls below tolerance. Compounding is
controlled by the two things the schedule provides -- a shrinking cone, and a
convergence test -- rather than by refusing to accumulate at all.

Why outer products, never bare vectors
--------------------------------------
The field is NEMATIC: v and -v are the same orientation, and the eigensolver
picks the sign independently at every voxel. Interpolating v componentwise
averages v against -v wherever two neighbours disagree and returns something
near zero -- not a direction at all. Measured on the crossing phantom, 13.2% of
adjacent voxel pairs are sign-opposed. T = v v^T is invariant under v -> -v, so
it interpolates correctly, and d^T T d = (d . v)^2 recovers the alignment.

Only the 6 unique components of the symmetric T are stored and interpolated.

Run
---
    python iterative_refine.py          # synthetic self-test
"""

import time

import numpy as np
from scipy.ndimage import map_coordinates

EPS = 1e-12

# Upper triangle of a symmetric 3x3, and the multiplicity each entry carries in
# d^T T d (the off-diagonals appear twice).
_DYAD = ((0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0),
         (0, 1, 2.0), (0, 2, 2.0), (1, 2, 2.0))

DEFAULT_SCHEDULE = [
    (30.0, 5.0),     # wide exploration
    (15.0, 3.0),     # moderate
    (8.0, 1.5),      # fine
    (4.0, 0.8),      # very fine
]
DEFAULT_L = 3.0
DEFAULT_TOL = 0.5            # degrees, 95th percentile
DEFAULT_MAX_SWEEPS = 20


# ---------------------------------------------------------------------------
# representation
# ---------------------------------------------------------------------------

def e3_to_outer(e3):
    """[X,Y,Z,3] unit vectors -> [X,Y,Z,3,3] outer-product field v v^T."""
    v = np.asarray(e3, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    v = np.where(n > EPS, v / np.maximum(n, EPS), 0.0)
    return (v[..., :, None] * v[..., None, :]).astype(np.float32)


def outer_to_e3(T_field):
    """[X,Y,Z,3,3] -> [X,Y,Z,3] the eigenvector of the LARGEST eigenvalue.

    T = v v^T is rank 1, so that eigenvector is v. The sign is then forced into
    the positive-z hemisphere purely so repeated calls agree with each other --
    the field is nematic, so this is cosmetic and carries no meaning.
    """
    T = np.asarray(T_field, float)
    w, V = np.linalg.eigh(T)                  # ascending
    e = V[..., :, -1]
    flip = e[..., 2] < 0
    e[flip] *= -1.0
    return e.astype(np.float32)


def _to6(T_field):
    """[X,Y,Z,3,3] -> (6, X, Y, Z), the components in `_DYAD` order."""
    T = np.asarray(T_field, float)
    return np.stack([T[..., i, j] for i, j, _w in _DYAD]).astype(np.float32)


def _from6(T6, shape):
    """(6, X, Y, Z) -> [X,Y,Z,3,3], symmetric."""
    out = np.zeros(tuple(shape) + (3, 3), np.float32)
    for k, (i, j, _w) in enumerate(_DYAD):
        out[..., i, j] = T6[k]
        out[..., j, i] = T6[k]
    return out


# ---------------------------------------------------------------------------
# cone
# ---------------------------------------------------------------------------

def _frame(v):
    """Two unit vectors completing an orthonormal frame with each row of v.

    The seed axis is chosen as the one LEAST aligned with v, so the cross
    product never degenerates -- picking a fixed axis fails wherever v happens
    to be parallel to it.
    """
    a = np.zeros_like(v)
    a[np.arange(len(v)), np.argmin(np.abs(v), axis=1)] = 1.0
    u1 = np.cross(v, a)
    u1 /= np.maximum(np.linalg.norm(u1, axis=1, keepdims=True), EPS)
    u2 = np.cross(v, u1)
    u2 /= np.maximum(np.linalg.norm(u2, axis=1, keepdims=True), EPS)
    return u1, u2


def cone_offsets(theta_deg, alpha_deg):
    """(tilt, azimuth) pairs for the candidate set, in radians.

    Rings at alpha, 2*alpha, ... up to floor(theta/alpha)*alpha, each sampled
    azimuthally at roughly `alpha` spacing along its own circumference -- so a
    wide ring gets more samples than a narrow one and the angular density is
    about uniform over the cap rather than over the ring index.

    The zero-tilt candidate (the voxel's current direction) is always first, so
    "no change" is always available and is chosen only if it genuinely wins.
    """
    out = [(0.0, 0.0)]
    al = np.radians(float(alpha_deg))
    n_rings = int(np.floor(float(theta_deg) / float(alpha_deg)))
    for i in range(1, n_rings + 1):
        tilt = i * al
        n_az = max(1, int(round(2.0 * np.pi * np.sin(tilt) / al)))
        for j in range(n_az):
            out.append((tilt, 2.0 * np.pi * j / n_az))
    return out


# ---------------------------------------------------------------------------
# one sweep
# ---------------------------------------------------------------------------

def _score(T6, pts, d):
    """sum_k w_k * T6_k(pts) * d_i * d_j  -- i.e. d^T T(pts) d, per point.

    `map_coordinates(order=1)` is trilinear. mode="nearest" clamps rather than
    wrapping; samples outside the volume therefore read the boundary value
    instead of folding round to the far side, which would invent a neighbour.
    """
    s = np.zeros(len(pts))
    for k, (i, j, w) in enumerate(_DYAD):
        s += w * map_coordinates(T6[k], pts.T, order=1, mode="nearest") \
            * d[:, i] * d[:, j]
    return s


def sweep(T6, idx, v, theta_deg, alpha_deg, L, shape, mask=None):
    """One Jacobi sweep. Returns the new directions at `idx`.

    idx : (N, 3) float voxel coordinates of the tissue voxels
    v   : (N, 3) their current unit directions

    Reads ONLY from `T6` (the old field) and returns a new array -- the caller
    swaps. Updating in place would let a voxel already visited this sweep
    influence its neighbour, which is Gauss-Seidel: the result would then depend
    on the traversal order, and a raster order biases the field along the scan
    direction.
    """
    u1, u2 = _frame(v)
    n_step = max(int(round(float(L))), 1)
    steps = np.arange(1, n_step + 1, dtype=float)

    best = np.full(len(v), -np.inf)
    best_d = v.copy()
    for tilt, az in cone_offsets(theta_deg, alpha_deg):
        if tilt == 0.0:
            d = v
        else:
            d = (np.cos(tilt) * v
                 + np.sin(tilt) * (np.cos(az) * u1 + np.sin(az) * u2))
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), EPS)
        sc = np.zeros(len(v))
        for s in steps:
            # SYMMETRIC lookahead: the field is nematic, so +d and -d are the
            # same orientation and both halves of the neighbourhood count.
            for sgn in (1.0, -1.0):
                p = idx + sgn * s * d
                inside = np.ones(len(p), bool)
                for c in range(3):
                    inside &= (p[:, c] >= 0) & (p[:, c] <= shape[c] - 1)
                if not inside.any():
                    continue
                contrib = np.zeros(len(p))
                contrib[inside] = _score(T6, p[inside], d[inside])
                if mask is not None:
                    # Samples that land outside tissue contribute nothing rather
                    # than contributing whatever the background happens to hold.
                    q = np.rint(p[inside]).astype(int)
                    ok = mask[q[:, 0], q[:, 1], q[:, 2]]
                    contrib[np.flatnonzero(inside)[~ok]] = 0.0
                sc += contrib
        better = sc > best
        best[better] = sc[better]
        best_d[better] = d[better]
    return best_d


def angular_change(d_new, T6_old, idx):
    """delta = arccos( sqrt( d_new^T T_old d_new ) ), in degrees.

    Nematic by construction: T_old carries no sign, so a voxel whose direction
    merely flipped reports 0 rather than 180.
    """
    q = np.zeros(len(idx))
    for k, (i, j, w) in enumerate(_DYAD):
        q += w * map_coordinates(T6_old[k], idx.T, order=1, mode="nearest") \
            * d_new[:, i] * d_new[:, j]
    return np.degrees(np.arccos(np.sqrt(np.clip(q, 0.0, 1.0))))


# ---------------------------------------------------------------------------

def refine_field(T_field, mask, schedule=None, L=DEFAULT_L,
                 convergence_tol=DEFAULT_TOL,
                 max_sweeps_per_stage=DEFAULT_MAX_SWEEPS, verbose=True):
    """Refine a nematic orientation field by iterative annealed cone search.

    Returns (T_refined, log). The field is updated after EVERY sweep and carried
    into the next, so the result at convergence is the accumulated refinement --
    that is the point of a fixed-point iteration, and it is safe here because
    the annealing schedule narrows the cone and the tolerance stops the loop.
    """
    schedule = schedule or DEFAULT_SCHEDULE
    T_field = np.asarray(T_field, np.float32)
    shape = T_field.shape[:3]
    mask = np.asarray(mask, bool)

    T6 = _to6(T_field)
    idx = np.argwhere(mask).astype(float)
    v = outer_to_e3(T_field)[mask].astype(float)
    if verbose:
        print(f"refine_field: {len(idx)} tissue voxels of {mask.size}, "
              f"L={L:g}, tol={convergence_tol:g} deg")

    log = dict(stages=[], n_voxels=int(len(idx)), L=float(L),
               convergence_tol=float(convergence_tol), schedule=list(schedule))
    for theta, alpha in schedule:
        st = dict(theta=float(theta), alpha=float(alpha), sweeps=[],
                  converged=False)
        n_cand = len(cone_offsets(theta, alpha))
        for it in range(int(max_sweeps_per_stage)):
            t0 = time.time()
            d_new = sweep(T6, idx, v, theta, alpha, L, shape, mask)
            delta = angular_change(d_new, T6, idx)
            # Jacobi: everything is computed from the OLD field, then swapped.
            T6[:, mask] = np.stack(
                [d_new[:, i] * d_new[:, j] for i, j, _w in _DYAD]).astype(
                    np.float32)
            v = d_new
            dt = time.time() - t0
            p95 = float(np.percentile(delta, 95))
            row = dict(sweep=int(it), median=float(np.median(delta)),
                       mean=float(delta.mean()), p95=p95,
                       n_changed=int((delta > alpha).sum()), seconds=dt,
                       n_candidates=int(n_cand))
            st["sweeps"].append(row)
            if verbose:
                print(f"  Stage theta={theta:g} alpha={alpha:g} | sweep {it} | "
                      f"median={row['median']:.2f}deg "
                      f"95pct={p95:.2f}deg | {dt:.1f}s")
            if p95 < convergence_tol:
                st["converged"] = True
                break
        log["stages"].append(st)

    return _from6(T6, shape), log


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("iterative_refine self-test -- 32^3, true direction (1,1,0)/sqrt(2)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    n = 32
    shape = (n, n, n)
    truth = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)

    # 1-2. random SIGN per voxel; the outer product must be identical anyway.
    signs = rng.choice([-1.0, 1.0], size=shape)[..., None]
    e3 = np.broadcast_to(truth, shape + (3,)) * signs
    T_clean = e3_to_outer(e3)
    spread = np.abs(T_clean - T_clean[0, 0, 0]).max()
    print(f"  sign flips leave T identical: max deviation {spread:.2e}")

    # 3. rotate each voxel by Normal(0, 15 deg) about a random axis
    ax = rng.normal(size=shape + (3,))
    ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
    ang = np.radians(rng.normal(0.0, 15.0, size=shape))[..., None]
    v0 = e3
    # Rodrigues
    noisy = (v0 * np.cos(ang)
             + np.cross(ax, v0) * np.sin(ang)
             + ax * (ax * v0).sum(-1, keepdims=True) * (1.0 - np.cos(ang)))
    noisy /= np.linalg.norm(noisy, axis=-1, keepdims=True)
    T_noisy = e3_to_outer(noisy)
    mask = np.ones(shape, bool)

    def err(T):
        e = outer_to_e3(T)
        c = np.abs(e @ truth)
        return float(np.degrees(np.arccos(np.clip(c, 0, 1)))[mask].mean())

    before = err(T_noisy)
    print(f"\n  mean angular error BEFORE: {before:.3f} deg\n")

    T_ref, log = refine_field(T_noisy, mask, L=DEFAULT_L,
                              convergence_tol=DEFAULT_TOL)
    after = err(T_ref)
    print(f"\n  mean angular error AFTER:  {after:.3f} deg   "
          f"({before:.3f} -> {after:.3f})")
    n_sweeps = sum(len(s["sweeps"]) for s in log["stages"])
    print(f"  {n_sweeps} sweeps over {len(log['stages'])} annealing stages")
    assert after < 3.0, f"refined error {after:.3f} deg is not below 3 deg"
    print("\n  PASS -- refined error is below 3 deg")
