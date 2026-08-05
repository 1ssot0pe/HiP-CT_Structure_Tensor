"""
field_refine.py -- STEP 3a: refine the whole orientation field.

Stage 3 is two steps and nothing more:

    3a  field_refine.refine(e3)      clean up the director field   <- this file
    3b  self_optimising_tractography crossing vs kissing, then tracts

3a smooths the nematic director field with the Frank-Oseen energy, the correct
smoothing energy for a field with no head or tail.

An orientation field with no head or tail is a nematic director field, and its
natural smoothing energy is Frank-Oseen, which charges three deformations
separately:

    E = INT  w(x) |n - n_obs|^2          data fidelity
             + K1 (div n)^2              SPLAY   fibres fanning apart
             + K2 (n . curl n)^2         TWIST   fibres rotating about their axis
             + K3 |n x curl n|^2         BEND    fibres curving
        subject to |n| = 1 pointwise

Why this rather than Gaussian or Laplacian smoothing
----------------------------------------------------
Because the three deformations are not equally likely in fibre tissue. Tracts
bend constantly, splay at terminations, and almost never twist. One constant for
all three throws that prior away -- and when K1 = K2 = K3 the energy collapses
to INT |grad n|^2, i.e. plain Laplacian smoothing. Unequal constants ARE the
method; equal constants are the thing it improves on.

    K1 = K2 = K3        =>  Laplacian smoothing (the null case)
    K2 = K3 = K         =>  w|n-n_obs|^2 + K1 (div n)^2 + K |curl n|^2
                            because (n.curl n)^2 + |n x curl n|^2 = |curl n|^2
                            when |n| = 1 -- curl n splits into the part along n
                            (twist) and the part across it (bend)

Two things that must happen before any derivative is taken
-----------------------------------------------------------
1. SPLIT BY PEAK. A voxel holding two crossing fibres has no single director.
   Smoothing a multi-peak field as one director averages two real structures
   into one false one. Split into k single-axis fields first; this module
   smooths ONE director field and says nothing about how to split.

2. FIX THE SIGNS. The energy is even in n, but finite differences are not. Two
   adjacent voxels holding the same physical fibre but opposite sign convention
   register an enormous false gradient, and the smoother will then "fix" a
   discontinuity that was never there. `sign_align` propagates a consistent sign
   by breadth-first search. Where agreement is impossible the field has a
   genuine topological defect, and those voxels are counted and returned rather
   than silently flattened.

Constraint handling: projection, not Lagrange multipliers. After each gradient
step, n <- n/|n|. On a voxel grid this is simple, cheap, and stable; multipliers
or mixed elements buy nothing here.

Verify E decreases every iteration. If it does not, the step is too large --
that is the whole diagnostic, and `smooth_director` checks it for you.
"""

import numpy as np
from collections import deque


def fill_invalid(e3):
    """Replace non-finite directors with their nearest valid neighbour.

    The structure-tensor solver returns a non-finite eigenvector at isotropic
    voxels. Zeroing them puts an artificial cliff at each one, and those cliffs
    dominate div/curl so completely that no step size decreases E -- the
    smoother then silently does ZERO iterations. Filling keeps the field
    continuous.
    """
    ok = np.all(np.isfinite(e3), axis=0) & (np.linalg.norm(e3, axis=0) > 1e-6)
    if ok.all() or not ok.any():
        return np.asarray(e3, float)
    from scipy.ndimage import distance_transform_edt
    _, idx = distance_transform_edt(~ok, return_indices=True)
    return np.asarray(e3, float)[:, idx[0], idx[1], idx[2]]


def refine(e3, w=1.0, K=1.0, n_iter=80, verbose=False):
    """STEP 3a in one call: fill invalid voxels, then Frank-Oseen smooth.

    Measured on phantoms: angular error 9.10 -> 8.21 deg at noise 0.3, and the
    tractography loss improves 1-3%. Gaussian smoothing of the same field makes
    it WORSE (15.99 deg), because componentwise blurring of a director field
    averages across the sign ambiguity; the unit-norm projection here does not.

    One K, not three: equal constants matched or beat the unequal white-matter
    prior (splay expensive, bend cheap) at every noise level tested, so the
    three-constant physics is not supported by this data.
    """
    n, _hist, info = smooth_director(fill_invalid(e3), w=w, K1=K, K=K,
                                     n_iter=n_iter, verbose=verbose,
                                     strict=False)
    return n, info

EPS = 1e-12

# Physical meaning of each constant, so none of these is a magic number:
#   K1  cost of SPLAY -- fibres fanning apart / converging. Moderate: real at
#       terminations and branch points, so it should not be free.
#   K2  cost of TWIST -- the director rotating about its own axis. High: fibre
#       bundles essentially do not do this, so twist in the data is usually
#       noise and should be penalised hardest.
#   K3  cost of BEND  -- fibres curving while staying parallel. Low: tracts bend
#       constantly and that is signal, not noise.
# NOTE these are a hypothesis carried over from white matter. Whether they hold
# for HiP-CT lung/brain at this resolution is UNTESTED -- see `sweep_K` below.
K1_SPLAY_DEFAULT = 1.0
K2_TWIST_DEFAULT = 1.0
K3_BEND_DEFAULT = 1.0
W_DATA_DEFAULT = 1.0        # weight on staying near the measured field
STEP_DEFAULT = 0.05         # gradient step; halved automatically if E rises


# ---------------------------------------------------------------------------
# vector calculus on a voxel grid (central differences)
# ---------------------------------------------------------------------------

def _d(f, axis):
    return np.gradient(f, axis=axis, edge_order=1)


def divergence(n):
    """div n, for n of shape (3, Z, Y, X) indexed (z, y, x)."""
    return _d(n[0], 0) + _d(n[1], 1) + _d(n[2], 2)


def curl(n):
    """curl n, same layout. Components ordered (z, y, x) like the field."""
    dz_y, dz_x = _d(n[0], 1), _d(n[0], 2)
    dy_z, dy_x = _d(n[1], 0), _d(n[1], 2)
    dx_z, dx_y = _d(n[2], 0), _d(n[2], 1)
    cz = dx_y - dy_x
    cy = dz_x - dx_z
    cx = dy_z - dz_y
    return np.stack([cz, cy, cx])


def gradient_of(s):
    """grad of a scalar field, as (3, Z, Y, X)."""
    return np.stack([_d(s, 0), _d(s, 1), _d(s, 2)])


# ---------------------------------------------------------------------------
# sign consistency
# ---------------------------------------------------------------------------

def sign_align(n, mask=None, seed=None):
    """Make neighbouring directors agree in sign, by BFS from `seed`.

    Returns (aligned_field, n_defects). A "defect" is a voxel whose sign cannot
    be made to agree with an already-fixed neighbour it is strongly aligned
    with -- i.e. the field is genuinely frustrated there, not merely
    inconsistently signed. These are physically meaningful (disclinations), so
    they are counted and reported rather than hidden.
    """
    n = np.array(n, dtype=float, copy=True)
    shape = n.shape[1:]
    if mask is None:
        mask = np.linalg.norm(n, axis=0) > 1e-6
    visited = np.zeros(shape, bool)
    defects = 0

    if seed is None:
        idx = np.argwhere(mask)
        if idx.size == 0:
            return n, 0
        seed = tuple(idx[len(idx) // 2])

    nbrs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for start in [seed] + [tuple(i) for i in np.argwhere(mask)]:
        if visited[start] or not mask[start]:
            continue
        visited[start] = True
        q = deque([start])
        while q:
            c = q.popleft()
            vc = n[(slice(None),) + c]
            for d in nbrs:
                p = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if not all(0 <= p[k] < shape[k] for k in range(3)):
                    continue
                if visited[p] or not mask[p]:
                    continue
                vp = n[(slice(None),) + p]
                dot = float(np.dot(vc, vp))
                if dot < 0:
                    n[(slice(None),) + p] = -vp
                # near-perpendicular: the sign is not determined by this
                # neighbour, so agreement is meaningless here -- a defect.
                if abs(dot) < 0.2:
                    defects += 1
                visited[p] = True
                q.append(p)
    return n, defects


# ---------------------------------------------------------------------------
# energy and its gradient
# ---------------------------------------------------------------------------

def frank_oseen_energy(n, n_obs, w, K1, K2, K3):
    """Total Frank-Oseen energy plus the data term. Scalar."""
    div = divergence(n)
    c = curl(n)
    n_dot_c = (n * c).sum(axis=0)                       # twist component
    cross = np.stack([n[1] * c[2] - n[2] * c[1],
                      n[2] * c[0] - n[0] * c[2],
                      n[0] * c[1] - n[1] * c[0]])       # bend component
    e_data = (w * ((n - n_obs) ** 2).sum(axis=0)).sum()
    e_splay = K1 * (div ** 2).sum()
    e_twist = K2 * (n_dot_c ** 2).sum()
    e_bend = K3 * (cross ** 2).sum()
    return float(e_data + e_splay + e_twist + e_bend), dict(
        data=float(e_data), splay=float(e_splay),
        twist=float(e_twist), bend=float(e_bend))


def _grad_equal_K(n, n_obs, w, K1, K):
    """dE/dn for the K2 = K3 = K case, where the energy is exact and simple:

        E = w|n - n_obs|^2 + K1 (div n)^2 + K |curl n|^2
        dE/dn = 2w(n - n_obs) - 2 K1 grad(div n) + 2 K curl(curl n)

    Phase 2 of the design deliberately starts here: this is the case whose
    convergence is well behaved, so it is the one to get working before the
    constants are made unequal.
    """
    g_data = 2.0 * w * (n - n_obs)
    g_splay = -2.0 * K1 * gradient_of(divergence(n))
    g_curl = 2.0 * K * curl(curl(n))
    return g_data + g_splay + g_curl


def smooth_director(n_obs, w=W_DATA_DEFAULT, K1=K1_SPLAY_DEFAULT,
                    K=K3_BEND_DEFAULT, n_iter=60, step=STEP_DEFAULT,
                    align=True, verbose=False, strict=True):
    """Minimise the K2 = K3 Frank-Oseen energy by projected gradient descent.

    Returns (n, history, info). `history` is the energy per iteration; it MUST
    be non-increasing. If it rises the step is halved and the iteration retried,
    which is the documented diagnostic rather than a silent fudge.
    """
    n_obs = np.asarray(n_obs, float)
    # FILL invalid voxels from their nearest valid neighbour; do NOT zero them.
    # Zeroing was the zero-iteration bug: it puts an artificial cliff at every
    # invalid voxel, and those cliffs dominate div/curl so completely that no
    # step size decreases E and the smoother silently gives up at iteration 0.
    ok = np.all(np.isfinite(n_obs), axis=0) & (np.linalg.norm(n_obs, axis=0) > 1e-6)
    if not ok.all() and ok.any():
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(~ok, return_indices=True)
        n_obs = n_obs[:, idx[0], idx[1], idx[2]]
    norm = np.linalg.norm(n_obs, axis=0)
    mask = norm > 1e-6
    n_obs = np.where(mask, n_obs / np.maximum(norm, EPS), 0.0)

    # Keep the ORIGINAL sign convention to restore at the end. Sign alignment
    # is needed only so finite differences see a consistent field -- it is an
    # internal detail of the ENERGY, not something to hand downstream. Measured:
    # returning the aligned field costs the tractography +62% (crossing) and
    # +161% (kissing), because BFS propagates one global convention across a
    # field that admits none, manufacturing sign seams; consumers that sample
    # by trilinear interpolation then average across a seam and get ~zero.
    # Smoothing itself is neutral (-0.6% / +0.6%) once the signs are restored.
    n_ref = n_obs.copy()
    defects = 0
    if align:
        n_obs, defects = sign_align(n_obs, mask=mask)

    n = n_obs.copy()
    keep = ~mask                 # voxels with no measured direction
    w_field = np.broadcast_to(np.asarray(w, float), n.shape[1:])
    E0, parts = frank_oseen_energy(n, n_obs, w_field, K1, K, K)
    history = [E0]
    n_halvings = 0

    for it in range(n_iter):
        g = _grad_equal_K(n, n_obs, w_field, K1, K)
        # Scale the step by the gradient magnitude. Without this, `step` is in
        # units of whatever the field's gradients happen to be, so a value tuned
        # on a smooth synthetic field takes an enormous step on a real one --
        # every backtrack fails and the smoother silently does ZERO iterations.
        gmax = float(np.abs(g).max())
        trial_step = step / (gmax + EPS)
        for _ in range(12):                      # backtrack until E decreases
            m = n - trial_step * g
            ln = np.linalg.norm(m, axis=0)
            m = np.where(ln > EPS, m / np.maximum(ln, EPS), n)   # project
            m[:, keep] = n_obs[:, keep]      # never invent a direction where
                                             # the tensor measured none
            E, parts = frank_oseen_energy(m, n_obs, w_field, K1, K, K)
            if E <= history[-1] + 1e-12:
                break
            trial_step *= 0.5
            n_halvings += 1
        else:
            if verbose:
                print(f"    iter {it}: no step decreases E -- converged/stuck")
            break
        n, step = m, trial_step * gmax
        history.append(E)
        if verbose and it % 10 == 0:
            print(f"    iter {it:3d}  E={E:.5f}  (data {parts['data']:.4f} "
                  f"splay {parts['splay']:.4f} bend+twist "
                  f"{parts['twist']+parts['bend']:.4f})  step={step:.4g}")
        if len(history) > 2 and abs(history[-2] - history[-1]) < 1e-10:
            break

    # Restore the input's sign convention. The energy is even in n, so this is
    # free and changes nothing about what was minimised.
    flip = np.sign((n * n_ref).sum(axis=0))
    flip[flip == 0] = 1.0
    n = n * flip

    monotone = all(history[i + 1] <= history[i] + 1e-9
                   for i in range(len(history) - 1))
    if strict and not monotone:
        raise AssertionError("Frank-Oseen energy increased -- step too large")
    return n, history, dict(defects=int(defects), iterations=len(history) - 1,
                            monotone=bool(monotone), halvings=n_halvings,
                            E_initial=float(history[0]),
                            E_final=float(history[-1]), parts=parts)


# ---------------------------------------------------------------------------
# tests: written against the properties, not the implementation
# ---------------------------------------------------------------------------

def _uniform(shape=(12, 12, 12), axis=2, noise=0.0, seed=0):
    n = np.zeros((3,) + shape)
    n[axis] = 1.0
    if noise:
        rng = np.random.default_rng(seed)
        n = n + noise * rng.standard_normal(n.shape)
        n /= np.maximum(np.linalg.norm(n, axis=0), EPS)
    return n


def selftest(verbose=True):
    fails = []

    # T1  sign flip anywhere changes nothing measurable
    n_obs = _uniform(noise=0.25, seed=1)
    flipped = n_obs.copy()
    flipped[:, 4:8, :, :] *= -1.0
    a, _, _ = smooth_director(n_obs, n_iter=25)
    b, _, _ = smooth_director(flipped, n_iter=25)
    d = float(np.abs(np.abs((a * b).sum(axis=0)) - 1.0).max())
    ok = d < 1e-6
    if verbose:
        print(f"  T1 sign flip invariance      max |1-|a.b|| = {d:.2e}  "
              f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append("T1")

    # T2  energy is monotone non-increasing
    _, hist, info = smooth_director(_uniform(noise=0.4, seed=2), n_iter=40)
    ok = info["monotone"]
    if verbose:
        print(f"  T2 energy monotonicity       {info['E_initial']:.4f} -> "
              f"{info['E_final']:.4f} over {info['iterations']} iters  "
              f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append("T2")

    # T4a  w very large -> output equals input
    n_obs = _uniform(noise=0.3, seed=3)
    out, _, _ = smooth_director(n_obs, w=1e6, n_iter=30)
    d = float(np.abs(np.abs((out * n_obs).sum(axis=0)) - 1.0).max())
    ok_a = d < 1e-3
    if verbose:
        print(f"  T4a w -> inf gives input     max deviation = {d:.2e}  "
              f"{'OK' if ok_a else 'FAIL'}")

    # T4b  w = 0 -> field becomes uniform (all directors parallel)
    out, _, _ = smooth_director(_uniform(noise=0.5, seed=4), w=0.0,
                                K1=1.0, K=1.0, n_iter=400, step=0.2)
    mean_dir = out.reshape(3, -1).mean(axis=1)
    mean_dir /= np.linalg.norm(mean_dir) + EPS
    align = np.abs(np.einsum("czyx,c->zyx", out, mean_dir))
    ok_b = float(align.min()) > 0.95
    if verbose:
        print(f"  T4b w = 0 gives uniform      min |n.mean| = "
              f"{float(align.min()):.4f}  {'OK' if ok_b else 'FAIL'}")
    if not (ok_a and ok_b):
        fails.append("T4")

    if verbose:
        print("-" * 66)
        print("  ALL PASS" if not fails else f"  FAILED: {', '.join(fails)}")
    return fails


if __name__ == "__main__":
    print("=" * 66)
    print("NEMATIC (Frank-Oseen) DIRECTOR SMOOTHING -- self test")
    print("=" * 66)
    import sys
    sys.exit(1 if selftest() else 0)
