"""
weight_fit.py -- fit per-streamline weights to the observed density.

The one convex piece of Stage 3
-------------------------------
Everything else in the optimisation is awkward: `theta` is non-differentiable
(streamlines terminate on a hard angular cutoff, seeds are drawn discretely),
so it can only be searched derivative-free. The tract WEIGHTS are different.
With the streamlines held fixed, each one contributes a fixed spatial pattern
and the density is linear in the weights:

    D_rec = sum_s w_s * phi_s        ->      D_rec = Phi w

so fitting them is

    min || Phi w - D_obs ||^2   subject to   w >= 0

which is a non-negative least squares problem: convex, with a unique minimum
that a real descent algorithm reaches. This is the same formulation COMMIT and
SIFT2 use to weight a tractogram against measured signal.

Why it is needed here specifically
----------------------------------
`seed_density` conflates two things it should not. Seeding more streamlines
improves COVERAGE and inflates AMPLITUDE together, and an unscaled MSE cannot
separate them: on i74, 1532 -> 24538 streams took coverage from 42% to 90% but
drove the loss from 1.21 to 16.95, because D_rec overshot the unit-mean target
2.5x into a quadratic penalty. The optimiser's rational response was to seed
FEWER tracts -- the opposite of what the density model claims.

A single fitted global scale already fixes the sign of that gradient (the same
three settings score 1.2053 / 1.1173 / 1.0865, so coverage helps monotonically).
Per-streamline weights are the full version: instead of one number correcting
the whole tractogram's amplitude, each tract is allowed its own, so a tract
that explains real density is kept and one that does not is driven to zero.
Tracts at w = 0 are the model saying, in the fit rather than by a threshold,
that this connection is not supported.

Non-negativity is not a convenience. A negative weight would be a tract that
REMOVES fibre density, which has no physical reading, and unconstrained least
squares will happily use them to cancel overshoot elsewhere.
"""

import numpy as np
import scipy.sparse as sp

EPS = 1e-12


def _splat_indices(shape, p, w):
    """Trilinear neighbours of p and their weights -- `tract_density.splat`,
    but returning the contributions instead of accumulating them, so they can
    be assembled into a matrix column."""
    z, y, x = p
    z0, y0, x0 = int(np.floor(z)), int(np.floor(y)), int(np.floor(x))
    dz, dy, dx = z - z0, y - y0, x - x0
    nz, ny, nx = shape
    out = []
    for kz in (0, 1):
        iz = z0 + kz
        if iz < 0 or iz >= nz:
            continue
        wz = dz if kz else 1.0 - dz
        for ky in (0, 1):
            iy = y0 + ky
            if iy < 0 or iy >= ny:
                continue
            wy = dy if ky else 1.0 - dy
            for kx in (0, 1):
                ix = x0 + kx
                if ix < 0 or ix >= nx:
                    continue
                wx = dx if kx else 1.0 - dx
                c = wz * wy * wx
                if c > 0:
                    out.append((iz, iy, ix, c * w))
    return out


def design_matrix(streamlines, shape, support, step=1.0):
    """Phi: (n_support_voxels, n_streamlines), sparse.

    Column s is streamline s rasterised at unit weight, restricted to the
    voxels the loss is scored on. Sparse because a tract touches a few hundred
    voxels out of a few hundred thousand -- dense would be 236133 x 6117
    floats (11 GB) for the i74 run, and is 99.9% zeros.

    Rows are ordered by `np.flatnonzero(support)`, so `d = D_obs[support]`
    lines up with Phi @ w without any further bookkeeping.

    Fully vectorised. A per-sample Python loop is unusable at the seed
    densities the optimiser now selects: ~50k streamlines x ~200 samples x 8
    trilinear neighbours is 80M iterations for ONE loss evaluation, and a
    coordinate sweep needs dozens.
    """
    support = np.asarray(support, bool)
    lut = np.full(support.size, -1, np.int64)
    lut[np.flatnonzero(support.ravel())] = np.arange(int(support.sum()))
    nz, ny, nx = shape
    n_str = len(streamlines)

    keep = [(s, np.asarray(sl, float)) for s, sl in enumerate(streamlines)
            if np.asarray(sl).ndim == 2 and np.asarray(sl).shape[0] >= 2]
    if not keep:
        return sp.csr_matrix((int(support.sum()), n_str))

    P = np.concatenate([sl for _s, sl in keep], axis=0)              # (N, 3)
    col = np.repeat([s for s, _sl in keep], [len(sl) for _s, sl in keep])

    base = np.floor(P).astype(np.int64)
    frac = P - base

    rows, cols, vals = [], [], []
    for kz in (0, 1):
        for ky in (0, 1):
            for kx in (0, 1):
                iz = base[:, 0] + kz
                iy = base[:, 1] + ky
                ix = base[:, 2] + kx
                # out-of-bounds samples are DROPPED, never clamped: clamping
                # would pile an escaped tail onto the boundary voxel and
                # invent density there (same rule as tract_density.splat).
                ok = ((iz >= 0) & (iz < nz) & (iy >= 0) & (iy < ny)
                      & (ix >= 0) & (ix < nx))
                if not ok.any():
                    continue
                wz = frac[:, 0] if kz else 1.0 - frac[:, 0]
                wy = frac[:, 1] if ky else 1.0 - frac[:, 1]
                wx = frac[:, 2] if kx else 1.0 - frac[:, 2]
                c = wz * wy * wx * float(step)
                r = lut[(iz[ok] * ny + iy[ok]) * nx + ix[ok]]
                inside = r >= 0
                if not inside.any():
                    continue
                rows.append(r[inside])
                cols.append(col[ok][inside])
                vals.append(c[ok][inside])

    n_vox = int(support.sum())
    if not rows:
        return sp.csr_matrix((n_vox, n_str))
    Phi = sp.coo_matrix((np.concatenate(vals),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n_vox, n_str)).tocsr()
    # duplicate (row, col) entries are summed by tocsr, which is what we want:
    # consecutive samples of one tract land in the same voxel and their arc
    # lengths add.
    return Phi


def fit_weights(Phi, d, l1=0.0, n_iter=200, tol=1e-8, w0=None, verbose=False):
    """Non-negative least squares by projected gradient with FISTA momentum.

        min ||Phi w - d||^2 + l1 * sum(w)      subject to  w >= 0

    `scipy.optimize.nnls` is dense-only and `lsq_linear`'s trust-region solver
    is far slower than needed here; projected gradient exploits the sparsity
    directly and the projection onto w >= 0 is just a clip.

    The step size is 1/L with L the largest eigenvalue of Phi^T Phi, estimated
    by a few power iterations -- the standard guarantee for gradient descent on
    a quadratic, so no step-size tuning enters the pipeline.

    Why l1 is not optional
    ----------------------
    Adding streamlines can essentially never INCREASE the fitted residual: more
    columns in Phi is a larger feasible set, and any tract that does not help
    takes weight zero. So the residual falls monotonically with seed density
    and "the loss got smaller" stops being evidence that the model got better
    -- with enough tracts the fit starts explaining noise. An L1 penalty prices
    each tract, which is what makes the residual interpretable again and the
    surviving tractogram sparse. This is the same role the penalty plays in
    SIFT2/COMMIT.

    Because w >= 0 the L1 term is just l1 * sum(w), whose proximal operator
    combines with the non-negativity projection into a single shifted clip:
    max(y - stepsize*l1, 0). No extra machinery.
    """
    d = np.asarray(d, float).ravel()
    n = Phi.shape[1]
    if n == 0:
        return np.zeros(0), dict(iterations=0, converged=True, residual=float(d @ d))

    # Lipschitz constant of the gradient = largest eigenvalue of Phi^T Phi
    v = np.random.default_rng(0).normal(size=n)
    v /= np.linalg.norm(v) + EPS
    for _ in range(30):
        v = Phi.T @ (Phi @ v)
        nv = np.linalg.norm(v)
        if nv < EPS:
            break
        v /= nv
    L = float(nv) if nv > EPS else 1.0
    stepsize = 1.0 / max(L, EPS)

    w = np.zeros(n) if w0 is None else np.maximum(np.asarray(w0, float), 0.0)
    y = w.copy()
    t = 1.0
    # `prev` starts as None, NOT inf: with prev = inf the convergence test
    # reads `inf <= tol * max(inf, 1)`, i.e. `inf <= inf`, which is True, so
    # the solver returns after a single iteration on every problem. A test on
    # four disjoint streamlines hides this completely -- disjoint tracts make
    # Phi^T Phi diagonal, so 1/L IS the exact Newton step and one iteration is
    # genuinely correct there.
    prev = None
    it = 0
    for it in range(1, n_iter + 1):
        grad = Phi.T @ (Phi @ y - d)
        w_new = np.maximum(y - stepsize * (grad + l1), 0.0)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
        r = Phi @ w - d
        obj = float(r @ r) + float(l1) * float(w.sum())
        if prev is not None and abs(prev - obj) <= tol * max(abs(prev), 1.0):
            prev = obj
            break
        prev = obj
        if verbose and it % 50 == 0:
            print(f"      nnls iter {it}: ||r||^2 = {obj:.6f}")

    r = Phi @ w - d
    return w, dict(iterations=it, converged=it < n_iter,
                   residual=float(r @ r), n_active=int((w > EPS).sum()),
                   n_total=n, l1=float(l1), weight_sum=float(w.sum()))
