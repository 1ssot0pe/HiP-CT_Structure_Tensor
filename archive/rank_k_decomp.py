"""
rank_k_decomp.py -- the REAL fix for multi-fibre resolution beyond k=2: an explicit
rank-k decomposition of the directional energy field, replacing host_fod.analyse's
"one scalar field + generic sphere peak-finder" approach.

Why: analyse() builds f(u) = max(E) - E(u) and hands it to dipy's peak_directions
with FIXED thresholds tuned only for k<=2 -- confirmed (see conversation) to fail on
a properly symmetric 3-fibre crossing even at the widest angle (90 deg) and low
noise, because the combined field for 3 populations doesn't have to show 3 clean,
equally-deep, well-separated local minima at all (superposing three "valley along
its axis" bowls does not preserve the individual valleys' locations in general).

The fix: derive a closed-form single-fibre energy model, then FIT a weighted sum of
k such models directly against the observed E(u) -- a symmetric-tensor / mixture
decomposition, not a peak search.

Closed-form single-fibre model
-------------------------------
For a straight fibre along unit direction v, intensity is ~constant along v, so its
gradients live in the plane PERPENDICULAR to v (structure-tensor logic: fibre axis =
direction of LOW gradient energy). Modelling that perpendicular gradient population
as an isotropic ring, the order-2n directional energy is

    E_v(u) = K_n * sin(theta)^order,    theta = angle(u, v), order = 2n

(derivation: decompose u = cos(theta) v + sin(theta) w with w perp v; since v.g=0
for every ring gradient g, (u.g)^order = sin(theta)^order * (w.g)^order, and
averaging (w.g)^order over the ring gives a THETA-INDEPENDENT constant K_n --
so E_v depends only on the angle to v, and is exactly 0 at u=v, maximal at u perp v).

For k crossing fibres with population weights pi_i, the MIXTURE energy is

    E(u) ~= sum_i pi_i * sin(angle(u, v_i))^order

This is what we fit -- k unit directions + k non-negative amplitudes -- by nonlinear
least squares against the measured E(u) sampled on the same fixed sphere host_fod
already uses, with multi-start (including a warm start from analyse()'s own,
possibly-degenerate, peak read) to manage the non-convex optimisation.
"""

import itertools

import numpy as np
from scipy.optimize import minimize, nnls

from two_orientation_test import patch_gradients
from host_fod import directional_energy
from fod_dipy import SPHERE

EPS = 1e-12
_V = SPHERE.vertices  # (Nv, 3)
# a coarser candidate set for the exhaustive direction search (below) -- the
# fine 724-vertex sphere makes exhaustive triples too expensive; a stride
# subsample gives ~90 roughly-evenly-spread candidates, plenty for a coarse
# global search that gets locally refined afterwards anyway.
_COARSE_IDX = np.arange(0, len(_V), 8)
_V_COARSE = _V[_COARSE_IDX]


# ---------------------------------------------------------------------------
# single-fibre basis: CALIBRATED empirically, not the idealised closed form
#
# The closed-form isotropic-ring derivation (sin(theta)^order) turned out to
# be measurably wrong for our actual Gaussian-tube phantoms once gradients are
# smoothed at sigma_grad=2.0: fitting it against a real 40deg crossing, the
# optimiser found a LOWER-residual (but wrong, ~90deg-separated) solution than
# the true directions -- i.e. the model's own optimum didn't coincide with the
# true crossing at all. Measuring the actual single-fibre profile directly
# (below) showed it rises more slowly and saturates lower near theta=90 than
# the idealised formula predicts. Calibrating from real data removes this
# model-mismatch instead of trying to patch the closed form.
# ---------------------------------------------------------------------------

_PROFILE_CACHE = {}


def calibrate_profile(order=6, sigma_grad=2.0, seeds=(0, 1, 2, 3, 4), n_bins=60):
    """Measure E_v(theta) from real isolated single fibres (averaged over several
    orientations/seeds), return (theta_grid_deg, E_grid) for interpolation."""
    key = (order, sigma_grad)
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
    from crossing_shape_3d import make_single_fibre_3d
    from two_orientation_test import _central_patch
    bins = np.linspace(0, 90, n_bins + 1)
    sums = np.zeros(n_bins); counts = np.zeros(n_bins)
    for i, seed in enumerate(seeds):
        theta_deg = 15.0 + 10.0 * i                    # vary orientation across seeds
        v, tr = make_single_fibre_3d(theta_deg=theta_deg, noise_sigma=0.0, seed=seed)
        patch = _central_patch(v, half=14)
        G = patch_gradients(patch, sigma_grad)
        E = directional_energy(G, order)
        E = E / (E.max() + EPS)
        theta = np.degrees(np.arccos(np.clip(np.abs(_V @ tr[0]), 0, 1)))
        idx = np.clip(np.digitize(theta, bins) - 1, 0, n_bins - 1)
        np.add.at(sums, idx, E)
        np.add.at(counts, idx, 1)
    grid = 0.5 * (bins[:-1] + bins[1:])
    prof = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    valid = ~np.isnan(prof)
    grid, prof = grid[valid], prof[valid]
    prof = np.maximum.accumulate(prof)                  # enforce monotonic (physical: energy
                                                         # only rises moving away from the fibre axis)
    _PROFILE_CACHE[key] = (grid, prof)
    return grid, prof


def _dirs_from_angles(theta, phi):
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)


def _basis_matrix(dirs, order):
    """(Nv, k) matrix: column i = calibrated single-fibre profile evaluated at
    the angle between every sphere vertex and dirs[i]."""
    grid, prof = calibrate_profile(order=order)
    cos_t = np.clip(_V @ dirs.T, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_t))
    return np.stack([np.interp(theta_deg[:, i], grid, prof) for i in range(dirs.shape[0])], axis=1)


def _model_energy(dirs, amps, order):
    """dirs (k,3) unit vectors, amps (k,) >=0 -> model E(u) at every sphere vertex,
    using the CALIBRATED single-fibre profile (not an idealised closed form)."""
    return _basis_matrix(dirs, order) @ amps


_COARSE_BASIS_CACHE = {}


def _coarse_basis(order):
    if order not in _COARSE_BASIS_CACHE:
        _COARSE_BASIS_CACHE[order] = _basis_matrix(_V_COARSE, order)
    return _COARSE_BASIS_CACHE[order]


def _unpack(params, k):
    theta = params[0:k]
    phi = params[k:2 * k]
    amp = params[2 * k:3 * k] ** 2                     # square -> non-negative
    return _dirs_from_angles(theta, phi), amp


def _residual(params, k, order, E_obs):
    dirs, amp = _unpack(params, k)
    pred = _model_energy(dirs, amp, order)
    return pred - E_obs


def _angles_from_dir(v):
    v = v / (np.linalg.norm(v) + EPS)
    theta = np.arccos(np.clip(v[2], -1.0, 1.0))
    phi = np.arctan2(v[1], v[0])
    return theta, phi


# ---------------------------------------------------------------------------
# fit k fibres to one patch
#
# This is a BILINEAR problem: linear in the amplitudes given fixed directions
# (a plain non-negative least squares), nonlinear only in the directions
# themselves. A generic multi-start nonlinear optimiser over ALL 3k parameters
# turned out to be unreliable even for the already-solved k=2 case -- it kept
# finding lower-residual but WRONG solutions. The robust fix (standard for
# bilinear problems): do the nonlinear part by EXHAUSTIVE grid search over a
# coarse candidate direction set (using the fast NNLS solve for amplitudes at
# every candidate combination, not a slow nonlinear step), then locally
# refine the single best candidate with continuous optimisation.
# ---------------------------------------------------------------------------

def fit_rank_k(patch, k, order=6, sigma_grad=2.0, min_sep_deg=15.0):
    """Fit k single-fibre energy lobes to the patch's directional energy field.
    Returns dict(dirs, amps, resid, n_distinct, n) or None if the patch has
    too little signal."""
    G = patch_gradients(patch, sigma_grad)
    if G.shape[0] < 40:
        return None
    E_obs = directional_energy(G, order)
    E_obs = E_obs / (E_obs.max() + EPS)                # scale-free fit

    basis_all = _coarse_basis(order)                   # (Nv, Ncoarse), fixed candidates
    Ncoarse = basis_all.shape[1]

    best = None                                         # (resid_norm, combo, amp)
    for combo in itertools.combinations(range(Ncoarse), k):
        B = basis_all[:, combo]
        amp, rnorm = nnls(B, E_obs)
        if best is None or rnorm < best[0]:
            best = (rnorm, combo, amp)
    _, combo, amp0 = best
    dirs0 = _V_COARSE[list(combo)]

    # local continuous refinement from the best coarse candidate
    theta_phi = [_angles_from_dir(d) for d in dirs0]
    x0 = np.concatenate([[tp[0] for tp in theta_phi], [tp[1] for tp in theta_phi],
                        np.sqrt(np.maximum(amp0, EPS))])
    res = minimize(lambda p: 0.5 * np.sum(_residual(p, k, order, E_obs) ** 2),
                   x0, method="L-BFGS-B")
    dirs, amp = _unpack(res.x, k)

    # collapse near-duplicate directions (a genuine failure mode of mixture
    # fitting: two of the k slots can converge onto the same true fibre,
    # leaving another true fibre completely unmodelled)
    order_idx = np.argsort(-amp)
    dirs, amp = dirs[order_idx], amp[order_idx]
    keep = [0]
    for i in range(1, k):
        ang = [np.degrees(np.arccos(np.clip(abs(dirs[i] @ dirs[j]), 0, 1))) for j in keep]
        if min(ang) > min_sep_deg:
            keep.append(i)
    dirs, amp = dirs[keep], amp[keep]

    resid = float(np.sqrt(np.mean(_residual(res.x, k, order, E_obs) ** 2)))
    return dict(dirs=dirs, amps=amp, resid=resid, n_distinct=len(keep), n=G.shape[0])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _match_errors(dirs, truths):
    errs, used = [], set()
    for t in truths:
        best_e, best_i = None, None
        for i, d in enumerate(dirs):
            if i in used:
                continue
            e = np.degrees(np.arccos(np.clip(abs(d @ t), 0, 1)))
            if best_e is None or e < best_e:
                best_e, best_i = e, i
        if best_i is not None:
            used.add(best_i); errs.append(best_e)
    return errs


def _equilateral(alpha_deg, n=3):
    a = np.deg2rad(alpha_deg)
    ct = np.sqrt(max((2 * np.cos(a) + 1) / 3.0, 0.0)) if n == 3 else np.cos(a)
    st = np.sqrt(max(1 - ct * ct, 0.0))
    return np.array([[ct, st * np.cos(np.deg2rad(360 * k / n)), st * np.sin(np.deg2rad(360 * k / n))]
                     for k in range(n)])


if __name__ == "__main__":
    from crossing_shape_3d import make_single_fibre_3d, make_crossing_phantom_3d
    from two_orientation_test import _central_patch
    from model_ladder import make_multifibre

    print("=" * 78)
    print("RANK-K DECOMPOSITION -- validation")
    print("=" * 78)

    print("\n--- k=1 sanity: single fibre ---")
    for seed in (0, 1, 2):
        v, tr = make_single_fibre_3d(theta_deg=20.0, noise_sigma=0.1, seed=seed)
        patch = _central_patch(v, half=14)
        r = fit_rank_k(patch, k=1)
        err = _match_errors(r["dirs"], tr) if r else None
        print(f"  seed={seed}: err={err} resid={r['resid']:.3f}" if r else "  FAILED")

    print("\n--- k=2: crossings at various separations (noise=0.1) ---")
    for sep in (30, 40, 50, 70, 90):
        errs_all = []
        for seed in (0, 1, 2):
            v, tr = make_crossing_phantom_3d(sep_deg=sep, noise_sigma=0.1, seed=seed, spacing2=12.0)
            patch = _central_patch(v, half=14)
            r = fit_rank_k(patch, k=2)
            if r:
                errs_all += _match_errors(r["dirs"], tr)
        print(f"  sep={sep:3d}: mean_err={np.mean(errs_all):5.1f} deg  (n={len(errs_all)})")

    print("\n--- k=3: EQUILATERAL crossings (the case that broke analyse() completely) ---")
    for alpha in (90, 70, 60, 50, 40, 30):
        truths = _equilateral(alpha)
        errs_all, n_distinct = [], []
        for seed in (0, 1, 2):
            v = make_multifibre(dirs=truths, noise_sigma=0.1, seed=seed)
            patch = _central_patch(v, half=14)
            r = fit_rank_k(patch, k=3)
            if r:
                errs_all += _match_errors(r["dirs"], truths)
                n_distinct.append(r["n_distinct"])
        me = np.mean(errs_all) if errs_all else float("nan")
        print(f"  alpha={alpha:3d}: mean_err={me:5.1f} deg  n_distinct~{np.mean(n_distinct):.1f}/3  "
              f"(n={len(errs_all)})")
    print("=" * 78)
