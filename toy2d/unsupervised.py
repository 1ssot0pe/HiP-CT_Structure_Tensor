"""
unsupervised.py -- Layer 3

Two unsupervised (no-ground-truth) criteria:

(a) Lindeberg gamma-normalised response for SIGMA selection: raw
    derivative-response amplitude decays with sigma, so comparing it
    directly across scales is invalid; multiplying by sigma^2 (gamma=1
    normalisation for a quantity built from squared first derivatives)
    compensates that decay so response CAN be compared across sigma, and
    its peak marks the characteristic scale of the feature. Uses the
    LARGER structure-tensor eigenvalue (the on-fibre gradient energy) at a
    fixed minimal rho -- this criterion probes sigma only.

(b) Gradient-orthogonality residual for RHO selection:
        R(sigma,rho) = sum((grad_L . e3)^2) / sum(|grad_L|^2)
    A fibre lies along the direction of LEAST intensity change, so grad_L
    should be perpendicular to e3 wherever the estimate is correct. This is
    the SAME formula already implemented (and found NOT to track truth on
    a single-population, spatially-varying phantom) in this repo's
    orthogonality_residual.py -- see that module's "VALIDATION FINDING"
    docstring: R can trivially -> 0 as rho -> 0, because the un-integrated
    structure tensor is exactly rank-1 (S = grad L (grad L)^T pointwise)
    and its null eigenvector is orthogonal to the local gradient BY
    CONSTRUCTION, regardless of whether the direction is actually correct.
    This layer re-tests the same formula on a CROSSING phantom, where that
    3D test could not be run, to see whether crossing geometry changes the
    outcome.
"""

import numpy as np
from scipy import ndimage
from structure_tensor import structure_tensor_2d, eig_special_2d
from forward import compute_e3, crop_margin, _crop
from supervised import per_pixel_angle_deg, angular_distance_deg


def lindeberg_response(image, sigmas, rho=0.5, truncate=4.0):
    """sigma^2 * mean(largest eigenvalue of S(sigma,rho)) for each sigma.

    The Lindeberg criterion picks sigma = argmax of this array.
    """
    response = np.full(len(sigmas), np.nan)
    for i, sigma in enumerate(sigmas):
        S = structure_tensor_2d(image.astype(np.float64), sigma, rho, truncate=truncate)
        val, _vec = eig_special_2d(S)          # val[0] <= val[1], ascending
        m = crop_margin(sigma, rho, truncate)
        largest = _crop(val[1], m)
        response[i] = sigma ** 2 * np.mean(largest)
    return response


def smoothed_gradient(image, sigma, truncate=4.0):
    """Derivative-of-Gaussian gradient at scale sigma, shape (2, rows, cols).
    Same axis-order convention as structure_tensor_2d, so component i lines
    up with e3's component i.
    """
    im = image.astype(np.float64)
    Ix = ndimage.gaussian_filter(im, sigma, order=(1, 0), mode="nearest", truncate=truncate)
    Iy = ndimage.gaussian_filter(im, sigma, order=(0, 1), mode="nearest", truncate=truncate)
    return np.stack([Ix, Iy])


def orthogonality_residual(image, sigma, rho, truncate=4.0):
    """R(sigma,rho) as a single float. Separate sums, then divide (so
    strong-gradient pixels dominate, matching the 3D module's convention).
    """
    e3 = compute_e3(image, sigma, rho, truncate=truncate)
    grad = smoothed_gradient(image, sigma, truncate=truncate)
    valid = np.isfinite(e3[0]) & np.isfinite(e3[1])
    e3 = np.where(valid, e3, 0.0)
    dot = np.sum(grad * e3, axis=0)
    magsq = np.sum(grad * grad, axis=0)

    m = crop_margin(sigma, rho, truncate)
    v = _crop(valid, m)
    dot = _crop(dot, m)[v]
    magsq = _crop(magsq, m)[v]
    num = np.sum(np.square(dot))
    den = np.sum(magsq)
    return float(num / den) if den > 0 else float("nan")


def sweep_residual(image, sigmas, rhos, truncate=4.0, verbose=True):
    sigmas = np.asarray(sigmas, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    R = np.full((sigmas.size, rhos.size), np.nan)
    for i, s in enumerate(sigmas):
        for j, r in enumerate(rhos):
            R[i, j] = orthogonality_residual(image, s, r, truncate)
        if verbose:
            print(f"  sigma={s:.3g} done")
    i, j = np.unravel_index(np.nanargmin(R), R.shape)
    best = dict(sigma=float(sigmas[i]), rho=float(rhos[j]), i=int(i), j=int(j),
                value=float(R[i, j]))
    if verbose:
        print(f"[residual]   argmin R = {best['value']:.4f} "
              f"at sigma={best['sigma']:.3g}, rho={best['rho']:.3g}")
    return R, best


# ---------------------------------------------------------------------------
# Alternative (b'): decoupled residual -- test gradient at a FIXED reference
# scale, independent of the swept sigma/rho, so R can't trivially self-
# satisfy as rho -> 0 (there, e3 is only orthogonal to the SAME-scale
# gradient that built it; a gradient computed at a different, fixed scale
# has no such built-in relationship to e3).
# ---------------------------------------------------------------------------

def orthogonality_residual_decoupled(image, sigma, rho, grad_ref, truncate=4.0):
    """R using a reference gradient computed ONCE, outside the (sigma,rho)
    that build e3. `grad_ref` must already be shape (2, rows, cols)."""
    e3 = compute_e3(image, sigma, rho, truncate=truncate)
    valid = np.isfinite(e3[0]) & np.isfinite(e3[1])
    e3 = np.where(valid, e3, 0.0)
    dot = np.sum(grad_ref * e3, axis=0)
    magsq = np.sum(grad_ref * grad_ref, axis=0)

    m = crop_margin(sigma, rho, truncate)
    v = _crop(valid, m)
    dot = _crop(dot, m)[v]
    magsq = _crop(magsq, m)[v]
    num = np.sum(np.square(dot))
    den = np.sum(magsq)
    return float(num / den) if den > 0 else float("nan")


def sweep_residual_decoupled(image, sigmas, rhos, sigma_ref=None, truncate=4.0, verbose=True):
    sigmas = np.asarray(sigmas, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    sigma_ref = float(min(sigmas)) if sigma_ref is None else float(sigma_ref)
    grad_ref = smoothed_gradient(image, sigma_ref, truncate=truncate)

    R = np.full((sigmas.size, rhos.size), np.nan)
    for i, s in enumerate(sigmas):
        for j, r in enumerate(rhos):
            R[i, j] = orthogonality_residual_decoupled(image, s, r, grad_ref, truncate)
        if verbose:
            print(f"  sigma={s:.3g} done")
    i, j = np.unravel_index(np.nanargmin(R), R.shape)
    best = dict(sigma=float(sigmas[i]), rho=float(rhos[j]), i=int(i), j=int(j),
                value=float(R[i, j]))
    if verbose:
        print(f"[decoupled]  argmin R = {best['value']:.4f} "
              f"at sigma={best['sigma']:.3g}, rho={best['rho']:.3g} "
              f"(sigma_ref={sigma_ref:.3g})")
    return R, best


# ---------------------------------------------------------------------------
# Alternative (b''): rho-stability plateau -- pick rho where the recovered
# angle field is LEAST sensitive to further changes in rho. Compares
# estimates ACROSS rho rather than trusting one (sigma,rho) evaluation in
# isolation, so it cannot inherit the rank-1-at-rho->0 triviality: at small
# rho the per-pixel angle is decided by a single point's gradient and
# changes sharply as rho ticks up (high sensitivity, penalised); once rho
# is large enough to see the crossing's OTHER family, the angle drifts
# again (also high sensitivity). The interior plateau is the candidate.
# ---------------------------------------------------------------------------

def angle_field(image, sigma, rho, truncate=4.0):
    """Per-pixel dominant angle (deg) and validity mask, uncropped."""
    vec = compute_e3(image, sigma, rho, truncate=truncate)
    valid = np.isfinite(vec[0]) & np.isfinite(vec[1])
    angle = per_pixel_angle_deg(np.where(valid, vec, 0.0))
    return angle, valid


def sweep_rho_stability(image, sigmas, rhos, truncate=4.0, verbose=True):
    """Sensitivity[i,k] = mean angular change per unit log(rho) between
    rhos[k] and rhos[k+1], at sigma=sigmas[i]. Lower = more stable = better.

    Returns Sens (len(sigmas), len(rhos)-1), best dict, and rho_mid (the
    geometric-mean rho representing each transition, for plotting/argmin).
    """
    sigmas = np.asarray(sigmas, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    n_r = rhos.size - 1
    rho_mid = np.sqrt(rhos[:-1] * rhos[1:])
    Sens = np.full((sigmas.size, n_r), np.nan)

    for i, s in enumerate(sigmas):
        fields = [angle_field(image, s, r, truncate) for r in rhos]
        for k in range(n_r):
            a0, v0 = fields[k]
            a1, v1 = fields[k + 1]
            m = crop_margin(s, max(rhos[k], rhos[k + 1]), truncate)
            v = _crop(v0, m) & _crop(v1, m)
            if not v.any():
                continue
            d = angular_distance_deg(_crop(a1, m)[v], _crop(a0, m)[v])
            dlogrho = np.log(rhos[k + 1]) - np.log(rhos[k])
            Sens[i, k] = np.mean(d) / dlogrho
        if verbose:
            print(f"  sigma={s:.3g} done")

    i, k = np.unravel_index(np.nanargmin(Sens), Sens.shape)
    best = dict(sigma=float(sigmas[i]), rho=float(rho_mid[k]), i=int(i), j=int(k),
                value=float(Sens[i, k]))
    if verbose:
        print(f"[stability]  argmin sensitivity = {best['value']:.4f} deg/e-fold "
              f"at sigma={best['sigma']:.3g}, rho~{best['rho']:.3g}")
    return Sens, best, rho_mid


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from phantom import make_phantom
    from supervised import sweep_angular_error, plot_heatmap, DEFAULT_SIGMAS, DEFAULT_RHOS

    # --- crossing phantom: two generic, non-axis-aligned angles, DIFFERENT
    # spacing per family so the two periods don't share an exact common
    # period (avoids resonance-like artifacts in the integration-scale sweep)
    THETA1, THETA2 = 20.0, 110.0
    image, true_angles = make_phantom(shape=(200, 200), theta_deg=THETA1,
                                       spacing=12.0, width=1.2,
                                       theta2_deg=THETA2, spacing2=17.0, width2=1.2,
                                       noise_sigma=0.0)
    print(f"crossing phantom true angles = {true_angles}")

    # --- (a) Lindeberg sigma criterion --------------------------------------
    sigmas_1d = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
    response = lindeberg_response(image, sigmas_1d, rho=0.5)
    i_peak = int(np.argmax(response))
    sigma_peak = float(sigmas_1d[i_peak])
    print(f"[Lindeberg]  peak at sigma={sigma_peak:.3g} "
          f"(response={response[i_peak]:.4g}); phantom line width=1.2")

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(sigmas_1d, response, "o-")
    ax1.axvline(sigma_peak, color="r", ls="--", label=f"peak sigma={sigma_peak:.2g}")
    ax1.set_xlabel("sigma")
    ax1.set_ylabel("sigma^2 * mean(largest eigenvalue)")
    ax1.set_title("Lindeberg gamma-normalised response (crossing phantom)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("layer3a_lindeberg.png", dpi=120)
    print("Saved layer3a_lindeberg.png")

    # --- (b) three rho criteria vs supervised error, same base grid --------
    A, bestA = sweep_angular_error(image, true_angles, DEFAULT_SIGMAS, DEFAULT_RHOS)
    R, bestR = sweep_residual(image, DEFAULT_SIGMAS, DEFAULT_RHOS)
    Rd, bestRd = sweep_residual_decoupled(image, DEFAULT_SIGMAS, DEFAULT_RHOS)
    Sens, bestSens, rho_mid = sweep_rho_stability(image, DEFAULT_SIGMAS, DEFAULT_RHOS)

    def verdict(cand, name, grid_based=True):
        if grid_based:
            d_cells = int(max(abs(cand["i"] - bestA["i"]), abs(cand["j"] - bestA["j"])))
            agree = d_cells <= 1
            print(f"{name:10s} argmin -> sigma={cand['sigma']:.3g}, rho={cand['rho']:.3g}  "
                  f"(grid-cell sep from supervised: {d_cells}, "
                  f"{'AGREE' if agree else 'DISAGREE'})")
        else:
            d_param = float(np.hypot(cand["sigma"] - bestA["sigma"],
                                      cand["rho"] - bestA["rho"]))
            print(f"{name:10s} argmin -> sigma={cand['sigma']:.3g}, rho={cand['rho']:.3g}  "
                  f"(param-space distance from supervised: {d_param:.3g})")

    print("-" * 70)
    print(f"{'supervised':10s} argmin -> sigma={bestA['sigma']:.3g}, rho={bestA['rho']:.3g}")
    verdict(bestR, "naive R")
    verdict(bestRd, "decoupled")
    verdict(bestSens, "stability", grid_based=False)
    print("-" * 70)

    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_heatmap(axes[0][0], A, DEFAULT_SIGMAS, DEFAULT_RHOS, bestA,
                 "Supervised angular error (deg)", "degrees", fig2)
    plot_heatmap(axes[0][1], R, DEFAULT_SIGMAS, DEFAULT_RHOS, bestR,
                 "Naive orthogonality residual R(sigma,rho)",
                 "R (lower = better)", fig2)
    plot_heatmap(axes[1][0], Rd, DEFAULT_SIGMAS, DEFAULT_RHOS, bestRd,
                 "Decoupled residual (fixed reference gradient)",
                 "R (lower = better)", fig2)
    plot_heatmap(axes[1][1], Sens, DEFAULT_SIGMAS, rho_mid, bestSens,
                 "Rho-stability sensitivity (deg per e-fold of rho)",
                 "sensitivity (lower = better)", fig2)
    fig2.suptitle(f"Crossing phantom ({THETA1} deg / {THETA2} deg): "
                  f"supervised vs three unsupervised rho criteria")
    fig2.tight_layout()
    fig2.savefig("layer3b_residual_vs_supervised.png", dpi=120)
    print("Saved layer3b_residual_vs_supervised.png")

    # --- combined criterion: Lindeberg picks sigma, stability picks rho ----
    # sigma_peak (1.5) came from the Lindeberg 1D sweep above; find the
    # matching row in the (sigma,rho) grid used for the stability sweep and
    # take that ROW's argmin over rho (not the global stability argmin,
    # which was free to also move sigma away from Lindeberg's choice).
    i_row = int(np.argmin(np.abs(DEFAULT_SIGMAS - sigma_peak)))
    sigma_combined = float(DEFAULT_SIGMAS[i_row])
    k_row = int(np.nanargmin(Sens[i_row]))
    rho_combined = float(rho_mid[k_row])
    from supervised import mean_angular_error
    err_combined = mean_angular_error(image, sigma_combined, rho_combined, true_angles)
    d_param_combined = float(np.hypot(sigma_combined - bestA["sigma"],
                                       rho_combined - bestA["rho"]))

    print("-" * 70)
    print(f"combined   argmin -> sigma={sigma_combined:.3g} (Lindeberg), "
          f"rho={rho_combined:.3g} (stability row)")
    print(f"  supervised error AT combined point = {err_combined:.3f} deg "
          f"(vs true optimum {bestA['value']:.3f} deg at "
          f"sigma={bestA['sigma']:.3g}, rho={bestA['rho']:.3g}; "
          f"param-space distance {d_param_combined:.3g})")
    print("-" * 70)
