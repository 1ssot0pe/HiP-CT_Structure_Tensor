"""
pick_sigma_rho.py

3D port of the toy2d Layer 3 criteria (see toy2d/unsupervised.py) that
validated against supervised ground truth on a crossing-fibre phantom:

  - Lindeberg gamma-normalised response picks SIGMA: peak of
        sigma^2 * mean(on-fibre-plane gradient energy)
    where "on-fibre-plane energy" is the sum of the two LARGEST
    structure-tensor eigenvalues -- the 3D analogue of the 2D toy's single
    larger eigenvalue (in 3D the plane perpendicular to the fibre carries
    two eigenvalues, not one).

  - Rho-stability plateau picks RHO: the interior minimum of how much the
    recovered fibre direction e3 changes, per unit log(rho), as rho
    increases. Same antipodally-aware idea as
    toy2d/unsupervised.py:sweep_rho_stability, generalised to 3D unit
    vectors via angle = arccos(|dot(e3_a, e3_b)|) instead of a signed 2D
    angle.

On the toy phantom this PAIR tracked supervised truth (param-space distance
1.07) where the naive/decoupled gradient-orthogonality residual did NOT
(see orthogonality_residual.py's "VALIDATION FINDING" and toy2d's Layer 3
diagnosis) -- so that residual is deliberately NOT used here.

Reuses this repo's existing Skielex-based 3D primitives (fibre_direction,
crop_margin, _crop3) from orthogonality_residual.py instead of
reimplementing them.

CAVEAT (from the toy's Layer 4): the Lindeberg sigma pick was NOT fully
noise-adaptive -- it stayed flat as noise grew while the true optimum's
sigma increased. There is no ground truth on real data to check this
against, so treat the picked (sigma, rho) as a starting point to sanity
-check visually, not a certified optimum.
"""

import os
import sys
import numpy as np
from structure_tensor import structure_tensor_3d, eig_special_3d

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root, for the two imports below
from core.orthogonality_residual import fibre_direction, crop_margin, _crop3
from core.reconstruction import iter_progress


def lindeberg_response_3d(volume, sigmas, rho=1.0, truncate=4.0, verbose=True):
    """sigma^2 * mean(sum of two LARGEST structure-tensor eigenvalues) for
    each sigma. Peak marks the characteristic cross-sectional scale (e.g.
    fibre diameter). Returns (sigmas, response) arrays."""
    sigmas = np.asarray(sigmas, dtype=float)
    response = np.full(sigmas.size, np.nan)
    for i, sigma in enumerate(iter_progress(sigmas, desc="lindeberg-3d", enable=verbose)):
        S = structure_tensor_3d(volume.astype(np.float32), sigma, rho, truncate=truncate)
        val, _vec = eig_special_3d(S)          # default order='desc': val[0] >= val[1] >= val[2]
        m = crop_margin(sigma, rho, truncate)
        plane_energy = _crop3(val[0] + val[1], m)
        response[i] = sigma ** 2 * np.mean(plane_energy)
    return sigmas, response


def _antipodal_angle_deg(e3_a, e3_b):
    """Per-voxel antipodally-aware angle (deg) between two unit-vector
    fields, each shape (3, *spatial)."""
    cos = np.clip(np.abs(np.sum(e3_a * e3_b, axis=0)), 0.0, 1.0)
    return np.degrees(np.arccos(cos))


def _crop_vec3(vec, m):
    """Crop a (3, *spatial) field by margin m using the scalar _crop3."""
    return np.stack([_crop3(vec[c], m) for c in range(3)])


def sweep_rho_stability_3d(volume, sigmas, rhos, truncate=4.0, verbose=True):
    """3D analogue of toy2d.unsupervised.sweep_rho_stability: for each
    sigma, mean antipodal angular change in e3 per unit log(rho) between
    consecutive rhos. Lower = e3 is more stable w.r.t. rho = better.

    Returns Sens (len(sigmas), len(rhos)-1), best dict, rho_mid (geometric
    mean of each consecutive rho pair, the effective rho axis).
    """
    sigmas = np.asarray(sigmas, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    n_r = rhos.size - 1
    rho_mid = np.sqrt(rhos[:-1] * rhos[1:])
    Sens = np.full((sigmas.size, n_r), np.nan)

    for i, s in enumerate(iter_progress(sigmas, desc="stability-3d", enable=verbose)):
        fields = []
        for r in rhos:
            e3 = fibre_direction(volume, s, r, truncate)
            valid = np.isfinite(e3).all(axis=0)
            fields.append((np.where(valid, e3, 0.0), valid))
        for k in range(n_r):
            e3_0, v0 = fields[k]
            e3_1, v1 = fields[k + 1]
            m = crop_margin(s, max(rhos[k], rhos[k + 1]), truncate)
            v = _crop3(v0, m) & _crop3(v1, m)
            if not v.any():
                continue
            ang = _antipodal_angle_deg(_crop_vec3(e3_0, m), _crop_vec3(e3_1, m))
            dlogrho = np.log(rhos[k + 1]) - np.log(rhos[k])
            Sens[i, k] = np.mean(ang[v]) / dlogrho

    i, k = np.unravel_index(np.nanargmin(Sens), Sens.shape)
    best = dict(sigma=float(sigmas[i]), rho=float(rho_mid[k]), i=int(i), j=int(k),
                value=float(Sens[i, k]))
    if verbose:
        print(f"[stability-3d] argmin sensitivity = {best['value']:.4f} deg/e-fold "
              f"at sigma={best['sigma']:.3g}, rho~{best['rho']:.3g}")
    return Sens, best, rho_mid


def first_local_max(sigmas, response):
    """Index of the first INTERIOR local maximum (response[i-1] < response[i]
    > response[i+1]), or None if the curve has none.

    On real (non-phantom) data, a global argmax over a wide sigma range can
    be hijacked by non-fibre, large-scale content (sample shading/curvature,
    background gradients) whose energy keeps growing with sigma long past
    the fibre's own characteristic scale -- the toy phantom has no such
    content, so this distinction never came up there. A genuine local
    maximum at a small sigma is a much safer bet for "the fibre's own
    characteristic scale" than the global endpoint.
    """
    for i in range(1, len(response) - 1):
        if response[i - 1] < response[i] > response[i + 1]:
            return i
    return None


def pick_sigma_rho(volume, sigmas_lindeberg, sigmas_grid, rhos_grid,
                    rho_ref=1.0, truncate=4.0, verbose=True, peak_mode="first_local"):
    """Full recipe: Lindeberg picks sigma (from sigmas_lindeberg, at fixed
    rho_ref), then the rho-stability ROW at the matching grid sigma picks
    rho (from sigmas_grid x rhos_grid). Returns (sigma_pick, rho_pick,
    diagnostics dict).

    peak_mode : "first_local" (recommended for real data, see
        first_local_max) or "global" (the toy-validated behaviour, only
        safe when there's no non-fibre multi-scale content to confound it).
    """
    _s, response = lindeberg_response_3d(volume, sigmas_lindeberg, rho=rho_ref,
                                          truncate=truncate, verbose=verbose)
    if peak_mode == "first_local":
        i_peak = first_local_max(sigmas_lindeberg, response)
        if i_peak is None:
            if verbose:
                print("[lindeberg-3d] WARNING: no interior local max found, "
                      "falling back to global argmax")
            i_peak = int(np.argmax(response))
    else:
        i_peak = int(np.argmax(response))
    sigma_peak = float(sigmas_lindeberg[i_peak])
    if verbose:
        print(f"[lindeberg-3d] peak ({peak_mode}) at sigma={sigma_peak:.3g}")

    Sens, _bestSens, rho_mid = sweep_rho_stability_3d(volume, sigmas_grid, rhos_grid,
                                                       truncate=truncate, verbose=verbose)
    sigmas_grid = np.asarray(sigmas_grid, dtype=float)
    i_row = int(np.argmin(np.abs(sigmas_grid - sigma_peak)))
    sigma_pick = float(sigmas_grid[i_row])
    rho_pick = float(rho_mid[int(np.nanargmin(Sens[i_row]))])

    if verbose:
        print(f"[combined]     pick -> sigma={sigma_pick:.3g}, rho={rho_pick:.3g}")

    return sigma_pick, rho_pick, dict(sigma_peak=sigma_peak, response=response,
                                       Sens=Sens, rho_mid=rho_mid)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.reconstruction import load_volume, DEFAULT_VOLUME_DIR

    # Cropped chunk for speed -- full stack is ~768^3, this repo's own
    # reconstruction.py defaults to a 256^3 crop for the same reason, but a
    # single fibre_direction() call there took ~16s and this script needs
    # ~57 of them; 128^3 (~2.5-3.5s/call) keeps the whole sweep to a few
    # minutes.
    print(f"loading cropped volume from {DEFAULT_VOLUME_DIR}")
    volume = load_volume(crop=(300, 428), dtype=np.float32)
    print(f"volume shape={volume.shape} dtype={volume.dtype} "
          f"range=[{volume.min():.1f},{volume.max():.1f}]")

    SIGMAS_LINDEBERG = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
    SIGMAS_GRID = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    RHOS_GRID = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0])

    sigma_pick, rho_pick, diag = pick_sigma_rho(
        volume, SIGMAS_LINDEBERG, SIGMAS_GRID, RHOS_GRID, rho_ref=1.0)

    print("=" * 70)
    print(f"FINAL PICK: sigma={sigma_pick:.3g}, rho={rho_pick:.3g}")
    print("=" * 70)

    # --- diagnostics: Lindeberg curve + a visual sanity check -------------
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(SIGMAS_LINDEBERG, diag["response"], "o-")
    ax1.axvline(diag["sigma_peak"], color="r", ls="--",
                label=f"peak sigma={diag['sigma_peak']:.2g}")
    ax1.set_xlabel("sigma")
    ax1.set_ylabel("sigma^2 * mean(plane energy)")
    ax1.set_title("Lindeberg response (real volume, cropped chunk)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("real_lindeberg_response.png", dpi=120)
    print("Saved real_lindeberg_response.png")

    # Visual sanity check: RGB-encoded e3 orientation map on the middle
    # slice at the picked (sigma, rho) -- direction, not correctness (no
    # ground truth on real data), but incoherent/noisy-looking orientation
    # would be a red flag for the pick.
    e3 = fibre_direction(volume, sigma_pick, rho_pick)
    z_mid = e3.shape[1] // 2
    rgb = np.abs(e3[:, z_mid]).transpose(1, 2, 0)   # (y,x,3) in [0,1]

    fig2, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(volume[z_mid], cmap="gray")
    axes[0].set_title(f"raw volume, slice z={z_mid}")
    axes[1].imshow(rgb)
    axes[1].set_title(f"|e3| RGB orientation, sigma={sigma_pick:.2g}, rho={rho_pick:.2g}")
    fig2.tight_layout()
    fig2.savefig("real_orientation_sanity_check.png", dpi=120)
    print("Saved real_orientation_sanity_check.png")
