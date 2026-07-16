"""
supervised.py -- Layer 2

Supervised angular error: the "ruler" that ground-truth ONLY gives us in a
toy phantom, never on real data. Everything downstream is judged against it.

Metric design
-------------
A phantom's ground truth is one or two GLOBAL angles (the whole image is a
single spatially-uniform fibre family, or two crossing families -- see
phantom.py). Rather than collapsing the recovered e3 field to one scalar
(that's what forward.dominant_angle_deg does, and it degenerates on a
symmetric crossing: averaging e.g. -20 and +20 deg cancels to 0 deg, which
is neither true direction and would make the ruler blind to real parameter
effects), we compare e3 PER PIXEL to the NEAREST known true angle, then
average. Rationale for a crossing phantom: away from exact intersections a
local neighbourhood is generally dominated by one family's gradient, so a
correctly-behaving e3 should sit close to ONE of the two true angles;
picking the nearer one per pixel doesn't require knowing a priori which
family dominates where. If integration is too aggressive the two families
blend and e3 drifts to a direction that is far from BOTH true angles, and
average error rises -- exactly the parameter-sensitivity we want to detect.
"""

import numpy as np
from forward import compute_e3, crop_margin, _crop

DEFAULT_SIGMAS = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
DEFAULT_RHOS = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0])


def angular_distance_deg(a, b):
    """Antipodally-aware distance between angles in degrees (theta ~ theta+180)."""
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def per_pixel_angle_deg(vec):
    """Angle (deg, in [0,180)) of each e3 vector. NaN where vec is non-finite."""
    return np.mod(np.degrees(np.arctan2(vec[0], vec[1])), 180.0)


def angular_error_map(image, sigma, rho, true_angles, truncate=4.0, crop=True):
    """Per-pixel angular error (deg) to the NEAREST true angle, plus valid mask.

    Returns (err, valid), both cropped 2D arrays; err is NaN outside valid.
    """
    vec = compute_e3(image, sigma, rho, truncate=truncate)
    valid = np.isfinite(vec[0]) & np.isfinite(vec[1])
    angle = per_pixel_angle_deg(np.where(valid, vec, 0.0))
    err_per_truth = np.stack([angular_distance_deg(angle, t) for t in true_angles])
    err = np.min(err_per_truth, axis=0)
    err = np.where(valid, err, np.nan)

    if crop:
        m = crop_margin(sigma, rho, truncate)
        err = _crop(err, m)
        valid = _crop(valid, m)
    return err, valid


def mean_angular_error(image, sigma, rho, true_angles, truncate=4.0, crop=True):
    """Mean angular error (deg) over valid pixels; NaN if none valid."""
    err, valid = angular_error_map(image, sigma, rho, true_angles, truncate, crop)
    return float(np.mean(err[valid])) if valid.any() else float("nan")


def sweep_angular_error(image, true_angles, sigmas, rhos, truncate=4.0, verbose=True):
    """Grid-evaluate mean supervised angular error over sigmas x rhos.

    Returns A (len(sigmas), len(rhos)) and best dict (argmin location).
    """
    sigmas = np.asarray(sigmas, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    A = np.full((sigmas.size, rhos.size), np.nan)
    for i, s in enumerate(sigmas):
        for j, r in enumerate(rhos):
            A[i, j] = mean_angular_error(image, s, r, true_angles, truncate)
        if verbose:
            print(f"  sigma={s:.3g} done")
    i, j = np.unravel_index(np.nanargmin(A), A.shape)
    best = dict(sigma=float(sigmas[i]), rho=float(rhos[j]), i=int(i), j=int(j),
                value=float(A[i, j]))
    if verbose:
        print(f"[supervised] argmin error = {best['value']:.3f} deg "
              f"at sigma={best['sigma']:.3g}, rho={best['rho']:.3g}")
    return A, best


def plot_heatmap(ax, M, sigmas, rhos, best, title, cbar_label, fig):
    extent = [rhos[0], rhos[-1], sigmas[-1], sigmas[0]]
    im = ax.imshow(M, origin="upper", aspect="auto", extent=extent, cmap="viridis")
    ax.set_xlabel("rho")
    ax.set_ylabel("sigma")
    ax.set_title(title)
    ax.plot(best["rho"], best["sigma"], "r*", ms=16, mec="white",
            label=f"argmin ({best['sigma']:.2g}, {best['rho']:.2g})")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from phantom import make_phantom

    image, true_angles = make_phantom(shape=(200, 200), theta_deg=30.0,
                                       spacing=12.0, width=1.2,
                                       noise_sigma=0.0)
    print(f"true angle(s) = {true_angles}")

    A, best = sweep_angular_error(image, true_angles, DEFAULT_SIGMAS, DEFAULT_RHOS)

    finite = np.isfinite(A)
    print(f"error range: min={np.nanmin(A):.3f} deg, max={np.nanmax(A):.3f} deg "
          f"(should be small and fairly flat on a single-angle phantom)")

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap(ax, A, DEFAULT_SIGMAS, DEFAULT_RHOS, best,
                 "Supervised angular error (deg) -- single-angle phantom",
                 "degrees", fig)
    fig.tight_layout()
    fig.savefig("layer2_heatmap.png", dpi=120)
    print("Saved layer2_heatmap.png")
