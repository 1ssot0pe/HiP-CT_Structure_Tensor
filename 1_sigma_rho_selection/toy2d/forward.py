"""
forward.py -- Layer 1

Forward structure-tensor pipeline: image -> principal (smallest-eigenvalue)
eigenvector field e3 -> optional collapse to a single dominant angle.

RESTORED from __pycache__/forward.cpython-313.pyc (the .py source went missing
on disk; toy2d is untracked by git so it could not be recovered from history).
Signatures, docstrings and logic reconstructed from the compiled bytecode and
the Skielex conventions the rest of the package relies on.

Axis convention
---------------
structure_tensor_2d builds the gradient in the array's [axis0, axis1] basis.
eig_special_2d returns `vec` = the eigenvector of the SMALLEST eigenvalue (the
fibre direction e3); the accompanying `val` is sorted ascending, so val[0] is
the smaller and val[1] the larger eigenvalue. per_pixel_angle_deg (supervised)
reads the angle as arctan2(vec[0], vec[1]); everything downstream stays in that
same basis, so no z/y/x juggling is needed here.
"""

import numpy as np
from structure_tensor import structure_tensor_2d, eig_special_2d


def compute_e3(image, sigma, rho, truncate=4.0):
    """Structure-tensor smallest-eigenvalue eigenvector field.

    Returns vec, shape (2, rows, cols), unit vectors (or NaN at isotropic
    voxels where Skielex's normalisation divides by ~0).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        S = structure_tensor_2d(image.astype(np.float64), sigma, rho, truncate=truncate)
        _val, vec = eig_special_2d(S)
    return vec


def crop_margin(sigma, rho, truncate=4.0):
    """Kernel-radius rule for how many border pixels are edge-contaminated."""
    return int(np.ceil((2.0 / 3.0) * truncate * max(sigma, rho)))


def _crop(arr2d, m):
    """Crop `m` pixels from every edge of a 2D array."""
    if m <= 0:
        return arr2d
    return arr2d[m:arr2d.shape[0] - m, m:arr2d.shape[1] - m]


def dominant_angle_deg(vec, mask=None):
    """Collapse a (2, rows, cols) e3 field to a single dominant angle (deg),
    antipodally-aware circular mean (theta and theta+180 treated as equal).

    Uses the double-angle trick: cos(2*theta) = vec1^2 - vec0^2 and
    sin(2*theta) = 2*vec0*vec1 are single-valued under theta -> theta+180,
    so averaging them (instead of averaging theta directly) sidesteps the
    wraparound that would otherwise cancel opposite-pointing eigenvectors.

    mask : optional bool array (rows, cols), True = include. Defaults to
    finite-e3 pixels only.
    """
    if mask is None:
        mask = np.isfinite(vec[0]) & np.isfinite(vec[1])
    v0 = vec[0][mask]
    v1 = vec[1][mask]
    cos2 = np.mean(v1 ** 2 - v0 ** 2)
    sin2 = np.mean(2.0 * v0 * v1)
    return float(np.mod(0.5 * np.degrees(np.arctan2(sin2, cos2)), 180.0))


def recover_angle(image, sigma, rho, truncate=4.0, crop=True):
    """Full forward pipeline: image -> dominant angle (deg)."""
    vec = compute_e3(image, sigma, rho, truncate=truncate)
    if crop:
        m = crop_margin(sigma, rho, truncate)
        vec = np.stack([_crop(vec[0], m), _crop(vec[1], m)])
    return dominant_angle_deg(vec)


if __name__ == "__main__":
    from phantom import make_phantom

    image, true_angles = make_phantom(shape=(200, 200), theta_deg=30.0,
                                       spacing=12.0, width=1.2, noise_sigma=0.0)
    true_theta = true_angles[0]
    sigma, rho = 1.5, 3.0
    recovered = recover_angle(image, sigma, rho)
    err = min(abs(recovered - true_theta), 180.0 - abs(recovered - true_theta))
    print(f"true={true_theta:.2f} recovered={recovered:.2f} err={err:.3f} deg "
          f"(sigma={sigma}, rho={rho})")
