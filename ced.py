"""
ced.py -- coherence-enhancing anisotropic diffusion (Weickert) for 3D fibre
structure, as an orientation-AWARE alternative to the generic TV denoiser.

Idea. Instead of lowering variance everywhere (TV, NLM), smooth ALONG the local
fibre axis in proportion to how coherent the structure is, and almost not at all
ACROSS it. The fibre axis is the structure tensor's smallest-eigenvalue
eigenvector e3 (the least-gradient direction = the tube axis). So each step
evolves  du/dt = div(D grad u)  with a diffusion tensor

    D = alpha * I  +  beta * (e3 e3^T),      beta = (1 - alpha) * cl

where cl = (mu2 - mu3)/(mu2 + mu3) is the fibre "coherence" (high when a clear
low-gradient axis exists, ~0 when isotropic). Because both cross-fibre
directions share the same weak diffusivity alpha, the full diffusion tensor
collapses to this rank-1-plus-isotropic form -- only e3 and the eigenvalues are
needed (no per-voxel eigendecomposition).

This REINFORCES the very oriented signal the HOST-FOD measures (unlike generic
denoising, which is orientation-agnostic), so it is the denoising-family change
most likely to help orientation estimation rather than just cut noise.

Reuses structure_tensor_3d + eig_special_3d (the same primitives used
throughout). Everything is kept in AXIS order (axis0,axis1,axis2) so e3 and the
finite-difference gradients share one basis -- no x/y/z reordering.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from structure_tensor import structure_tensor_3d, eig_special_3d

EPS = 1e-12


def coherence_enhancing_diffusion(vol, n_iter=8, dt=0.12, sigma=1.0, rho=4.0,
                                  alpha=0.001, verbose=False):
    """Coherence-enhancing diffusion on a 3D volume (float, any range).

    n_iter/dt : explicit-scheme steps and time step (dt <= ~1/6 for 3D stability).
    sigma/rho : structure-tensor noise/integration scales (recomputed each step).
    alpha     : baseline isotropic diffusivity (small).
    """
    u = np.asarray(vol, np.float32).copy()
    for it in range(n_iter):
        S = structure_tensor_3d(u, sigma, rho)
        val, vec = eig_special_3d(S)                    # val desc; vec = e3 (smallest)
        cl = (val[1] - val[2]) / (val[1] + val[2] + EPS)   # fibre coherence in [0,1]
        beta = (1.0 - alpha) * np.nan_to_num(cl, nan=0.0)
        e = np.nan_to_num(vec, nan=0.0)                 # (3, *) axis-order components

        # D = alpha*I + beta * e e^T  (symmetric, 6 unique components)
        D00 = alpha + beta * e[0] * e[0]
        D11 = alpha + beta * e[1] * e[1]
        D22 = alpha + beta * e[2] * e[2]
        D01 = beta * e[0] * e[1]
        D02 = beta * e[0] * e[2]
        D12 = beta * e[1] * e[2]

        g0, g1, g2 = np.gradient(u)                     # d/axis0, d/axis1, d/axis2
        f0 = D00 * g0 + D01 * g1 + D02 * g2             # flux components
        f1 = D01 * g0 + D11 * g1 + D12 * g2
        f2 = D02 * g0 + D12 * g1 + D22 * g2
        div = (np.gradient(f0, axis=0) + np.gradient(f1, axis=1)
               + np.gradient(f2, axis=2))
        u = u + dt * div
        if verbose:
            print(f"  CED iter {it+1}/{n_iter}: mean coherence cl={np.nanmean(cl):.3f}")
    return u


def mean_coherence(vol, sigma=1.0, rho=4.0):
    """Region-mean fibre coherence cl -- a scalar to show CED increased it."""
    S = structure_tensor_3d(np.asarray(vol, np.float32), sigma, rho)
    val, _ = eig_special_3d(S)
    cl = (val[1] - val[2]) / (val[1] + val[2] + EPS)
    return float(np.nanmean(cl))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.restoration import denoise_tv_chambolle
    from selfcalib_threshold import load_interior

    Z, HALF = 100, 20
    vol = load_interior()
    slab = vol[Z - HALF:Z + HALF + 1].astype(np.float32)

    tv = denoise_tv_chambolle(slab / 255.0, weight=0.08) * 255.0
    ced = coherence_enhancing_diffusion(slab.copy(), verbose=True)

    print(f"mean fibre coherence cl:  original={mean_coherence(slab):.3f}  "
          f"TV={mean_coherence(tv):.3f}  CED={mean_coherence(ced):.3f}")

    mid = HALF
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    for a, img, t in [(ax[0], slab[mid], "original"),
                      (ax[1], tv[mid], "TV denoise (weight 0.08)"),
                      (ax[2], ced[mid], "coherence-enhancing diffusion")]:
        a.imshow(img, cmap="gray"); a.set_xticks([]); a.set_yticks([])
        a.set_title(t, fontsize=11)
    fig.suptitle("i74 slab (z=100): original vs TV vs coherence-enhancing diffusion",
                 fontsize=13)
    fig.tight_layout(); fig.savefig("ced_comparison.png", dpi=130)
    print("Saved ced_comparison.png")
