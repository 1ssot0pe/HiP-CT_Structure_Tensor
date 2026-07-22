"""
bayes_tracto_scope.py -- Phase-0 proof-of-concept for global tractography.

Claim to demonstrate: at a fibre CROSSING the per-voxel structure tensor is
locally AMBIGUOUS (near-isotropic: it cannot say which way the fibre goes), yet a
tracker that carries a CONTINUITY prior (inertia) traverses the crossing and
correctly reconstructs the through-going fibre -- while a greedy per-voxel tracker
derails onto the other bundle. This is the core "global > local" argument for the
"too complex" bucket, in the simplest possible 2D setting.

No noise model / half-map needed: the data term here is the image itself.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12


def make_crossing(n=96, width=2.0, noise=0.15, seed=0):
    """Two Gaussian bands crossing at the centre at +/-45 deg."""
    y, x = np.indices((n, n)).astype(float)
    c = (n - 1) / 2.0
    yc, xc = y - c, x - c
    d1 = np.abs((xc - yc) / np.sqrt(2))        # distance to the 45deg line
    d2 = np.abs((xc + yc) / np.sqrt(2))        # distance to the 135deg line
    img = np.exp(-d1 ** 2 / (2 * width ** 2)) + np.exp(-d2 ** 2 / (2 * width ** 2))
    if noise:
        img = img + noise * np.random.default_rng(seed).standard_normal((n, n))
    return (img - img.min()) / (np.ptp(img) + EPS)


def structure_tensor_2d(img, sigma=1.0, rho=2.0):
    """Return (Jxx, Jxy, Jyy) fields (x=axis1, y=axis0) and the per-pixel
    coherence (l1-l2)/(l1+l2). Fibre = smaller-eigenvalue eigenvector."""
    Iy = gaussian_filter(img, sigma, order=(1, 0))
    Ix = gaussian_filter(img, sigma, order=(0, 1))
    Jxx = gaussian_filter(Ix * Ix, rho)
    Jyy = gaussian_filter(Iy * Iy, rho)
    Jxy = gaussian_filter(Ix * Iy, rho)
    R = np.sqrt(((Jxx - Jyy) / 2) ** 2 + Jxy ** 2)
    coh = 2 * R / (Jxx + Jyy + EPS)
    return Jxx, Jxy, Jyy, coh


def _fibre_dir_coh(a, b, c):
    """Fibre direction (x,y) and coherence for one 2x2 tensor [[a,b],[b,c]]."""
    theta_grad = 0.5 * np.arctan2(2 * b, a - c)
    theta_fib = theta_grad + np.pi / 2.0            # fibre perp to gradient
    R = np.sqrt(((a - c) / 2) ** 2 + b ** 2)
    coh = 2 * R / (a + c + EPS)
    return np.array([np.cos(theta_fib), np.sin(theta_fib)]), float(coh)


def track(J, seed_yx, dir0_xy, mode, steps=120, step=1.0):
    """Propagate a streamline. mode='continuity' blends the local fibre direction
    with the previous direction by (1-coherence) -> at an ambiguous (low-coh)
    junction, inertia carries it straight. mode='greedy' always snaps to the
    local fibre direction (only sign-resolved) -> derails where coherence is low."""
    Jxx, Jxy, Jyy = J
    pos = np.array(seed_yx, float)                  # (y, x)
    prev = np.array(dir0_xy, float); prev /= np.linalg.norm(prev) + EPS
    path = [pos.copy()]
    for _ in range(steps):
        cc = [[pos[0]], [pos[1]]]
        a = float(map_coordinates(Jxx, cc, order=1, mode="nearest")[0])
        b = float(map_coordinates(Jxy, cc, order=1, mode="nearest")[0])
        c = float(map_coordinates(Jyy, cc, order=1, mode="nearest")[0])
        v, coh = _fibre_dir_coh(a, b, c)
        if np.dot(v, prev) < 0:                     # sign: keep travelling forward
            v = -v
        if mode == "continuity":
            d = coh * v + (1 - coh) * prev          # inertia when ambiguous
        else:
            d = v                                   # greedy: trust the local pixel
        d /= np.linalg.norm(d) + EPS
        pos = pos + step * np.array([d[1], d[0]])   # (y,x) += (dy,dx)
        prev = d
        path.append(pos.copy())
        if not (0 <= pos[0] < Jxx.shape[0] and 0 <= pos[1] < Jxx.shape[1]):
            break
    return np.array(path)


def main(save="bayes_tracto_scope.png"):
    n = 96
    img = make_crossing(n=n, noise=0.15)
    Jxx, Jxy, Jyy, coh = structure_tensor_2d(img)
    J = (Jxx, Jxy, Jyy)
    c = (n - 1) / 2.0

    coh_centre = float(coh[int(c), int(c)])
    coh_arm = float(coh[20, 20])
    print("=" * 68)
    print("Phase-0: is the crossing centre locally AMBIGUOUS to the structure tensor?")
    print(f"  coherence at an ARM (20,20)   = {coh_arm:.3f}  (high -> one clear fibre)")
    print(f"  coherence at the CENTRE       = {coh_centre:.3f}  (low -> ambiguous)")
    print(f"  -> the per-voxel tensor {'CANNOT' if coh_centre < 0.4*coh_arm else 'can'} "
          f"resolve the fibre at the crossing.")

    # seed near one end of the 45deg bundle (band1: x=y), heading toward SE.
    # Correct behaviour = END on band1 (|x-y| small). Derailing onto band2
    # (the 135deg line x+y=n-1) = |x+y-(n-1)| small.
    seed = (14.0, 14.0); dir0 = (1.0, 1.0)
    p_cont = track(J, seed, dir0, "continuity")
    p_greedy = track(J, seed, dir0, "greedy")

    def band1_dist(p): return abs(p[1] - p[0]) / np.sqrt(2)          # to x=y
    def band2_dist(p): return abs(p[1] + p[0] - (n - 1)) / np.sqrt(2)  # to x+y=n-1

    ec, eg = p_cont[-1], p_greedy[-1]
    print("-" * 68)
    print(f"  CONTINUITY endpoint ({ec[0]:.0f},{ec[1]:.0f}): dist to band1={band1_dist(ec):.1f}, "
          f"band2={band2_dist(ec):.1f} -> on {'band1 (CORRECT, traversed)' if band1_dist(ec)<band2_dist(ec) else 'band2 (derailed)'}")
    print(f"  GREEDY     endpoint ({eg[0]:.0f},{eg[1]:.0f}): dist to band1={band1_dist(eg):.1f}, "
          f"band2={band2_dist(eg):.1f} -> on {'band1 (traversed)' if band1_dist(eg)<band2_dist(eg) else 'band2 (DERAILED onto other bundle)'}")
    verdict = ("CONTINUITY traverses the crossing (stays on band1); "
               "greedy derails onto band2"
               if band1_dist(ec) < band2_dist(ec) and band2_dist(eg) < band1_dist(eg)
               else "inconclusive on this seed/noise (see figure)")
    print(f"  => {verdict}")
    print("=" * 68)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img, cmap="gray", origin="upper")
    ax[0].set_title("crossing phantom (noisy)")
    im = ax[1].imshow(coh, cmap="magma", origin="upper", vmin=0, vmax=1)
    ax[1].plot(c, c, "c+", ms=14, mew=2)
    ax[1].set_title(f"coherence: LOW at centre ({coh_centre:.2f})\n= locally ambiguous")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[2].imshow(img, cmap="gray", origin="upper")
    ax[2].plot(p_greedy[:, 1], p_greedy[:, 0], "-", color="#eb6834", lw=2.5,
               label="greedy per-voxel (derails)")
    ax[2].plot(p_cont[:, 1], p_cont[:, 0], "-", color="#2a78d6", lw=2.5,
               label="continuity (traverses)")
    ax[2].plot(seed[1], seed[0], "wo", ms=8, mec="k", label="seed")
    ax[2].set_title("streamlines through the ambiguous centre")
    ax[2].legend(fontsize=8, loc="lower right")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Phase-0: global continuity resolves what per-voxel analysis cannot",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    main()
