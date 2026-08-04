"""
two_orientation_test.py

Does a TWO-orientation model explain the real i74 high-c_s voxels much better
than ONE? That is the direct "do we actually need a multi-fixel model / is
single-e3 the wrong model" test.

Per local patch, from the fine-scale gradient vectors g_i (weighted by |g|^2):
  one-fibre residual  R1 = lambda_min / sum(lambda)          (= c_s / 3)
      energy the best SINGLE perpendicular-plane model leaves unexplained;
      ~0 for a clean single fibre (gradients fill one plane), high for a
      crossing OR disorganised tissue.
  two-fibre residual  R2 = min_{a1,a2} sum_i min_k (g_i . a_k)^2 / sum_i |g_i|^2
      fit TWO fibre axes, assign each gradient to the axis it is most
      perpendicular to; ~0 if two orientations really explain the gradients.

Discriminator (only meaningful where R1 is already high -- the voxels in
question):
  R1 high & R2 LOW   -> genuine two-fibre CROSSING  (single-e3 is wrong here)
  R1 high & R2 HIGH  -> DISORGANISED / noise        (neither model fits)
  gain = 1 - R2/R1   -> near 1 for a crossing, near 0 for disorganised.

Validated on synthetic (single / crossing / noise) BEFORE being applied to real
high-c_s vs low-c_s voxels. Reuses the 3D phantoms + sphericity field.
"""

import numpy as np
from scipy import ndimage

from crossing_shape_3d import (make_crossing_phantom_3d, make_single_fibre_3d,
                               _noise_volume)
# NOTE: `from selfcalib_threshold import load_interior, sphericity_field` used to
# sit here. Neither name was referenced anywhere in this file. Because host_fod
# imports this module, that one dead line pulled selfcalib_threshold -- and
# through it noise_floor and reconstruction -- into every pipeline run.

EPS = 1e-12


# ---------------------------------------------------------------------------
# Patch gradient analysis
# ---------------------------------------------------------------------------

def patch_gradients(patch, sigma_grad=2.0, mag_frac=0.1):
    """Fine-scale gradient vectors of a patch, shape (M, 3), keeping only voxels
    whose |grad| exceeds mag_frac * max|grad| (drop flat, info-less voxels)."""
    p = patch.astype(np.float32)
    gz = ndimage.gaussian_filter(p, sigma_grad, order=(1, 0, 0), mode="nearest")
    gy = ndimage.gaussian_filter(p, sigma_grad, order=(0, 1, 0), mode="nearest")
    gx = ndimage.gaussian_filter(p, sigma_grad, order=(0, 0, 1), mode="nearest")
    G = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)   # (N,3)
    mag = np.linalg.norm(G, axis=1)
    keep = mag > mag_frac * (mag.max() + EPS)
    return G[keep]


def residual_one(G):
    """R1 = lambda_min / sum(lambda) of the gradient structure tensor."""
    T = G.T @ G
    lam = np.linalg.eigvalsh(T)                 # ascending, >=0
    tot = lam.sum()
    return float(lam[0] / (tot + EPS)), float(tot)


def residual_two(G, n_restarts=5, n_iter=15, seed=0):
    """R2 = min over two fibre axes of the along-axis gradient energy fraction.

    Two-means-like: assign each gradient to the axis it is most perpendicular to
    (smallest (g.a)^2), update each axis to the smallest-eigenvalue eigenvector
    of its cluster's gradient tensor (the direction most orthogonal to it)."""
    tot = float(np.sum(G * G))
    if tot <= 0 or G.shape[0] < 6:
        return 0.0
    T = G.T @ G
    _lam, vec = np.linalg.eigh(T)               # ascending
    rng = np.random.default_rng(seed)
    inits = [(vec[:, 0], vec[:, 1])]            # two smallest eigenvectors
    for _ in range(n_restarts - 1):
        inits.append((rng.standard_normal(3), rng.standard_normal(3)))

    best = np.inf
    for a1, a2 in inits:
        a1 = a1 / (np.linalg.norm(a1) + EPS); a2 = a2 / (np.linalg.norm(a2) + EPS)
        for _ in range(n_iter):
            p1 = (G @ a1) ** 2; p2 = (G @ a2) ** 2
            c1 = p1 <= p2
            for aidx, mask in ((0, c1), (1, ~c1)):
                Gk = G[mask]
                if Gk.shape[0] < 3:
                    ak = rng.standard_normal(3)
                else:
                    _l, v = np.linalg.eigh(Gk.T @ Gk)
                    ak = v[:, 0]
                if aidx == 0:
                    a1 = ak / (np.linalg.norm(ak) + EPS)
                else:
                    a2 = ak / (np.linalg.norm(ak) + EPS)
        obj = float(np.sum(np.minimum((G @ a1) ** 2, (G @ a2) ** 2)))
        best = min(best, obj)
    return best / (tot + EPS)


def analyse_patch(patch, sigma_grad=2.0):
    G = patch_gradients(patch, sigma_grad)
    if G.shape[0] < 20:
        return None
    R1, _tot = residual_one(G)
    R2 = residual_two(G)
    gain = 1.0 - R2 / (R1 + EPS)
    return dict(R1=R1, R2=R2, gain=gain, n=G.shape[0])


# ---------------------------------------------------------------------------
# Synthetic validation
# ---------------------------------------------------------------------------

def _central_patch(vol, half=12):
    c = [s // 2 for s in vol.shape]
    sl = tuple(slice(ci - half, ci + half + 1) for ci in c)
    return vol[sl]


def validate_synthetic(noise_sigma=0.3, seeds=(0, 1, 2)):
    print("=" * 70)
    print("SYNTHETIC VALIDATION -- does the two-fibre fit behave as designed?")
    print(f"  {'case':>16} | {'R1':>6} {'R2':>6} {'gain':>6}")
    cases = {}
    for seed in seeds:
        specs = [("single", None), ("cross30", 30), ("cross50", 50),
                 ("cross70", 70), ("cross90", 90), ("noise", None)]
        for name, sep in specs:
            if name == "single":
                vol, _ = make_single_fibre_3d(theta_deg=20.0, noise_sigma=noise_sigma, seed=seed)
            elif name == "noise":
                vol = _noise_volume((96, 96, 96), seed=seed)
            else:
                vol, _ = make_crossing_phantom_3d(sep_deg=sep, noise_sigma=noise_sigma, seed=seed)
            r = analyse_patch(_central_patch(vol))
            cases.setdefault(name, []).append(r)
    summ = {}
    for name, rs in cases.items():
        R1 = np.mean([r["R1"] for r in rs]); R2 = np.mean([r["R2"] for r in rs])
        gain = np.mean([r["gain"] for r in rs])
        summ[name] = (R1, R2, gain)
        print(f"  {name:>16} | {R1:6.3f} {R2:6.3f} {gain:6.3f}")
    # gate: crossings must show big gain (two-fibre helps); noise must not.
    cross_gain = np.mean([summ[k][2] for k in ("cross50", "cross70", "cross90")])
    noise_gain = summ["noise"][2]
    passed = cross_gain > 0.5 and noise_gain < cross_gain - 0.2
    print(f"  crossing mean gain={cross_gain:.3f}  vs noise gain={noise_gain:.3f}")
    print(f"  GATE {'PASS' if passed else 'FAIL'}: two-fibre fit "
          f"{'separates crossings from noise' if passed else 'does NOT separate'}")
    print("=" * 70)
    return summ, passed


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if stage == "validate":
        validate_synthetic()
