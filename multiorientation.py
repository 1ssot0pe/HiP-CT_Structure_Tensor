"""
multiorientation.py -- proper local multi-orientation model with STABILITY.

The failed residual test (two_orientation_test.py) showed a second axis always
lowers the fit residual, most on noise -- so "does the residual drop" cannot
tell a real crossing from disorganised tissue. The fix: judge the RECOVERED
AXES, not the residual.

Model. A straight tube's gradients fill the plane perpendicular to its axis, so
on the sphere the gradient directions form a GIRDLE (great circle) whose POLE is
the fibre axis. One fibre -> one girdle; a crossing -> two girdles. We fit two
girdle-poles (assign each gradient to the pole it is most perpendicular to,
update the pole to that cluster's least-gradient direction), and then ask the
question the residual could not:

  Are the two recovered axes STABLE under resampling?
    genuine crossing  -> two axes reproduce every bootstrap  (low dispersion)
    disorganised/noise-> axes scatter every bootstrap         (high dispersion)
  Are they SEPARATED?
    one fibre         -> the two poles collapse together      (small sep)
    crossing          -> poles sit at the true crossing angle (sep ~ true)

So per patch we report: R1 (single-plane residual = c_s/3, "is one plane
enough"), sep (are there two distinct axes), disp (are those two axes stable).
Classification: R1 low -> single fibre; R1 high & stable & separated ->
CROSSING; R1 high & unstable -> DISORGANISED.

Reuses patch_gradients (two_orientation_test), the 3D phantoms, and the
sphericity field. Validated on synthetic before touching real data.
"""

import numpy as np

from two_orientation_test import patch_gradients, _central_patch
from crossing_shape_3d import (make_crossing_phantom_3d, make_single_fibre_3d,
                               _noise_volume)
from selfcalib_threshold import load_interior, sphericity_field

EPS = 1e-12


def _unit(v):
    return v / (np.linalg.norm(v) + EPS)


def _ang(a, b):
    """Antipodal angle (deg) between unit axes."""
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def _fit_two_poles(G, init, n_iter=12, rng=None):
    """Girdle two-means: assign each gradient to the pole it is most
    perpendicular to, update pole = least-gradient direction of its cluster."""
    a1, a2 = _unit(init[0]), _unit(init[1])
    for _ in range(n_iter):
        c1 = (G @ a1) ** 2 <= (G @ a2) ** 2
        for idx, mask in ((0, c1), (1, ~c1)):
            Gk = G[mask]
            if Gk.shape[0] < 3:
                ak = rng.standard_normal(3) if rng is not None else a1 + a2
            else:
                _l, v = np.linalg.eigh(Gk.T @ Gk)   # smallest eigvec = pole
                ak = v[:, 0]
            if idx == 0:
                a1 = _unit(ak)
            else:
                a2 = _unit(ak)
    return a1, a2


def _antipodal_2means(axes, n_iter=25):
    """Cluster antipodal axes (K,3) into two groups; return the two mean axes,
    the mean within-cluster angular dispersion (deg), and their separation."""
    # init: axes[0] and the axis most different from it
    c1 = axes[0]
    c2 = axes[int(np.argmax([1 - (c1 @ a) ** 2 for a in axes]))]
    for _ in range(n_iter):
        m = (1 - (axes @ c1) ** 2) <= (1 - (axes @ c2) ** 2)
        for grp, mask in ((0, m), (1, ~m)):
            Ak = axes[mask]
            if Ak.shape[0] == 0:
                continue
            _l, v = np.linalg.eigh(Ak.T @ Ak)       # largest eigvec = mean axis
            c = v[:, -1]
            if grp == 0:
                c1 = c
            else:
                c2 = c
    d1 = np.degrees(np.arccos(np.clip(np.abs(axes @ c1), 0, 1)))
    d2 = np.degrees(np.arccos(np.clip(np.abs(axes @ c2), 0, 1)))
    m = d1 <= d2
    disp = float(np.mean(np.where(m, d1, d2)))
    return c1, c2, disp, _ang(c1, c2)


def analyse_patch(patch, sigma_grad=2.0, n_boot=15, seed=0):
    """Return dict(R1, sep, disp, n) or None if too few gradients."""
    G = patch_gradients(patch, sigma_grad)
    if G.shape[0] < 40:
        return None
    T = G.T @ G
    lam, vec = np.linalg.eigh(T)                     # ascending
    R1 = float(lam[0] / (lam.sum() + EPS))           # single-plane residual (c_s/3)
    init = (vec[:, 0], vec[:, 1])                    # same init every bootstrap
    rng = np.random.default_rng(seed)
    N = G.shape[0]
    poles = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)                  # bootstrap resample
        a1, a2 = _fit_two_poles(G[idx], init, rng=rng)
        poles.append(a1); poles.append(a2)
    _c1, _c2, disp, sep = _antipodal_2means(np.array(poles))
    return dict(R1=R1, sep=sep, disp=disp, n=N)


def classify(res, r1_hi=0.10, disp_stable=18.0, sep_min=20.0):
    """Label a patch from its (R1, sep, disp)."""
    if res is None:
        return "n/a"
    if res["R1"] < r1_hi:
        return "single"
    if res["sep"] >= sep_min and res["disp"] <= disp_stable:
        return "crossing"
    return "disorganised"


# ---------------------------------------------------------------------------
# Synthetic validation: does STABILITY separate crossings from noise?
# ---------------------------------------------------------------------------

def validate(noise_sigma=0.3, seeds=(0, 1, 2)):
    print("=" * 74)
    print("VALIDATION -- axis stability separates crossing from noise?")
    print(f"  {'case':>14} | {'R1':>6} {'sep':>6} {'disp':>6} | typical label")
    specs = [("single", None), ("cross30", 30), ("cross50", 50),
             ("cross70", 70), ("cross90", 90), ("noise", None)]
    summ = {}
    for name, sep in specs:
        rs = []
        for seed in seeds:
            if name == "single":
                vol, _ = make_single_fibre_3d(theta_deg=20.0, noise_sigma=noise_sigma, seed=seed)
            elif name == "noise":
                vol = _noise_volume((96, 96, 96), seed=seed)
            else:
                vol, _ = make_crossing_phantom_3d(sep_deg=sep, noise_sigma=noise_sigma, seed=seed)
            rs.append(analyse_patch(_central_patch(vol, half=14)))
        R1 = np.mean([r["R1"] for r in rs]); sepm = np.mean([r["sep"] for r in rs])
        disp = np.mean([r["disp"] for r in rs])
        lab = classify(dict(R1=R1, sep=sepm, disp=disp))
        summ[name] = (R1, sepm, disp, lab)
        print(f"  {name:>14} | {R1:6.3f} {sepm:6.1f} {disp:6.1f} | {lab}")

    cross_disp = np.mean([summ[k][2] for k in ("cross50", "cross70", "cross90")])
    noise_disp = summ["noise"][2]
    # gate: crossings stable (low disp), noise unstable (high disp), and the
    # labels come out right for the clear cases.
    labels_ok = (summ["single"][3] == "single"
                 and summ["cross70"][3] == "crossing"
                 and summ["cross90"][3] == "crossing"
                 and summ["noise"][3] == "disorganised")
    passed = (noise_disp > cross_disp + 8.0) and labels_ok
    print(f"  crossing disp={cross_disp:.1f} vs noise disp={noise_disp:.1f} "
          f"(want noise >> crossing)")
    print(f"  GATE {'PASS' if passed else 'FAIL'}: stability "
          f"{'separates crossing from noise + labels correct' if passed else 'does NOT'}")
    print("=" * 74)
    return summ, passed


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if stage == "validate":
        validate()
