"""
fod_dipy.py -- fibre orientation distribution via dipy (option 2: use a library).

My hand-rolled back-projection (fod.py) put spurious peaks at the in-plane
perpendicular axes for crossings and MISSED the true second fibre, because it
summed the single-fibre girdle response instead of deconvolving it. The fix is a
proper spherical deconvolution; here it is built on dipy's battle-tested
spherical-harmonic transforms and peak extractor.

Pipeline per patch:
  1. gradient-direction density on a sphere:
        Q(v) = sum_i |g_i|^2 * exp(kappa * (v . ghat_i)^2)     (Watson, antipodal)
     For a tube fibre the gradients fill the plane perpendicular to it, so Q is a
     GIRDLE (equatorial band) whose pole is the fibre direction.
  2. Q -> spherical harmonics (dipy sf_to_sh).
  3. Funk-Radon deconvolution: multiply the order-l coefficients by 2*pi*P_l(0).
     P_l(0) alternates sign with l, which turns the equatorial girdle into a
     PEAK at the fibre pole -- i.e. the ODF now peaks at fibre directions.
  4. back to the sphere (sh_to_sf) and dipy peak_directions with a relative-peak
     threshold + minimum-separation angle.

Discriminators for the real question:
  n_peaks (1 vs 2 vs 3), peak separations, and gfa-like anisotropy (isotropic =
  noise). Crossing = >=2 well-separated peaks; dispersed = 1 peak but low
  anisotropy / broad; single = 1 sharp peak; noise = near-isotropic ODF.

Validated on the synthetic ladder before real data. Reuses patch_gradients
(two_orientation_test) and the phantoms (crossing_shape_3d / model_ladder).
"""

import warnings

import numpy as np
from scipy.special import eval_legendre

warnings.filterwarnings("ignore")

from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf, sph_harm_ind_list
from dipy.direction import peak_directions

from core.two_orientation_test import patch_gradients, _central_patch

EPS = 1e-12
SH_ORDER = 8
try:
    SPHERE = get_sphere(name="repulsion724")
except TypeError:
    SPHERE = get_sphere("repulsion724")
_M, _L = sph_harm_ind_list(SH_ORDER)
FRT = 2.0 * np.pi * eval_legendre(_L, 0.0)          # Funk-Radon factor per coef


def _grad_density(G, kappa=8.0, mag_frac=0.1):
    mag = np.linalg.norm(G, axis=1)
    keep = mag > mag_frac * (mag.max() + EPS)
    Gn = G[keep] / mag[keep, None]
    w = mag[keep] ** 2
    w = w / (w.sum() + EPS)
    dot = SPHERE.vertices @ Gn.T                    # (Nv, M)
    return np.exp(kappa * dot ** 2) @ w             # (Nv,)


def fod_of_patch(patch, sigma_grad=2.0, kappa=8.0):
    G = patch_gradients(patch, sigma_grad)
    if G.shape[0] < 40:
        return None, None
    Q = _grad_density(G, kappa)
    q_sh = sf_to_sh(Q, SPHERE, sh_order_max=SH_ORDER)
    fod = sh_to_sf(q_sh * FRT, SPHERE, sh_order_max=SH_ORDER)
    return fod, G.shape[0]


def analyse(patch, sigma_grad=2.0, kappa=8.0, rel=0.45, min_sep=22.0):
    fod, n = fod_of_patch(patch, sigma_grad, kappa)
    if fod is None:
        return None
    f = fod - fod.min()
    dirs, vals, _ = peak_directions(f, SPHERE, relative_peak_threshold=rel,
                                    min_separation_angle=min_sep)
    # anisotropy: normalised std of the ODF over the sphere (0 isotropic .. high peaky)
    fn = f / (f.mean() + EPS)
    aniso = float(fn.std() / (fn.mean() + EPS))
    npk = len(vals)
    sep = float(np.degrees(np.arccos(np.clip(abs(dirs[0] @ dirs[1]), 0, 1)))) if npk >= 2 else np.nan
    amp2 = float(vals[1] / (vals[0] + EPS)) if npk >= 2 else np.nan
    return dict(n_peaks=npk, sep=sep, amp2=amp2, aniso=aniso, dirs=dirs, vals=vals, n=n)


def classify(m, aniso_iso=0.55, amp2_min=0.45, sep_min=22.0):
    if m is None:
        return "n/a"
    if m["aniso"] < aniso_iso:
        return "disorganised"
    if m["n_peaks"] >= 2 and m["amp2"] >= amp2_min and m["sep"] >= sep_min:
        return "crossing"
    return "single/dispersed"


# ---------------------------------------------------------------------------
# Validation on the synthetic ladder
# ---------------------------------------------------------------------------

def validate(noise_sigma=0.3, seeds=(0, 1, 2)):
    from core.crossing_shape_3d import (make_single_fibre_3d, make_crossing_phantom_3d,
                                   _noise_volume)
    from core.model_ladder import make_multifibre, _inplane, _cone_dirs

    def patch_of(kind, seed):
        if kind == "single":
            v, tr = make_single_fibre_3d(theta_deg=20.0, noise_sigma=noise_sigma, seed=seed)
        elif kind == "noise":
            v, tr = _noise_volume((96, 96, 96), seed=seed), []
        elif kind.startswith("cross"):
            v, tr = make_crossing_phantom_3d(sep_deg=float(kind.split("_")[1]),
                                             noise_sigma=noise_sigma, seed=seed)
        elif kind.startswith("disp"):
            v = make_multifibre(dirs=_cone_dirs(float(kind.split("_")[1]), 6, seed=seed + 1),
                                noise_sigma=noise_sigma, seed=seed); tr = []
        elif kind == "three":
            v = make_multifibre(dirs=[_inplane(0), _inplane(60), _inplane(120)],
                                noise_sigma=noise_sigma, seed=seed); tr = []
        return _central_patch(v, half=14), tr

    print("=" * 90)
    print("FOD (dipy) VALIDATION -- peaks at true dirs + crossing vs dispersion?")
    print(f"  {'case':>14} | {'npk':>3} {'sep':>5} {'amp2':>5} {'aniso':>6} | "
          f"{'peak err(deg)':>13} | label   (truth)")
    cases = [("single", "single/dispersed"), ("cross_30", "crossing"),
             ("cross_50", "crossing"), ("cross_70", "crossing"),
             ("cross_90", "crossing"), ("disp_20", "single/dispersed"),
             ("disp_40", "single/dispersed"), ("three", "crossing"),
             ("noise", "disorganised")]
    correct = 0
    for kind, truth in cases:
        ms, errs = [], []
        for s in seeds:
            patch, tr = patch_of(kind, s)
            m = analyse(patch)
            ms.append(m)
            if tr and m and m["n_peaks"] >= 1:
                # mean angular error of recovered peaks to nearest truth
                e = [min(np.degrees(np.arccos(np.clip(abs(d @ t), 0, 1))) for t in tr)
                     for d in m["dirs"][:len(tr)]]
                errs.append(np.mean(e))
        avg = {k: np.nanmean([m[k] for m in ms]) for k in ("n_peaks", "sep", "amp2", "aniso")}
        labels = [classify(m) for m in ms]
        lab = max(set(labels), key=labels.count)
        ok = (lab == truth)
        correct += ok
        errtxt = f"{np.mean(errs):13.1f}" if errs else f"{'-':>13}"
        print(f"  {kind:>14} | {avg['n_peaks']:3.0f} {avg['sep']:5.0f} {avg['amp2']:5.2f} "
              f"{avg['aniso']:6.2f} | {errtxt} | {lab:16s} ({truth})  {'OK' if ok else 'XX'}")
    print(f"  correct labels: {correct}/{len(cases)}")
    passed = correct >= len(cases) - 1
    print(f"  GATE {'PASS' if passed else 'FAIL'}: dipy FOD "
          f"{'recovers structure + separates crossing/dispersion' if passed else 'does NOT'}")
    print("=" * 90)
    return passed


if __name__ == "__main__":
    import sys
    if (sys.argv[1] if len(sys.argv) > 1 else "validate") == "validate":
        validate()
