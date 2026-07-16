"""
fod.py -- local fibre orientation distribution (FOD) from gradients, with the
crossing-vs-dispersion discriminator the scalar c_s could not provide.

Model. A tube fibre along d has gradients filling the plane perpendicular to d,
so every gradient g "votes" for the great circle of directions perpendicular to
it (its fibre could be anywhere on that circle). Back-projecting all gradients
builds an orientation function on the sphere

    F(u) = sum_i |g_i|^2 * exp( -(ghat_i . u)^2 / (2 tau^2) )

which is LARGE where many gradients are perpendicular to u = candidate fibre
directions. Fibre directions are the PEAKS of F. Unlike the structure tensor
(u^T S u, quadratic, one extremum), F can carry several peaks.

Discriminators the scalar c_s is degenerate on:
  * number of peaks           1 / 2 / 3 ...
  * VALLEY between two peaks   deep valley = two DISCRETE fibres (crossing);
                              shallow (F stays high between) = one BROAD mode
                              (dispersion / fanning).  <-- the key test
  * top-peak angular spread    sharp = single fibre; broad = dispersed
  * isotropy mean(F)/max(F)    ~1 = no orientation (noise / disorganised)

Reuses patch_gradients (two_orientation_test) and the synthetic ladder
(model_ladder / crossing_shape_3d). Validated on single / crossing / dispersed /
3-fibre / noise before any real data.
"""

import numpy as np

from two_orientation_test import patch_gradients, _central_patch

EPS = 1e-12


def _unit(v):
    return v / (np.linalg.norm(v) + EPS)


def _fib_sphere(n=600):
    """n roughly-uniform directions on the unit sphere (Fibonacci)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


_SPHERE = _fib_sphere(600)


def _grad_dirs_weights(G, mag_frac=0.1):
    mag = np.linalg.norm(G, axis=1)
    keep = mag > mag_frac * (mag.max() + EPS)
    return G[keep] / mag[keep, None], mag[keep] ** 2


def _F(dirs, Gn, w, tau):
    """Evaluate the (unnormalised) FOD at arbitrary unit directions."""
    dot = Gn @ dirs.T                      # (M, K)
    return w @ np.exp(-(dot ** 2) / (2 * tau ** 2))


def _slerp(a, b, t):
    a, b = _unit(a), _unit(np.array(b) if (a @ b) >= 0 else -np.array(b))
    om = np.arccos(np.clip(a @ b, -1, 1))
    if om < 1e-6:
        return a
    return _unit(np.sin((1 - t) * om) * a + np.sin(t * om) * b)


def _extract_peaks(F, U, min_sep_deg=25.0, rel_thresh=0.55, max_peaks=3):
    """Greedy non-max suppression on the sphere. Returns list of (dir, amp)."""
    cos_sep = np.cos(np.deg2rad(min_sep_deg))
    order = np.argsort(F)[::-1]
    peaks = []
    for i in order:
        if F[i] < rel_thresh * F[order[0]]:
            break
        u = U[i]
        if all(abs(u @ p) < cos_sep for p, _ in peaks):
            peaks.append((u, float(F[i])))
        if len(peaks) >= max_peaks:
            break
    return peaks


def analyse_fod(patch, sigma_grad=2.0, tau=0.25):
    """FOD metrics for one patch. Returns dict or None."""
    G = patch_gradients(patch, sigma_grad)
    if G.shape[0] < 40:
        return None
    Gn, w = _grad_dirs_weights(G)
    F = _F(_SPHERE, Gn, w, tau)
    Fn = F / (F.max() + EPS)
    iso = float(Fn.mean())                              # ~1 => isotropic/noise
    peaks = _extract_peaks(Fn, _SPHERE)
    p1 = peaks[0][0]
    ang1 = np.degrees(np.arccos(np.clip(np.abs(_SPHERE @ p1), 0, 1)))
    cap = ang1 < 50
    spread1 = float(np.sqrt(np.sum(Fn[cap] * ang1[cap] ** 2) / (np.sum(Fn[cap]) + EPS)))

    n_peaks = len(peaks)
    sep = amp_ratio = valley = np.nan
    if n_peaks >= 2:
        p2, a2 = peaks[1]
        sep = float(np.degrees(np.arccos(np.clip(abs(p1 @ p2), 0, 1))))
        amp_ratio = a2 / (peaks[0][1] + EPS)
        arc = np.array([_slerp(p1, p2, t) for t in np.linspace(0, 1, 15)])
        Fa = _F(arc, Gn, w, tau)
        valley = float(Fa[1:-1].min() / (min(Fa[0], Fa[-1]) + EPS))
    return dict(iso=iso, n_peaks=n_peaks, sep=sep, amp_ratio=amp_ratio,
                valley=valley, spread1=spread1, n=G.shape[0])


def classify(m, iso_hi=0.80, amp_two=0.6, valley_deep=0.80,
             spread_sharp=22.0, sep_min=22.0):
    """Label from FOD metrics."""
    if m is None:
        return "n/a"
    if m["iso"] > iso_hi:
        return "disorganised"
    two = (m["n_peaks"] >= 2 and m["amp_ratio"] >= amp_two and m["sep"] >= sep_min)
    if two and m["valley"] <= valley_deep:
        return "crossing"
    if two and m["valley"] > valley_deep:
        return "dispersed"          # two lobes but no valley = one broad mode
    return "single" if m["spread1"] <= spread_sharp else "dispersed"


# ---------------------------------------------------------------------------
# Validation on the synthetic ladder
# ---------------------------------------------------------------------------

def validate(noise_sigma=0.3, seeds=(0, 1, 2)):
    from crossing_shape_3d import (make_single_fibre_3d, make_crossing_phantom_3d,
                                   _noise_volume)
    from model_ladder import make_multifibre, _inplane, _cone_dirs

    print("=" * 92)
    print("FOD VALIDATION -- recover peaks + separate crossing from dispersion")
    print(f"  {'case':>16} | {'iso':>5} {'npk':>3} {'sep':>5} {'amp2':>5} "
          f"{'valley':>6} {'sprd':>5} | label   (truth)")

    def patch_of(kind, seed):
        if kind == "single":
            v, _ = make_single_fibre_3d(theta_deg=20.0, noise_sigma=noise_sigma, seed=seed)
        elif kind == "noise":
            v = _noise_volume((96, 96, 96), seed=seed)
        elif kind.startswith("cross"):
            sep = float(kind.split("_")[1])
            v, _ = make_crossing_phantom_3d(sep_deg=sep, noise_sigma=noise_sigma, seed=seed)
        elif kind.startswith("disp"):
            kap = float(kind.split("_")[1])
            v = make_multifibre(dirs=_cone_dirs(kap, 6, seed=seed + 1),
                                noise_sigma=noise_sigma, seed=seed)
        elif kind == "three":
            v = make_multifibre(dirs=[_inplane(0), _inplane(60), _inplane(120)],
                                noise_sigma=noise_sigma, seed=seed)
        return _central_patch(v, half=14)

    cases = [("single", "single"), ("cross_30", "crossing"), ("cross_50", "crossing"),
             ("cross_70", "crossing"), ("cross_90", "crossing"),
             ("disp_20", "dispersed"), ("disp_40", "dispersed"),
             ("three", "crossing"), ("noise", "disorganised")]
    correct = 0
    for kind, truth in cases:
        ms = [analyse_fod(patch_of(kind, s)) for s in seeds]
        avg = {k: np.nanmean([m[k] for m in ms]) for k in
               ("iso", "n_peaks", "sep", "amp_ratio", "valley", "spread1")}
        labels = [classify(m) for m in ms]
        lab = max(set(labels), key=labels.count)      # majority label
        ok = (lab == truth) or (truth == "crossing" and kind == "three" and lab in ("crossing",))
        correct += ok
        print(f"  {kind:>16} | {avg['iso']:5.2f} {avg['n_peaks']:3.0f} {avg['sep']:5.0f} "
              f"{avg['amp_ratio']:5.2f} {avg['valley']:6.2f} {avg['spread1']:5.0f} | "
              f"{lab:12s} ({truth})  {'OK' if ok else 'XX'}")
    print(f"  correct labels: {correct}/{len(cases)}")
    passed = correct >= len(cases) - 1
    print(f"  GATE {'PASS' if passed else 'FAIL'}: FOD "
          f"{'recovers structure + separates crossing from dispersion' if passed else 'does NOT'}")
    print("=" * 92)
    return passed


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if stage == "validate":
        validate()
