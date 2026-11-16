"""
host_fod.py -- multi-fibre orientation via the HIGHER-ORDER STRUCTURE TENSOR.

Domain-correct method (Schultz & Seidel, "A Higher-Order Structure Tensor"):
the ordinary structure tensor is the 2nd-order moment of the gradient and its
directional energy E2(u) = <(grad.u)^2> = u^T S u is quadratic -- one minimum,
one fibre. Going to higher even order,

    E_2n(u) = < (grad_hat . u)^(2n) >            (weighted by |grad|^2)

is a degree-2n polynomial on the sphere that CAN have several minima. A fibre is
a direction of LOW gradient energy (intensity ~constant along it), so fibres are
the MINIMA of E_2n. n>=2 therefore resolves crossings that the structure tensor
(n=1) cannot -- and, unlike diffusion-MRI spherical deconvolution (dipy), it
needs no calibrated response and has no Funk-Radon bisector artifact (verified:
E4/E6 minima land on both fibres of a 90 deg crossing to <5 deg, even at noise).

We form the fibre-ness ODF  f(u) = max(E) - E(u)  (high at fibres) and extract
peaks with dipy's peak_directions. Classification: isotropic E => disorganised;
>=2 separated peaks => crossing; one broad peak => dispersed; one sharp => single.

Reuses patch_gradients (two_orientation_test), the dipy sphere + peak_directions,
and the synthetic ladder. Validated across separation AND noise before real data.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from dipy.direction import peak_directions

from dipy.data import get_sphere

# The repulsion724 sphere (it has edges, which peak_directions needs). Built
# here rather than imported from fod_dipy: that one constant was the only
# reason the live pipeline loaded fod_dipy, a dipy cross-check experiment.
try:
    SPHERE = get_sphere(name="repulsion724")
except TypeError:                                 # older dipy: positional only
    SPHERE = get_sphere("repulsion724")
from core.two_orientation_test import patch_gradients, _central_patch

EPS = 1e-12
_V = SPHERE.vertices


def _ipow(a, n):
    """a ** n for a small positive integer n, by repeated multiplication.

    numpy's `**` calls pow() elementwise. On the (M, 724) array below that is
    ~600 ms versus ~216 ms for three multiplies -- 2.8x, and this term is 99%
    of analyse()'s cost. Exact for integer exponents: max observed difference
    against `**` is 8e-17, i.e. float rounding.
    """
    r, b = None, a
    while n:
        if n & 1:
            r = b if r is None else r * b
        n >>= 1
        if n:
            b = b * b
    return r


def directional_energy(G, order=4, mag_frac=0.1):
    """E_2n(u) over the sphere: |grad|^2-weighted mean of (grad_hat . u)^order."""
    mag = np.linalg.norm(G, axis=1)
    keep = mag > mag_frac * (mag.max() + EPS)
    Gn = G[keep] / mag[keep, None]
    w = mag[keep] ** 2
    w = w / (w.sum() + EPS)
    dot = Gn @ _V.T                               # (M, Nv)
    return (w[:, None] * _ipow(dot, int(order))).sum(0)   # (Nv,)


def _scatter_eigvals(G, mag_frac=0.1):
    """Ascending eigenvalues of the gradient scatter matrix -- no sphere needed."""
    mag = np.linalg.norm(G, axis=1)
    keep = mag > mag_frac * (mag.max() + EPS)
    return np.sort(np.linalg.eigvalsh(G[keep].T @ G[keep]))


def _c_s(lam):
    """Sphericity 3*l3/(l1+l2+l3): how much the patch has NO dominant direction."""
    return float(3.0 * lam[0] / (lam.sum() + EPS))


def _fa(lam):
    """Fractional anisotropy from the same three eigenvalues."""
    return float(np.sqrt(0.5) * np.sqrt((lam[0] - lam[1]) ** 2
                                        + (lam[1] - lam[2]) ** 2
                                        + (lam[2] - lam[0]) ** 2)
                 / (np.sqrt((lam ** 2).sum()) + EPS))


def analyse(patch, sigma_grad=2.0, order=6, rel=0.30, min_sep=30.0, sharpen=3.0,
            peaks=True):
    """Metrics for one patch. `peaks=False` skips the sphere entirely.

    c_s and FA come from a 3x3 eigendecomposition of the gradient scatter and
    need no sphere at all; only n_peaks/sep/amp2/aniso/dirs do. Measured split
    per call: patch_gradients 5.9 ms, 3x3 eigvalsh 1.6 ms, sphere work 600 ms.
    So a caller that only wants the c_s vs FA routing decision should pass
    peaks=False and skip 99% of the work.
    """
    G = patch_gradients(patch, sigma_grad)
    if G.shape[0] < 40:
        return None
    if not peaks:
        lam = _scatter_eigvals(G)
        return dict(n_peaks=None, sep=np.nan, amp2=np.nan, aniso=np.nan,
                    c_s=_c_s(lam), FA=_fa(lam), dirs=[], vals=[], n=G.shape[0])
    E = directional_energy(G, order)
    f = E.max() - E                               # fibre-ness ODF (>=0)
    fn = (f / (f.max() + EPS)) ** sharpen         # sharpen minima into distinct peaks
    # min_sep=30: spurious large-sigma peak-SPLITTING clusters at ~18-22 deg,
    # whereas every crossing we validated is >=40 deg apart -- so a 30 deg floor
    # merges the split lobes without losing any real crossing.
    dirs, vals, _ = peak_directions(fn, SPHERE, relative_peak_threshold=rel,
                                    min_separation_angle=min_sep)
    aniso = float((E.max() - E.min()) / (E.mean() + EPS))   # 0 = isotropic/noise
    # 2nd-order sphericity c_s from the same gradients: single fibre is planar
    # (c_s ~ 0), a dispersed population smears the plane (c_s higher). Splits the
    # two 1-peak cases the HOST peak-count alone cannot.
    lam = _scatter_eigvals(G)
    c_s = _c_s(lam)
    # fractional anisotropy of the SAME structure tensor -- a second normalised
    # descriptor from the same eigenvalues. classify() compares c_s vs FA as a
    # parameter-free single/dispersed rule (both local to this voxel).
    fa = _fa(lam)
    npk = len(vals)
    sep = amp2 = np.nan
    if npk >= 2:
        sep = float(np.degrees(np.arccos(np.clip(abs(dirs[0] @ dirs[1]), 0, 1))))
        amp2 = float(vals[1] / (vals[0] + EPS))
    return dict(n_peaks=npk, sep=sep, amp2=amp2, aniso=aniso, c_s=c_s, FA=fa,
                dirs=dirs, vals=vals, n=G.shape[0])


def classify(m, aniso_iso=0.9, amp2_min=0.4, sep_min=30.0, cs_single=None):
    if m is None:
        return "n/a"

    if m["aniso"] < aniso_iso:
        return "disorganised"

    # EXACTLY 2 clean peaks = a crossing we can trust. n_peaks >= 3 is either a
    # genuine 3+ crossing (beyond this method's validated 2-fibre ceiling) or
    # spurious splitting -- either way not trustworthy, so label as complex.
    if m["n_peaks"] == 2 and m["amp2"] >= amp2_min and m["sep"] >= sep_min:
        return "crossing"
    # The comment above was the intent from the start, but the branch was never
    # written: a >=3-peak patch fell through to the c_s vs FA rule below and came
    # back "single" whenever c_s <= FA. That is the most confident label in the
    # set, handed to a patch the method has explicitly given up on -- and
    # downstream it is a FILTER, not a display string (mass_conservation builds
    # its calibration pool from classify(m) == "single"), so such patches were
    # entering a clean-single-fibre pool. Measured on real i74: rare, 1 of 196
    # sampled patches, but that one did land in the pool.
    if m["n_peaks"] >= 3:
        return "complex"
    # single vs dispersed: PARAMETER-FREE rule -- a voxel is a single fibre when
    # it is more anisotropic (FA) than spherical (c_s). Both are normalised
    # descriptors of the SAME structure-tensor eigenvalues, so this is purely
    # local to the voxel, needs no external threshold, and uses the full
    # eigenvalue structure (not just c_s). Validated on i74: gives ~29% single vs
    # the implausible 3% a fixed c_s<=0.15 gives, and the classes overlap in c_s
    # (so FA genuinely adds information a c_s threshold cannot). Pass a numeric
    # cs_single to override with a hard c_s threshold instead.
    if cs_single is None:
        return "single" if m["c_s"] <= m["FA"] else "dispersed"
    return "single" if m["c_s"] <= cs_single else "dispersed"


# ---------------------------------------------------------------------------
# Validation across the ladder AND noise
# ---------------------------------------------------------------------------

def _patch_of(kind, seed, noise):
    from core.crossing_shape_3d import (make_single_fibre_3d, make_crossing_phantom_3d,
                                   _noise_volume)
    from core.model_ladder import make_multifibre, _inplane, _cone_dirs
    if kind == "single":
        v, tr = make_single_fibre_3d(theta_deg=20.0, noise_sigma=noise, seed=seed)
    elif kind == "noise":
        v, tr = _noise_volume((96, 96, 96), seed=seed), []
    elif kind.startswith("cross"):
        v, tr = make_crossing_phantom_3d(sep_deg=float(kind.split("_")[1]),
                                         noise_sigma=noise, seed=seed, spacing2=12.0)
    elif kind.startswith("disp"):
        v = make_multifibre(dirs=_cone_dirs(float(kind.split("_")[1]), 6, seed=seed + 1),
                            noise_sigma=noise, seed=seed); tr = []
    elif kind == "three":
        v = make_multifibre(dirs=[_inplane(0), _inplane(70), _inplane(140)],
                            noise_sigma=noise, seed=seed); tr = []
    return _central_patch(v, half=14), tr


def validate(order=4, noises=(0.1, 0.3, 0.5), seeds=(0, 1, 2)):
    cases = [("single", "single"), ("cross_40", "crossing"), ("cross_50", "crossing"),
             ("cross_70", "crossing"), ("cross_90", "crossing"),
             ("disp_40", "dispersed"), ("noise", "disorganised")]
    print("=" * 78)
    print(f"HOST-FOD VALIDATION (order={order}) across noise")
    total = correct = 0
    for noise in noises:
        print(f"  --- noise={noise} ---")
        for kind, truth in cases:
            ms, errs = [], []
            for s in seeds:
                patch, tr = _patch_of(kind, s, noise)
                m = analyse(patch, order=order)
                ms.append(m)
                if tr and m:
                    errs.append(np.mean([min(np.degrees(np.arccos(np.clip(abs(d @ t), 0, 1)))
                                             for t in tr) for d in m["dirs"][:len(tr)]]))
            labels = [classify(x) for x in ms]
            lab = max(set(labels), key=labels.count)
            ok = (lab == truth)
            total += 1; correct += ok
            avg = {k: np.nanmean([x[k] for x in ms]) for k in ("n_peaks", "sep", "amp2", "aniso", "c_s")}
            et = f"{np.mean(errs):4.1f}" if errs else "  - "
            print(f"    {kind:>9}->{truth:12s} npk={avg['n_peaks']:.0f} sep={avg['sep']:4.0f} "
                  f"amp2={avg['amp2']:.2f} aniso={avg['aniso']:.2f} cs={avg['c_s']:.2f} "
                  f"err={et} | {lab:12s} {'OK' if ok else 'XX'}")
    print(f"  TOTAL correct: {correct}/{total}")
    print(f"  GATE {'PASS' if correct >= total - 2 else 'FAIL'}")
    print("=" * 78)
    return correct, total


if __name__ == "__main__":
    import sys
    order = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    validate(order=order)
