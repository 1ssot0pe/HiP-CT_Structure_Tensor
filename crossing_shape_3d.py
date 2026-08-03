"""
crossing_shape_3d.py

Does the structure-tensor eigenvalue SHAPE distinguish a single fibre from a
fused narrow crossing? The 2D toy could not answer this: a 2x2 tensor has only
two eigenvalues (one shape number = coherence), and that number does NOT
separate fused-narrow from resolved-wide crossings. This module moves the test
to 3D, where a THIRD eigenvalue exists, and applies the corrected convention.

Convention (corrected from the DTI mapping)
-------------------------------------------
For the STRUCTURE tensor (gradient outer product), a single straight fibre is
PLANAR, not linear: its gradients fill the plane perpendicular to the fibre, so
lambda1 ~ lambda2 >> lambda3 ~ 0, and the fibre axis is the SMALLEST-eigenvalue
eigenvector e3. Energy leaking into lambda3 (SPHERICITY) is the signature of a
crossing / multi-orientation region. So the discriminator to test is
sphericity c_s = 3*lambda3 / sum(lambda) rising on fused crossings -- NOT
planarity (which is high for the clean single fibre).

Westin shape measures (val descending, l1>=l2>=l3):
    linearity  c_l = (l1 - l2) / (l1 + l2 + l3)      -- edge/sheet-like
    planarity  c_p = 2*(l2 - l3) / (l1 + l2 + l3)    -- single-fibre-like (HIGH)
    sphericity c_s = 3*l3 / (l1 + l2 + l3)           -- isotropic / multi-orient

Reuses: structure_tensor_3d + eig_special_3d (3 eigenvalues); fibre_direction,
crop_margin, _crop3 (orthogonality_residual); the coherence-onset knee detector
(toy2d/coherence_sigma.knee_sigma) unchanged. The only NEW thing is a 3D
crossing phantom with a tunable separation angle -- no existing phantom has that.
"""

import os
import sys

import numpy as np
from structure_tensor import structure_tensor_3d, eig_special_3d

from orthogonality_residual import fibre_direction, crop_margin, _crop3

# Reuse the exact knee detector from the 2D selector (pure function of a curve).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "1_sigma_rho_selection", "toy2d"))
from coherence_sigma import knee_sigma  # noqa: E402

EPS = 1e-12


# ---------------------------------------------------------------------------
# 3D crossing phantom with a tunable separation angle
# ---------------------------------------------------------------------------

def _perp_basis(d):
    """Two orthonormal vectors spanning the plane perpendicular to unit d."""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = a - d * np.dot(a, d); u /= np.linalg.norm(u)
    v = np.cross(d, u); v /= np.linalg.norm(v)
    return u, v


def _tube_family(coords_xyz, d, spacing, width):
    """Periodic Gaussian tubes along unit direction d. coords_xyz = (x, y, z)
    arrays with x=axis2, y=axis1, z=axis0 (the structure-tensor e3 basis)."""
    u, v = _perp_basis(d)
    x, y, z = coords_xyz
    a = x * u[0] + y * u[1] + z * u[2]
    b = x * v[0] + y * v[1] + z * v[2]
    da = ((a + spacing / 2) % spacing) - spacing / 2
    db = ((b + spacing / 2) % spacing) - spacing / 2
    return np.exp(-(da ** 2 + db ** 2) / (2.0 * width ** 2))


def _coords(shape):
    """(x, y, z) coordinate arrays centred at the volume middle, with
    x=axis2, y=axis1, z=axis0 so vectors live in the e3 component basis."""
    idx = np.indices(shape).astype(np.float32)
    c = [idx[a] - shape[a] / 2 for a in range(3)]   # c[0]=axis0=z, c[1]=y, c[2]=x
    return (c[2], c[1], c[0])                        # return as (x, y, z)


def _random_rotation(seed):
    """A uniform-ish random 3D rotation matrix (QR of a Gaussian matrix)."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def make_crossing_phantom_3d(shape=(96, 96, 96), sep_deg=40.0, spacing=12.0,
                             spacing2=17.0, width=1.5, noise_sigma=0.0, seed=0,
                             rot=None):
    """Two straight Gaussian-tube families separated by `sep_deg`.

    Built symmetric about the x-axis in the x-y plane (z is the plane normal,
    the direction into which a crossing leaks lambda3), with different spacings
    to avoid a shared period. If `rot` (3x3) is given, BOTH directions are
    rotated by it -- a skew / arbitrarily-oriented crossing whose plane is no
    longer axis-aligned (the generalisation test: the guard must not depend on
    the leakage direction lining up with a volume axis).

    Returns (volume float32 in [0,1], truths) with truths the two unit fibre
    directions in the SAME (x,y,z) basis as fibre_direction's e3.
    """
    half = np.deg2rad(sep_deg / 2.0)
    d1 = np.array([np.cos(half),  np.sin(half), 0.0])
    d2 = np.array([np.cos(half), -np.sin(half), 0.0])
    if rot is not None:
        d1, d2 = rot @ d1, rot @ d2
    coords = _coords(shape)
    vol = (_tube_family(coords, d1, spacing, width)
           + _tube_family(coords, d2, spacing2, width))
    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        vol = vol + noise_sigma * rng.standard_normal(shape).astype(np.float32)
    vol = (vol - vol.min()) / (np.ptp(vol) + EPS)
    return vol.astype(np.float32), [d1, d2]


def make_single_fibre_3d(shape=(96, 96, 96), theta_deg=20.0, spacing=12.0,
                         width=1.5, noise_sigma=0.0, seed=0, rot=None):
    """One straight tube family at `theta_deg` in the x-y plane, optionally
    rotated by `rot` (3x3). Returns (volume, [truth_direction])."""
    th = np.deg2rad(theta_deg)
    d = np.array([np.cos(th), np.sin(th), 0.0])
    if rot is not None:
        d = rot @ d
    coords = _coords(shape)
    vol = _tube_family(coords, d, spacing, width)
    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        vol = vol + noise_sigma * rng.standard_normal(shape).astype(np.float32)
    vol = (vol - vol.min()) / (np.ptp(vol) + EPS)
    return vol.astype(np.float32), [d]


# ---------------------------------------------------------------------------
# LAYER 1 -- eigenvalue-shape measures (Westin) + 3D coherence for the selector
# ---------------------------------------------------------------------------

def shape_measures(vol, sigma, rho, truncate=4.0):
    """Region-mean Westin measures + a 3D coherence, cropped to the valid border.

    C3 = normalised eigenvalue variance in [0,1] (0 isotropic/noise, high for
    any anisotropy); it is what the coherence-onset selector's knee runs on.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        S = structure_tensor_3d(vol.astype(np.float32), sigma, rho, truncate=truncate)
        val, _vec = eig_special_3d(S)                 # desc: val[0]>=val[1]>=val[2]
    l1, l2, l3 = val[0], val[1], val[2]
    s = l1 + l2 + l3 + EPS
    cl = (l1 - l2) / s
    cp = 2.0 * (l2 - l3) / s
    cs = 3.0 * l3 / s
    C3 = ((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l1 - l3) ** 2) / (2.0 * (l1 ** 2 + l2 ** 2 + l3 ** 2) + EPS)
    m = crop_margin(sigma, rho, truncate)
    return dict(cl=float(np.mean(_crop3(cl, m))), cp=float(np.mean(_crop3(cp, m))),
                cs=float(np.mean(_crop3(cs, m))), C3=float(np.mean(_crop3(C3, m))))


def _noise_volume(shape=(96, 96, 96), seed=0):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(shape).astype(np.float32)
    return (n - n.min()) / (np.ptp(n) + EPS)


def layer1(sigma=1.5, rho=3.0):
    """GATE (corrected convention): single fibre -> PLANARITY high, sphericity
    low; pure noise -> SPHERICITY high."""
    print("=" * 70)
    print("LAYER 1 -- Westin shape sanity (structure-tensor convention)")
    fib, _ = make_single_fibre_3d(theta_deg=20.0, noise_sigma=0.0)
    noise = _noise_volume(seed=0)
    sf = shape_measures(fib, sigma, rho)
    sn = shape_measures(noise, sigma, rho)
    print(f"  single fibre : c_l={sf['cl']:.3f}  c_p={sf['cp']:.3f}  c_s={sf['cs']:.3f}")
    print(f"  pure noise   : c_l={sn['cl']:.3f}  c_p={sn['cp']:.3f}  c_s={sn['cs']:.3f}")
    fibre_planar = sf["cp"] > 0.5 and sf["cp"] > sf["cl"] and sf["cp"] > sf["cs"]
    noise_spherical = sn["cs"] > 0.5 and sn["cs"] > sn["cp"]
    passed = fibre_planar and noise_spherical
    print(f"  single fibre is PLANAR (c_p dominant)? {fibre_planar}")
    print(f"  pure noise is SPHERICAL (c_s dominant)? {noise_spherical}")
    print(f"  GATE {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    return passed


# ---------------------------------------------------------------------------
# selector + angular error (reused pieces)
# ---------------------------------------------------------------------------

SIGMAS = np.arange(0.5, 6.01, 0.5)


def onset_sigma(vol, sigmas=SIGMAS, rho_ratio=2.0, truncate=4.0):
    """Coherence-onset sigma: knee of the 3D coherence(C3) vs sigma curve
    (rho = rho_ratio*sigma). Returns (sigma_onset, sigmas, C3_curve, shapes)."""
    shapes = [shape_measures(vol, s, rho_ratio * s, truncate) for s in sigmas]
    C3 = np.array([sh["C3"] for sh in shapes])
    s_on, _info = knee_sigma(sigmas, C3)
    return float(s_on), sigmas, C3, shapes


def crossing_angular_error(vol, truths, sigma, rho, truncate=4.0):
    """Mean angular error (deg) of e3 to the NEAREST of the known global fibre
    directions, over cropped valid voxels."""
    e3 = fibre_direction(vol, sigma, rho, truncate)      # (3,*), e3 basis
    valid = np.isfinite(e3).all(axis=0)
    e3 = np.where(valid, e3, 0.0)
    errs = []
    for t in truths:
        t = np.asarray(t, float); t = t / np.linalg.norm(t)
        cos = np.clip(np.abs(np.einsum("i...,i->...", e3, t)), 0.0, 1.0)
        errs.append(np.degrees(np.arccos(cos)))
    err = np.min(np.stack(errs), axis=0)
    m = crop_margin(sigma, rho, truncate)
    v = _crop3(valid, m)
    err = _crop3(err, m)
    return float(np.mean(err[v])) if v.any() else float("nan")


# ---------------------------------------------------------------------------
# LAYER 2 -- THE DISCRIMINATOR TEST (make-or-break)
# ---------------------------------------------------------------------------

SEPS = [20.0, 30.0, 40.0, 50.0, 70.0, 90.0]
SEEDS = [0, 1, 2]
NOISE = 0.3
RESOLVE_THRESH = 10.0


def _shape_at(shapes, sigmas, s_on):
    i = int(np.argmin(np.abs(np.asarray(sigmas) - s_on)))
    return shapes[i]


def layer2(save="crossing_shape_layer2.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print(f"LAYER 2 -- discriminator: shape at onset sigma vs crossing separation")
    print(f"  noise={NOISE}, seeds={SEEDS}, evaluated AT the onset sigma the selector picks")
    rows = []

    # single-fibre baseline (also checks the truth/e3 basis: err must be ~0)
    for seed in SEEDS:
        vol, tr = make_single_fibre_3d(theta_deg=20.0, noise_sigma=NOISE, seed=seed)
        s_on, sigmas, C3, shapes = onset_sigma(vol)
        sh = _shape_at(shapes, sigmas, s_on)
        err = crossing_angular_error(vol, tr, s_on, 2 * s_on)
        rows.append(dict(sep=None, kind="single", seed=seed, s_on=s_on,
                         cp=sh["cp"], cs=sh["cs"], cl=sh["cl"], err=err,
                         resolved=err < RESOLVE_THRESH))

    for sep in SEPS:
        for seed in SEEDS:
            vol, tr = make_crossing_phantom_3d(sep_deg=sep, noise_sigma=NOISE, seed=seed)
            s_on, sigmas, C3, shapes = onset_sigma(vol)
            sh = _shape_at(shapes, sigmas, s_on)
            err = crossing_angular_error(vol, tr, s_on, 2 * s_on)
            rows.append(dict(sep=sep, kind="cross", seed=seed, s_on=s_on,
                             cp=sh["cp"], cs=sh["cs"], cl=sh["cl"], err=err,
                             resolved=err < RESOLVE_THRESH))

    # ---- table (mean over seeds) ----
    print(f"  {'group':>10} | {'onset':>6} | {'c_p':>6} {'c_s':>6} {'c_l':>6} | "
          f"{'err':>6} | resolved")
    groups = [("single", None)] + [("cross", s) for s in SEPS]
    for kind, sep in groups:
        rs = [r for r in rows if r["kind"] == kind and r["sep"] == sep]
        m = lambda k: np.mean([r[k] for r in rs])
        res_frac = np.mean([r["resolved"] for r in rs])
        name = "single" if kind == "single" else f"sep={sep:.0f}"
        print(f"  {name:>10} | {m('s_on'):6.2f} | {m('cp'):6.3f} {m('cs'):6.3f} "
              f"{m('cl'):6.3f} | {m('err'):6.2f} | {res_frac:.0%}")

    # ---- does any shape measure separate FUSED from RESOLVED? ----
    fused = [r for r in rows if not r["resolved"]]
    resolved = [r for r in rows if r["resolved"]]
    print("-" * 70)
    if not fused or not resolved:
        print("  (need both fused and resolved cases to test a threshold)")
    for meas in ("cs", "cp", "cl"):
        fv = np.array([r[meas] for r in fused])
        rv = np.array([r[meas] for r in resolved])
        # A clean threshold exists iff the two groups' ranges don't overlap.
        gap = fv.min() - rv.max() if fv.mean() > rv.mean() else rv.min() - fv.max()
        direction = "fused>resolved" if fv.mean() > rv.mean() else "resolved>fused"
        sep_ok = gap > 0
        print(f"  {meas}: fused={fv.mean():.3f}+/-{fv.std():.3f} [{fv.min():.3f},{fv.max():.3f}]  "
              f"resolved={rv.mean():.3f}+/-{rv.std():.3f} [{rv.min():.3f},{rv.max():.3f}]  "
              f"({direction}, clean-split={'YES' if sep_ok else 'no'}, gap={gap:+.3f})")

    # Verdict: is sphericity a clean discriminator (fused separable from resolved)?
    cs_fused = np.array([r["cs"] for r in fused]) if fused else np.array([])
    cs_res = np.array([r["cs"] for r in resolved]) if resolved else np.array([])
    viable = fused and resolved and (cs_fused.min() > cs_res.max() or cs_res.min() > cs_fused.max())
    print(f"  VERDICT: sphericity {'CLEANLY SEPARATES' if viable else 'does NOT cleanly separate'} "
          f"fused from resolved")
    print("=" * 70)

    # ---- plot: c_s and c_p at onset vs separation, coloured by resolved ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, meas, title in ((ax[0], "cs", "sphericity c_s at onset"),
                           (ax[1], "cp", "planarity c_p at onset")):
        for r in rows:
            xx = 0 if r["sep"] is None else r["sep"]
            a.scatter(xx, r[meas], c=("tab:green" if r["resolved"] else "tab:red"),
                      s=55, edgecolors="k", linewidths=0.4,
                      marker=("o" if r["kind"] == "cross" else "D"))
        a.set_xlabel("crossing separation (deg)  [0 = single fibre, D]")
        a.set_ylabel(meas); a.set_title(title)
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", mfc="tab:green", mec="k", label="resolved (<10 deg)"),
                          Line2D([], [], marker="o", ls="", mfc="tab:red", mec="k", label="fused (>=10 deg)"),
                          Line2D([], [], marker="D", ls="", mfc="gray", mec="k", label="single fibre")],
                 fontsize=8)
    fig.suptitle("Layer 2: eigenvalue shape at onset sigma -- does it flag the fused crossings?")
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return rows, viable


# ---------------------------------------------------------------------------
# LAYER 3 + 4 -- build the sphericity guard, test catch / false-positive rates,
# and characterise the boundary (train/test seeds kept disjoint).
# ---------------------------------------------------------------------------

# Onset sigma is always small (<=2.5); a grid to 4.0 covers it and skips the
# slow large-sigma/large-rho structure-tensor calls, so the stress sweeps run
# in minutes rather than tens of minutes.
SIGMAS_STRESS = np.arange(0.5, 4.01, 0.5)


def _evaluate_one(kind, sep, seed, noise=NOISE, rot=None, sigmas=SIGMAS):
    """Onset sigma, sphericity at onset, angular error at onset for one phantom."""
    if kind == "single":
        vol, tr = make_single_fibre_3d(theta_deg=20.0, noise_sigma=noise, seed=seed, rot=rot)
    else:
        vol, tr = make_crossing_phantom_3d(sep_deg=sep, noise_sigma=noise, seed=seed, rot=rot)
    s_on, sig, C3, shapes = onset_sigma(vol, sigmas=sigmas)
    sh = _shape_at(shapes, sig, s_on)
    err = crossing_angular_error(vol, tr, s_on, 2 * s_on)
    return dict(kind=kind, sep=sep, seed=seed, s_on=s_on, cs=sh["cs"],
                err=err, resolved=err < RESOLVE_THRESH)


def _threshold_from(rows):
    """Midpoint of the sphericity gap between resolved and fused on TRAIN rows."""
    fused = [r["cs"] for r in rows if not r["resolved"]]
    resolved = [r["cs"] for r in rows if r["resolved"]]
    return 0.5 * (max(resolved) + min(fused)), max(resolved), min(fused)


def layer3(train_seeds=(0, 1, 2), test_seeds=(3, 4, 5)):
    print("=" * 70)
    print("LAYER 3 -- sphericity guard: catch rate + false-positive rate")
    specs = [("single", None)] + [("cross", s) for s in SEPS]
    train = [_evaluate_one(k, s, sd) for (k, s) in specs for sd in train_seeds]
    thr, res_max, fus_min = _threshold_from(train)
    print(f"  threshold c_s* = {thr:.3f}  (train: max resolved={res_max:.3f}, "
          f"min fused={fus_min:.3f}, gap={fus_min-res_max:+.3f})")

    test = [_evaluate_one(k, s, sd) for (k, s) in specs for sd in test_seeds]
    fused = [r for r in test if not r["resolved"]]
    trust = [r for r in test if r["resolved"]]
    catch = np.mean([r["cs"] > thr for r in fused]) if fused else float("nan")
    fp = np.mean([r["cs"] > thr for r in trust]) if trust else float("nan")
    print(f"  TEST seeds {test_seeds}:")
    print(f"  caught {int(np.sum([r['cs']>thr for r in fused]))}/{len(fused)} fused "
          f"(catch rate {catch:.0%})")
    print(f"  false-flagged {int(np.sum([r['cs']>thr for r in trust]))}/{len(trust)} "
          f"trustworthy (false-positive rate {fp:.0%})")
    passed = catch >= 0.8 and fp <= 0.2
    print(f"  GATE {'PASS' if passed else 'FAIL'}: guard "
          f"{'catches fusions without crying wolf' if passed else 'not clean enough'}")
    print("=" * 70)
    return thr, catch, fp, passed


def layer4(seeds=(0, 1, 2, 3, 4), fine_seps=(30, 35, 40, 45, 50, 55, 60, 70),
           save="crossing_shape_layer4.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("LAYER 4 -- boundary: where does the guard fire vs where fusion begins?")
    # threshold from the Layer-3 training seeds, applied here
    specs = [("single", None)] + [("cross", s) for s in SEPS]
    thr, _, _ = _threshold_from([_evaluate_one(k, s, sd) for (k, s) in specs for sd in (0, 1, 2)])

    err_m, cs_m, fire_frac, res_frac = [], [], [], []
    for sep in fine_seps:
        rs = [_evaluate_one("cross", sep, sd) for sd in seeds]
        err_m.append(np.mean([r["err"] for r in rs]))
        cs_m.append(np.mean([r["cs"] for r in rs]))
        fire_frac.append(np.mean([r["cs"] > thr for r in rs]))
        res_frac.append(np.mean([r["resolved"] for r in rs]))
    for sep, e, c, ff, rf in zip(fine_seps, err_m, cs_m, fire_frac, res_frac):
        print(f"  sep={sep:2d} | err={e:5.2f} | c_s={c:.3f} | guard-fires={ff:.0%} | "
              f"resolved={rf:.0%}")
    print(f"  threshold c_s* = {thr:.3f}; fusion (err>{RESOLVE_THRESH:.0f}) begins "
          f"where resolved% drops below 100%.")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(RESOLVE_THRESH, color="0.6", ls=":", label="resolve thresh 10 deg")
    ax.plot(fine_seps, err_m, "k^-", label="angular error at onset (deg)")
    ax2 = ax.twinx()
    ax2.plot(fine_seps, cs_m, "o-", color="tab:red", label="sphericity c_s")
    ax2.axhline(thr, color="tab:red", ls="--", label=f"guard threshold c_s*={thr:.2f}")
    ax.set_xlabel("crossing separation (deg)")
    ax.set_ylabel("angular error at onset (deg)")
    ax2.set_ylabel("sphericity c_s", color="tab:red")
    ax.set_title("Layer 4: error and sphericity vs separation (guard boundary)")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return thr, fine_seps, err_m, cs_m, fire_frac


# ---------------------------------------------------------------------------
# STRESS A -- noise sweep: does the sphericity guard survive higher noise?
# ---------------------------------------------------------------------------

STRESS_SEPS = [30.0, 40.0, 50.0, 70.0, 90.0]
STRESS_SPECS = [("single", None)] + [("cross", s) for s in STRESS_SEPS]


def _guard_eval(specs, train_seeds, test_seeds, noise, rot_train=None, rot_test=None):
    """Fit c_s threshold on train, score catch/FP on test. Returns a dict."""
    train = [_evaluate_one(k, s, sd, noise=noise, rot=rot_train, sigmas=SIGMAS_STRESS)
             for (k, s) in specs for sd in train_seeds]
    tf = [r["cs"] for r in train if not r["resolved"]]
    tr = [r["cs"] for r in train if r["resolved"]]
    if not tf or not tr:
        return dict(ok=False, note="train lacks both classes")
    thr = 0.5 * (max(tr) + min(tf))
    gap = min(tf) - max(tr)
    test = [_evaluate_one(k, s, sd, noise=noise, rot=rot_test, sigmas=SIGMAS_STRESS)
            for (k, s) in specs for sd in test_seeds]
    fused = [r for r in test if not r["resolved"]]
    trust = [r for r in test if r["resolved"]]
    catch = float(np.mean([r["cs"] > thr for r in fused])) if fused else float("nan")
    fp = float(np.mean([r["cs"] > thr for r in trust])) if trust else float("nan")
    return dict(ok=True, thr=thr, gap=gap, catch=catch, fp=fp,
                n_fused=len(fused), n_trust=len(trust))


def stress_noise(noises=(0.2, 0.3, 0.5, 0.7, 1.0), train_seeds=(0, 1, 2),
                 test_seeds=(3, 4, 5), save="stress_noise.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("STRESS A -- noise sweep (axis-aligned crossings)")
    rows = []
    for ns in noises:
        d = _guard_eval(STRESS_SPECS, train_seeds, test_seeds, ns)
        rows.append((ns, d))
        if d["ok"]:
            print(f"  noise={ns:4.2f} | thr={d['thr']:.3f} gap={d['gap']:+.3f} | "
                  f"catch={d['catch']:.0%} ({d['n_fused']}) | FP={d['fp']:.0%} ({d['n_trust']})")
        else:
            print(f"  noise={ns:4.2f} | {d['note']}")
    good = [(ns, d) for ns, d in rows if d["ok"] and d["catch"] >= 0.8 and d["fp"] <= 0.2]
    print(f"  guard clean (catch>=80%, FP<=20%) up to noise = "
          f"{max(ns for ns,_ in good) if good else 'NONE':}")
    print("=" * 70)

    ns_ok = [ns for ns, d in rows if d["ok"]]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ns_ok, [d["catch"] for _, d in rows if d["ok"]], "o-", color="tab:green", label="catch rate")
    ax.plot(ns_ok, [d["fp"] for _, d in rows if d["ok"]], "s-", color="tab:red", label="false-positive rate")
    ax.axhline(0.8, color="0.7", ls=":"); ax.axhline(0.2, color="0.7", ls=":")
    ax2 = ax.twinx()
    ax2.plot(ns_ok, [d["gap"] for _, d in rows if d["ok"]], "^--", color="tab:blue", label="c_s gap")
    ax2.axhline(0, color="tab:blue", ls=":", lw=0.8)
    ax.set_xlabel("phantom noise_sigma"); ax.set_ylabel("rate")
    ax2.set_ylabel("sphericity gap (fused-resolved)", color="tab:blue")
    ax.set_title("Stress A: guard vs noise")
    ax.legend(loc="center left", fontsize=8); ax2.legend(loc="center right", fontsize=8)
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return rows


# ---------------------------------------------------------------------------
# STRESS B -- skew crossings: threshold trained on axis-aligned, tested on
# arbitrarily-oriented crossing planes (guard must be rotation-invariant).
# ---------------------------------------------------------------------------

def stress_skew(noise=0.3, train_seeds=(0, 1, 2), test_seeds=(0, 1, 2),
                orient_seeds=(1, 2, 3, 4, 5), save="stress_skew.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("STRESS B -- skew / arbitrarily-oriented crossing planes")
    print(f"  threshold trained on AXIS-ALIGNED crossings, tested on tilted ones")
    # Train threshold once on axis-aligned crossings.
    base = _guard_eval(STRESS_SPECS, train_seeds, test_seeds, noise)  # rot=None both
    thr = base["thr"]
    print(f"  axis-aligned: thr={thr:.3f} gap={base['gap']:+.3f} "
          f"catch={base['catch']:.0%} FP={base['fp']:.0%}  (reference)")

    catches, fps = [], []
    for osd in orient_seeds:
        R = _random_rotation(osd)
        test = [_evaluate_one(k, s, sd, noise=noise, rot=R, sigmas=SIGMAS_STRESS)
                for (k, s) in STRESS_SPECS for sd in test_seeds]
        fused = [r for r in test if not r["resolved"]]
        trust = [r for r in test if r["resolved"]]
        catch = float(np.mean([r["cs"] > thr for r in fused])) if fused else float("nan")
        fp = float(np.mean([r["cs"] > thr for r in trust])) if trust else float("nan")
        catches.append(catch); fps.append(fp)
        print(f"  orient#{osd} | catch={catch:.0%} ({len(fused)}) | FP={fp:.0%} ({len(trust)})")
    mc, mf = np.nanmean(catches), np.nanmean(fps)
    passed = mc >= 0.8 and mf <= 0.2
    print(f"  MEAN over tilted orientations: catch={mc:.0%}, FP={mf:.0%}")
    print(f"  GATE {'PASS' if passed else 'FAIL'}: axis-aligned threshold "
          f"{'transfers to skew crossings' if passed else 'does NOT transfer'}")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = list(orient_seeds)
    ax.plot(x, catches, "o-", color="tab:green", label="catch rate")
    ax.plot(x, fps, "s-", color="tab:red", label="false-positive rate")
    ax.axhline(0.8, color="0.7", ls=":"); ax.axhline(0.2, color="0.7", ls=":")
    ax.set_ylim(-0.05, 1.05); ax.set_xlabel("random orientation seed")
    ax.set_ylabel("rate"); ax.set_title(f"Stress B: axis-aligned guard on skew crossings (thr={thr:.2f})")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return catches, fps


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    if stage == "1":
        layer1()
    elif stage == "2":
        layer1()
        layer2()
    elif stage == "3":
        layer3()
    elif stage == "4":
        layer4()
    elif stage == "34":
        layer3()
        layer4()
    elif stage == "noise":
        stress_noise()
    elif stage == "skew":
        stress_skew()
    elif stage == "stress":
        stress_noise()
        stress_skew()
