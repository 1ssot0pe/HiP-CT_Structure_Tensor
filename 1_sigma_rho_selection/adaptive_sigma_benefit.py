"""
adaptive_sigma_benefit.py

Does per-voxel Lindeberg-adaptive sigma actually CHANGE anything downstream?

lindeberg_feature_size.py produces sigma*(x), a per-voxel certified scale, and
its docstring proposes it as "the voxel-continuous adaptive sigma field that
sigma_rho_adaptive could only deliver piecewise-constant". That is a claim about
what the field IS, not about whether using it changes any answer. This file
measures the change.

The measurement
---------------
Build the fibre-direction field e3 several ways on the SAME real i74 block and
report the angular difference distribution against one reference field:

    ref   sigma = 0.85, rho = 0.85      the discretisation floor, constant
    A     sigma = sigma*(x)             per-voxel adaptive, UNCLAMPED
    B     sigma = max(sigma*(x), 0.85)  per-voxel adaptive, CLAMPED to the floor
                                        -- the only version that is deployable,
                                        since sigma_rho_global REJECTS any pick
                                        below 0.85
    C     sigma = 0.90, rho = 0.90      the other dense-box pick (0.88/0.90 is
                                        the entire real spread behind the
                                        "regional drift" finding)
    D     sigma = 1.15, rho = 1.15      a uniform +35% -- wider than the whole
                                        spatial spread of sigma*
    E     sigma = 0.85, rho = 3.46      the rho end of SIGMA_RHO.md's "loosely
                                        pinned" 2-3.5 range

C, D and E are the SCALE BAR. An angular difference means nothing on its own;
it only means something next to how much the field moves under changes we
already accept as within the parameter's uncertainty. If adaptive-vs-constant
is small compared to the rho ambiguity (E), adaptivity is not the thing to
spend effort on.

rho is tied to rho = 1*sigma in arms A-D, matching the convention sigma*(x) was
measured under (lindeberg_feature_size.RHO_RATIO). Comparing an adaptive sigma
against a constant sigma at a DIFFERENT rho would confound the two parameters.

    F     adaptive max(sigma*, 0.85), rho = 3.46 pinned
    G     adaptive sigma*(x) unclamped,  rho = 3.46 pinned
    H     sigma = 0.90,                  rho = 3.46 pinned   (scale bar)

F, G and H are the CONTROL, against a reference at sigma=0.85, rho=3.46. Arm E
turns out to dominate every sigma effect by an order of magnitude, which means
the rho=1*sigma reference is still under-integrated and jittery -- a sigma
verdict that only held against a noisy reference would be worth nothing. With
rho pinned at a settled value, only sigma varies, and F vs H is the comparison
the whole question reduces to.

Building e3 at a per-voxel sigma
--------------------------------
There is no "structure tensor at a spatially varying scale" primitive. The
tensor is computed on the same geometric grid sigma* was selected on, and the
per-voxel result is the linear blend of the two bracketing grid tensors,
weighted by a tent function in LOG sigma (the grid is geometric, so log spacing
is uniform and the tent is symmetric). Blending the tensor -- not the direction
-- is the right level: e3 is a nonlinear readout of S, and averaging directions
across scales would average over the sign ambiguity of an eigenvector.

The HOST fire-fraction question
-------------------------------
Asked, and answered structurally first: the i74 gate is
refined_field.cheap_gate -> two_orientation_test.patch_gradients(patch,
sigma_ref=3.0), i.e. lambda3 of the patch gradient matrix at a FIXED
sigma_grad = 3.0, computed straight from the image. The e3 field's sigma never
enters it. So switching e3 from a constant sigma to sigma*(x) changes the HOST
fire fraction by exactly zero -- that is a property of the code path, not an
empirical result, and no experiment is needed to establish it.

The empirical question worth asking instead is the one adaptive_triage.py
already poses: what if the map SET the gate's sigma_grad per patch? That is
measured here (gate_arm) as the fire-set flip rate under the percentile-65
operating rule that run_i74_gated actually uses. Note the percentile rule pins
the fire FRACTION at 65% by construction, so the meaningful quantity is which
patches flip, not how many fire.

Run
---
    python adaptive_sigma_benefit.py                  # 96^3 core at the volume centre
    python adaptive_sigma_benefit.py --core 128       # more voxels, ~2x the RAM
    python adaptive_sigma_benefit.py --centre 384 384 384

Writes <out>_results.json (all statistics) and <out>_fields.npz (the per-voxel
angle maps and sigma* on the core) for the figure step.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from structure_tensor import structure_tensor_3d, eig_special_3d

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from core.reconstruction import VOXEL_SIZE_UM, DEFAULT_VOLUME_DIR
from lindeberg_feature_size import (SIGMAS, RHO_RATIO, TRUNCATE, RESP_SMOOTH,
                                    AIR_FRAC, scale_select, kernel_halo,
                                    load_calibration, sigma_to_um)

SIGMA_FLOOR = 0.85          # discretisation bound (sigma_rho_global.py)
RHO_LOOSE = 3.46            # the rho end of the "loosely pinned" range
EPS = 1e-12

# Gate arm (mirrors refined_field's patch geometry; STEP reduced because the
# block here is a 96^3 core, not a full slice -- see gate_arm docstring).
GATE_HALF = 20
GATE_SIGMA_REF = 3.0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_block(folder, z, y, x, verbose=True):
    """Load volume[z0:z1, y0:y1, x0:x1] by reading only the z slices needed."""
    import tifffile
    from pathlib import Path
    files = sorted(Path(folder).glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"no *.tif in {folder}")
    t0 = time.time()
    blk = np.stack([tifffile.imread(f)[y[0]:y[1], x[0]:x[1]] for f in files[z[0]:z[1]]])
    blk = blk.astype(np.float32)
    if verbose:
        print(f"  block {blk.shape} z{z} y{y} x{x}  mean={blk.mean():.1f} "
              f"({time.time() - t0:.1f}s)")
    return blk


# ---------------------------------------------------------------------------
# e3 at a constant scale and at a per-voxel scale
# ---------------------------------------------------------------------------

def e3_constant(vol, sigma, rho, truncate=TRUNCATE):
    """Fibre direction at one global (sigma, rho). vec IS e3 (smallest-eigenvalue
    eigenvector) -- same call orthogonality_residual.fibre_direction makes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        S = structure_tensor_3d(vol, float(sigma), float(rho), truncate=truncate)
        _val, vec = eig_special_3d(S)
    del S
    return vec


def e3_adaptive(vol, sigma_star, sigmas=SIGMAS, rho_ratio=RHO_RATIO,
                truncate=TRUNCATE, floor=None, rho_fixed=None, verbose=True):
    """Fibre direction at a per-voxel sigma = sigma_star(x) (optionally floored).

    S(x) is the tent-weighted blend in log sigma of the tensors on `sigmas`;
    with a geometric grid the log spacing is uniform, so the weights of the two
    bracketing scales sum to 1 everywhere and the blend is an interpolation, not
    a smoothing. sigma_star outside the grid is clamped to it first (those are
    the rail voxels, which are censoring bounds anyway).

    rho_fixed pins the integration scale instead of tying it to sigma. The map
    was measured at rho = 1*sigma and that is the default, but arm E shows rho
    dominates the field by an order of magnitude, so the sigma question has to
    be re-asked with rho held constant -- otherwise "adaptive sigma" is being
    compared against a reference whose rho is too small to have settled.
    """
    sig = np.asarray(sigma_star, dtype=np.float32)
    if floor is not None:
        sig = np.maximum(sig, np.float32(floor))
    us = np.log(np.asarray(sigmas, dtype=np.float64))
    du = float(us[1] - us[0])
    u = np.log(np.clip(sig, sigmas[0], sigmas[-1])).astype(np.float32)
    del sig

    S_acc = None
    for j, s in enumerate(sigmas):
        w = np.abs(u - np.float32(us[j]))
        w *= np.float32(-1.0 / du)
        w += np.float32(1.0)
        np.clip(w, 0.0, 1.0, out=w)             # tent in log sigma
        if not w.any():
            if verbose:
                print(f"    sigma={s:5.2f}  (no voxels)")
            continue
        t0 = time.time()
        rho_s = float(rho_fixed) if rho_fixed is not None else float(rho_ratio * s)
        S = structure_tensor_3d(vol, float(s), rho_s, truncate=truncate)
        S *= w                                   # broadcast over the 6 components
        if S_acc is None:
            S_acc = S
        else:
            S_acc += S
            del S
        if verbose:
            print(f"    sigma={s:5.2f}  w>0 on {float((w > 0).mean()):5.1%} "
                  f"({time.time() - t0:4.1f}s)", flush=True)
        del w
    del u
    with np.errstate(divide="ignore", invalid="ignore"):
        _val, vec = eig_special_3d(S_acc)
    del S_acc
    return vec


def angle_deg(a, b):
    """Angle between two unit axis fields, in degrees, sign-ambiguity removed."""
    d = np.abs(np.einsum("i...,i...->...", a, b))
    np.clip(d, 0.0, 1.0, out=d)
    return np.degrees(np.arccos(d)).astype(np.float32)


def crop(arr, h):
    """Drop the halo from the last three axes."""
    if h <= 0:
        return arr
    return arr[..., h:-h, h:-h, h:-h]


def summarise(ang, mask):
    """Percentiles over `mask`, dropping voxels where e3 is undefined.

    Skielex returns a non-finite eigenvector at isotropic voxels (the
    normalisation divides by a zero norm), so those carry no direction and are
    excluded rather than counted as agreement or disagreement. The excluded
    fraction is reported -- if it were large the percentiles would be a
    statement about a biased subset.
    """
    v = ang[mask]
    n_all = v.size
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, frac_undefined=1.0)
    q = np.percentile(v, [50, 75, 90, 95, 99])
    return dict(n=int(v.size),
                frac_undefined=float(1.0 - v.size / max(n_all, 1)),
                mean=float(v.mean()), median=float(q[0]),
                p75=float(q[1]), p90=float(q[2]), p95=float(q[3]),
                p99=float(q[4]), max=float(v.max()),
                frac_gt_5=float((v > 5).mean()), frac_gt_10=float((v > 10).mean()),
                frac_gt_20=float((v > 20).mean()))


def method_control(shape=(64, 64, 64), width=2.0, noise=0.05, seed=0):
    """How much angular difference does the BLEND ITSELF invent?

    Two controls on a noisy tube phantom, both in the same degrees the real arms
    are reported in, so the real numbers can be read against them:

      consistency   : sigma*(x) set to an exact grid point must reproduce
                      e3_constant at that scale -- any difference is a bug.
      interpolation : sigma*(x) set to the geometric MIDPOINT between two grid
                      points, the worst case for a tent blend, compared against
                      the tensor actually computed at that scale. This is the
                      approximation error of building a per-voxel-sigma field by
                      interpolation, and it is the floor below which a measured
                      adaptive-vs-constant difference means nothing.

    The phantom is noisy on purpose: on a NOISELESS tube the smallest
    eigenvalue is ~0 everywhere, the eigenvector normalisation divides by zero
    and e3 is undefined at every voxel.
    """
    from lindeberg_sigma_phantoms import make_tube_phantom
    rng = np.random.default_rng(seed)
    vol = (make_tube_phantom(shape=shape, width=width)
           + noise * rng.standard_normal(shape)).astype(np.float32)
    m = shape[0] // 4

    def core_stats(a):
        c = a[m:-m, m:-m, m:-m]
        c = c[np.isfinite(c)]
        return dict(median=float(np.median(c)), p95=float(np.percentile(c, 95)),
                    max=float(c.max()))

    g = float(SIGMAS[5])
    cons = angle_deg(e3_constant(vol, g, g),
                     e3_adaptive(vol, np.full(shape, g, np.float32), verbose=False))
    g2 = float(np.sqrt(SIGMAS[5] * SIGMAS[6]))
    interp = angle_deg(e3_constant(vol, g2, g2),
                       e3_adaptive(vol, np.full(shape, g2, np.float32), verbose=False))
    return dict(consistency_at_grid_point=core_stats(cons),
                interpolation_at_midpoint=core_stats(interp),
                grid_sigma=g, midpoint_sigma=g2)


# ---------------------------------------------------------------------------
# HOST gate arm
# ---------------------------------------------------------------------------

def gate_arm(core_vol, sigma_star_core, half=GATE_HALF, step=12, n_z=5,
             verbose=True):
    """Does the sigma* map change which patches fire the HOST gate?

    Not the same question as the e3 arms above -- see the module docstring. The
    gate's lambda3 is computed at a fixed sigma_grad=3.0 and cannot see the e3
    field's sigma at all. This asks the adaptive_triage question instead: if the
    map SET sigma_grad per patch (patch-median sigma*), which patches change
    side of the percentile-65 fire rule that run_i74_gated applies?

    Patches are the refined_field geometry (side 2*half+1) but on a 96^3 core,
    so they overlap heavily at step=12; n_patches is reported and the effective
    independent sample is ~ (core/patch)^3, far smaller. Treat these as a
    directional check, not a precise rate.
    """
    from core.two_orientation_test import patch_gradients

    v = core_vol.astype(np.float32)
    v = (v - v.min()) / (np.ptp(v) + EPS)        # [0,1], as run_i74_gated does
    n = v.shape[0]
    centres = list(range(half, n - half, step))
    if not centres:
        return dict(n_patches=0, note="core too small for the gate patch size")
    zc = centres[:: max(1, len(centres) // n_z)][:n_z]

    def lam3(patch, sg):
        G = patch_gradients(patch, sg)
        if G.shape[0] < 40:
            return 0.0
        return float(np.linalg.eigvalsh(G.T @ G)[0])

    rows = []
    for cz in zc:
        for cy in centres:
            for cx in centres:
                sl = (slice(cz - half, cz + half + 1),
                      slice(cy - half, cy + half + 1),
                      slice(cx - half, cx + half + 1))
                p = v[sl]
                sg_ad = float(np.median(sigma_star_core[sl]))
                rows.append((lam3(p, GATE_SIGMA_REF), lam3(p, sg_ad), sg_ad))
    if verbose:
        print(f"  {len(rows)} patches (side {2*half+1}, step {step}, {len(zc)} z-levels)")

    fixed = np.array([r[0] for r in rows])
    adapt = np.array([r[1] for r in rows])
    sgs = np.array([r[2] for r in rows])

    # the operating rule run_i74_gated actually applies: fire the top 65%
    gate_pct = 35.0
    fire_f = fixed > np.percentile(fixed, gate_pct)
    fire_a = adapt > np.percentile(adapt, gate_pct)
    flip = float(np.mean(fire_f != fire_a))

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        ra, rb = ra - ra.mean(), rb - rb.mean()
        return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + EPS))

    return dict(n_patches=len(rows), step=step, half=half,
                sigma_grad_fixed=GATE_SIGMA_REF,
                sigma_grad_adaptive_median=float(np.median(sgs)),
                sigma_grad_adaptive_min=float(sgs.min()),
                sigma_grad_adaptive_max=float(sgs.max()),
                lam3_fixed_median=float(np.median(fixed)),
                lam3_adaptive_median=float(np.median(adapt)),
                lam3_ratio_median=float(np.median(adapt / (fixed + EPS))),
                spearman_rank=spearman(fixed, adapt),
                fire_fraction_fixed=float(fire_f.mean()),
                fire_fraction_adaptive=float(fire_a.mean()),
                fire_flip_rate=flip)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(centre=(384, 384, 384), core=96, out="adaptive_sigma_benefit",
        folder=DEFAULT_VOLUME_DIR, verbose=True):
    t_start = time.time()
    halo = kernel_halo(SIGMAS, RHO_RATIO, TRUNCATE, RESP_SMOOTH)
    h2 = core // 2
    rng = [(c - h2 - halo, c + h2 + halo) for c in centre]

    print("=" * 84)
    print("DOES PER-VOXEL ADAPTIVE SIGMA CHANGE THE FIBRE FIELD?")
    print(f"  core {core}^3 at centre {centre}, halo {halo} "
          f"(block {core + 2*halo}^3), {VOXEL_SIZE_UM:g} um/voxel")
    print("=" * 84)

    print("\n[0] method control on a phantom -- what the blend itself invents")
    ctrl = method_control()
    print(f"    consistency (sigma* = exact grid point {ctrl['grid_sigma']:.2f}): "
          f"median {ctrl['consistency_at_grid_point']['median']:.4f} deg "
          f"(must be ~0)")
    print(f"    interpolation error (sigma* = midpoint {ctrl['midpoint_sigma']:.2f}): "
          f"median {ctrl['interpolation_at_midpoint']['median']:.4f} deg, "
          f"p95 {ctrl['interpolation_at_midpoint']['p95']:.4f}")
    print("    -> any real difference below this is method noise, not adaptivity.")

    blk = load_block(folder, *rng, verbose=verbose)

    # ---- 1. the per-voxel scale field -------------------------------------
    print("\n[1] per-voxel Lindeberg scale selection (the sigma* map)")
    t0 = time.time()
    sel = scale_select(blk, SIGMAS, RHO_RATIO, TRUNCATE, RESP_SMOOTH,
                       verbose=verbose)
    print(f"    ({time.time() - t0:.0f}s)")
    sig_core = crop(sel["sigma_star"], halo)
    rail_lo_core = crop(sel["rail_lo"], halo)
    rail_hi_core = crop(sel["rail_hi"], halo)
    sigma_star = sel["sigma_star"]                      # full block, for the blend
    del sel

    core_vol = crop(blk, halo)
    tissue = core_vol > AIR_FRAC * 255.0
    certified = tissue & ~rail_lo_core & ~rail_hi_core

    calib = load_calibration()
    um = sigma_to_um(sig_core[certified], calib)
    sstat = dict(
        tissue_frac=float(tissue.mean()),
        certified_frac_of_tissue=float(certified.sum() / max(tissue.sum(), 1)),
        rail_lo_frac_of_tissue=float(rail_lo_core[tissue].mean()),
        rail_hi_frac_of_tissue=float(rail_hi_core[tissue].mean()),
        sigma_median=float(np.median(sig_core[certified])),
        sigma_p05=float(np.percentile(sig_core[certified], 5)),
        sigma_p95=float(np.percentile(sig_core[certified], 95)),
        um_median=float(np.median(um)),
        um_p05=float(np.percentile(um, 5)),
        um_p95=float(np.percentile(um, 95)),
        frac_below_floor=float((sig_core[certified] < SIGMA_FLOOR).mean()),
    )
    print(f"    tissue {sstat['tissue_frac']:.0%} of core; of tissue: "
          f"certified {sstat['certified_frac_of_tissue']:.0%}, "
          f"rail_lo {sstat['rail_lo_frac_of_tissue']:.1%}, "
          f"rail_hi {sstat['rail_hi_frac_of_tissue']:.1%}")
    print(f"    sigma* over certified tissue: median {sstat['sigma_median']:.3f} vox "
          f"({sstat['um_median']:.0f} um), 5-95% "
          f"{sstat['sigma_p05']:.3f}-{sstat['sigma_p95']:.3f} vox "
          f"({sstat['um_p05']:.0f}-{sstat['um_p95']:.0f} um)")
    print(f"    BELOW the {SIGMA_FLOOR} discretisation floor: "
          f"{sstat['frac_below_floor']:.1%} of certified tissue voxels")
    del um

    # ---- 2. the reference field -------------------------------------------
    print(f"\n[2] reference e3 at constant sigma={SIGMA_FLOOR}, rho={SIGMA_FLOOR}")
    ref = crop(e3_constant(blk, SIGMA_FLOOR, SIGMA_FLOOR), halo).copy()

    arms = {}
    angles = {}

    def add_arm(key, label, e3_block, reference=None, ref_label="ref sigma=0.85 rho=0.85"):
        e3c = crop(e3_block, halo)
        a = angle_deg(ref if reference is None else reference, e3c)
        arms[key] = dict(label=label, vs=ref_label, **summarise(a, certified))
        angles[key] = a
        s = arms[key]
        print(f"    {label:<46} median {s['median']:6.2f} deg   "
              f"p95 {s['p95']:6.2f}   >5deg {s['frac_gt_5']:5.1%}")
        del e3c, e3_block

    # ---- 3. the adaptive arms ---------------------------------------------
    print("\n[3] adaptive arms (per-voxel sigma*, rho = sigma*)")
    print("    A: UNCLAMPED sigma*(x)")
    add_arm("A_adaptive_raw", "A  adaptive sigma*(x), unclamped",
            e3_adaptive(blk, sigma_star, floor=None, verbose=verbose))
    print(f"    B: CLAMPED to the {SIGMA_FLOOR} floor (the deployable version)")
    add_arm("B_adaptive_floored", f"B  adaptive max(sigma*, {SIGMA_FLOOR})",
            e3_adaptive(blk, sigma_star, floor=SIGMA_FLOOR, verbose=verbose))

    # ---- 4. the scale bar --------------------------------------------------
    print("\n[4] scale bar: constant-sigma changes we already accept")
    add_arm("C_sigma090", "C  constant sigma=0.90 (other dense-box pick)",
            e3_constant(blk, 0.90, 0.90))
    add_arm("D_sigma115", "D  constant sigma=1.15 (+35%, > sigma* spread)",
            e3_constant(blk, 1.15, 1.15))
    add_arm("E_rho346", f"E  constant sigma=0.85, rho={RHO_LOOSE} (rho ambiguity)",
            e3_constant(blk, SIGMA_FLOOR, RHO_LOOSE))
    del ref

    # ---- 4b. the same sigma question at a SETTLED rho ----------------------
    # Arm E says rho moves the field far more than any sigma here, which means
    # the rho=1*sigma reference above is under-integrated and still jittery.
    # If the sigma verdict only holds against a noisy reference it is worth
    # nothing, so both adaptive arms are re-run with rho pinned at RHO_LOOSE
    # for every voxel, against a reference at the same rho. Only sigma varies.
    print(f"\n[4b] control: the same comparison with rho pinned at {RHO_LOOSE}")
    ref2 = crop(e3_constant(blk, SIGMA_FLOOR, RHO_LOOSE), halo).copy()
    rl = f"ref2 sigma=0.85 rho={RHO_LOOSE}"
    add_arm("F_adaptive_floored_rho346",
            f"F  adaptive max(sigma*,{SIGMA_FLOOR}), rho={RHO_LOOSE}",
            e3_adaptive(blk, sigma_star, floor=SIGMA_FLOOR,
                        rho_fixed=RHO_LOOSE, verbose=False),
            reference=ref2, ref_label=rl)
    add_arm("G_adaptive_raw_rho346",
            f"G  adaptive sigma*(x) unclamped, rho={RHO_LOOSE}",
            e3_adaptive(blk, sigma_star, floor=None,
                        rho_fixed=RHO_LOOSE, verbose=False),
            reference=ref2, ref_label=rl)
    add_arm("H_sigma090_rho346",
            f"H  constant sigma=0.90, rho={RHO_LOOSE} (scale bar at same rho)",
            e3_constant(blk, 0.90, RHO_LOOSE), reference=ref2, ref_label=rl)

    del blk, ref2, sigma_star

    # ---- 5. HOST gate ------------------------------------------------------
    print("\n[5] HOST gate arm")
    print("    NOTE the fire fraction is INVARIANT to the e3 sigma by construction:")
    print("    cheap_gate -> patch_gradients(patch, sigma_ref=3.0) reads the IMAGE,")
    print("    not the tensor field. Measuring instead: sigma* SETTING sigma_grad.")
    gate = gate_arm(core_vol, sig_core, verbose=verbose)
    if gate.get("n_patches"):
        print(f"    sigma_grad: fixed {gate['sigma_grad_fixed']:.2f} vs "
              f"patch-median sigma* {gate['sigma_grad_adaptive_median']:.2f} "
              f"(range {gate['sigma_grad_adaptive_min']:.2f}-"
              f"{gate['sigma_grad_adaptive_max']:.2f})")
        print(f"    lambda3 median ratio adaptive/fixed: "
              f"{gate['lam3_ratio_median']:.3g}")
        print(f"    rank correlation of the two lambda3 orderings: "
              f"{gate['spearman_rank']:+.3f}")
        print(f"    fire-set FLIP rate under the percentile-65 rule: "
              f"{gate['fire_flip_rate']:.1%}  "
              f"(fire fraction pinned at {gate['fire_fraction_fixed']:.0%} both ways)")

    # ---- 6. save -----------------------------------------------------------
    res = dict(centre=list(centre), core=core, halo=halo,
               voxel_um=VOXEL_SIZE_UM, sigma_floor=SIGMA_FLOOR,
               sigmas=np.asarray(SIGMAS).tolist(), rho_ratio=RHO_RATIO,
               resp_smooth=RESP_SMOOTH, truncate=TRUNCATE,
               method_control=ctrl, sigma_star=sstat, arms=arms, gate=gate,
               runtime_min=(time.time() - t_start) / 60.0)
    with open(f"{out}_results.json", "w") as f:
        json.dump(res, f, indent=2)
    mid = core // 2
    np.savez_compressed(
        f"{out}_fields.npz",
        sigma_star_mid=sig_core[mid], certified_mid=certified[mid],
        vol_mid=core_vol[mid],
        sigma_star_certified=sig_core[certified].astype(np.float32),
        **{f"ang_{k}_mid": angles[k][mid] for k in angles},
        **{f"ang_{k}_certified": angles[k][certified].astype(np.float32)
           for k in angles})
    print(f"\nwrote {out}_results.json, {out}_fields.npz "
          f"({res['runtime_min']:.1f} min)")
    print("=" * 84)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--core", type=int, default=96)
    p.add_argument("--centre", nargs=3, type=int, default=[384, 384, 384])
    p.add_argument("--out", default="adaptive_sigma_benefit")
    a = p.parse_args()
    run(centre=tuple(a.centre), core=a.core, out=a.out)
