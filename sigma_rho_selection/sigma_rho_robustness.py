"""
sigma_rho_robustness.py -- targeted follow-ups to sigma_rho_global.py's
(sigma, rho) pick, each closing a weakness this project already documented
but never actually measured.

UPDATE: FSC output-quality corroboration (originally item (4) below, "refined
FSC grid") has been REMOVED. Running sigma_rho_global.py's own trust gate on
real i74 showed FSC does not track direction quality even on a clean phantom
(argmax 5 grid cells from the supervised angular-error argmin, surface
correlation only -0.10) -- so it was deleted from sigma_rho_global.py, and the
refinement experiment built on top of it is deleted here too. (2) below now
checks Lindeberg-only consistency across tissue boxes instead of also
tracking an FSC pick.

  (1) REAL-NOISE ADAPTIVITY. toy2d's Layer-4 finding: Lindeberg's sigma pick
      stayed flat as (synthetic white) noise grew, while a "true" optimal
      sigma appeared to increase -- flagged as untested on real data. A first
      pass here (fixed rho=1.0 while sweeping sigma) reproduced that same
      "CONFIRMED" verdict -- but that test was flawed: holding rho fixed
      forces sigma to inflate as the only compensating knob. Corrected: let
      rho vary jointly (matching what the real pipeline does via the
      rho-stability sweep) and compare pick_sigma_rho's full (sigma,rho) pick
      against the true 2D angular-error optimum, using lung_A.npy/lung_B.npy
      REAL measured noise injected into a clean single-width tube phantom
      (known ground-truth direction). Result: the joint true optimum barely
      moves with noise; Lindeberg's bias relative to it is small and does not
      grow with noise -- not a real adaptivity failure.

  (2) MULTI-BOX CONSISTENCY. sigma_rho_global.py certifies one (sigma, rho)
      from a single 112^3 box. i74_calibration_bias_check.py (mass-
      conservation work) showed real i74 tissue density/complexity varies
      sharply by region. adaptive_triage.py exists precisely because a
      single global sigma_grad misfires region-to-region for HOST-FOD. Does
      the certified structure-tensor (sigma, rho) hold across a few
      different real tissue boxes, or drift the same way?

  (3) HIGH-PASS PRE-FILTER. first_local_max is a downstream patch: it stops
      large-scale shading/curvature from hijacking the sigma pick, but only
      after the fact. Removing the confound at the source (subtract a
      heavily-smoothed background before the Lindeberg sweep) should make a
      plain global argmax agree with first_local_max again on real data --
      check whether it actually does.

Kept at reduced grids/box sizes relative to sigma_rho_global.py so the whole
script runs in a few minutes -- these are diagnostic checks on the known
weaknesses, not a re-certification.
"""

import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root
from reconstruction import load_volume, VOXEL_SIZE_UM
from orthogonality_residual import fibre_direction, crop_margin, _crop3
from mass_conservation_validate import real_noise  # lung_A/B measured noise

from pick_sigma_rho import (lindeberg_response_3d, first_local_max,
                                     sweep_rho_stability_3d, pick_sigma_rho, _crop_vec3)
from lindeberg_sigma_phantoms import make_tube_phantom, refined_peak_sigma

RHO_REF = 1.0
TRUNCATE = 4.0


# ===========================================================================
# (1) Real-noise adaptivity
# ===========================================================================
#
# FIX HISTORY: the first version of this check held rho FIXED at 1.0 while
# calling the resulting best sigma "the true optimum" -- which forces sigma to
# inflate as noise grows, since sigma was the only knob available to
# compensate. That produced a false-positive "Lindeberg isn't noise-adaptive"
# verdict. Corrected version: let rho vary too and compare against the JOINT
# 2D (sigma, rho) angular-error optimum, using pick_sigma_rho -- the actual
# recipe the pipeline runs (Lindeberg sigma + rho-stability sweep), not a
# sigma-only strawman. Result: the true joint optimum barely moves with noise
# (only the achievable error grows, which no scale pick can undo); Lindeberg's
# pick carries a modest, roughly noise-INVARIANT bias relative to it, not a
# noise-adaptivity failure. So the fix here IS the corrected test.

def _angular_error_surface(vol, true_dir, sigmas, rhos, truncate=TRUNCATE):
    """Mean angular error (deg) between recovered e3 and the KNOWN true
    direction, over the full (sigma, rho) grid -- the fair ground truth to
    compare pick_sigma_rho's JOINT pick against."""
    sigmas = np.asarray(sigmas, float); rhos = np.asarray(rhos, float)
    A = np.full((len(sigmas), len(rhos)), np.nan)
    td = np.asarray(true_dir, float).reshape(3, 1, 1, 1)
    for i, s in enumerate(sigmas):
        for j, r in enumerate(rhos):
            e3 = fibre_direction(vol, s, r, truncate)
            m = crop_margin(s, r, truncate)
            e3c = _crop_vec3(e3, m) if m > 0 else e3
            tdc = np.broadcast_to(td, e3.shape)
            tdc = _crop_vec3(tdc, m) if m > 0 else tdc
            valid = np.isfinite(e3c).all(axis=0)
            if not valid.any():
                continue
            cos = np.clip(np.abs(np.sum(e3c * tdc, axis=0)), 0.0, 1.0)
            ang = np.degrees(np.arccos(cos))
            A[i, j] = float(np.mean(ang[valid]))
    i, j = np.unravel_index(np.nanargmin(A), A.shape)
    return float(sigmas[i]), float(rhos[j]), float(A[i, j]), A


def real_noise_adaptivity(width=2.0, shape=(64, 96, 96), axis=0,
                           amps=(0.0, 0.03, 0.07, 0.15), seed=0,
                           sigmas_lind=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0),
                           sigmas_grid=(1.0, 2.0, 3.0, 4.0), rhos_grid=(1.0, 2.0, 3.0)):
    print("=" * 88)
    print("(1) REAL-NOISE ADAPTIVITY -- Lindeberg+rho-stability's JOINT (sigma,rho) pick "
          "vs the true 2D angular-error optimum, measured lung noise")
    print("=" * 88)
    sigmas_lind = np.asarray(sigmas_lind, float)
    sigmas_grid = np.asarray(sigmas_grid, float)
    rhos_grid = np.asarray(rhos_grid, float)
    true_dir = np.zeros(3); true_dir[2 - axis] = 1.0   # tubes run along `axis`

    # tiny synthetic dither (1e-3, negligible next to the width=2.0 signal) breaks
    # the EXACT axis-alignment symmetry of a pure straight tube lattice -- with
    # zero along-axis gradient everywhere, the structure tensor is exactly
    # block-diagonal and eig_special_3d's closed-form eigenvector solve divides
    # by zero identically (confirmed: 100% NaN e3 field with noise_sigma=0.0,
    # <0.01% NaN once any real variation, e.g. the amp>0 real noise below, is
    # present). This dither is NOT the real-noise signal being tested -- it's
    # only there so the amp=0.0 "clean" reference point is numerically solvable.
    clean = make_tube_phantom(shape=shape, width=width, axis=axis, noise_sigma=1e-3)
    rows = []
    for amp in amps:
        vol = clean + (real_noise(shape, amp, seed) if amp > 0 else 0.0)
        sigma_L, rho_L, _diag = pick_sigma_rho(vol, sigmas_lind, sigmas_grid, rhos_grid,
                                                rho_ref=RHO_REF, verbose=False)
        sigma_t, rho_t, err_t, _A = _angular_error_surface(vol, true_dir, sigmas_grid, rhos_grid)
        rows.append(dict(amp=amp, sigma_L=sigma_L, rho_L=rho_L,
                          sigma_true=sigma_t, rho_true=rho_t, err_at_true=err_t))
        print(f"  amp={amp:5.2f} | Lindeberg (sigma={sigma_L:.2f}, rho={rho_L:.2f})  |  "
              f"true joint optimum (sigma={sigma_t:.2f}, rho={rho_t:.2f}, err={err_t:.1f} deg)  |  "
              f"gap=({sigma_L - sigma_t:+.2f}, {rho_L - rho_t:+.2f})")

    amps_arr = np.array([r["amp"] for r in rows])
    gap_sig = np.array([r["sigma_L"] - r["sigma_true"] for r in rows])
    gap_rho = np.array([r["rho_L"] - r["rho_true"] for r in rows])
    true_sig_moved = float(np.ptp([r["sigma_true"] for r in rows]))
    true_rho_moved = float(np.ptp([r["rho_true"] for r in rows]))
    # "grows" = |gap| at the highest noise amplitude exceeds |gap| at the lowest by
    # a meaningful margin -- NOT just np.ptp(|gap|), which flags a SHRINKING gap
    # (bias improving as noise grows) identically to a growing one. Trend, not range.
    gap_grows = (abs(gap_sig[-1]) > abs(gap_sig[0]) + 0.5) or (abs(gap_rho[-1]) > abs(gap_rho[0]) + 0.5)

    print(f"\n  true joint optimum movement across noise levels: "
          f"sigma range={true_sig_moved:.2f}, rho range={true_rho_moved:.2f}")
    print(f"  Lindeberg bias vs true optimum: sigma gap {gap_sig[0]:+.2f} -> {gap_sig[-1]:+.2f}, "
          f"rho gap {gap_rho[0]:+.2f} -> {gap_rho[-1]:+.2f} (lowest -> highest noise amplitude)")
    if true_sig_moved < 0.75 and true_rho_moved < 0.75 and not gap_grows:
        print("  VERDICT: NOT a noise-adaptivity failure -- the true (sigma,rho) optimum "
              "barely moves with noise once rho is allowed to vary jointly (only the "
              "achievable error grows, which no scale pick can undo). Lindeberg carries a "
              "modest, roughly noise-INVARIANT bias relative to the true point, not a "
              "growing failure to adapt. (The earlier sigma-only, fixed-rho test that "
              "reported 'CONFIRMED' was measuring an artifact of holding rho fixed.)")
    else:
        print("  VERDICT: the true optimum DOES move with noise even jointly, and/or "
              "Lindeberg's bias grows with it -- a genuine adaptivity gap.")
    print("=" * 88)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].plot(amps_arr, [r["sigma_L"] for r in rows], "o-", label="Lindeberg sigma_L")
    ax[0].plot(amps_arr, [r["sigma_true"] for r in rows], "s--", label="true joint-optimal sigma")
    ax[0].set_xlabel("real-noise amplitude"); ax[0].set_ylabel("sigma (voxels)")
    ax[0].set_title("sigma: pick vs true joint optimum"); ax[0].legend(fontsize=9)
    ax[1].plot(amps_arr, [r["rho_L"] for r in rows], "o-", label="Lindeberg rho_L")
    ax[1].plot(amps_arr, [r["rho_true"] for r in rows], "s--", label="true joint-optimal rho")
    ax[1].set_xlabel("real-noise amplitude"); ax[1].set_ylabel("rho (voxels)")
    ax[1].set_title("rho: pick vs true joint optimum"); ax[1].legend(fontsize=9)
    fig.suptitle("Real-noise adaptivity (corrected, joint sigma+rho comparison)")
    fig.tight_layout()
    fig.savefig("sigma_rho_noise_adaptivity.png", dpi=130)
    print("Saved sigma_rho_noise_adaptivity.png")
    return rows


# ===========================================================================
# (2) Multi-box consistency (Lindeberg only -- FSC removed, see module docstring)
# ===========================================================================

BOXES = {
    "A (original)": (slice(60, 260), slice(240, 440), slice(240, 440)),
    "B": (slice(300, 500), slice(100, 300), slice(100, 300)),
    "C": (slice(400, 600), slice(450, 650), slice(450, 650)),
}
SUB = 96
SIG_LIND = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
SIG_GRID = np.array([1.0, 2.0, 3.0, 4.0])
RHO_GRID = np.array([1.0, 2.0, 3.0])


def _sub_crop(full, box, sub=SUB):
    b = full[box].astype(np.float32)
    c = [(s - sub) // 2 for s in b.shape]
    return b[c[0]:c[0] + sub, c[1]:c[1] + sub, c[2]:c[2] + sub].copy()


def multi_box_consistency():
    print("=" * 88)
    print("(2) MULTI-BOX CONSISTENCY -- same Lindeberg recipe, different real tissue locations")
    print("=" * 88)
    print("  loading full i74 volume ...")
    full = load_volume(crop=None, dtype=np.uint8)

    rows = []
    for name, box in BOXES.items():
        vol = _sub_crop(full, box)
        print(f"\n  --- box {name}: sub-crop {vol.shape}, mean={vol.mean():.0f} ---")
        if vol.mean() < 5.0:
            print("      (looks like background/air -- skipping)")
            continue

        _s, response = lindeberg_response_3d(vol, SIG_LIND, rho=RHO_REF, verbose=False)
        sigma_L = refined_peak_sigma(SIG_LIND, response)
        Sens, _b, rho_mid = sweep_rho_stability_3d(vol, SIG_GRID, RHO_GRID, verbose=False)
        row_i = int(np.argmin(np.abs(SIG_GRID - sigma_L)))
        rho_L = float(rho_mid[int(np.nanargmin(Sens[row_i]))])

        print(f"      Lindeberg: sigma_L={sigma_L:.2f}, rho_L={rho_L:.2f}")
        rows.append(dict(box=name, sigma_L=sigma_L, rho_L=rho_L))
    del full

    if len(rows) >= 2:
        sigL = np.array([r["sigma_L"] for r in rows])
        rhoL = np.array([r["rho_L"] for r in rows])
        print(f"\n  across {len(rows)} boxes:")
        print(f"    sigma_L range={np.ptp(sigL):.2f} (values: {np.round(sigL,2)})")
        print(f"    rho_L   range={np.ptp(rhoL):.2f} (values: {np.round(rhoL,2)})")
        drifted = np.ptp(sigL) > 1.0 or np.ptp(rhoL) > 1.0
        print("  VERDICT: " + ("PICK DRIFTS across tissue regions -- a single global "
              "(sigma,rho) is not well-supported; consider per-region calibration." if drifted
              else "pick is reasonably STABLE across the tested regions."))
    print("=" * 88)
    return rows


# ===========================================================================
# (3) High-pass pre-filter
# ===========================================================================

def highpass_prefilter_check(box=BOXES["A (original)"], sigma_bg=18.0,
                              sigmas=SIG_LIND):
    print("=" * 88)
    print("(3) HIGH-PASS PRE-FILTER -- does removing large-scale background content "
          "make first_local_max agree with a plain global argmax?")
    print("=" * 88)
    full = load_volume(crop=None, dtype=np.uint8)
    vol = _sub_crop(full, box)
    del full

    _s, resp_raw = lindeberg_response_3d(vol, sigmas, rho=RHO_REF, verbose=False)
    i_loc_raw = first_local_max(sigmas, resp_raw)
    i_glob_raw = int(np.argmax(resp_raw))
    ramp_raw = i_loc_raw is None

    background = gaussian_filter(vol, sigma_bg)
    vol_hp = vol - background
    _s, resp_hp = lindeberg_response_3d(vol_hp, sigmas, rho=RHO_REF, verbose=False)
    i_loc_hp = first_local_max(sigmas, resp_hp)
    i_glob_hp = int(np.argmax(resp_hp))
    ramp_hp = i_loc_hp is None

    print(f"  RAW      : first-local-max sigma="
          f"{sigmas[i_loc_raw] if i_loc_raw is not None else 'NONE (ramp)'}, "
          f"global-argmax sigma={sigmas[i_glob_raw]:.2f}, "
          f"agree={(i_loc_raw == i_glob_raw) if not ramp_raw else False}")
    print(f"  HIGH-PASS: first-local-max sigma="
          f"{sigmas[i_loc_hp] if i_loc_hp is not None else 'NONE (ramp)'}, "
          f"global-argmax sigma={sigmas[i_glob_hp]:.2f}, "
          f"agree={(i_loc_hp == i_glob_hp) if not ramp_hp else False}")

    fixed = (ramp_raw or i_loc_raw != i_glob_raw) and (not ramp_hp and i_loc_hp == i_glob_hp)
    if fixed:
        print("  VERDICT: high-pass pre-filtering REMOVES the confound at the source -- "
              "first_local_max and global argmax now agree, where they didn't on raw data.")
    elif ramp_raw and ramp_hp:
        print("  VERDICT: inconclusive on this box -- no interior peak either way "
              "(may need a wider sigma range).")
    else:
        print("  VERDICT: no clear improvement from high-pass on this box/setting.")
    print("=" * 88)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sigmas, resp_raw / np.nanmax(resp_raw), "o-", label="raw")
    ax.plot(sigmas, resp_hp / np.nanmax(resp_hp), "s--", label=f"high-pass (bg sigma={sigma_bg})")
    ax.set_xlabel("sigma"); ax.set_ylabel("Lindeberg response (peak-normalised)")
    ax.set_title("High-pass pre-filter: does it clean up the response curve?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("sigma_rho_highpass_check.png", dpi=130)
    print("Saved sigma_rho_highpass_check.png")
    return dict(ramp_raw=ramp_raw, ramp_hp=ramp_hp, fixed=fixed)


if __name__ == "__main__":
    r1 = real_noise_adaptivity()
    r2 = multi_box_consistency()
    r3 = highpass_prefilter_check()
    print("\n" + "#" * 88)
    print("ALL FOUR ROBUSTNESS CHECKS COMPLETE")
    print("#" * 88)
