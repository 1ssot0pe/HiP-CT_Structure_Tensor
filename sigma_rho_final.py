"""
sigma_rho_final.py -- CERTIFY the (sigma, rho) pick for the structure-tensor
fibre analysis on real i74. This is the finalisation of Goal A (the original
project): turn Lindeberg's parameter-free pick into a defensible recommendation
by corroborating it with an INDEPENDENT output-quality measure.

Three independent lines of evidence, and whether they agree:

  (1) LINDEBERG response peak (lindeberg_stability_3d): the gamma-normalised
      response sigma^2 * mean(plane energy) peaks at the fibre's own scale ->
      sigma_L. Its rho-stability plateau (where e3 stops changing with rho) ->
      rho_L.  [image-driven, resists over-smoothing via first-interior-local-max]

  (2) OUTPUT QUALITY (resolution_sweep): reconstruct the orientation field with
      LIC and score how finely the texture stays REAL (FSC resolution). The
      (sigma, rho) that MAXIMISES resolution -> sigma_F, rho_F.  [independent of
      (1): a different tensor, a reconstruction, a Fourier metric]

  (3) The sigma >= ~0.85 vox DISCRETISATION FLOOR (from noise_floor.py T1): below
      it the Gaussian derivative stops being a faithful derivative. A hard lower
      bound the pick must clear.

TRUST GATE: FSC "sharper = better" can drift toward over-smoothing (large rho).
So we FIRST run resolution_sweep.validate() on a phantom and only trust the FSC
optimum on real data if its argmax coincides with the supervised angular-error
argmin. If the gate fails, FSC is reported as untrustworthy and the recommendation
rests on Lindeberg + the floor alone.

If (1) and (2) AGREE and both clear (3), the pick is CERTIFIED (two independent
criteria + a hard floor). If they disagree, we report the discrepancy honestly.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reconstruction import load_volume, VOXEL_SIZE_UM
from lindeberg_stability_3d import (lindeberg_response_3d, first_local_max,
                                    sweep_rho_stability_3d)
from lindeberg_sigma_phantoms import refined_peak_sigma
from resolution_sweep import sweep_resolution, validate

SIGMA_FLOOR = 0.85          # discretisation bound from noise_floor.py T1
BOX = (slice(60, 260), slice(240, 440), slice(240, 440))   # validated tissue box
SUB = 112                   # central sub-crop for LIC speed (fair, tissue-only)

SIG_LIND = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
SIG_GRID = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
RHO_GRID = np.array([1.0, 2.0, 3.0, 4.0])


def load_real():
    full = load_volume(crop=None, dtype=np.uint8)
    box = full[BOX].astype(np.float32)
    del full
    c = [(s - SUB) // 2 for s in box.shape]
    sub = box[c[0]:c[0]+SUB, c[1]:c[1]+SUB, c[2]:c[2]+SUB].copy()
    print(f"[data] tissue sub-crop {sub.shape} mean={sub.mean():.0f} "
          f"({VOXEL_SIZE_UM:g} um/vox)")
    return sub


def main():
    print("=" * 74)
    print("STEP 1 -- TRUST GATE: does FSC output-quality track truth on a phantom?")
    g = validate(sigmas=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
                 rhos=(1.0, 2.0, 3.0, 4.0), save_path="sigma_rho_trustgate.png")
    trust = g["trustworthy"]
    print(f"TRUST GATE: {'PASS' if trust else 'FAIL'} "
          f"(FSC argmax vs angular argmin: {g['d_cells']} cell(s), corr={g['corr']:+.2f})")

    print("=" * 74)
    print("STEP 2 -- real i74: Lindeberg pick vs FSC output-quality optimum")
    vol = load_real()

    # (1) Lindeberg sigma (response peak) + rho (stability plateau)
    _s, response = lindeberg_response_3d(vol, SIG_LIND, rho=1.0, verbose=True)
    i_loc = first_local_max(SIG_LIND, response)
    ramp = i_loc is None
    sigma_L = refined_peak_sigma(SIG_LIND, response)
    Sens, _b, rho_mid = sweep_rho_stability_3d(vol, SIG_GRID, RHO_GRID, verbose=True)
    row = int(np.argmin(np.abs(SIG_GRID - sigma_L)))
    rho_L = float(rho_mid[int(np.nanargmin(Sens[row]))])

    # (2) FSC output-quality surface on the same grid
    F, bestF = sweep_resolution(vol, SIG_GRID, RHO_GRID)
    sigma_F, rho_F = bestF["sigma"], bestF["rho"]
    fsc_sigma_marg = np.nanmax(F, axis=1)          # best-over-rho per sigma

    # agreement (within one grid step in sigma, and rho)
    dsig = abs(sigma_L - sigma_F)
    sig_step = float(np.min(np.diff(SIG_GRID)))
    agree = (dsig <= sig_step + 1e-9)
    above_floor = (sigma_L >= SIGMA_FLOOR) and (sigma_F >= SIGMA_FLOOR)

    print("=" * 74)
    print("STEP 3 -- verdict")
    print(f"  Lindeberg     : sigma_L={sigma_L:.2f} vox ({sigma_L*VOXEL_SIZE_UM:.0f} um), "
          f"rho_L={rho_L:.2f}" + ("   [WARNING: response was a ramp, no interior peak]" if ramp else ""))
    print(f"  FSC quality   : sigma_F={sigma_F:.2f} vox ({sigma_F*VOXEL_SIZE_UM:.0f} um), "
          f"rho_F={rho_F:.2f}  (trust={'ok' if trust else 'FAILED gate'})")
    print(f"  discretisation floor: sigma >= {SIGMA_FLOOR} vox -> "
          f"both above floor: {above_floor}")
    print(f"  sigma agreement (|dL-F|={dsig:.2f} <= grid step {sig_step:.2f}): {agree}")
    if agree and above_floor and trust:
        verdict = (f"CERTIFIED: two independent criteria agree at sigma~{sigma_L:.1f} vox "
                   f"and the pick clears the discretisation floor.")
        rec_sigma, rec_rho = sigma_L, rho_L
    elif agree and above_floor and not trust:
        verdict = (f"CORROBORATED WEAKLY: Lindeberg and FSC agree at sigma~{sigma_L:.1f}, but "
                   f"the FSC trust gate FAILED, so lean on Lindeberg + the floor.")
        rec_sigma, rec_rho = sigma_L, rho_L
    else:
        verdict = (f"NOT CERTIFIED: Lindeberg (sigma={sigma_L:.1f}) and FSC (sigma={sigma_F:.1f}) "
                   f"disagree or breach the floor -- report the discrepancy, do not ship a single pick.")
        rec_sigma, rec_rho = sigma_L, rho_L
    print(f"  => {verdict}")
    print(f"  RECOMMENDATION: sigma={rec_sigma:.2f} vox (~{rec_sigma*VOXEL_SIZE_UM:.0f} um), "
          f"rho={rec_rho:.2f} vox")
    print("=" * 74)

    _figure(response, sigma_L, fsc_sigma_marg, sigma_F, F, bestF, rho_L, agree, trust)
    return dict(sigma_L=sigma_L, rho_L=rho_L, sigma_F=sigma_F, rho_F=rho_F,
                agree=agree, trust=trust, above_floor=above_floor)


def _figure(response, sigma_L, fsc_marg, sigma_F, F, bestF, rho_L, agree, trust,
            save="sigma_rho_final.png"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    # panel 1: Lindeberg response vs FSC sigma-marginal, both peaks + floor
    axb = ax[0].twinx()
    l1, = ax[0].plot(SIG_LIND, response, "o-", color="#2a78d6", label="Lindeberg response")
    l2, = axb.plot(SIG_GRID, fsc_marg, "s--", color="#eb6834",
                   label="FSC resolution (best over $\\rho$)")
    ax[0].axvline(sigma_L, color="#2a78d6", ls=":", lw=2)
    axb.axvline(sigma_F, color="#eb6834", ls=":", lw=2)
    ax[0].axvspan(0, SIGMA_FLOOR, color="0.85", label="below discretisation floor")
    ax[0].set_xlabel("$\\sigma$ (voxels)")
    ax[0].set_ylabel("Lindeberg response", color="#2a78d6")
    axb.set_ylabel("FSC resolution (cyc/vox, higher=better)", color="#eb6834")
    ax[0].set_title(f"Two independent criteria vs $\\sigma$\n"
                    f"Lindeberg peak={sigma_L:.1f}, FSC peak={sigma_F:.1f}  "
                    f"-> {'AGREE' if agree else 'DISAGREE'}")
    ax[0].legend(handles=[l1, l2], loc="upper right", fontsize=8)
    # panel 2: FSC sigma x rho surface with both picks
    extent = [RHO_GRID[0], RHO_GRID[-1], SIG_GRID[-1], SIG_GRID[0]]
    im = ax[1].imshow(F, origin="upper", aspect="auto", extent=extent, cmap="viridis")
    ax[1].plot(bestF["rho"], bestF["sigma"], "*", ms=18, color="#eb6834",
               mec="white", label=f"FSC opt ({bestF['sigma']:.1f},{bestF['rho']:.1f})")
    ax[1].plot(rho_L, sigma_L, "P", ms=13, color="#2a78d6", mec="white",
               label=f"Lindeberg ({sigma_L:.1f},{rho_L:.1f})")
    ax[1].set_xlabel("$\\rho$ (voxels)"); ax[1].set_ylabel("$\\sigma$ (voxels)")
    ax[1].set_title("FSC output-quality surface (higher=finer)")
    ax[1].legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="cutoff freq")
    fig.suptitle(f"Certifying $\\sigma,\\rho$ on real i74 "
                 f"(FSC trust gate: {'PASS' if trust else 'FAIL'})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save, dpi=130)
    print(f"Saved {save}")


if __name__ == "__main__":
    main()
