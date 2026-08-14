"""

  (1) The gamma-normalised response sigma^2 * mean(plane energy) peaks at the fibre's own scale ->
      sigma_L. Its rho-stability plateau (where e3 stops changing with rho) ->
      rho_L. Resists over-smoothing via first-interior-local-max, not a
      global argmax (see pick_sigma_rho.first_local_max).

  (2) The sigma >= ~0.85 vox DISCRETISATION FLOOR (from noise_floor.py T1):
      below it the Gaussian derivative stops being a faithful derivative. A
      hard lower bound the pick must clear.

"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root, for reconstruction
from reconstruction import load_volume, VOXEL_SIZE_UM
from pick_sigma_rho import (lindeberg_response_3d, first_local_max,
                                    sweep_rho_stability_3d)
from lindeberg_sigma_phantoms import refined_peak_sigma

SIGMA_FLOOR = 0.85          # discretisation bound from noise_floor.py T1
BOX = (slice(60, 260), slice(240, 440), slice(240, 440))   # validated tissue box
SUB = 112                   # central sub-crop (kept from the FSC-era setting)

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
    print("STEP 1 -- real i74: Lindeberg (sigma, rho) pick")
    vol = load_real()

    _s, response = lindeberg_response_3d(vol, SIG_LIND, rho=1.0, verbose=True)
    i_loc = first_local_max(SIG_LIND, response)
    ramp = i_loc is None
    sigma_L = refined_peak_sigma(SIG_LIND, response)
    Sens, _b, rho_mid = sweep_rho_stability_3d(vol, SIG_GRID, RHO_GRID, verbose=True)
    row = int(np.argmin(np.abs(SIG_GRID - sigma_L)))
    sens_row = Sens[row]
    rho_L = float(rho_mid[int(np.nanargmin(sens_row))])

    above_floor = sigma_L >= SIGMA_FLOOR

    print("=" * 74)
    print("STEP 2 -- verdict")
    print(f"  Lindeberg     : sigma_L={sigma_L:.2f} vox ({sigma_L*VOXEL_SIZE_UM:.0f} um), "
          f"rho_L={rho_L:.2f}" + ("   [WARNING: response was a ramp, no interior peak]" if ramp else ""))
    print(f"  discretisation floor: sigma >= {SIGMA_FLOOR} vox -> above floor: {above_floor}")
    if above_floor and not ramp:
        verdict = (f"ACCEPTED: Lindeberg has a genuine interior peak at sigma~{sigma_L:.1f} vox "
                   f"and clears the discretisation floor. (Single-criterion pick -- FSC "
                   f"corroboration was tried and rejected, see module docstring.)")
    elif above_floor and ramp:
        verdict = (f"FLAGGED: no interior local max in the response curve (used global "
                   f"argmax at sigma~{sigma_L:.1f}) -- treat this pick as less certain, "
                   f"sanity-check the orientation field visually.")
    else:
        verdict = (f"REJECTED: sigma_L={sigma_L:.2f} is below the {SIGMA_FLOOR} vox "
                   f"discretisation floor -- do not use this pick.")
    print(f"  => {verdict}")
    print(f"  RECOMMENDATION: sigma={sigma_L:.2f} vox (~{sigma_L*VOXEL_SIZE_UM:.0f} um), "
          f"rho={rho_L:.2f} vox")
    print("=" * 74)

    _figure(SIG_LIND, response, sigma_L, ramp, SIG_GRID, RHO_GRID, rho_mid, sens_row, rho_L)
    return dict(sigma_L=sigma_L, rho_L=rho_L, above_floor=above_floor, ramp=ramp)


def _figure(sig_lind, response, sigma_L, ramp, sig_grid, rho_grid, rho_mid, sens_row, rho_L,
            save="sigma_rho_final.png"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    # panel 1: Lindeberg response vs sigma, peak + floor marked
    ax[0].plot(sig_lind, response, "o-", color="#2a78d6", label="Lindeberg response")
    ax[0].axvline(sigma_L, color="#2a78d6", ls=":", lw=2,
                  label=f"pick sigma={sigma_L:.2f}" + (" [ramp]" if ramp else ""))
    ax[0].axvspan(0, SIGMA_FLOOR, color="0.85", label="below discretisation floor")
    ax[0].set_xlabel("$\\sigma$ (voxels)")
    ax[0].set_ylabel("Lindeberg response  $\\sigma^2 \\cdot$ mean(plane energy)")
    ax[0].set_title("Sigma pick (single criterion, no FSC)")
    ax[0].legend(fontsize=8)
    # panel 2: rho-stability sensitivity at the picked-sigma row, plateau marked
    ax[1].plot(rho_mid, sens_row, "s-", color="#c0392b")
    ax[1].axvline(rho_L, color="#c0392b", ls=":", lw=2, label=f"pick rho={rho_L:.2f}")
    ax[1].set_xlabel("$\\rho$ (voxels)")
    ax[1].set_ylabel("angular drift of e3 per unit log(rho)  (deg/e-fold)")
    ax[1].set_title(f"Rho pick: stability plateau at sigma~{sigma_L:.1f}")
    ax[1].legend(fontsize=8)
    fig.suptitle("sigma/rho pick on real i74 -- Lindeberg + discretisation floor only "
                 "(FSC rejected)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save, dpi=130)
    print(f"Saved {save}")


if __name__ == "__main__":
    main()
