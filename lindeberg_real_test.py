"""
lindeberg_real_test.py

Apply the Lindeberg sigma criterion to REAL data: the HiP-CT
sensitivity_chunk_IC_768 volume (subject i74, 20 um/voxel), the same stack the
Structure Tensor Extraction notebook and reconstruction.py target.

This is the end of the validation chain:
  lindeberg_sigma_phantoms.py  -- clean Gaussian tubes: peak recovers scale
                                  (sigma ~ 0.87 * width), the criterion works.
  lindeberg_apc_test.py        -- solid random-diameter cylinders: peak reports
                                  WALL/edge scale, not diameter -- a caveat for
                                  filled, sharp-walled structures.
  THIS FILE                    -- no ground truth. We can only ask whether the
                                  response is well-behaved (a single interior
                                  peak, not a monotone ramp) and read off the
                                  sigma it implies, then eyeball the orientation
                                  field at that sigma for sanity.

There is NO ground truth on real data, so the picked sigma is a starting point
to sanity-check visually, NOT a certified optimum (see the caveat at the top of
lindeberg_stability_3d.py). We pick via the first interior local maximum
(first_local_max), which on real data is safer than a global argmax that
large-scale sample shading/curvature can hijack.
"""

import numpy as np

from reconstruction import load_volume, DEFAULT_VOLUME_DIR, VOXEL_SIZE_UM
from lindeberg_stability_3d import lindeberg_response_3d, first_local_max
from lindeberg_sigma_phantoms import refined_peak_sigma
from orthogonality_residual import fibre_direction

# INTERIOR tissue box (per-axis slices), chosen to sit fully inside the sample
# so the Lindeberg response reads FIBRE scale, not the tissue/background
# boundary. A symmetric whole-cube crop through this stack straddles that
# boundary, whose huge step edge dominates the energy and turns the response
# into a monotone ramp (the failure mode first_local_max exists to survive).
# This 200^3 box (mean~183, <1% background) shows clear fibre striations.
BOX = (slice(60, 260), slice(240, 440), slice(240, 440))
RHO = 1.0
SIGMAS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
                   6.0, 7.0, 8.0, 10.0, 12.0, 14.0])


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"loading {DEFAULT_VOLUME_DIR.name}  interior box={BOX}")
    full = load_volume(crop=None, dtype=np.uint8)     # full 768^3, small in uint8
    vol = full[BOX].astype(np.float32)
    del full
    print(f"volume shape={vol.shape} dtype={vol.dtype} "
          f"range=[{vol.min():.0f},{vol.max():.0f}]  ({VOXEL_SIZE_UM:g} um/vox)")

    # --- Lindeberg response ------------------------------------------------
    print("computing Lindeberg response (a couple of minutes)...")
    _s, response = lindeberg_response_3d(vol, SIGMAS, rho=RHO, verbose=True)

    i_local = first_local_max(SIGMAS, response)
    used_fallback = i_local is None
    sigma_peak = refined_peak_sigma(SIGMAS, response)
    print("-" * 62)
    if used_fallback:
        print("WARNING: no interior local max -- response is a ramp; using global "
              "argmax. Treat this sigma with suspicion (large-scale content may "
              "dominate).")
    print(f"Lindeberg peak sigma = {sigma_peak:.2f} vox "
          f"(~{sigma_peak * VOXEL_SIZE_UM:.0f} um)")
    print("NOTE: no ground truth on real data -- this is a starting point to "
          "sanity-check, not a certified optimum.")
    print("-" * 62)

    # --- figure: raw slice + response + orientation sanity at the pick -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    z_mid = vol.shape[0] // 2
    axes[0].imshow(vol[z_mid], cmap="gray")
    axes[0].set_title(f"raw volume, slice z={z_mid}")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].plot(SIGMAS, response, "o-")
    axes[1].axvline(sigma_peak, color="r", ls="--",
                    label=f"peak sigma={sigma_peak:.2f}")
    axes[1].set_xlabel("sigma")
    axes[1].set_ylabel("sigma^2 * mean(plane energy)")
    axes[1].set_title("Lindeberg response (real HiP-CT chunk)")
    axes[1].legend(fontsize=8)

    # |e3| RGB orientation at the picked sigma (rho=RHO). Not a correctness
    # check (no truth), but coherent, spatially smooth colour = plausible pick;
    # salt-and-pepper colour = the sigma is too small / noise-dominated.
    e3 = fibre_direction(vol, sigma_peak, RHO)
    rgb = np.abs(e3[:, z_mid]).transpose(1, 2, 0)
    axes[2].imshow(rgb)
    axes[2].set_title(f"|e3| RGB orientation @ sigma={sigma_peak:.2f}, rho={RHO:g}")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    fig.suptitle("Lindeberg sigma criterion on real data (sensitivity_chunk_IC_768)")
    fig.tight_layout()
    fig.savefig("lindeberg_real_test.png", dpi=120)
    print("Saved lindeberg_real_test.png")
