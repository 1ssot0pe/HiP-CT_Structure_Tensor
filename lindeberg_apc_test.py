"""
lindeberg_apc_test.py

Apply the Lindeberg sigma criterion (validated on clean Gaussian-tube phantoms
in lindeberg_sigma_phantoms.py) to a more realistic synthetic phantom:
Stack_Synthetic_Data_Anti-Parallel_Cylinders_Random_Dia_512X512.tif -- bundles
of straight SOLID cylinders with RANDOM diameters.

Two differences from the clean phantom make this a real test rather than a
tautology:
  * solid binary cylinders (sharp walls), not soft Gaussian ridges -- the
    structure-tensor energy sits on the cylinder WALLS, so the "characteristic
    scale" the response reports is edge/cross-section scale, not a profile sigma;
  * a DISTRIBUTION of diameters, not one width -- so there is no single true
    answer, only a typical scale. A single clean peak landing near the median
    cylinder size is the success criterion.

We crop to ONE bundle (the cylinders running along axis 0) so the response
reflects a single orientation + a single size distribution, measure that
distribution directly (connected-component disk sizes on cross-sections) as
ground truth, then overlay the Lindeberg peak on it.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

from lindeberg_stability_3d import lindeberg_response_3d
from lindeberg_sigma_phantoms import refined_peak_sigma

TIF = Path("Stack_Synthetic_Data_Anti-Parallel_Cylinders_Random_Dia_512X512.tif")

# Sub-volume containing only the axis-0 cylinder bundle (excludes the
# x-running bars at y<245 and stays inside the bundle's y/x footprint).
CROP = (slice(100, 300), slice(250, 360), slice(170, 360))

# rho fixed and minimal: Lindeberg probes SIGMA only.
RHO = 1.0
SIGMAS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
                   6.0, 7.0, 8.0, 10.0, 12.0, 14.0])


def measure_diameters(sub, thresh=127, step=10):
    """Equivalent disk diameters of cylinder cross-sections, pooled over every
    `step`-th axis-0 slice. Area -> diameter via area = pi (d/2)^2."""
    diams = []
    for z in range(0, sub.shape[0], step):
        lab, n = ndimage.label(sub[z] > thresh)
        for i in range(1, n + 1):
            area = int((lab == i).sum())
            diams.append(2.0 * np.sqrt(area / np.pi))
    return np.asarray(diams)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile

    print(f"loading {TIF.name}")
    vol = tifffile.imread(str(TIF)).astype(np.float32)
    sub = vol[CROP]
    print(f"full {vol.shape} -> axis-0 bundle crop {sub.shape}, "
          f"fill fraction {(sub > 127).mean():.3f}")

    # --- ground truth: the cylinder diameter distribution -----------------
    diams = measure_diameters(sub)
    d_med, r_med = float(np.median(diams)), float(np.median(diams) / 2)
    print(f"measured cross-sections: n={diams.size}, "
          f"median diameter={d_med:.2f} vox (radius {r_med:.2f})")

    # --- Lindeberg response -----------------------------------------------
    print("computing Lindeberg response (this takes a couple of minutes)...")
    _s, response = lindeberg_response_3d(sub, SIGMAS, rho=RHO, verbose=True)
    sigma_peak = refined_peak_sigma(SIGMAS, response)
    print("-" * 60)
    print(f"Lindeberg peak sigma = {sigma_peak:.2f} vox")
    print(f"  vs median cylinder radius {r_med:.2f}  "
          f"(sigma/radius = {sigma_peak / r_med:.2f})")
    print(f"  vs median cylinder diameter {d_med:.2f}  "
          f"(sigma/diameter = {sigma_peak / d_med:.2f})")
    print("-" * 60)

    # --- figure: slice + diameter histogram + response --------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    z_mid = sub.shape[0] // 2
    axes[0].imshow(sub[z_mid], cmap="gray")
    axes[0].set_title(f"axis-0 bundle, cross-section z={z_mid}")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].hist(diams, bins=30, color="steelblue", edgecolor="white")
    axes[1].axvline(d_med, color="k", ls="--", label=f"median diam={d_med:.1f}")
    axes[1].set_xlabel("cylinder diameter (vox)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Measured diameter distribution (ground truth)")
    axes[1].legend(fontsize=8)

    axes[2].plot(SIGMAS, response, "o-")
    axes[2].axvline(sigma_peak, color="r", ls="--",
                    label=f"Lindeberg peak sigma={sigma_peak:.2f}")
    axes[2].axvline(r_med, color="g", ls=":", label=f"median radius={r_med:.2f}")
    axes[2].set_xlabel("sigma")
    axes[2].set_ylabel("sigma^2 * mean(plane energy)")
    axes[2].set_title("Lindeberg response (random-diameter cylinders)")
    axes[2].legend(fontsize=8)

    fig.suptitle("Lindeberg sigma criterion on random-diameter solid cylinders")
    fig.tight_layout()
    fig.savefig("lindeberg_apc_test.png", dpi=120)
    print("Saved lindeberg_apc_test.png")
