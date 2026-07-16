"""
premise_check.py -- question the premise (option 1)

Is the real i74 chunk's pervasive high sphericity (median c_s~0.33) GENUINE
multi-orientation structure -- meaning single-e3 fibre extraction is the wrong
model for much of the volume -- or is it noise/PSF inflation?

Decisive test: process THREE volumes identically across a sigma sweep (rho=2*sigma)
and compare mean sphericity vs sigma:
  * pure noise      -- noise-driven c_s should DECAY as sigma grows (rho
                       integration averages random gradients toward isotropy...
                       actually toward the noise floor; the point is it MOVES).
  * clean single    -- genuine single orientation; c_s stays ~0 at all sigma.
  * real interior   -- if it stays HIGH while noise behaves differently and the
                       single fibre stays low, the high c_s is real structure.

Plus a spatial/visual read: are high-c_s voxels organised (coherent regions =
genuine crossings/junctions) or salt-and-pepper (noise)?

Reuses load_interior + sphericity_field (selfcalib_threshold), make_single_fibre_3d
+ _noise_volume (crossing_shape_3d), fibre_direction (orthogonality_residual).
"""

import numpy as np

from selfcalib_threshold import load_interior, sphericity_field
from crossing_shape_3d import make_single_fibre_3d, _noise_volume, shape_measures
from orthogonality_residual import fibre_direction, crop_margin, _crop3

SIGMAS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _mean_cs_vs_sigma(vol, label):
    means, meds = [], []
    for s in SIGMAS:
        cs = sphericity_field(vol, s, 2 * s).ravel()
        cs = cs[np.isfinite(cs)]
        means.append(float(cs.mean())); meds.append(float(np.median(cs)))
        print(f"  {label:14s} sigma={s:.0f} rho={2*s:.0f} | mean c_s={cs.mean():.3f}  median={np.median(cs):.3f}")
    return np.array(means), np.array(meds)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("PREMISE CHECK -- is real high c_s genuine structure or noise?")

    noise = _noise_volume((96, 96, 96), seed=0)
    single, _ = make_single_fibre_3d(theta_deg=20.0, noise_sigma=0.0)
    real = load_interior()
    print(f"  real interior {real.shape}")

    print("  --- mean sphericity vs sigma ---")
    n_m, n_md = _mean_cs_vs_sigma(noise, "pure noise")
    s_m, s_md = _mean_cs_vs_sigma(single, "single fibre")
    r_m, r_md = _mean_cs_vs_sigma(real, "REAL i74")

    # How much of the real volume is single-e3-inappropriate at the onset sigma
    # (c_s high => lambda3 ~ lambda1,2 => no clear plane => e3 ill-defined)?
    cs4 = sphericity_field(real, 4.0, 8.0).ravel()
    cs4 = cs4[np.isfinite(cs4)]
    for thr in (0.3, 0.5):
        print(f"  REAL: fraction with c_s>{thr} at sigma=4 = {(cs4 > thr).mean():.2%}")
    # single-fibre-like fraction (clean planar, trustworthy for single e3)
    print(f"  REAL: fraction with c_s<0.15 (single-fibre-like) = {(cs4 < 0.15).mean():.2%}")
    print("=" * 70)

    # ---- figure: (a) c_s vs sigma curves; (b) raw slice; (c) c_s map ----
    fig = plt.figure(figsize=(15, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(SIGMAS, n_m, "o-", color="0.5", label="pure noise")
    ax1.plot(SIGMAS, s_m, "s-", color="tab:green", label="clean single fibre")
    ax1.plot(SIGMAS, r_m, "^-", color="tab:red", label="REAL i74")
    ax1.set_xlabel("sigma (rho=2*sigma)"); ax1.set_ylabel("mean sphericity c_s")
    ax1.set_title("c_s vs sigma: does real behave like noise or structure?")
    ax1.legend(fontsize=8)

    # spatial read at the onset sigma
    z = real.shape[0] // 2
    m = crop_margin(4.0, 8.0)
    cs_map = sphericity_field(real, 4.0, 8.0)
    zc = cs_map.shape[0] // 2
    raw_c = _crop3(real, m)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(raw_c[zc], cmap="gray"); ax2.set_title(f"real raw (cropped) z-mid")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax3 = fig.add_subplot(1, 3, 3)
    im = ax3.imshow(cs_map[zc], cmap="magma", vmin=0, vmax=1)
    ax3.set_title("sphericity c_s map (sigma=4)"); ax3.set_xticks([]); ax3.set_yticks([])
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    fig.suptitle("Premise check: is single-e3 the wrong model for real i74?")
    fig.tight_layout(); fig.savefig("premise_check.png", dpi=120)
    print("  saved premise_check.png")
