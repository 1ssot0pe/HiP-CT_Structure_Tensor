"""
lindeberg_sigma_phantoms.py

A CLEAN, isolated test of the Lindeberg sigma criterion (see
lindeberg_stability_3d.lindeberg_response_3d): on a phantom whose fibres have
ONE known cross-sectional width, does the gamma-normalised response

        sigma^2 * mean(sum of two LARGEST structure-tensor eigenvalues)

peak at a sigma that tracks that width? And when the width changes, does the
peak move with it, monotonically?

Why a new phantom rather than reusing the existing ones
-------------------------------------------------------
The two 3D phantoms already in the repo each bundle a confounder that muddies a
pure sigma read:

  * reconstruction.make_phantom stacks straight + crossing + isotropic-noise
    regions along x. The crossing region carries TWO scales/orientations and the
    noise region has energy at every scale, so a whole-volume Lindeberg average
    mixes them -- you cannot say "the peak is the fibre width".
  * orthogonality_residual.make_fanning_phantom rotates the fibre orientation
    across the volume (by design, to give rho a real optimum). That spatial
    orientation change is exactly the large-scale content the
    lindeberg_stability_3d.first_local_max docstring warns can hijack the
    response at large sigma.

This module strips all of that away: a SINGLE population of straight, parallel
Gaussian tubes, one width, well separated, (optionally) light noise. One scale
in, so there should be one clean peak out -- the ideal setting to confirm the
criterion behaves before trusting it on real data.

What "meaningful" means here
----------------------------
A single phantom only tells you WHERE the peak lands. The real evidence that the
criterion measures scale (rather than some fixed grid artefact) is that the peak
MOVES with the width: sweep several widths and the peak-sigma vs width curve
should be monotonic and roughly linear. That sweep is the main() experiment.

Reuses lindeberg_response_3d + first_local_max from lindeberg_stability_3d so the
exact response being tested here is the same code the real-data recipe uses.
"""

import numpy as np

from lindeberg_stability_3d import lindeberg_response_3d, first_local_max


# ---------------------------------------------------------------------------
# Phantom: a single population of straight parallel Gaussian tubes
# ---------------------------------------------------------------------------

def make_tube_phantom(shape=(96, 96, 96), width=2.0, spacing=None, axis=0,
                      noise_sigma=0.0, seed=0):
    """Straight parallel Gaussian tubes (one fibre population) along `axis`.

    Every tube shares the SAME cross-sectional width, so the volume has exactly
    ONE characteristic scale -- the clean input the Lindeberg sigma criterion is
    meant for. Tubes sit on a square lattice in the plane perpendicular to
    `axis`; the intensity is a periodic 2D Gaussian of that width (identical
    profile to reconstruction.make_phantom's `tubes`, just single-population).

    Parameters
    ----------
    shape : (N0, N1, N2) volume shape.
    width : Gaussian sigma of each tube's cross-section, in voxels. This is the
        scale the Lindeberg peak should recover.
    spacing : centre-to-centre tube spacing, in voxels. Defaults to 6*width so
        neighbouring tubes barely overlap (the same 6x rule used in
        toy2d/width_sense_check.py) -- close enough to fill the volume, far
        enough that the spacing does not introduce a second competing scale.
    axis : fibre axis (0, 1 or 2). The tubes run along it; the lattice lives in
        the other two axes.
    noise_sigma : std of additive Gaussian noise (0 = clean). Kept OFF by
        default so this is a pure-scale test; the lindeberg_stability_3d caveat
        is that the pick is not fully noise-adaptive, which is a separate
        experiment.
    seed : RNG seed for the noise.

    Returns
    -------
    volume : float32 array, normalised to [0, 1].
    """
    if spacing is None:
        spacing = 6.0 * width
    coords = np.indices(shape).astype(np.float32)
    perp = [a for a in range(3) if a != axis]
    a, b = coords[perp[0]], coords[perp[1]]

    da = ((a + spacing / 2) % spacing) - spacing / 2
    db = ((b + spacing / 2) % spacing) - spacing / 2
    vol = np.exp(-(da ** 2 + db ** 2) / (2.0 * width ** 2))

    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        vol = vol + noise_sigma * rng.standard_normal(shape).astype(np.float32)
    vol = (vol - vol.min()) / (np.ptp(vol) + 1e-12)
    return vol.astype(np.float32)


# ---------------------------------------------------------------------------
# Peak location, refined to sub-grid resolution
# ---------------------------------------------------------------------------

def refined_peak_sigma(sigmas, response):
    """Sigma at the Lindeberg peak, refined off the discrete grid.

    Picks the first interior local maximum (first_local_max -- robust to any
    slow rise at large sigma from residual boundary content), then fits a
    parabola through that point and its two neighbours and returns the vertex.
    So the recovered peak is continuous in the phantom width instead of being
    quantised to the sigma grid, which matters for a clean peak-vs-width trend.

    Falls back to the plain grid argmax if there is no interior local max.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    response = np.asarray(response, dtype=float)
    i = first_local_max(sigmas, response)
    if i is None:
        return float(sigmas[int(np.nanargmax(response))])
    x0, x1, x2 = sigmas[i - 1], sigmas[i], sigmas[i + 1]
    y0, y1, y2 = response[i - 1], response[i], response[i + 1]
    # Vertex of the parabola through the three (x, y) points.
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return float(x1)
    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    B = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / denom
    if A >= 0:                       # not a maximum -> keep the grid point
        return float(x1)
    xv = -B / (2 * A)
    # Guard against a vertex thrown outside the bracketing interval.
    return float(np.clip(xv, x0, x2))


# ---------------------------------------------------------------------------
# Experiment: peak sigma vs true tube width
# ---------------------------------------------------------------------------

def sweep_width_vs_peak(widths, shape=(96, 96, 96), sigmas=None, rho=1.0,
                        noise_sigma=0.0, verbose=True):
    """For each tube width, build the phantom and record the Lindeberg peak sigma.

    Returns (widths, peak_sigmas, responses, sigmas) where responses[k] is the
    full response curve for widths[k], so the caller can plot the curves too.
    """
    widths = np.asarray(widths, dtype=float)
    if sigmas is None:
        sigmas = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
                           6.0, 8.0, 10.0, 12.0])
    sigmas = np.asarray(sigmas, dtype=float)

    peak_sigmas = np.full(widths.size, np.nan)
    responses = []
    for k, w in enumerate(widths):
        vol = make_tube_phantom(shape=shape, width=float(w),
                                noise_sigma=noise_sigma)
        _s, resp = lindeberg_response_3d(vol, sigmas, rho=rho, verbose=False)
        responses.append(resp)
        peak_sigmas[k] = refined_peak_sigma(sigmas, resp)
        if verbose:
            print(f"  width={w:5.2f}  ->  peak sigma={peak_sigmas[k]:6.3f}  "
                  f"(sigma/width={peak_sigmas[k] / w:5.3f})")
    return widths, peak_sigmas, responses, sigmas


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SHAPE = (96, 96, 96)
    WIDTHS = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    print("Lindeberg sigma criterion on clean single-scale tube phantoms")
    print(f"shape={SHAPE}, single straight Gaussian-tube population, no noise\n")
    widths, peaks, responses, sigmas = sweep_width_vs_peak(WIDTHS, shape=SHAPE)

    # --- gallery of the actual phantoms fed to the sweep -------------------
    # Tubes run along axis 0, so a mid-slice at fixed axis 0 shows the tube
    # cross-sections (the lattice of blobs whose size IS the width). Same
    # display extent for every panel so the growing blob size is comparable.
    fig0, axes0 = plt.subplots(1, len(widths), figsize=(2.4 * len(widths), 2.8))
    view = 32   # show a fixed 32x32 corner so tube size is visually comparable
    for ax, w in zip(np.atleast_1d(axes0), widths):
        vol = make_tube_phantom(shape=SHAPE, width=float(w))
        ax.imshow(vol[vol.shape[0] // 2, :view, :view], cmap="gray",
                  vmin=0, vmax=1)
        ax.set_title(f"width={w:g}\n(spacing={6 * w:g})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig0.suptitle("Phantoms used: cross-section through the tubes "
                  "(axis-0 mid-slice, 32x32 corner)")
    fig0.tight_layout()
    fig0.savefig("lindeberg_sigma_phantoms_gallery.png", dpi=120)
    print("Saved lindeberg_sigma_phantoms_gallery.png")

    # Straight-line fit peak_sigma = m*width + b (the "does it track scale?" test).
    m, b = np.polyfit(widths, peaks, 1)
    print("-" * 60)
    print(f"linear fit: peak_sigma = {m:.3f} * width + {b:+.3f}")
    print(f"mean sigma/width ratio = {np.mean(peaks / widths):.3f}")
    print("-" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) response curves, normalised to their own peak so all widths overlay.
    for w, resp in zip(widths, responses):
        r = np.asarray(resp, float)
        axes[0].plot(sigmas, r / np.nanmax(r), "o-", alpha=0.8,
                     label=f"width={w:g}")
    axes[0].set_xlabel("sigma")
    axes[0].set_ylabel("Lindeberg response (peak-normalised)")
    axes[0].set_title("Response per width -- peak marches right with width")
    axes[0].legend(fontsize=8)

    # (B) recovered peak sigma vs true width + the fitted line.
    axes[1].plot(widths, peaks, "o", ms=9, label="Lindeberg peak")
    xl = np.array([widths.min(), widths.max()])
    axes[1].plot(xl, m * xl + b, "r--",
                 label=f"fit  sigma={m:.2f}·width{b:+.2f}")
    axes[1].plot(xl, xl, "k:", alpha=0.5, label="sigma = width")
    axes[1].set_xlabel("true tube width")
    axes[1].set_ylabel("recovered peak sigma")
    axes[1].set_title("Peak sigma tracks fibre width (clean, monotonic)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Lindeberg sigma criterion: clean single-scale phantom sweep")
    fig.tight_layout()
    fig.savefig("lindeberg_sigma_phantoms.png", dpi=120)
    print("Saved lindeberg_sigma_phantoms.png")
