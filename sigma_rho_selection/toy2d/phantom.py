"""
phantom.py -- Layer 0

2D phantom generator for structure-tensor parameter-selection toy.

Coordinate convention (read this before touching angles anywhere downstream)
------------------------------------------------------------------------
Image arrays are indexed [row, col] = [axis0, axis1].
We define x := col index (axis1), y := row index (axis0).
A fibre angle theta is the direction of the bright lines as a vector

    d = (dx, dy) = (cos(theta), sin(theta))     in (x, y)

so theta=0 is horizontal lines (running left-right), theta=90 is vertical
lines (running top-to-bottom). Because y = row increases DOWNWARD on screen,
a positive theta tilts the line downward to the right when displayed with
imshow's default origin='upper' -- exactly what you'd draw by hand with a
protractor on the image, so "visually confirm 30 degrees" just means: does
the line look tilted 30 degrees off horizontal, sloping down to the right.

Angles are stored and returned in degrees, taken modulo 180 (fibre
orientation is antipodal: theta and theta+180 describe the same line).
"""

import numpy as np


def wrap_angle_deg(theta_deg):
    """Wrap an angle (or array of angles) into [0, 180)."""
    return np.mod(theta_deg, 180.0)


def make_phantom(shape=(200, 200), theta_deg=30.0, spacing=12.0, width=1.2,
                  noise_sigma=0.0, theta2_deg=None, spacing2=None, width2=None,
                  seed=0):
    """Generate a 2D phantom of parallel bright lines.

    Parameters
    ----------
    shape : (rows, cols)
    theta_deg : angle of the first line family, see module docstring for
        convention.
    spacing : distance between adjacent line centres (pixels).
    width : Gaussian sigma of each line's brightness profile (pixels). This
        IS the "Gaussian blur" -- each line is a Gaussian ridge, not a hard
        edge, so `width` controls how sharp/blurred the lines are.
    noise_sigma : std dev of additive Gaussian noise, added after the clean
        pattern is built (0.0 = noiseless).
    theta2_deg : if given, a SECOND line family at this angle is added,
        producing a crossing-fibre phantom.
    spacing2, width2 : spacing/width for the second family. Default to
        `spacing`/`width` (identical to the first family) if not given.
        Using a DIFFERENT spacing2 avoids the two families sharing an exact
        common period, which can otherwise cause resonance-like artifacts
        in downstream integration-scale sweeps.
    seed : RNG seed for the noise.

    Returns
    -------
    image : (rows, cols) float array, normalised to [0, 1].
    true_angles : list of the ground-truth angle(s) in degrees, each
        wrapped to [0, 180). Length 1 for a single-angle phantom, length 2
        for a crossing phantom.
    """
    rows, cols = shape
    y, x = np.indices(shape).astype(float)  # y=axis0=row, x=axis1=col

    def line_family(theta_deg, spacing, width):
        theta = np.deg2rad(theta_deg)
        nx, ny = -np.sin(theta), np.cos(theta)   # unit normal to the lines
        phase = x * nx + y * ny
        d = np.mod(phase + spacing / 2.0, spacing) - spacing / 2.0
        return np.exp(-(d ** 2) / (2.0 * width ** 2))

    image = line_family(theta_deg, spacing, width)
    true_angles = [float(wrap_angle_deg(theta_deg))]

    if theta2_deg is not None:
        s2 = spacing if spacing2 is None else spacing2
        w2 = width if width2 is None else width2
        image = image + line_family(theta2_deg, s2, w2)
        true_angles.append(float(wrap_angle_deg(theta2_deg)))

    image = image / image.max()

    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        image = image + noise_sigma * rng.standard_normal(shape)

    return image, true_angles


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image, true_angles = make_phantom(shape=(200, 200), theta_deg=30.0,
                                       spacing=12.0, width=1.2,
                                       noise_sigma=0.0)
    print(f"Ground-truth angle(s): {true_angles} degrees")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_title(f"Layer 0 phantom: theta = {true_angles[0]:.1f} deg")
    fig.tight_layout()
    fig.savefig("layer0_phantom.png", dpi=120)
    print("Saved layer0_phantom.png")
