"""
kissing_phantom.py -- a PROPER kissing phantom: two CURVED fibre bundles that bulge
toward each other and come very close in the MIDDLE but DO NOT touch (a gap), then
curve away. At the closest approach the two curves are TANGENT (locally parallel) --
the classic tractography trap: a continuity tracker is tempted to bridge the gap and
merge them, but they are separate fibres.

Shape (z-projection):   )(   -- left curve bulges right, right curve bulges left,
their bulges nearest at the centre with `gap` voxels of empty space between them.

Each bundle is a family of parallel curved fibres of a chosen RADIUS, so the two can be
SAME size or DIFFERENT sizes.

Test: (B) HOST should call every patch "single" (each bundle is locally one fibre,
never "crossing"); (C) global tractography must leave TWO separate tracts and NOT link
across the gap (no merge) -- even though the curves are tangent and close there.
"""

import numpy as np

from crossing_shape_3d import _coords

EPS = 1e-12


def _finalise(vol):
    """Shift to zero and scale to [0, 1].

    Copied from host_geometry_test rather than imported: this three-line helper
    was the only reason the live pipeline loaded that 298-line phantom suite.
    """
    vol = vol - vol.min()
    return (vol / (vol.max() + EPS)).astype(np.float32)

SHAPE = (80, 80, 80)


def _curve_pts(side, gap, curv, ymax=34.0, n=140):
    """Polyline of a parabolic fibre in the x-y plane (centred coords). side=-1 -> the
    left ')' curve (bulge tip at x=-gap/2); side=+1 -> the right '(' curve."""
    s = np.linspace(-ymax, ymax, n)
    x = side * (gap / 2.0) + side * curv * s ** 2      # bulge toward centre at s=0
    return np.stack([x, s, np.zeros_like(s)], axis=1)   # (n,3) = (x,y,z)


def curved_bundle(C, pts, radius):
    """A single curved Gaussian tube of the given radius, following polyline `pts`."""
    X, Y, Z = C
    d2 = None
    for px, py, pz in pts:
        dm = (X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2
        d2 = dm if d2 is None else np.minimum(d2, dm)
    return np.exp(-d2 / (2 * radius ** 2))


def add_poisson(vol, bg=0.02, photons=2000.0, seed=0):
    """Shot noise on a non-zero background -- what a real reconstruction has.

    The noiseless phantom is a sum of Gaussian tubes, so beyond ~21 voxels it
    underflows float32 to EXACTLY zero over half the volume. A structure tensor
    there is the zero matrix: no direction exists, and an eigensolver asked for
    one returns whatever rounding noise it decomposed. That is what put
    single-voxel direction streaks in the empty corners.

    A real scan never has this problem, because the background is not zero and
    not constant. Reproducing that is more honest than special-casing it:

        counts ~ Poisson((vol + bg) * photons)  ->  intensity = counts / photons

    Poisson alone would not do it -- Poisson(0) is exactly 0 -- so the constant
    `bg` pedestal is the part that matters. Background noise is then
    sqrt(bg / photons) in intensity units; at the defaults that is 0.3% of the
    fibre peak.

    Returns float32; values are NOT renormalised, so the fibre peak stays ~1 + bg
    and callers that use `vol.max()` as a reference see what they expect.
    """
    rng = np.random.default_rng(seed)
    lam = np.clip((vol.astype(np.float64) + bg) * photons, 0.0, None)
    return (rng.poisson(lam) / photons).astype(np.float32)


def build_kissing(same_size=True, gap=8.0, curv=0.028, bg=0.0, photons=2000.0,
                  seed=0, halo=0):
    """Two curved fibres bulging toward each other, closest in the middle (gap),
    tangent there. same_size -> both radius 1.6; else A thin 1.1, B thick 2.6.

    bg > 0 adds Poisson shot noise on a `bg` pedestal (see `add_poisson`).
    Default 0.0 leaves the phantom exactly as it was for existing callers.

    halo > 0 grows the volume by `halo` voxels on every face AND lengthens the
    fibres so they run out through the enlarged boundary rather than stopping
    at it. Process the padded volume, then crop the halo away: every voxel kept
    was measured with a full filter footprint of real data. Without it the
    outer shell is measured against whatever the filter's boundary mode
    invents -- 30 voxels deep for the gate (a 16-voxel gradient kernel at
    sigma_grad=4 feeding a 14-voxel box), which on an 80-voxel cube is most of
    the volume.

    `ymax` grows with the halo so the curve still leaves the box; at halo=0 it
    is 34.0 exactly as before, so existing callers see an unchanged phantom.
    """
    shape = tuple(s + 2 * halo for s in SHAPE)
    C = _coords(shape)

    rA, rB = (1.6, 1.6)

    ymax = 34.0 + halo
    n = int(round(140 * ymax / 34.0))          # keep the point spacing constant
    ptsA = _curve_pts(-1, gap, curv, ymax=ymax, n=n)
    ptsB = _curve_pts(+1, gap, curv, ymax=ymax, n=n)
    A = curved_bundle(C, ptsA, rA)   # ) on the left
    B = curved_bundle(C, ptsB, rB)   # ( on the right

    # honest minimum true gap between the two curve centrelines (not just the
    # parametric tip separation) -- report it so we KNOW they don't touch
    dmin = float(np.min(np.linalg.norm(ptsA[:, None, :] - ptsB[None, :, :], axis=-1)))

    vol = _finalise(A + B)
    meta = dict(rA=rA, rB=rB, gap=gap, curv=curv, true_min_gap=dmin, bg=bg,
                halo=halo, shape=shape, ymax=ymax)
    if bg > 0:
        vol = add_poisson(vol, bg=bg, photons=photons, seed=seed)
        meta.update(photons=photons, seed=seed,
                    bg_noise=float(np.sqrt(bg / photons)))
    return vol, meta


def crop_halo(arr, halo):
    """Drop `halo` voxels from every face of the first three axes."""
    if halo <= 0:
        return arr
    s = (slice(halo, -halo),) * 3
    return arr[s]


if __name__ == "__main__":
    # geometry check only: save z-MIP of both variants
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    for a, same in zip(ax, (True, False)):
        S, meta = build_kissing(same_size=same)
        print(f"  {'same' if same else 'diff'}-size: true min centreline gap = "
              f"{meta['true_min_gap']:.2f} vox (radii {meta['rA']:.1f}/{meta['rB']:.1f})")
        a.imshow(S.mean(0), cmap="gray", origin="upper")
        a.axhline(SHAPE[1] / 2, color="cyan", ls=":", lw=0.8)
        a.set_title(("SAME-size" if same else "DIFFERENT-size") +
                    f" fibres, true gap={meta['true_min_gap']:.1f} vox")
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Proper kissing phantom (z-projection): two bundles approach but a gap "
                 "separates them", fontsize=12)
    fig.tight_layout(); fig.savefig("kissing_phantom_geom.png", dpi=130); plt.close(fig)
    print("Saved kissing_phantom_geom.png")


build_kissing(same_size=True, gap=12.0, curv=0.028)