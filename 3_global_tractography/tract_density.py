"""
tract_density.py -- Stage 3 density forward model.

Turns a set of streamlines into a voxel-wise reconstructed density D_rec, the
quantity the self-optimisation loop compares against its target.

Rasterisation
-------------
Trilinear splatting, not nearest-voxel counting. A streamline is a continuous
curve sampled at sub-voxel steps; dropping each sample into its nearest voxel
makes the density depend on where the sample points happen to land, which shows
up as stripes along the tract and adds variance to exactly the junction values
the loop scores. Splatting each sample over its eight neighbours by volume
weight removes that.

Each sample deposits its arc length, so a tract crossing a voxel edge-to-edge
deposits ~1.0 regardless of how finely it was stepped. That normalisation is
what makes "density 2 at a crossing" literally mean two fibres -- without it,
halving the step size would double the apparent density and the MSE against a
fixed target would be meaningless.

This module decides nothing. No hypotheses, no scoring, no iteration --
`self_optimising_tractography.py` owns all of that. Kept separate so the
rasteriser can be tested on its own, which `_selftest` below does.
"""

import numpy as np

EPS = 1e-12


def splat(tdi, p, w):
    """Add weight w at continuous position p=(z,y,x) by trilinear distribution.

    Out-of-bounds samples are dropped rather than clamped: clamping would pile a
    tract's escaped tail onto the boundary voxel and invent density there.
    """
    z, y, x = p
    z0, y0, x0 = int(np.floor(z)), int(np.floor(y)), int(np.floor(x))
    dz, dy, dx = z - z0, y - y0, x - x0
    nz, ny, nx = tdi.shape
    for kz in (0, 1):
        iz = z0 + kz
        if iz < 0 or iz >= nz:
            continue
        wz = dz if kz else 1.0 - dz
        for ky in (0, 1):
            iy = y0 + ky
            if iy < 0 or iy >= ny:
                continue
            wy = dy if ky else 1.0 - dy
            for kx in (0, 1):
                ix = x0 + kx
                if ix < 0 or ix >= nx:
                    continue
                wx = dx if kx else 1.0 - dx
                tdi[iz, iy, ix] += w * wz * wy * wx


def rasterise(streamlines, shape, step=1.0, weights=None):
    """Streamlines -> D_rec, in units of fibres per voxel.

    streamlines : iterable of (M, 3) arrays of (z, y, x) positions, in voxels.
    step        : arc length between consecutive samples, in voxels.
    weights     : optional per-streamline weight, so a half-believed tract
                  deposits half a fibre.
    """
    tdi = np.zeros(tuple(shape), dtype=np.float64)
    for i, sl in enumerate(streamlines):
        sl = np.asarray(sl, dtype=float)
        if sl.ndim != 2 or sl.shape[0] < 2:
            continue
        w = float(step) * (1.0 if weights is None else float(weights[i]))
        if w <= 0:
            continue
        for p in sl:
            splat(tdi, p, w)
    return tdi


def density_at(tdi, centre, half=0):
    """Read D_rec at a junction centre.

    half=0 reads one voxel; half>0 averages a small cube, which is what you want
    when the junction centre is only known to within a voxel or two -- a point
    read of a splatted density is noisy at the sub-voxel scale.
    """
    c = [int(round(v)) for v in centre]
    if half <= 0:
        c = [int(np.clip(v, 0, s - 1)) for v, s in zip(c, tdi.shape)]
        return float(tdi[c[0], c[1], c[2]])
    sl = tuple(slice(max(v - half, 0), min(v + half + 1, s))
               for v, s in zip(c, tdi.shape))
    return float(tdi[sl].mean())


def mse(predicted, measured):
    """Mean squared error, ignoring non-finite entries on either side."""
    p = np.asarray(predicted, dtype=float).ravel()
    m = np.asarray(measured, dtype=float).ravel()
    ok = np.isfinite(p) & np.isfinite(m)
    if not ok.any():
        return np.nan
    return float(np.mean((p[ok] - m[ok]) ** 2))


def _selftest():
    print("tract_density self-test")
    shape = (32, 32, 32)
    xs = np.arange(4.0, 28.0, 0.5)
    sl = np.stack([np.full_like(xs, 16.0), np.full_like(xs, 16.0), xs], axis=1)
    d = density_at(rasterise([sl], shape, step=0.5), (16, 16, 16))
    print(f"  1 straight tract  -> {d:.3f}   (want ~1.0)")

    ys = np.arange(4.0, 28.0, 0.5)
    sl2 = np.stack([np.full_like(ys, 16.0), ys, np.full_like(ys, 16.0)], axis=1)
    d2 = density_at(rasterise([sl, sl2], shape, step=0.5), (16, 16, 16))
    print(f"  2 crossing tracts -> {d2:.3f}   (want ~2.0)")

    sl3 = sl.copy(); sl3[:, 1] += 2.0
    sl4 = sl2.copy(); sl4[:, 2] += 2.0
    d3 = density_at(rasterise([sl3, sl4], shape, step=0.5), (16, 16, 16))
    print(f"  2 tangent tracts  -> {d3:.3f}   (want <1.0)")

    xf = np.arange(4.0, 28.0, 0.125)
    slf = np.stack([np.full_like(xf, 16.0), np.full_like(xf, 16.0), xf], axis=1)
    df = density_at(rasterise([slf], shape, step=0.125), (16, 16, 16))
    print(f"  same tract 4x finer sampling -> {df:.3f}  (must match {d:.3f})")

    dw = density_at(rasterise([sl], shape, step=0.5, weights=[0.5]), (16, 16, 16))
    print(f"  weight 0.5        -> {dw:.3f}   (want ~{d/2:.3f})")

    ok = (abs(d - 1.0) < 0.05 and abs(d2 - 2.0) < 0.1 and d3 < 1.0
          and abs(df - d) < 0.05 and abs(dw - d / 2) < 0.02)
    print("  " + ("ALL PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    _selftest()
