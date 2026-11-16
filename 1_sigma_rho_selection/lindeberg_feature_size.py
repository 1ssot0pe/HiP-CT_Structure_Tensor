"""
lindeberg_feature_size.py

PER-VOXEL Lindeberg feature-size map on the real i74 volume -- i.e. a heat map
in which every voxel is colour-coded by the characteristic size of the structure
it sits in, in microns.

Why this and not sigma_rho_adaptive.py's block map
--------------------------------------------------
sigma_rho_adaptive.compute_adaptive_map already showed the (sigma, rho) pick
drifts across the volume, so it tiles the region into 96^3 blocks and certifies
one (sigma, rho) per block. That is the right *finding* but the wrong
*resolution*: a 96^3 block is 1.9 mm on a side at 20 um/voxel, far coarser than
the tissue's own compartments, and every block boundary is a hard discontinuity
in the scale that is used downstream.

Lindeberg's scale-selection principle does not actually need blocks. The
gamma-normalised response

        R(x; sigma) = sigma^2 * ( lambda_1 + lambda_2 )(x; sigma, rho)

(lambda_1 >= lambda_2 = the two largest structure-tensor eigenvalues, i.e. the
energy in the plane PERPENDICULAR to the fibre -- the same quantity
pick_sigma_rho.lindeberg_response_3d averages over a whole block) is defined
AT EVERY VOXEL. Averaging it over a block and then taking one argmax is a
deliberate loss of information. Taking the argmax over sigma *per voxel*

        sigma*(x) = argmax_sigma R(x; sigma)

is the textbook form of the criterion and gives a scale field at the data's own
resolution. That field, converted to microns by the phantom calibration below,
IS the feature-size heat map; and since sigma*(x) is exactly the differentiation
scale the criterion certifies for that voxel, the map doubles as the
voxel-continuous adaptive sigma (and rho = RHO_RATIO * sigma*) field that
sigma_rho_adaptive could only deliver piecewise-constant.

Three things this file gets right that a naive per-voxel argmax gets wrong
-------------------------------------------------------------------------
(1) rho is tied to sigma (rho = RHO_RATIO * sigma), not held fixed.
    Scale selection is only meaningful if the operator is SELF-SIMILAR across
    the sweep -- a fixed rho makes the large-sigma end of the sweep a
    differently-shaped operator, not the same operator at a bigger scale, and
    the argmax then partly tracks that shape change instead of the structure.
    (pick_sigma_rho sweeps at a fixed rho_ref=1.0, which is defensible for a
    block MEAN but biases a per-voxel argmax.)
    RHO_RATIO >= 1 is also what makes the response non-degenerate ON a fibre:
    the intensity gradient vanishes on a ridge crest, so with rho -> 0 the
    response would be near-zero exactly along every fibre centreline and the
    argmax there would be pure noise. rho ~ sigma pulls in the flank energy.

(2) The sigma grid is GEOMETRIC and the sub-grid refinement is done in
    log(sigma). Scale is a multiplicative quantity; a parabola fitted in linear
    sigma over a geometric grid is biased towards the coarse end.

(3) Voxels whose argmax lands on the first or last grid point are FLAGGED, not
    reported. A rail means the true scale is outside the swept range, so the
    number is a censoring bound, not a measurement. They are masked out of the
    heat map (shown as "unresolved") rather than being silently painted with
    sigma_min / sigma_max, which would invent a plateau of fake feature size at
    both ends of the colour bar.

Calibration (sigma* -> microns)
-------------------------------
sigma* is a scale-space parameter, not a length. calibrate() recovers the
proportionality on lindeberg_sigma_phantoms.make_tube_phantom, whose tubes have
a KNOWN Gaussian cross-sectional width w: it measures the per-voxel sigma*
inside the tubes for a range of w and fits sigma* = k * w through the origin.
The map then reports the FWHM-equivalent diameter

        d(x) = 2*sqrt(2*ln 2) * (sigma*(x) / k) * VOXEL_SIZE_UM   [microns]

which is a real, physically-checkable length (a fibre whose greyscale profile is
Gaussian with s.d. w voxels is 2.355*w voxels across at half maximum). The raw
sigma* field is always saved alongside, so the calibration can be revised
without recomputing anything.

Memory / cost
-------------
The full stack is 768^3. A per-voxel sweep over n scales cannot hold n response
volumes (n * 1.8 GB), so the peak search is STREAMED over a 3-deep rolling
window in scale, and the volume is processed in overlapping tiles whose halo is
the full composed kernel support (see kernel_halo) so each tile's interior is
what an untiled run would give -- no seams. Output goes to a .npy memmap on
disk.

Run
---
    python lindeberg_feature_size.py --calibrate            # phantom fit (~2 min)
    python lindeberg_feature_size.py --crop 256 512 --tile 256 --out PREFIX
    python lindeberg_feature_size.py --tile 192 --out PREFIX   # full 768^3

then  python render_feature_size_map.py PREFIX  for the heat map.

Point --out at a LOCAL path, not into this repo: the sigma map for the full
volume is 1.8 GB, and allocating a file that size inside the OneDrive-synced
tree makes the sync client saturate the disk -- a full run stalled at 0.1 s of
CPU until the output was moved off OneDrive.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from structure_tensor import structure_tensor_3d, eig_special_3d

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root
from core.reconstruction import VOXEL_SIZE_UM, DEFAULT_VOLUME_DIR

# --- sweep parameters -------------------------------------------------------
# Geometric sigma grid, 10 points (ratio 1.31 between neighbours -- fine enough
# that the log-parabola refinement is interpolating, not extrapolating).
#
# The ENDS are chosen to bracket the two block-level picks this repo already
# has, so the per-voxel map can resolve which of them a voxel belongs to rather
# than re-deriving them: sigma_rho_robustness certified sigma_L = 0.88-0.90 in
# the two dense-tissue boxes and 6.00 in the sparse box B. 6.0 is therefore the
# top. 0.5 is the bottom because it is BELOW sigma_rho_adaptive's SIGMA_FLOOR of
# 0.85 -- a voxel whose structure is genuinely sub-voxel then rails visibly
# instead of being silently clipped into the trustworthy range.
SIGMAS = np.geomspace(0.5, 6.0, 10)
RHO_RATIO = 1.0          # rho = RHO_RATIO * sigma  (see docstring point 1)
RESP_SMOOTH = 2.0        # response pooled over a Gaussian of RESP_SMOOTH*sigma
TRUNCATE = 3.0           # kernel support in s.d.; 3.0 keeps the tile halo sane
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.3548

# Background rejection. Air in this stack sits near 0 on a uint8 range whose
# tissue occupies the upper half, so a fraction-of-max cut is enough; it is
# applied to the volume smoothed at sigma_max so the mask is defined on the same
# scale the coarsest response is.
AIR_FRAC = 0.15

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "lindeberg_feature_size_calib.json")


# ---------------------------------------------------------------------------
# The per-voxel response
# ---------------------------------------------------------------------------

def lindeberg_response_voxel(volume, sigma, rho, truncate=TRUNCATE,
                             resp_smooth=0.0):
    """sigma^2 * (lambda_1 + lambda_2) at EVERY voxel, float32.

    Identical quantity to pick_sigma_rho.lindeberg_response_3d, minus its
    mean() over the volume and minus its boundary crop (cropping is the
    caller's job here -- the tiling supplies a real halo instead of discarding
    an interior margin).

    resp_smooth > 0 additionally blurs the response with a Gaussian of s.d.
    resp_smooth*sigma. This is the per-voxel replacement for the block MEAN in
    pick_sigma_rho: that mean pools ~10^6 voxels before looking for a peak, so
    its curve is smooth; a single voxel's curve is not, and a noise wiggle can
    both invent a spurious local max and destroy a real one. Pooling
    PROPORTIONALLY to sigma keeps the whole operator scale-covariant (a block
    of fixed size would not), so the calibration below still holds, and the
    field stays voxel-continuous instead of piecewise-constant.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        S = structure_tensor_3d(volume.astype(np.float32, copy=False),
                                float(sigma), float(rho), truncate=truncate)
        val, _vec = eig_special_3d(S)     # order='desc': val[0] >= val[1] >= val[2]
    del S
    resp = val[0] + val[1]               # in-plane (perpendicular) energy
    del val
    resp *= np.float32(sigma) ** 2       # gamma-normalisation, gamma = 1
    if resp_smooth > 0.0:
        from scipy.ndimage import gaussian_filter
        resp = gaussian_filter(resp, resp_smooth * sigma, truncate=truncate,
                               mode="nearest", output=resp)
    return resp


def kernel_halo(sigmas, rho_ratio, truncate, resp_smooth):
    """Voxels of context a tile needs so its interior matches an untiled run.

    The three stages (derivative at sigma, tensor window at rho, response blur
    at resp_smooth*sigma) are Gaussians in series, so they compose to a single
    Gaussian of s.d. sqrt(sigma^2 + rho^2 + smooth^2) -- adding the three
    supports linearly would over-pad by ~40% and cost that much extra compute.
    """
    s = float(np.max(sigmas))
    return int(np.ceil(truncate * s * np.sqrt(1.0 + rho_ratio**2 + resp_smooth**2)))


# ---------------------------------------------------------------------------
# Streaming per-voxel argmax over scale, with sub-grid refinement in log(sigma)
# ---------------------------------------------------------------------------

def _parabola_vertex_log(s_lo, s_mid, s_hi, y_lo, y_mid, y_hi):
    """Vectorised vertex of the parabola through the three (log sigma, response)
    points, returned as a sigma (not a log sigma), clipped to [s_lo, s_hi].

    Fitting in log(sigma) is the point: the grid is geometric, so in log space
    the three abscissae are equispaced and the vertex is unbiased. In linear
    sigma the right-hand sample sits further away than the left-hand one and the
    fit is pulled towards coarse scales.
    """
    u0, u1, u2 = np.log(s_lo), np.log(s_mid), np.log(s_hi)
    denom = (u0 - u1) * (u0 - u2) * (u1 - u2)
    A = (u2 * (y_mid - y_lo) + u1 * (y_lo - y_hi) + u0 * (y_hi - y_mid)) / denom
    B = (u2**2 * (y_lo - y_mid) + u1**2 * (y_hi - y_lo)
         + u0**2 * (y_mid - y_hi)) / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.where(A < 0, -B / (2.0 * A), u1)   # A >= 0 is not a max -> keep grid point
    uv = np.clip(uv, u0, u2)
    return np.exp(uv).astype(np.float32)


def scale_select(volume, sigmas=SIGMAS, rho_ratio=RHO_RATIO, truncate=TRUNCATE,
                 resp_smooth=RESP_SMOOTH, peak_mode="first_local", verbose=False):
    """Per-voxel Lindeberg scale selection over `sigmas`.

    peak_mode="first_local" (default) is the per-voxel form of
    pick_sigma_rho.first_local_max: take the FIRST interior local maximum
    R(i-1) < R(i) > R(i+1), not the global argmax. That module's docstring
    explains why on real data -- non-fibre content (sample shading, the
    tissue/air boundary, gross curvature) keeps adding energy as sigma grows
    long past the fibre's own scale, so a global argmax gets dragged to the
    coarse end of the grid. It matters MORE per-voxel than per-block, because
    a block average pools thousands of voxels whose fibre peak reinforces while
    that slow background rise is common to all of them. Measured on a 256^3
    i74 crop, global-argmax left only ~10-40% of tissue voxels with an interior
    peak; first-local-max recovers the fibre peak that was sitting there all
    along, underneath the rise.

    peak_mode="highest_local" keeps the strongest interior local max instead of
    the first -- a diagnostic, not the recommended setting.

    Streams the sweep with a 3-deep rolling window over scale, so peak memory
    is a fixed handful of volume copies no matter how many scales are swept
    (n response volumes would be n * 1.8 GB at 768^3).

    Returns dict, all shaped like `volume`:
        sigma_star : float32, sub-grid-refined selected scale
        rail_lo    : bool, no interior peak and the response FALLS with sigma --
                     the structure is finer than sigmas[0] (or the voxel is
                     noise-dominated, whose response falls like sigma^-3).
                     sigma_star there is a bound, not a measurement.
        rail_hi    : bool, no interior peak and the response RISES -- coarser
                     than sigmas[-1], or a non-fibre large-scale gradient.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    n = sigmas.size
    shape = volume.shape
    if n < 3:
        raise ValueError("need at least 3 scales to have an interior peak")

    sigma_star = np.zeros(shape, dtype=np.float32)
    peak_val = np.zeros(shape, dtype=np.float32) if peak_mode == "highest_local" else None
    found = np.zeros(shape, dtype=bool)
    r_first = None
    rm2 = rm1 = None                               # responses at i-2, i-1

    for i, s in enumerate(sigmas):
        t0 = time.time()
        r = lindeberg_response_voxel(volume, s, rho_ratio * s, truncate,
                                     resp_smooth)
        if i == 0:
            r_first = r.copy()

        if i >= 2:
            # Test the MIDDLE of the rolling window, scale index i-1.
            hit = (rm2 < rm1) & (rm1 > r)
            if peak_mode == "first_local":
                hit &= ~found                      # keep the FIRST such peak
            elif peak_mode == "highest_local":
                hit &= rm1 > peak_val              # keep the STRONGEST instead
            else:
                raise ValueError(f"unknown peak_mode {peak_mode!r}")
            if hit.any():
                # Refine and store IMMEDIATELY: the three responses the parabola
                # needs are all in the window right now, so carrying them to the
                # end of the sweep in three more volume-sized accumulators (the
                # obvious implementation) would cost 2 extra copies of the tile
                # for nothing. That is what lets a 192^3 tile fit in RAM here.
                sigma_star[hit] = _parabola_vertex_log(
                    sigmas[i - 2], sigmas[i - 1], sigmas[i],
                    rm2[hit].astype(np.float64), rm1[hit].astype(np.float64),
                    r[hit].astype(np.float64))
                if peak_val is not None:
                    peak_val[hit] = rm1[hit]
                found |= hit

        rm2, rm1 = rm1, r
        if verbose:
            print(f"    sigma={s:5.2f} rho={rho_ratio * s:5.2f}  "
                  f"({time.time() - t0:4.1f}s)", flush=True)

    # No interior peak anywhere on the curve: the response is (near-)monotone.
    # Which way it runs says which side of the grid the true scale is on.
    rising = rm1 > r_first
    rail_hi = ~found & rising
    rail_lo = ~found & ~rising
    del rm2, rm1, r, r_first, rising, peak_val

    sigma_star[rail_hi] = sigmas[-1]
    sigma_star[rail_lo] = sigmas[0]
    return dict(sigma_star=sigma_star, rail_lo=rail_lo, rail_hi=rail_hi)


# ---------------------------------------------------------------------------
# Calibration: sigma* -> physical feature size, on a known-width phantom
# ---------------------------------------------------------------------------

def calibrate(widths=(1.0, 1.5, 2.0, 3.0, 4.0), shape=(160, 160, 160),
              sigmas=SIGMAS, rho_ratio=RHO_RATIO, truncate=TRUNCATE,
              resp_smooth=RESP_SMOOTH, save=CALIB_PATH, verbose=True):
    """Fit sigma* = k * w on straight Gaussian tubes of known width w.

    Measured the same way the real map is read: the per-voxel sigma* restricted
    to tube voxels (intensity above half maximum -- the part of the volume that
    IS the fibre), summarised by its median. Fitted through the origin because
    scale selection is scale-covariant: a phantom scaled by a factor should give
    a sigma* scaled by the same factor, so an intercept would be a bug, not a
    parameter.
    """
    from lindeberg_sigma_phantoms import make_tube_phantom

    widths = np.asarray(widths, dtype=float)
    picks = np.full(widths.size, np.nan)
    m = kernel_halo(sigmas, rho_ratio, truncate, resp_smooth)
    for j, w in enumerate(widths):
        vol = make_tube_phantom(shape=shape, width=float(w))
        out = scale_select(vol, sigmas, rho_ratio, truncate, resp_smooth)
        core = (slice(m, -m),) * 3                       # drop kernel-contaminated faces
        s = out["sigma_star"][core]
        on_fibre = vol[core] > 0.5                       # above half maximum
        ok = on_fibre & ~out["rail_lo"][core] & ~out["rail_hi"][core]
        picks[j] = np.median(s[ok]) if ok.any() else np.nan
        if verbose:
            print(f"  width={w:4.1f} vox -> median sigma*={picks[j]:6.3f} "
                  f"(ratio {picks[j] / w:5.3f}, {ok.mean():5.1%} of core usable)")

    good = np.isfinite(picks)
    k = float(np.sum(picks[good] * widths[good]) / np.sum(widths[good] ** 2))
    resid = picks[good] - k * widths[good]
    r2 = float(1.0 - np.sum(resid**2) / np.sum((picks[good] - picks[good].mean())**2))

    calib = dict(k=k, r2=r2, widths=widths.tolist(), sigma_star=picks.tolist(),
                 sigmas=np.asarray(sigmas).tolist(), rho_ratio=rho_ratio,
                 truncate=truncate, resp_smooth=resp_smooth,
                 voxel_um=VOXEL_SIZE_UM, fwhm_factor=FWHM)
    if verbose:
        print(f"  fit through origin: sigma* = {k:.3f} * width   (R^2 = {r2:.3f})")
        print(f"  => feature size [um] = {FWHM:.3f} * sigma* / {k:.3f} * "
              f"{VOXEL_SIZE_UM:g} = {FWHM * VOXEL_SIZE_UM / k:.2f} * sigma*")
    if save:
        with open(save, "w") as f:
            json.dump(calib, f, indent=2)
        if verbose:
            print(f"  saved {save}")
    return calib


def load_calibration(path=CALIB_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python lindeberg_feature_size.py "
            f"--calibrate` first (it takes about a minute).")
    with open(path) as f:
        return json.load(f)


def sigma_to_um(sigma_star, calib):
    """FWHM-equivalent feature diameter in microns (see module docstring)."""
    return (np.float32(calib["fwhm_factor"] * calib["voxel_um"] / calib["k"])
            * np.asarray(sigma_star, dtype=np.float32))


# ---------------------------------------------------------------------------
# Tiled whole-volume run
# ---------------------------------------------------------------------------

def _tile_starts(n, tile):
    starts = list(range(0, max(n - tile, 0) + 1, tile))
    if starts[-1] + tile < n:
        starts.append(n - tile)
    return starts


def _load_z_slab(folder, z0, z1, crop=None):
    """Read only TIFF slices [z0, z1) and (optionally) crop them in-plane.

    Reading per-slice is what keeps a 768^3 run inside 15 GB: the full stack is
    never resident, only the slab currently being swept.
    """
    import tifffile
    from pathlib import Path
    files = sorted(Path(folder).glob("*.tif"))
    if crop is None:
        return np.stack([tifffile.imread(f) for f in files[z0:z1]])
    a, b = crop
    return np.stack([tifffile.imread(f)[a:b, a:b] for f in files[z0:z1]])


def feature_size_volume(folder=DEFAULT_VOLUME_DIR, crop=None, tile=192,
                        sigmas=SIGMAS, rho_ratio=RHO_RATIO, truncate=TRUNCATE,
                        resp_smooth=RESP_SMOOTH,
                        out_prefix="lindeberg_feature_size", verbose=True):
    """Whole-volume per-voxel scale selection, tiled with a full-support halo.

    The halo is truncate*(sigma_max + rho_max), i.e. the entire kernel reach of
    the coarsest scale in the sweep, so every tile's interior equals what an
    untiled run would produce -- no seams, unlike the piecewise-constant block
    stitching in sigma_rho_adaptive.

    Writes <out_prefix>_sigma.npy (float32) and <out_prefix>_flags.npy (uint8:
    bit0 tissue, bit1 rail-low, bit2 rail-high) as memmaps and returns their
    paths plus the volume shape.
    """
    import tifffile
    from pathlib import Path

    files = sorted(Path(folder).glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"no *.tif slices found in {folder}")
    nz_full = len(files)
    ny_full, nx_full = tifffile.imread(files[0]).shape
    if crop is None:
        z_range = (0, nz_full)
        shape = (nz_full, ny_full, nx_full)
    else:
        a, b = crop
        z_range = (a, b)
        shape = (b - a, b - a, b - a)

    sigmas = np.asarray(sigmas, dtype=float)
    halo = kernel_halo(sigmas, rho_ratio, truncate, resp_smooth)

    sig_path = f"{out_prefix}_sigma.npy"
    flg_path = f"{out_prefix}_flags.npy"
    sig_map = np.lib.format.open_memmap(sig_path, mode="w+", dtype=np.float32,
                                        shape=shape)
    flg_map = np.lib.format.open_memmap(flg_path, mode="w+", dtype=np.uint8,
                                        shape=shape)

    zs = _tile_starts(shape[0], tile)
    ys = _tile_starts(shape[1], tile)
    xs = _tile_starts(shape[2], tile)
    if verbose:
        print(f"volume {shape}, tile={tile}, halo={halo}, "
              f"{len(zs)}x{len(ys)}x{len(xs)} = {len(zs)*len(ys)*len(xs)} tiles, "
              f"{sigmas.size} scales")
        print(f"sigmas: {np.array2string(sigmas, precision=2)}  "
              f"(rho = {rho_ratio:g} * sigma)")

    t_start = time.time()
    n_tiles = len(zs) * len(ys) * len(xs)
    done = 0
    for z0 in zs:
        # Load the z-slab once (with halo) and reuse it for every in-plane tile.
        zp0, zp1 = max(z0 - halo, 0), min(z0 + tile + halo, shape[0])
        slab = _load_z_slab(folder, z_range[0] + zp0, z_range[0] + zp1,
                            crop=crop).astype(np.float32)
        air = AIR_FRAC * 255.0
        for y0 in ys:
            for x0 in xs:
                yp0, yp1 = max(y0 - halo, 0), min(y0 + tile + halo, shape[1])
                xp0, xp1 = max(x0 - halo, 0), min(x0 + tile + halo, shape[2])
                sub = slab[:, yp0:yp1, xp0:xp1]

                out = scale_select(sub, sigmas, rho_ratio, truncate, resp_smooth)
                core = (slice(z0 - zp0, z0 - zp0 + tile),
                        slice(y0 - yp0, y0 - yp0 + tile),
                        slice(x0 - xp0, x0 - xp0 + tile))

                tissue = sub[core] > air
                flags = tissue.astype(np.uint8)
                flags |= out["rail_lo"][core].astype(np.uint8) << 1
                flags |= out["rail_hi"][core].astype(np.uint8) << 2

                sig_map[z0:z0 + tile, y0:y0 + tile, x0:x0 + tile] = out["sigma_star"][core]
                flg_map[z0:z0 + tile, y0:y0 + tile, x0:x0 + tile] = flags

                done += 1
                if verbose:
                    el = time.time() - t_start
                    print(f"  tile {done:3d}/{n_tiles} (z={z0},y={y0},x={x0}) "
                          f"tissue={tissue.mean():5.1%} "
                          f"rail_lo={out['rail_lo'][core].mean():5.1%} "
                          f"rail_hi={out['rail_hi'][core].mean():5.1%} "
                          f"| {el / 60:5.1f} min elapsed, "
                          f"~{el / done * (n_tiles - done) / 60:5.1f} min left",
                          flush=True)
                del out
        del slab

    sig_map.flush(); flg_map.flush()
    if verbose:
        print(f"done in {(time.time() - t_start) / 60:.1f} min -> "
              f"{sig_path}, {flg_path}")
    return dict(sigma_path=sig_path, flags_path=flg_path, shape=shape,
                sigmas=sigmas.tolist(), rho_ratio=rho_ratio,
                resp_smooth=resp_smooth, truncate=truncate, halo=halo,
                crop=crop, folder=str(folder))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--calibrate", action="store_true",
                   help="fit sigma* -> microns on the tube phantom and exit")
    p.add_argument("--crop", nargs=2, type=int, default=None, metavar=("A", "B"),
                   help="cube crop [A:B] on all three axes (default: full volume)")
    p.add_argument("--tile", type=int, default=192)
    p.add_argument("--out", default="lindeberg_feature_size")
    args = p.parse_args()

    if args.calibrate:
        print("Calibrating sigma* against known tube widths")
        calibrate()
        sys.exit(0)

    calib = load_calibration()
    print(f"calibration: sigma* = {calib['k']:.3f} * width  "
          f"(R^2 = {calib['r2']:.3f})  ->  "
          f"{FWHM * calib['voxel_um'] / calib['k']:.2f} um per unit sigma*")
    meta = feature_size_volume(crop=tuple(args.crop) if args.crop else None,
                               tile=args.tile, out_prefix=args.out)
    meta["calib"] = calib
    with open(f"{args.out}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out}_meta.json")
