"""export_nifti.py -- Step 2 as a dense NIfTI orientation field, one vector per voxel.

    <out>/kissing_phantom.nii.gz   greyscale volume        float32 (z, y, x)
    <out>/kissing_peaks.nii.gz     orientations            float32 (z, y, x, 6)
    <out>/kissing_ndirs.nii.gz     directions per voxel    uint8   (z, y, x)
    <out>/kissing_cs.nii.gz        sphericity c_s          float32 (z, y, x)
    <out>/kissing_fa.nii.gz        fractional anisotropy   float32 (z, y, x)

`peaks` follows the MRtrix `sh2peaks` convention -- 3 * n_peaks components, so
[..., 0:3] is the first direction and [..., 3:6] the second, all-zero where
there is no second. A voxel carries two directions exactly where the gate fires
(c_s > FA) and HOST-FOD resolves a second lobe there.

Everything is per-voxel
-----------------------
An earlier version took the gate and the second peak from `build_gated_field`'s
stride-6 patch grid and stamped each patch's answer across its neighbourhood.
That put rectangular blocks in the output either side of every branching point
-- an artifact of the sampling grid, nothing to do with the structure tensor --
and left the c_s/FA decision piecewise-constant on 6-voxel cells.

Nothing here uses a patch grid. c_s and FA are continuous fields
(`dense_field.dense_gate`), the gate is thresholded per voxel, and HOST's
directions are computed per voxel at the voxels that pass
(`dense_field.dense_host`). See `dense_field` for how the per-patch arithmetic
becomes a box filter, and for the check that it reproduces `host_fod.analyse`.

Peak 1 where the gate does NOT fire is Step 1's own e3, measured per voxel from
the structure tensor at each block's certified (sigma, rho).

Every value is measured; nothing is invented
--------------------------------------------
No interpolation, smoothing or extrapolation anywhere in this file. Where the
background carries only shot noise the tensor is still well defined, and its e3
is written as-is -- a real, and uniformly random, measurement of a region with
no structure. `kissing_measured.nii.gz` marks where that measurement is signal
rather than noise; mask with it rather than trusting every voxel equally.

The floor is measured too, not assumed: the p99 of the tensor trace over pure
background, so 1% of background passes by construction.

Two things this file used to do, and no longer does. It filled empty space from
the nearest measured voxel, which partitions the background into Voronoi cells
whose walls show up as hard single-voxel direction seams -- 58% of the axis-0
discontinuities in an earlier export. And it extrapolated a smooth orientation
from the director field, which is smoother but is invented data. Both are gone.

Two phantom changes remove the underlying problem rather than papering over it:

* `--bg` puts Poisson shot noise on a small pedestal. The noiseless phantom
  underflows float32 to EXACTLY zero over half the volume, where the tensor is
  the zero matrix and an eigensolver returns whatever rounding noise it
  decomposed -- traces down to 1e-111 of peak, and the streaks that came with
  them. A real reconstruction never has this problem. Now neither does this one.

* `--halo` builds, processes and then crops a padded volume, with the fibres
  extended so they run out through the enlarged boundary. Without it the outer
  30 voxels are measured against whatever the filter's boundary mode invents (a
  16-voxel gradient kernel at sigma_grad=4 feeding a 14-voxel box), which on an
  80-voxel cube is most of the volume.

Frame: directions come out of `host_fod` and the tensor in (x, y, z) while the
volume is (z, y, x), so they are reversed on the way out -- the same `[::-1]`
`field_from_gated` applies and `test_frame_convention.py` guards, for both the
patch path and this one.

Sign: e3 and -e3 describe the same fibre. Nothing canonicalises the sign; render
as line segments, not arrows (see `render_field.py`).

Why e3 comes from `numpy.linalg.eigh`
-------------------------------------
Not from `eig_special_3d`, which returns a NaN eigenvector on 37.5% of the
kissing phantom's fibre-core voxels and 95% of the volume. Its EIGENVALUES are
right there -- e.g. (3.65e-2, 1.28e-2, 7.3e-5), an unambiguous stick -- so the
fault is its closed-form eigenvector step, and it strikes where lambda3 -> 0:
the cleanest single-fibre voxels, not the isotropic ones. float64 alone lifts
coverage from 4.7% to 36.6% but leaves the core at 62.5%. Stacked `eigh` is
finite at 100% of voxels, matches the library's eigenvalues to 7e-15, and gives
an identical direction wherever the library answers at all.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage
from structure_tensor import structure_tensor_3d

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection")):
    sys.path.insert(0, _p)

from kissing_phantom import build_kissing, crop_halo
from sigma_rho_adaptive import compute_adaptive_map, SIGMA_FLOOR
from dense_field import (dense_gate, dense_host, tissue_windows, GATE_HALF,
                         GATE_SIGMA_GRAD, AIR_FRAC, validate_against_patches)

# Every voxel the gate touches: a gradient kernel of ceil(truncate * sigma_grad)
# feeding a box of `half`. Process a volume padded by this much, crop it away,
# and no exported voxel was ever measured against an invented boundary.
DEFAULT_HALO = int(np.ceil(4.0 * GATE_SIGMA_GRAD)) + GATE_HALF        # 30

SLAB = 16          # z-slab height for the eigh pass, to bound peak memory

# Relative trace below which a structure tensor carries no measurement.
#
# The image is float32, so the smallest gradient distinguishable from rounding
# is about eps_f32 of the image peak; the tensor is quadratic in the gradient,
# so its own noise floor is eps_f32^2 of the peak trace. Below that, eigh is
# decomposing rounding noise and returns a direction with no relation to the
# image -- which is exactly what produced single-voxel streaks in the empty
# corners of the kissing phantom, at traces down to 1e-111 of peak. On a real
# reconstruction the noise never lets the tensor get this small and this floor
# never binds.
TRACE_FLOOR_REL = float(np.finfo(np.float32).eps) ** 2      # 1.42e-14


def signal_mask(trace, background, pct=99.0, verbose=True):
    """Voxels whose tensor is signal, not noise.

    Proximity to a fibre is the wrong test: a voxel 8 away from a 1.6-radius
    tube sees fibre intensity ~4e-6 against shot noise ~3e-3, so its tensor is
    pure noise even though tissue is "nearby". What separates them is the
    tensor's own magnitude against the magnitude noise alone produces.

    So the floor is measured, not assumed: take the `pct` percentile of the
    trace over voxels that contain only background, and keep what exceeds it.
    At pct=99 that admits 1% of pure background by construction -- a stated
    false-positive rate rather than a tuned threshold.
    """
    if not background.any():
        return trace > 0
    floor = float(np.percentile(trace[background], pct))
    mask = trace > floor
    if verbose:
        print(f"       tensor-trace noise floor (p{pct:g} of background) = "
              f"{floor:.3e}, peak = {float(trace.max()):.3e} "
              f"-> SNR cut keeps {mask.mean():.1%} of voxels")
    return mask


def e3_from_tensor(S, slab=SLAB, trace_floor_rel=TRACE_FLOOR_REL):
    """Smallest-eigenvalue eigenvector of a (6, nz, ny, nx) structure tensor.

    S carries (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz) with x = axis 2, y = axis 1,
    z = axis 0 (see `structure_tensor_3d`), so the result is (3, nz, ny, nx) in
    (x, y, z) -- the frame `host_fod` uses.

    Returns (e3, measured). Voxels whose tensor trace falls below
    `trace_floor_rel` of the volume's peak carry no measurement: e3 is zeroed
    and `measured` is False there. Zeroing only the exactly-zero tensors is not
    enough -- see TRACE_FLOOR_REL.

    Slabbed: the full (nz, ny, nx, 3, 3) float64 stack is 36x the volume.
    """
    sxx, syy, szz, sxy, sxz, syz = S
    shape = sxx.shape
    trace = sxx + syy + szz
    floor = trace_floor_rel * float(trace.max())
    measured = trace > floor
    e3 = np.zeros((3,) + shape, dtype=np.float32)
    for z0 in range(0, shape[0], slab):
        sl = slice(z0, min(z0 + slab, shape[0]))
        M = np.empty(sxx[sl].shape + (3, 3), dtype=np.float64)
        M[..., 0, 0] = sxx[sl]
        M[..., 1, 1] = syy[sl]
        M[..., 2, 2] = szz[sl]
        M[..., 0, 1] = M[..., 1, 0] = sxy[sl]
        M[..., 0, 2] = M[..., 2, 0] = sxz[sl]
        M[..., 1, 2] = M[..., 2, 1] = syz[sl]
        w, Q = np.linalg.eigh(M)
        v = np.moveaxis(Q[..., 0], -1, 0)
        v[:, ~measured[sl]] = 0.0
        e3[:, sl] = v
        del M, w, Q, v
    return e3, measured


def dense_e3(vol, block=32, sigma_fallback=SIGMA_FLOOR, verbose=True):
    """Step 1's e3 at every voxel, each block at its own certified (sigma, rho).

    Uses the same searchsorted block lookup and below-floor fallback as
    `build_gated_field.block_scale`. Returns (3, nz, ny, nx) in (x, y, z).
    """
    m = compute_adaptive_map(vol, block=block, verbose=False)
    zs, ys, xs = m["zs"], m["ys"], m["xs"]

    def _bidx(starts, n):
        return np.clip(np.searchsorted(starts, np.arange(n), "right") - 1,
                       0, len(starts) - 1)

    ix = np.ix_(_bidx(zs, vol.shape[0]), _bidx(ys, vol.shape[1]),
                _bidx(xs, vol.shape[2]))
    sig, rho = m["sigma_map"][ix], m["rho_map"][ix]
    bad = ~np.isfinite(sig) | (sig < sigma_fallback)
    sig = np.where(bad, sigma_fallback, sig)
    rho = np.where(bad, sigma_fallback, np.where(np.isfinite(rho), rho, sig))

    e3 = np.zeros((3,) + vol.shape, dtype=np.float32)
    measured = np.zeros(vol.shape, dtype=bool)
    trace = np.zeros(vol.shape, dtype=np.float32)
    scales = np.unique(np.stack([sig.ravel(), rho.ravel()], 1), axis=0)
    if verbose:
        print(f"  [e3] over {len(scales)} distinct (sigma, rho)")
    volf = vol.astype(np.float64)          # tensors here are ~1e-5
    for s, r in scales:
        sel = (sig == s) & (rho == r)
        S = structure_tensor_3d(volf, float(s), float(r), truncate=4.0)
        v3, mk = e3_from_tensor(S)
        e3[:, sel] = v3[:, sel]
        measured[sel] = mk[sel]
        trace[sel] = (S[0] + S[1] + S[2])[sel]
        del S, v3, mk
    return e3, measured, trace


def _to_zyx(v):
    """(3, ...) or (..., 3) in (x,y,z) -> unit (z,y,x); zeros stay zeros."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.zeros_like(v)
    ok = (n[..., 0] > 1e-12) & np.all(np.isfinite(v), axis=-1)
    out[ok] = (v[ok] / n[ok])[..., ::-1]
    return out


def build_peaks(vol, e3, measured, gated, host_dirs, host_n, verbose=True):
    """Assemble the (z, y, x, 6) peaks image, its uint8 count map, and the mask
    saying which voxels the orientation is a measurement of SIGNAL.

    Every value written here is measured. Nothing is interpolated, smoothed or
    extrapolated: where the background carries only shot noise the tensor is
    still well defined and its e3 is reported as-is, which is a real (and
    uniformly random) measurement of a region with no structure. `measured`
    marks where that measurement is signal rather than noise -- mask with it
    rather than trusting every voxel equally.
    """
    shape = vol.shape
    peaks = np.zeros((*shape, 6), dtype=np.float32)
    peaks[..., 0:3] = _to_zyx(np.moveaxis(e3, 0, -1))

    # where the gate fired and HOST answered, HOST's answer replaces it --
    # that is what routing to HOST means
    gi = np.argwhere(gated)
    if len(gi):
        z, y, x = gi[:, 0], gi[:, 1], gi[:, 2]
        d1 = _to_zyx(host_dirs[:, 0])
        d2 = _to_zyx(host_dirs[:, 1])
        h1, h2 = host_n >= 1, host_n >= 2
        peaks[z[h1], y[h1], x[h1], 0:3] = d1[h1]
        peaks[z[h2], y[h2], x[h2], 3:6] = d2[h2]

    n_dirs = np.zeros(shape, dtype=np.uint8)
    n_dirs[np.linalg.norm(peaks[..., 0:3], axis=-1) > 0] = 1
    n_dirs[np.linalg.norm(peaks[..., 3:6], axis=-1) > 0] = 2
    return peaks, n_dirs, measured | gated


def save_nifti(arr, path, vox, dtype=np.float32):
    """Diagonal affine of isotropic `vox`. A phantom has no physical scale, so
    vox=1 leaves voxel index == world coordinate and vectors in voxel units."""
    img = nib.Nifti1Image(np.asarray(arr, dtype=dtype),
                          np.diag([vox, vox, vox, 1.0]))
    img.header.set_xyzt_units("mm")
    nib.save(img, path)
    return path


def export(vol, out, stem, vox=1.0, block=32, half=GATE_HALF,
           halo=0, verbose=True):
    """vol -> dense gate -> dense HOST -> crop the halo -> NIfTI.

    `vol` is the PADDED volume. Everything is measured on it, and only then is
    the halo cropped, so no exported voxel saw a filter boundary.
    """
    os.makedirs(out, exist_ok=True)
    e3, numeric_ok, trace = dense_e3(vol, block=block, verbose=verbose)
    # A tensor built entirely from background noise is numerically fine and
    # physically meaningless: it returns a uniformly random direction, which
    # fills empty space with speckle. Keep only what rises above the noise's
    # own tensor magnitude, with the floor measured from the background.
    reach = 2 * int(np.ceil(4.0 * SIGMA_FLOOR))
    background = ~tissue_windows(vol, half=reach)
    measured = numeric_ok & signal_mask(trace, background, verbose=verbose)
    c_s, fa, _ = dense_gate(vol, half=half, verbose=verbose)
    gated = c_s > fa
    dirs, n_found = dense_host(vol, gated, half=half, verbose=verbose)
    peaks, n_dirs, meas = build_peaks(vol, e3, measured, gated, dirs, n_found,
                                      verbose=verbose)

    if halo > 0 and verbose:
        print(f"  [crop] {vol.shape} -> "
              f"{tuple(s - 2 * halo for s in vol.shape)}, dropping the {halo}-voxel "
              f"halo every voxel of which saw a filter boundary")
    c = lambda a: crop_halo(a, halo)                              # noqa: E731
    vol, peaks, n_dirs, meas = c(vol), c(peaks), c(n_dirs), c(meas)
    c_s, fa, gated = c(c_s), c(fa), c(gated)

    paths = dict(
        vol=save_nifti(vol, os.path.join(out, f"{stem}_phantom.nii.gz"), vox),
        peaks=save_nifti(peaks, os.path.join(out, f"{stem}_peaks.nii.gz"), vox),
        ndirs=save_nifti(n_dirs, os.path.join(out, f"{stem}_ndirs.nii.gz"),
                         vox, dtype=np.uint8),
        measured=save_nifti(meas, os.path.join(out, f"{stem}_measured.nii.gz"),
                            vox, dtype=np.uint8),
        cs=save_nifti(c_s, os.path.join(out, f"{stem}_cs.nii.gz"), vox),
        fa=save_nifti(fa, os.path.join(out, f"{stem}_fa.nii.gz"), vox))
    return dict(paths=paths, vol=vol, n_dirs=n_dirs, gated=gated, c_s=c_s,
                FA=fa, measured=meas, n_found=n_found)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gap", type=float, default=8.0,
                   help="voxels of empty space between the two bundles")
    p.add_argument("--curv", type=float, default=0.028, help="bundle curvature")
    p.add_argument("--bg", type=float, default=0.02,
                   help="background pedestal, as a fraction of the fibre peak, "
                        "carrying Poisson shot noise; 0 disables (and brings "
                        "back the exactly-zero-tensor regions)")
    p.add_argument("--photons", type=float, default=2000.0,
                   help="expected photon count at intensity 1; higher = quieter")
    p.add_argument("--seed", type=int, default=0, help="noise seed")
    p.add_argument("--block", type=int, default=32, help="Step 1 block size")
    p.add_argument("--half", type=int, default=GATE_HALF,
                   help="gate/HOST window radius (window is 2*half+1 cubed)")
    p.add_argument("--halo", type=int, default=DEFAULT_HALO,
                   help="voxels of padding to build, process and then crop, so "
                        "no exported voxel saw a filter boundary; 0 disables")
    p.add_argument("--validate", action="store_true",
                   help="also check the dense fields against host_fod.analyse")
    p.add_argument("--vox", type=float, default=1.0,
                   help="isotropic voxel size for the NIfTI affine (mm)")
    p.add_argument("--out", default=os.path.join(_ROOT, "results"))
    a = p.parse_args()

    print("=" * 78)
    print("KISSING PHANTOM -> DENSE STEP 1 + STEP 2 -> NIfTI  (per voxel, no patch grid)")
    print("=" * 78)
    vol, meta = build_kissing(same_size=True, gap=a.gap, curv=a.curv, bg=a.bg,
                              photons=a.photons, seed=a.seed, halo=a.halo)
    print(f"  phantom {vol.shape} {vol.dtype}  radii {meta['rA']:.1f}/{meta['rB']:.1f}"
          f"  true min centreline gap {meta['true_min_gap']:.2f} vox")
    if a.halo:
        print(f"  halo {a.halo} vox: built {vol.shape}, will crop to "
              f"{tuple(s - 2 * a.halo for s in vol.shape)}; fibres run out "
              f"through the padded boundary (ymax {meta['ymax']:.0f})")
    if a.bg > 0:
        print(f"  background: pedestal {a.bg:g} of peak, {a.photons:g} photons "
              f"-> shot noise {meta['bg_noise']:.5f} ({meta['bg_noise'] * 100:.2f}% "
              f"of peak); zero voxels: {int((vol == 0).sum())}")

    r = export(vol, a.out, "kissing", vox=a.vox, block=a.block, half=a.half,
               halo=a.halo)

    n_dirs, gated, vol, meas = r["n_dirs"], r["gated"], r["vol"], r["measured"]
    print("-" * 78)
    print(f"  gate (c_s > FA): {int(gated.sum())} voxels ({gated.mean():.2%})")
    print(f"  vectors        : {int((n_dirs >= 1).sum())}/{vol.size} voxels "
          f"({(n_dirs >= 1).mean():.1%}) -- every one measured, none invented")
    print(f"  of those, signal rather than noise: {int(meas.sum())} "
          f"({meas.mean():.1%})   [kissing_measured.nii.gz]")
    print(f"  two directions : {int((n_dirs == 2).sum())} voxels "
          f"({(n_dirs == 2).mean():.2%})")
    if a.validate:
        validate_against_patches(vol)
    print("-" * 78)
    for k, q in r["paths"].items():
        print(f"  wrote {q}  ({os.path.getsize(q) / 1024:.0f} KB)")
    print("=" * 78)


if __name__ == "__main__":
    main()
