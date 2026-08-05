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

Coverage
--------
The structure tensor is undefined where the image is exactly constant: no
gradient anywhere in the window, so no direction exists to measure. On the
kissing phantom that is ~29% of the volume, because a Gaussian tube underflows
float32 about 21 voxels out; on a real noisy reconstruction it does not arise.
Those voxels are filled from the nearest voxel that does have a direction, so
the exported field has no holes. `--no-fill` leaves them zero, and the count is
always reported.

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

from kissing_phantom import build_kissing
from sigma_rho_adaptive import compute_adaptive_map, SIGMA_FLOOR
from dense_field import dense_gate, dense_host, GATE_HALF, validate_against_patches

SLAB = 16          # z-slab height for the eigh pass, to bound peak memory


def e3_from_tensor(S, slab=SLAB):
    """Smallest-eigenvalue eigenvector of a (6, nz, ny, nx) structure tensor.

    S carries (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz) with x = axis 2, y = axis 1,
    z = axis 0 (see `structure_tensor_3d`), so the result is (3, nz, ny, nx) in
    (x, y, z) -- the frame `host_fod` uses.

    Voxels whose tensor is identically zero come back as zeros; `eigh` would
    otherwise hand back an arbitrary basis for vacuum.

    Slabbed: the full (nz, ny, nx, 3, 3) float64 stack is 36x the volume.
    """
    sxx, syy, szz, sxy, sxz, syz = S
    shape = sxx.shape
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
        v[:, w.sum(-1) <= 0] = 0.0
        e3[:, sl] = v
        del M, w, Q, v
    return e3


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
    scales = np.unique(np.stack([sig.ravel(), rho.ravel()], 1), axis=0)
    if verbose:
        print(f"  [e3] over {len(scales)} distinct (sigma, rho)")
    volf = vol.astype(np.float64)          # tensors here are ~1e-5
    for s, r in scales:
        sel = (sig == s) & (rho == r)
        S = structure_tensor_3d(volf, float(s), float(r), truncate=4.0)
        e3[:, sel] = e3_from_tensor(S)[:, sel]
        del S
    return e3


def _to_zyx(v):
    """(3, ...) or (..., 3) in (x,y,z) -> unit (z,y,x); zeros stay zeros."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.zeros_like(v)
    ok = (n[..., 0] > 1e-12) & np.all(np.isfinite(v), axis=-1)
    out[ok] = (v[ok] / n[ok])[..., ::-1]
    return out


def build_peaks(vol, e3, gated, host_dirs, host_n, fill=True, verbose=True):
    """Assemble the (z, y, x, 6) peaks image and its uint8 count map."""
    shape = vol.shape
    peaks = np.zeros((*shape, 6), dtype=np.float32)

    # peak 1 everywhere the tensor defines one
    peaks[..., 0:3] = _to_zyx(np.moveaxis(e3, 0, -1))

    # where the gate fired and HOST answered, HOST's answer replaces it --
    # that is what routing to HOST means
    gi = np.argwhere(gated)
    if len(gi):
        z, y, x = gi[:, 0], gi[:, 1], gi[:, 2]
        d1 = _to_zyx(host_dirs[:, 0])
        d2 = _to_zyx(host_dirs[:, 1])
        h1 = host_n >= 1
        h2 = host_n >= 2
        peaks[z[h1], y[h1], x[h1], 0:3] = d1[h1]
        peaks[z[h2], y[h2], x[h2], 3:6] = d2[h2]

    defined = np.linalg.norm(peaks[..., 0:3], axis=-1) > 0
    n_filled = 0
    if fill and not defined.all() and defined.any():
        # nearest voxel that HAS a direction. Only peak 1 is carried across:
        # a vacuum voxel inherits an orientation, not a crossing.
        _, ind = ndimage.distance_transform_edt(~defined, return_indices=True)
        src = peaks[..., 0:3][tuple(ind)]
        n_filled = int((~defined).sum())
        peaks[..., 0:3] = np.where(defined[..., None], peaks[..., 0:3], src)
        if verbose:
            print(f"  [fill] {n_filled} voxels ({n_filled / vol.size:.1%}) had no "
                  f"defined tensor; filled from the nearest voxel that did")

    n_dirs = np.zeros(shape, dtype=np.uint8)
    n_dirs[np.linalg.norm(peaks[..., 0:3], axis=-1) > 0] = 1
    n_dirs[np.linalg.norm(peaks[..., 3:6], axis=-1) > 0] = 2
    return peaks, n_dirs, n_filled


def save_nifti(arr, path, vox, dtype=np.float32):
    """Diagonal affine of isotropic `vox`. A phantom has no physical scale, so
    vox=1 leaves voxel index == world coordinate and vectors in voxel units."""
    img = nib.Nifti1Image(np.asarray(arr, dtype=dtype),
                          np.diag([vox, vox, vox, 1.0]))
    img.header.set_xyzt_units("mm")
    nib.save(img, path)
    return path


def export(vol, out, stem, vox=1.0, block=32, half=GATE_HALF, fill=True,
           verbose=True):
    """vol -> dense gate -> dense HOST -> NIfTI. Returns a summary dict."""
    os.makedirs(out, exist_ok=True)
    e3 = dense_e3(vol, block=block, verbose=verbose)
    c_s, fa, _ = dense_gate(vol, half=half, verbose=verbose)
    gated = c_s > fa
    dirs, n_found = dense_host(vol, gated, half=half, verbose=verbose)
    peaks, n_dirs, n_filled = build_peaks(vol, e3, gated, dirs, n_found,
                                          fill=fill, verbose=verbose)

    paths = dict(
        vol=save_nifti(vol, os.path.join(out, f"{stem}_phantom.nii.gz"), vox),
        peaks=save_nifti(peaks, os.path.join(out, f"{stem}_peaks.nii.gz"), vox),
        ndirs=save_nifti(n_dirs, os.path.join(out, f"{stem}_ndirs.nii.gz"),
                         vox, dtype=np.uint8),
        cs=save_nifti(c_s, os.path.join(out, f"{stem}_cs.nii.gz"), vox),
        fa=save_nifti(fa, os.path.join(out, f"{stem}_fa.nii.gz"), vox))
    return dict(paths=paths, n_dirs=n_dirs, gated=gated, c_s=c_s, FA=fa,
                n_filled=n_filled, n_found=n_found)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gap", type=float, default=8.0,
                   help="voxels of empty space between the two bundles")
    p.add_argument("--curv", type=float, default=0.028, help="bundle curvature")
    p.add_argument("--block", type=int, default=32, help="Step 1 block size")
    p.add_argument("--half", type=int, default=GATE_HALF,
                   help="gate/HOST window radius (window is 2*half+1 cubed)")
    p.add_argument("--no-fill", action="store_true",
                   help="leave voxels with no defined tensor at zero")
    p.add_argument("--validate", action="store_true",
                   help="also check the dense fields against host_fod.analyse")
    p.add_argument("--vox", type=float, default=1.0,
                   help="isotropic voxel size for the NIfTI affine (mm)")
    p.add_argument("--out", default=os.path.join(_ROOT, "results"))
    a = p.parse_args()

    print("=" * 78)
    print("KISSING PHANTOM -> DENSE STEP 1 + STEP 2 -> NIfTI  (per voxel, no patch grid)")
    print("=" * 78)
    vol, meta = build_kissing(same_size=True, gap=a.gap, curv=a.curv)
    print(f"  phantom {vol.shape} {vol.dtype}  radii {meta['rA']:.1f}/{meta['rB']:.1f}"
          f"  true min centreline gap {meta['true_min_gap']:.2f} vox")

    r = export(vol, a.out, "kissing", vox=a.vox, block=a.block, half=a.half,
               fill=not a.no_fill)

    n_dirs, gated = r["n_dirs"], r["gated"]
    solid = vol > 0
    print("-" * 78)
    print(f"  gate (c_s > FA): {int(gated.sum())} voxels ({gated.mean():.2%})")
    print(f"  vectors        : {int((n_dirs >= 1).sum())}/{vol.size} voxels "
          f"({(n_dirs >= 1).mean():.1%}); "
          f"{(n_dirs[solid] >= 1).mean():.1%} of the {int(solid.sum())} non-zero")
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
