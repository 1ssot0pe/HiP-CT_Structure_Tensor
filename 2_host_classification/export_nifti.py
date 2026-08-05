"""export_nifti.py -- Step 2's orientation field as NIfTI, one vector per voxel.

`GatedField` answers per PATCH, on a stride-6 grid. This writes the same answer
as a dense image a viewer can open:

    <out>/kissing_phantom.nii.gz   greyscale volume        float32 (z, y, x)
    <out>/kissing_peaks.nii.gz     orientations            float32 (z, y, x, 6)
    <out>/kissing_ndirs.nii.gz     directions per voxel    uint8   (z, y, x)

`peaks` follows the MRtrix `sh2peaks` convention -- 3 * n_peaks components, so
[..., 0:3] is the first direction and [..., 3:6] the second, all-zero where
there is no second. `ndirs` is that count as its own map, for masking.

Where each vector comes from
----------------------------
Peak 1 is measured at every voxel, not interpolated: Step 1's structure tensor
is already dense, so `dense_e3` recomputes it at each block's certified (σ, ρ)
and reads e3 everywhere.

Only the gate decision and HOST's second orientation are patch-scale, and those
reach the voxels by nearest-patch lookup (the cKDTree idiom
`self_optimising_tractography` uses), bounded by `--max-dist` so a patch's
answer stays local. Running the gate per voxel is not an alternative: at
~0.25 s per `analyse` call, an 80³ volume would take ~35 hours.

Where the gate fired, HOST's answer replaces the tensor's, because that is what
Step 2 means -- "gated patches carry HOST's multi-orientation answer INSTEAD".

Frame: directions come out of `host_fod.analyse` and the tensor in (x, y, z)
while the volume is (z, y, x), so they are reversed on the way out -- the same
`[::-1]` that `field_from_gated` applies and `test_frame_convention.py` guards.
Vector component c then indexes image axis c.

Sign: e3 and -e3 describe the same fibre. Nothing here canonicalises the sign;
render as line segments, not arrows (see `render_field.py`).

Why e3 comes from `numpy.linalg.eigh`
-------------------------------------
Not from `eig_special_3d`, which returns a NaN eigenvector on 37.5% of the
kissing phantom's fibre-core voxels (I > 0.5) and 95% of the volume. Its
EIGENVALUES are right there -- e.g. (3.65e-2, 1.28e-2, 7.3e-5), an unambiguous
stick -- so the fault is in its closed-form eigenvector step, and it strikes
where λ3 → 0: the cleanest single-fibre voxels, not the isotropic ones. Its own
docstring warns the analytic solution needs float64 on small tensors; float64
alone lifts whole-volume coverage from 4.7% to 36.6% but leaves the core at
62.5%.

Stacked `eigh` returns a finite e3 at 100% of voxels, matches the library's
eigenvalues to 7e-15, gives an identical direction (|dot| = 1.000000) wherever
the library does answer, and costs ~0.4 s for 512k voxels.

`build_gated_field` still calls `eig_special_3d`, and its non-finite fallback --
commented there as "Skielex returns a non-finite eigenvector at isotropic
voxels" -- is firing on this same bug, on the most anisotropic voxels rather
than isotropic ones. Changing it would move Step 2 results repo-wide, so this
exporter leaves that call alone.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
from structure_tensor import structure_tensor_3d

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection")):
    sys.path.insert(0, _p)

from kissing_phantom import build_kissing
from sigma_rho_gated_field import build_gated_field, DEFAULT_STRIDE
from sigma_rho_adaptive import compute_adaptive_map, SIGMA_FLOOR

SLAB = 16          # z-slab height for the eigh pass, to bound peak memory


def e3_from_tensor(S, slab=SLAB):
    """Smallest-eigenvalue eigenvector of a (6, nz, ny, nx) structure tensor.

    S carries (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz) with x = axis 2, y = axis 1,
    z = axis 0 (see `structure_tensor_3d`), so the returned (3, nz, ny, nx)
    field is in (x, y, z) -- the same frame `host_fod` uses.

    Voxels whose tensor is identically zero carry no direction and come back as
    zeros; `eigh` would otherwise hand back an arbitrary basis for vacuum.

    Solved in z-slabs: the full (nz, ny, nx, 3, 3) float64 stack is 36x the
    volume in memory, which a real i74 crop cannot hold.
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
        w, Q = np.linalg.eigh(M)                 # eigenvalues ascending
        v = np.moveaxis(Q[..., 0], -1, 0)        # eigenvector of the smallest
        v[:, w.sum(-1) <= 0] = 0.0               # vacuum -> no direction
        e3[:, sl] = v
        del M, w, Q, v
    return e3


def dense_e3(vol, block=32, sigma_fallback=SIGMA_FLOOR, verbose=True):
    """Step 1's e3 at EVERY voxel, each block at its own certified (σ, ρ).

    Mirrors `build_gated_field`'s `block_scale` -- same searchsorted lookup,
    same below-floor fallback -- so dense and patch answers agree where they
    overlap. Returns (3, nz, ny, nx) in (x, y, z).
    """
    m = compute_adaptive_map(vol, block=block, verbose=False)
    zs, ys, xs = m["zs"], m["ys"], m["xs"]

    def _bidx(starts, n):
        return np.clip(np.searchsorted(starts, np.arange(n), "right") - 1,
                       0, len(starts) - 1)

    ix = np.ix_(_bidx(zs, vol.shape[0]), _bidx(ys, vol.shape[1]),
                _bidx(xs, vol.shape[2]))
    sig = m["sigma_map"][ix]
    rho = m["rho_map"][ix]
    bad = ~np.isfinite(sig) | (sig < sigma_fallback)
    sig = np.where(bad, sigma_fallback, sig)
    rho = np.where(bad, sigma_fallback, np.where(np.isfinite(rho), rho, sig))

    e3 = np.zeros((3,) + vol.shape, dtype=np.float32)
    scales = np.unique(np.stack([sig.ravel(), rho.ravel()], 1), axis=0)
    if verbose:
        print(f"  [dense] e3 over {len(scales)} distinct (sigma, rho)")
    # float64: the tensors here are ~1e-5, small enough that any 3x3 eigensolve
    # in float32 loses the eigenvector.
    volf = vol.astype(np.float64)
    for s, r in scales:
        sel = (sig == s) & (rho == r)
        S = structure_tensor_3d(volf, float(s), float(r), truncate=4.0)
        e3[:, sel] = e3_from_tensor(S)[:, sel]
        del S
    if verbose:
        n = int((np.linalg.norm(e3, axis=0) > 0).sum())
        print(f"          {n}/{vol.size} voxels ({n / vol.size:.1%}) have a "
              f"defined e3")
    return e3


def _unit_zyx(d):
    """(x, y, z) direction -> unit (z, y, x), or zeros if degenerate."""
    d = np.asarray(d, dtype=float)
    if d.shape != (3,) or not np.all(np.isfinite(d)):
        return np.zeros(3, dtype=np.float32)
    n = np.linalg.norm(d)
    return np.zeros(3, np.float32) if n < 1e-12 else (d / n)[::-1].astype(np.float32)


def dense_peaks(vol, field, e3, max_dist, mask_thresh=0.0):
    """Dense (z, y, x, 6) peaks image + the (z, y, x) uint8 count."""
    shape = vol.shape
    peaks = np.zeros((*shape, 6), dtype=np.float32)

    norm = np.linalg.norm(e3, axis=0)
    ok = norm > 1e-12
    for c in range(3):                            # (x,y,z) -> (z,y,x)
        peaks[..., c][ok] = (e3[2 - c] / np.where(ok, norm, 1.0))[ok]

    centres = np.array([p["centre"] for p in field.patches], dtype=float)
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"),
                    axis=-1).reshape(-1, 3).astype(float)
    dist, idx = cKDTree(centres).query(grid, workers=-1)
    reach = (dist <= max_dist).reshape(shape)
    idx = idx.reshape(shape)

    for i, p in enumerate(field.patches):
        if not p["gated"]:
            continue
        sel = reach & (idx == i)
        if not sel.any():
            continue
        d0 = _unit_zyx(p["dirs"][0])
        if np.any(d0):
            peaks[sel, 0:3] = d0
        if p["n_dirs"] >= 2:
            d1 = _unit_zyx(p["dirs"][1])
            if np.any(d1):
                peaks[sel, 3:6] = d1

    if mask_thresh > 0:
        peaks[vol < mask_thresh * float(vol.max())] = 0.0

    # counted off the FINAL image: tallying inside the loop above would
    # double-count overlaps and ignore whatever the mask then blanks.
    n_dirs = np.zeros(shape, dtype=np.uint8)
    n_dirs[np.linalg.norm(peaks[..., 0:3], axis=-1) > 0] = 1
    n_dirs[np.linalg.norm(peaks[..., 3:6], axis=-1) > 0] = 2
    return peaks, n_dirs


def save_nifti(arr, path, vox, dtype=np.float32):
    """Diagonal affine of isotropic `vox`. A phantom has no physical scale, so
    vox=1 leaves voxel index == world coordinate and vectors in voxel units."""
    img = nib.Nifti1Image(np.asarray(arr, dtype=dtype),
                          np.diag([vox, vox, vox, 1.0]))
    img.header.set_xyzt_units("mm")
    nib.save(img, path)
    return path


def export(vol, field, out, stem, vox=1.0, block=32, max_dist=None,
           mask_thresh=0.0, verbose=True):
    """Write vol + dense peaks + ndirs as NIfTI. Returns the three paths."""
    os.makedirs(out, exist_ok=True)
    e3 = dense_e3(vol, block=block, verbose=verbose)
    peaks, n_dirs = dense_peaks(vol, field, e3,
                                max_dist=DEFAULT_STRIDE if max_dist is None
                                else max_dist,
                                mask_thresh=mask_thresh)
    return (save_nifti(vol, os.path.join(out, f"{stem}_phantom.nii.gz"), vox),
            save_nifti(peaks, os.path.join(out, f"{stem}_peaks.nii.gz"), vox),
            save_nifti(n_dirs, os.path.join(out, f"{stem}_ndirs.nii.gz"), vox,
                       dtype=np.uint8),
            n_dirs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gap", type=float, default=8.0,
                   help="voxels of empty space between the two bundles")
    p.add_argument("--curv", type=float, default=0.028, help="bundle curvature")
    p.add_argument("--block", type=int, default=32, help="Step 1 block size")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                   help="Step 2 patch spacing; the gate cost scales as stride^-3")
    p.add_argument("--max-dist", type=float, default=None,
                   help="how far a gated patch's answer may travel to a voxel "
                        "(default: one stride)")
    p.add_argument("--mask-thresh", type=float, default=0.0,
                   help="blank voxels below this fraction of the max intensity "
                        "(0 = keep every voxel that has a direction)")
    p.add_argument("--vox", type=float, default=1.0,
                   help="isotropic voxel size for the NIfTI affine (mm)")
    p.add_argument("--out", default=os.path.join(_ROOT, "results"))
    a = p.parse_args()
    max_dist = float(a.stride) if a.max_dist is None else a.max_dist

    print("=" * 78)
    print("KISSING PHANTOM -> STEP 1 (adaptive sigma/rho) -> STEP 2 (gate) -> NIfTI")
    print("=" * 78)
    vol, meta = build_kissing(same_size=True, gap=a.gap, curv=a.curv)
    print(f"  phantom {vol.shape} {vol.dtype}  radii {meta['rA']:.1f}/{meta['rB']:.1f}"
          f"  true min centreline gap {meta['true_min_gap']:.2f} vox")

    field = build_gated_field(vol, block=a.block, stride=a.stride)
    if field.n == 0:
        raise SystemExit("empty field -- nothing to export")

    vp, pp, np_, n_dirs = export(vol, field, a.out, "kissing", vox=a.vox,
                                 block=a.block, max_dist=max_dist,
                                 mask_thresh=a.mask_thresh)
    field.save(os.path.join(a.out, "kissing_gated_field.npz"))

    n_patch = np.array([q["n_dirs"] for q in field.patches])
    solid = vol > 0
    print("-" * 78)
    print(f"  patches: {int((n_patch == 1).sum())} single, "
          f"{int((n_patch >= 2).sum())} double  (stride {a.stride})")
    print(f"  voxels : {int((n_dirs >= 1).sum())}/{vol.size} carry a vector "
          f"({(n_dirs >= 1).mean():.1%}); "
          f"{(n_dirs[solid] >= 1).mean():.1%} of the {int(solid.sum())} "
          f"non-zero voxels")
    print(f"           {int((n_dirs == 2).sum())} carry a second, within "
          f"{max_dist:g} vox of a gated patch")
    print("-" * 78)
    for q in (vp, pp, np_):
        print(f"  wrote {q}  ({os.path.getsize(q) / 1024:.0f} KB)")
    print("=" * 78)


if __name__ == "__main__":
    main()
