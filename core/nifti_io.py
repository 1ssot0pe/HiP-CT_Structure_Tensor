"""
nifti_io.py -- writing (z, y, x)-indexed arrays out as NIfTI, correctly.

Why this is at root
-------------------
These three functions used to live in `3_global_tractography/field_export.py`,
the LAST stage of the pipeline. They are now needed by the first: Step 0 writes
the raw crop, Step 1 writes the (sigma, rho) maps and Step 2 writes the two e3
fields, all before Stage 3 exists in the run. Importing `3_*` from `1_*` to get
a file-format helper is backwards, so the primitive moved to where every stage
can reach it. `field_export` re-exports the same names, so nothing that already
imported them from there had to change.

The two traps, both silent
--------------------------
1. NO CROP OFFSET. The pipeline runs on a cube carved out of the 768^3 volume
   (`load_volume(crop=(a, b))` slices [a:b] on ALL THREE axes). With a pure
   diagonal affine the crop is placed at the world origin, so it will not
   overlay on the full volume or on anything else derived from it. The
   translation column must carry `crop_origin * voxel_size`.

2. AXIS ORDER. The volume is stacked (Nz, Ny, Nx), so array axis 0 is z. A
   diagonal affine maps array axis 0 to world X, which silently transposes the
   volume. For a scalar map that only mirrors the picture; for a VECTOR field it
   rotates every direction into the wrong axis and still looks plausible.
   Arrays are therefore transposed to (x, y, z) on the way out and vector
   COMPONENTS are reordered to match.

Component-order conventions in this repo (getting these backwards is the
documented failure mode -- see `3_global_tractography/test_frame_convention.py`):

    tensor.tensor_field() e3               (z, y, x)   <- one convention at source
    GatedField patch["dirs"]               (x, y, z)   <- from host_fod
    what NIfTI wants                       (x, y, z)

so a (z, y, x) vector field is reversed here and patch dirs are passed through
untouched.
"""

import numpy as np

__all__ = ["nifti_affine", "save_nifti"]


def nifti_affine(crop_origin=(0, 0, 0), voxel_um=20.0):
    """4x4 index->world affine, in mm, for an array written as (x, y, z).

    crop_origin : (z0, y0, x0) index of the crop's first voxel in the FULL
                  volume, in array order. `load_volume(crop=(a, b))` slices
                  [a:b] on every axis, so that origin is (a, a, a).
    voxel_um    : isotropic voxel edge in micrometres (i74: 20).

    The scale is diagonal because the voxels are isotropic; the work this
    function does that a bare `np.diag` does not is the translation column.
    """
    mm = float(voxel_um) / 1000.0
    z0, y0, x0 = (int(v) for v in crop_origin)
    A = np.diag([mm, mm, mm, 1.0])
    # The array is written (x, y, z), so the translation is ordered to match.
    A[:3, 3] = np.array([x0, y0, z0], float) * mm
    return A


def _to_world_array(arr_zyx, vector=False):
    """(z,y,x[,3]) in (z,y,x) components -> (x,y,z[,3]) in (x,y,z) components."""
    a = np.asarray(arr_zyx)
    if not vector:
        return np.ascontiguousarray(a.transpose(2, 1, 0))
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"vector array must be (z, y, x, 3), got {a.shape}")
    # transpose the grid, then reverse the component axis (z,y,x) -> (x,y,z)
    return np.ascontiguousarray(a.transpose(2, 1, 0, 3)[..., ::-1])


def save_nifti(arr_zyx, path, affine, vector=False, dtype=None):
    """Write one (z, y, x)-indexed array as NIfTI in (x, y, z) world order.

    Vector maps are written 4D as (X, Y, Z, 3) -- the same layout as FSL's
    `dti_V1`, which every viewer in common use reads. The stricter NIFTI_INTENT_
    VECTOR 5D form is more correct on paper and less portable in practice.
    """
    import nibabel as nib

    out = _to_world_array(arr_zyx, vector=vector)
    if dtype is not None:
        out = out.astype(dtype)
    img = nib.Nifti1Image(out, affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, path)
    return path


def save_vector_field(e3_zyx, path, affine, dtype=np.float32):
    """Write a (3, Z, Y, X) direction field -- the layout `tensor_field` returns.

    A convenience over `save_nifti(..., vector=True)`, which wants the component
    axis LAST. Doing that move at every call site is exactly how a component
    order gets reversed by accident.
    """
    a = np.moveaxis(np.asarray(e3_zyx, float), 0, -1)      # (Z, Y, X, 3)
    return save_nifti(a, path, affine, vector=True, dtype=dtype)
