"""
field_export.py -- write the dense per-voxel field to NIfTI.

Why this exists
---------------
`GatedField` is a PATCH LIST, not a volume: a few hundred centres, each with 1-2
directions and a HOST label. Nothing downstream could look at it in a viewer,
and nothing could diff "before optimisation" against "after". This module writes
the painted field to disk with a correct spatial affine.

Where the painting now happens
------------------------------
`dense_fields` used to build the dense maps here, as an export-only convenience.
It no longer does: the dense field became the actual Stage 2 -> Stage 3 carrier,
so the construction moved to `dense_field.DenseField` and this module is a thin
layout conversion plus file I/O. `LABEL_CODES` / `CODE_LABELS` moved with it and
are re-exported below, so callers that import them from here still work.

Two things changed in the written maps as a result, both because the carrier
must be trackable rather than merely viewable:

  * `vec1` is filled at isotropic voxels, where the tensor solver returns a
    non-finite eigenvector, instead of being left at zero -- a tracker cannot
    step through NaN and a hole in the middle of measured tissue is not a
    faithful picture of it either;
  * `n_dirs` counts directions that EXIST, so oriented tissue outside every
    routed patch reads 1 rather than 0. The demotion test below (`n_dirs
    2 -> 1`) is unaffected: those voxels read the same before and after.

The "per-voxel" caveat is unchanged and lives in `dense_field`: `vec2` and
`labels` are patch-resolution evidence upsampled by nearest-centre fill, not
measurements at the voxel.

The affine, the axis-order trap, and where that code now lives
--------------------------------------------------------------
`nifti_affine` / `save_nifti` used to be defined here. They moved to root
`nifti_io.py` because Steps 0-2 (raw crop, scale maps, e3 fields) all need to
write NIfTI before Stage 3 exists in the run, and importing this module from
`1_sigma_rho_selection` to get a file-format helper is backwards. They are
re-exported below, so callers that import them from here still work. The two
silent traps they exist to avoid -- missing crop offset, and a diagonal affine
transposing (z,y,x) into world (x,y,z) -- are documented in that module.

Component-order conventions in this repo (getting these backwards is the
documented failure mode -- see `test_frame_convention.py`):

    GatedField patch["dirs"]              (x, y, z)   <- from host_fod
    tensor.tensor_field() e3              (z, y, x)   <- one convention at source
    what NIfTI wants                      (x, y, z)

so e3 is reversed on the way out and patch dirs are passed through untouched.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    sys.path.insert(0, _p)
from nifti_io import nifti_affine, save_nifti, save_vector_field  # noqa: F401,E402

EPS = 1e-12

# Re-exported from the carrier; see the module docstring.
from dense_field import (DenseField, LABEL_CODES, CODE_LABELS,  # noqa: F401,E402
                         _patch_assignment)


# ---------------------------------------------------------------------------
# sparse patches -> dense volumes
# ---------------------------------------------------------------------------

def dense_fields(gated, e3, tissue=None, host_only=True, max_radius=14.0,
                 label_overrides=None):
    """Paint a GatedField onto the voxel grid, in the layout NIfTI wants.

    A thin wrapper on `DenseField.build` -- see that for what the arguments
    mean. Kept because the exporters, the before/after diff and the figures all
    want `(z, y, x, 3)` arrays in a dict, while the tracker wants
    `(3, Z, Y, X)`; converting once here beats every caller doing it.
    """
    return DenseField.build(gated, e3, tissue=tissue, host_only=host_only,
                            max_radius=max_radius,
                            label_overrides=label_overrides).as_export_dict()


# ---------------------------------------------------------------------------

def export_field(gated, e3, outdir, tag, crop_origin=(0, 0, 0), voxel_um=20.0,
                 tissue=None, host_only=True, max_radius=14.0,
                 label_overrides=None, dense=None, verbose=True):
    """Write vec1 / vec2 / labels for one field state. Returns the paths.

    `dense` : an already-built DenseField to write instead of building one.
    Stage 3 now tracks through that object, so handing it in is what guarantees
    the exported map IS the field the tracts were integrated through, rather
    than a second construction that agrees with it only as long as nobody
    changes an argument at one call site and not the other.
    """
    os.makedirs(outdir, exist_ok=True)
    A = nifti_affine(crop_origin, voxel_um)
    F = (dense.as_export_dict() if dense is not None else
         dense_fields(gated, e3, tissue=tissue, host_only=host_only,
                      max_radius=max_radius, label_overrides=label_overrides))

    paths = {}
    paths["vec1"] = save_nifti(F["vec1"], os.path.join(outdir, f"vec1_{tag}.nii.gz"),
                               A, vector=True, dtype=np.float32)
    paths["vec2"] = save_nifti(F["vec2"], os.path.join(outdir, f"vec2_{tag}.nii.gz"),
                               A, vector=True, dtype=np.float32)
    paths["labels"] = save_nifti(F["labels"], os.path.join(outdir, f"labels_{tag}.nii.gz"),
                                 A, dtype=np.uint8)
    paths["n_dirs"] = save_nifti(F["n_dirs"], os.path.join(outdir, f"ndirs_{tag}.nii.gz"),
                                 A, dtype=np.uint8)
    if verbose:
        counts = {CODE_LABELS[c]: int((F["labels"] == c).sum())
                  for c in np.unique(F["labels"])}
        print(f"  [export {tag}] {counts}")
        for k, v in paths.items():
            print(f"      {k}: {v}")
    return paths, F


def export_difference(F_before, F_after, outdir, tag, crop_origin=(0, 0, 0),
                      voxel_um=20.0, verbose=True):
    """Write the map of what the optimiser changed.

    0 = unchanged, 1 = demoted, 2 = any other relabel.

    A demotion is detected as `n_dirs 2 -> 1`, NOT as the label going
    "crossing" -> "kissing". On real i74 HOST never emits the literal label
    "crossing": the multi-orientation patches come back "disorganised" or
    "dispersed", so a label-based test found zero demotions on exactly the data
    that has them, while the 13407 genuinely demoted voxels were miscounted as
    incidental relabels. The physical claim being retracted is "two fibres pass
    through here", and that claim lives in n_dirs.

    Kept separate from the two state maps because "which voxels moved" is the
    actual result; diffing two label maps by eye does not scale.
    """
    b, a = F_before["labels"], F_after["labels"]
    nb, na = F_before["n_dirs"], F_after["n_dirs"]
    diff = np.zeros_like(b)
    demoted = (nb >= 2) & (na < nb)
    other = ((b != a) | (nb != na)) & ~demoted
    diff[demoted] = 1
    diff[other] = 2
    p = save_nifti(diff, os.path.join(outdir, f"demoted_{tag}.nii.gz"),
                   nifti_affine(crop_origin, voxel_um), dtype=np.uint8)
    if verbose:
        print(f"  [export {tag}] demoted {int(demoted.sum())} voxels, "
              f"{int(other.sum())} other relabels -> {p}")
    return p
