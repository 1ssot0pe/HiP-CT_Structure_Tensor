"""
field_export.py -- turn the sparse GatedField into dense per-voxel maps and
write them as NIfTI.

Why this exists
---------------
`GatedField` is the object that carries state across Stages 1-3, but it is a
PATCH LIST, not a volume: a few hundred centres, each with 1-2 directions and a
HOST label. Nothing downstream could look at it in a viewer, and nothing could
diff "before optimisation" against "after". This module paints it back onto the
voxel grid and writes the result with a correct spatial affine.

What "per-voxel" honestly means here
------------------------------------
Two different resolutions are being combined, and conflating them would
overstate the result:

  * the PRIMARY direction is genuinely per-voxel -- it is Stage 1's dense e3
    field, one orientation at every voxel;
  * the SECOND direction and the LABEL are per-PATCH. HOST evaluates a patch of
    half-width GATE_HALF (=14, so a 29^3 neighbourhood) to resolve two peaks.
    There is no per-voxel second orientation to be had; assigning each voxel the
    value of its nearest patch centre is an upsampling of patch-resolution
    evidence, not a measurement at that voxel.

So `vec2` and `label` are nearest-patch fills, capped at `max_radius` so a
patch never claims voxels it did not see. Voxels beyond that stay background.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nifti_io import nifti_affine, save_nifti, save_vector_field  # noqa: F401,E402

EPS = 1e-12

# Label codes for the discrete map. 0 is reserved for "no evidence" so an
# unsigned map reads naturally in a viewer with 0 as background.
LABEL_CODES = {
    "background":   0,
    "single":       1,
    "crossing":     2,
    "kissing":      3,     # a crossing the optimiser demoted
    "dispersed":    4,
    "disorganised": 5,
}
CODE_LABELS = {v: k for k, v in LABEL_CODES.items()}


# ---------------------------------------------------------------------------
# sparse patches -> dense volumes
# ---------------------------------------------------------------------------

def _patch_assignment(shape, centres, max_radius):
    """Index of the nearest patch centre for every voxel, -1 beyond max_radius.

    A KD-tree query over the whole grid rather than painting a cube per patch:
    the patches OVERLAP (half=14 against a stride of 4-8), so cube painting
    makes the result depend on the order the patches happen to be stored in.
    Nearest-centre is order-independent.
    """
    from scipy.spatial import cKDTree

    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"),
                    axis=-1).reshape(-1, 3)
    d, i = cKDTree(np.asarray(centres, float)).query(grid, k=1)
    i = np.where(d <= max_radius, i, -1)
    return i.reshape(shape), d.reshape(shape)


def dense_fields(gated, e3, tissue=None, host_only=True, max_radius=14.0,
                 label_overrides=None):
    """Paint a GatedField onto the voxel grid.

    gated       : GatedField (patch centres in (z,y,x), dirs in (x,y,z))
    e3          : dense (3, Z, Y, X) primary orientation, components (z,y,x)
    tissue      : optional boolean mask; everything outside is background
    host_only   : if True, only patches the routing gate passed (`gated`) may
                  write a second direction or a label. This is the "strictly on
                  HOST-identified voxels" constraint -- without it, single
                  patches that never went through HOST would also paint.
    max_radius  : a patch may not claim voxels further than this from its
                  centre (default = GATE_HALF, the neighbourhood HOST saw)
    label_overrides : {patch_index: "kissing"} from the optimiser's demotions

    Returns a dict of (z,y,x)-indexed arrays.
    """
    shape = tuple(gated.shape)
    overrides = label_overrides or {}

    # --- primary direction: dense, straight from Stage 1 --------------------
    e3 = np.asarray(e3, float)
    if e3.shape[0] != 3:
        raise ValueError(f"e3 must be (3, Z, Y, X), got {e3.shape}")
    vec1 = np.moveaxis(e3, 0, -1).copy()               # (z, y, x, 3), (z,y,x) comps
    n = np.linalg.norm(vec1, axis=-1, keepdims=True)
    vec1 = np.where(n > EPS, vec1 / (n + EPS), 0.0)

    vec2 = np.zeros_like(vec1)
    labels = np.zeros(shape, np.uint8)
    n_dirs = np.zeros(shape, np.uint8)

    idx = [i for i, p in enumerate(gated.patches)
           if (p["gated"] or not host_only)]
    if not idx:
        return dict(vec1=vec1, vec2=vec2, labels=labels, n_dirs=n_dirs,
                    patch_index=np.full(shape, -1, np.int32))

    centres = [gated.patches[i]["centre"] for i in idx]
    assign, _dist = _patch_assignment(shape, centres, max_radius)

    for slot, pi in enumerate(idx):
        p = gated.patches[pi]
        m = assign == slot
        if not m.any():
            continue
        lab = overrides.get(pi, p["label"])
        labels[m] = LABEL_CODES.get(lab, 0)
        n_dirs[m] = int(p["n_dirs"])
        dirs = [np.asarray(d, float)[::-1] for d in p["dirs"]]   # ->(z,y,x)
        if dirs:
            d0 = dirs[0] / (np.linalg.norm(dirs[0]) + EPS)
            vec1[m] = d0
        if len(dirs) >= 2 and lab != "kissing":
            # A demoted patch carries ONE fibre by definition, so its second
            # direction must not survive into the optimised field -- that is
            # the whole point of the demotion.
            d1 = dirs[1] / (np.linalg.norm(dirs[1]) + EPS)
            vec2[m] = d1

    patch_index = np.full(shape, -1, np.int32)
    for slot, pi in enumerate(idx):
        patch_index[assign == slot] = pi

    if tissue is not None:
        out = ~np.asarray(tissue, bool)
        labels[out] = LABEL_CODES["background"]
        vec1[out] = 0.0
        vec2[out] = 0.0
        n_dirs[out] = 0
        patch_index[out] = -1

    return dict(vec1=vec1, vec2=vec2, labels=labels, n_dirs=n_dirs,
                patch_index=patch_index)


# ---------------------------------------------------------------------------

def export_field(gated, e3, outdir, tag, crop_origin=(0, 0, 0), voxel_um=20.0,
                 tissue=None, host_only=True, max_radius=14.0,
                 label_overrides=None, verbose=True):
    """Write vec1 / vec2 / labels for one field state. Returns the paths."""
    os.makedirs(outdir, exist_ok=True)
    A = nifti_affine(crop_origin, voxel_um)
    F = dense_fields(gated, e3, tissue=tissue, host_only=host_only,
                     max_radius=max_radius, label_overrides=label_overrides)

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
