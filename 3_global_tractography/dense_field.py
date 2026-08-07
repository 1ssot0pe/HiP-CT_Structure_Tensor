"""
dense_field.py -- the Stage 2 -> Stage 3 carrier: one direction per VOXEL.

What changed and why
--------------------
Stage 3 used to consume `GatedField` directly, and `field_from_gated(...,
host_only=True)` dropped every patch the routing gate had NOT routed. On i74
that is 344 of 1331 patches: the 987 single-fibre patches carrying Stage 1's e3
never entered the tracking field at all. Streamlines then existed only inside
`assoc_radius` bubbles around HOST patches -- which is why `host_support` had to
exist, to stop the loss from scoring tracts on voxels they were forbidden to
enter.

This module inverts that. The field handed to Stage 3 is DENSE:

    vec1   Stage 1's e3 at EVERY voxel, overwritten by HOST's first peak
           inside the patches the gate routed
    vec2   HOST's second peak, zero everywhere else

so a tracker walks continuously through single-fibre tissue on Stage 1's own
measurement and only sees a second option where Stage 2 actually found one.
Nothing is invented in between and nothing is discarded.

What "per-voxel" honestly means (unchanged from field_export)
-------------------------------------------------------------
Two resolutions are being combined and conflating them would overstate the
result:

  * `vec1` is genuinely per-voxel WHERE IT IS STAGE 1's e3 -- one orientation
    measured at every voxel;
  * `vec2`, `labels`, and `vec1` inside a routed patch are per-PATCH. HOST
    evaluates a 29^3 neighbourhood (half=14) to resolve two peaks. There is no
    per-voxel second orientation to be had; giving each voxel the value of its
    nearest patch centre is an UPSAMPLING of patch-resolution evidence, not a
    measurement at that voxel.

Nearest-patch fill, capped at `max_radius`, so a patch never claims voxels it
did not see.

Component order
---------------
    GatedField patch["dirs"]     (x, y, z)     <- from host_fod
    tensor.tensor_field() e3     (z, y, x)     <- the contract at source
    vec1 / vec2 here             (z, y, x)     <- Stage 3 indexes positions this
                                                  way, so directions match
    what NIfTI wants             (x, y, z)     <- reversed on the way out only

Patch dirs are reversed on the way IN (the `[::-1]` below); e3 passes through.
`test_frame_convention.py` guards both.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "2_host_classification")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from sigma_rho_gated_field import BUCKET_CANDIDATE
except ImportError:                      # standalone use, no Stage 2 on the path
    BUCKET_CANDIDATE = "candidate"

EPS = 1e-12

# Label codes for the discrete map. 0 is reserved for "no HOST evidence" so an
# unsigned map reads naturally in a viewer with 0 as background. These live here
# rather than in field_export because the carrier is now upstream of the export;
# field_export re-exports them, so existing importers still work.
LABEL_CODES = {
    "background":   0,
    "single":       1,
    "crossing":     2,
    "kissing":      3,     # a crossing the optimiser demoted
    "dispersed":    4,
    "disorganised": 5,
    # `classify` can return "complex" (>=3 peaks) and it was missing here, so
    # `LABEL_CODES.get(lab, 0)` painted those voxels as BACKGROUND while vec1
    # and vec2 were still written at them -- a label map that disagreed with the
    # vector maps it ships beside. Measured on i74: 5 patches, silently.
    "complex":      6,
    "n/a":          7,
    # Oriented by Stage 1, never evaluated by HOST -- the voxels between patch
    # centres, or beyond `max_radius` of any of them. They are the bulk of the
    # dense field (25,796 of 35,033 oriented voxels on the crossing phantom) and
    # calling them "background" would repeat the fault the `complex` entry above
    # fixes: a label map claiming nothing is there while vec1 ships a measured
    # direction at the same voxel. They are NOT "single" -- that is a HOST
    # verdict and nothing tested these.
    "unclassified": 8,
}
CODE_LABELS = {v: k for k, v in LABEL_CODES.items()}


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


def _unit(v, axis=0):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(n > EPS, v / (n + EPS), 0.0)


def fill_invalid(e3):
    """Replace non-finite directors with their nearest valid neighbour.

    The structure-tensor solver returns a non-finite eigenvector at isotropic
    voxels. Zeroing them puts a hole in the middle of otherwise measured tissue
    and a tracker cannot step through one.

    This lives here now. It was `field_refine.fill_invalid`, and that module has
    been archived -- with three live importers still pointing at it, so the
    pipeline does not currently start. The dense carrier is the natural home:
    it is the thing that has to be free of holes, it is upstream of every
    consumer, and a function this central should not depend on a module marked
    dead.
    """
    ok = np.all(np.isfinite(e3), axis=0) & (np.linalg.norm(e3, axis=0) > 1e-6)
    if ok.all() or not ok.any():
        return np.asarray(e3, float)
    from scipy.ndimage import distance_transform_edt
    _, idx = distance_transform_edt(~ok, return_indices=True)
    return np.asarray(e3, float)[:, idx[0], idx[1], idx[2]]


class DenseField:
    """Per-voxel orientations: vec1 always, vec2 where Stage 2 found two.

    vec1, vec2  (3, Z, Y, X) float32, components (z, y, x), unit or zero
    n_dirs      (Z, Y, X) uint8 -- 0 background, 1 one direction, 2 two
    labels      (Z, Y, X) uint8 -- LABEL_CODES, 0 where HOST never looked
    patch_index (Z, Y, X) int32 -- which GatedField patch painted this voxel,
                                   -1 where none did

    `n_dirs` counts DIRECTIONS THAT EXIST, not patch metadata: a single-fibre
    voxel outside every routed patch has n_dirs 1, because it has Stage 1's e3.
    The old patch-derived map called that 0. The demotion test both exports use
    (`n_dirs 2 -> 1`) is unaffected -- those voxels read the same before and
    after -- and a viewer no longer shows measured tissue as having no
    direction.
    """

    def __init__(self, vec1, vec2, n_dirs, labels, patch_index, shape):
        self.vec1 = vec1
        self.vec2 = vec2
        self.n_dirs = n_dirs
        self.labels = labels
        self.patch_index = patch_index
        self.shape = tuple(shape)

    # -- construction -------------------------------------------------------

    @classmethod
    def build(cls, gated, e3, tissue=None, host_only=True, max_radius=14.0,
              label_overrides=None, fill=True):
        """Paint a GatedField onto the voxel grid over Stage 1's dense e3.

        gated       : GatedField (centres (z,y,x), dirs (x,y,z))
        e3          : dense (3, Z, Y, X) Stage-1 field, components (z,y,x)
        tissue      : optional boolean mask; outside it everything is background
        host_only   : only patches the gate routed may write vec2 or a label.
                      Note this no longer removes single-fibre tissue from the
                      field -- that tissue is carried by e3 -- it only decides
                      who may OVERWRITE e3.
        max_radius  : a patch may not claim voxels further than this from its
                      centre (default = GATE_HALF, the neighbourhood HOST saw)
        label_overrides : {patch_index: "kissing"} from the optimiser
        fill        : replace non-finite e3 (isotropic voxels, where the solver
                      returns NaN) with the nearest valid director. A tracker
                      cannot step through NaN, and zeroing instead puts a hole
                      in the middle of otherwise measured tissue.
        """
        shape = tuple(gated.shape)
        overrides = label_overrides or {}

        e3 = np.asarray(e3, float)
        if e3.shape[0] != 3:
            raise ValueError(f"e3 must be (3, Z, Y, X), got {e3.shape}")
        if tuple(e3.shape[1:]) != shape:
            raise ValueError(f"e3 grid {e3.shape[1:]} != field shape {shape}")
        v1 = fill_invalid(e3) if fill else e3
        v1 = _unit(np.where(np.isfinite(v1), v1, 0.0))          # (3, Z, Y, X)

        v1 = np.moveaxis(v1, 0, -1).reshape(-1, 3).astype(np.float32)
        v2 = np.zeros_like(v1)
        labels = np.zeros(v1.shape[0], np.uint8)
        pidx = np.full(v1.shape[0], -1, np.int32)

        idx = [i for i, p in enumerate(gated.patches)
               if (p["gated"] or not host_only)]
        if idx:
            centres = [gated.patches[i]["centre"] for i in idx]
            assign, _d = _patch_assignment(shape, centres, max_radius)
            flat = assign.ravel()
            live = flat >= 0

            # One fancy-indexed write per field instead of a boolean mask per
            # patch. The old loop was O(n_patches * n_voxels) -- 344 x 884,736
            # on i74, twice over -- which is affordable once per run but not
            # once per optimiser iteration.
            d0 = np.zeros((len(idx), 3), np.float32)
            d1 = np.zeros((len(idx), 3), np.float32)
            lab = np.zeros(len(idx), np.uint8)
            pin = np.zeros(len(idx), np.int32)
            for slot, pi in enumerate(idx):
                p = gated.patches[pi]
                name = overrides.get(pi, p["label"])
                lab[slot] = LABEL_CODES.get(name, 0)
                pin[slot] = pi
                dirs = [np.asarray(d, float)[::-1] for d in p["dirs"]]  # ->(z,y,x)
                if dirs:
                    d0[slot] = dirs[0] / (np.linalg.norm(dirs[0]) + EPS)
                # A second direction requires Stage 2's CANDIDATE bucket, not
                # merely two entries in `dirs`. A "disorganised" patch (an ODF
                # too flat for any direction in it to mean much) or a "complex"
                # one (>=3 peaks, past this method's validated two-fibre
                # ceiling) must not offer the tracker a second option -- that is
                # the same claim `field_from_gated` refuses to turn into a
                # junction, and the field the tracker walks has to agree with
                # the junction list or the two disagree about what is there.
                # A demoted patch carries ONE fibre by definition, so its second
                # direction must not survive into the optimised field either --
                # that is the whole point of the demotion.
                if (len(dirs) >= 2 and name != "kissing"
                        and p.get("bucket", BUCKET_CANDIDATE) == BUCKET_CANDIDATE):
                    d1[slot] = dirs[1] / (np.linalg.norm(dirs[1]) + EPS)

            sel = flat[live]
            # A patch whose HOST direction came back degenerate keeps e3 rather
            # than being blanked: d0 == 0 would punch a hole in the field.
            keep = np.linalg.norm(d0[sel], axis=1) > EPS
            rows = np.nonzero(live)[0]
            v1[rows[keep]] = d0[sel][keep]
            v2[rows] = d1[sel]
            labels[rows] = lab[sel]
            pidx[rows] = pin[sel]

        n_dirs = np.zeros(v1.shape[0], np.uint8)
        n_dirs[np.linalg.norm(v1, axis=1) > EPS] = 1
        n_dirs[np.linalg.norm(v2, axis=1) > EPS] = 2
        labels[(n_dirs >= 1) & (labels == 0)] = LABEL_CODES["unclassified"]

        if tissue is not None:
            out = ~np.asarray(tissue, bool).ravel()
            v1[out] = 0.0
            v2[out] = 0.0
            n_dirs[out] = 0
            labels[out] = LABEL_CODES["background"]
            pidx[out] = -1

        def _grid(a):
            return np.moveaxis(a.reshape(shape + (3,)), -1, 0).copy()

        return cls(_grid(v1), _grid(v2), n_dirs.reshape(shape),
                   labels.reshape(shape), pidx.reshape(shape), shape)

    @classmethod
    def from_e3(cls, e3, tissue=None, fill=True):
        """A one-direction field: Stage 1's e3 and nothing else.

        The control case. Every comparison between "with HOST" and "without"
        has to run the same tracker over the same code path, or the difference
        measured is the code path.
        """
        class _Empty:
            shape = tuple(np.asarray(e3).shape[1:])
            patches = []
        return cls.build(_Empty(), e3, tissue=tissue, fill=fill)

    # -- use ----------------------------------------------------------------

    def demote(self, patch_indices):
        """Retract the second direction at the given patches, in place.

        The physical claim being withdrawn is "two fibres pass through here",
        and that claim lives in vec2 -- so the demotion zeroes vec2 and drops
        n_dirs to 1. A tracker then simply has nothing to switch onto, which is
        how a TANGENCY verdict stops a tract crossing through without any
        special case in the tracker itself.
        """
        pi = np.asarray(sorted(set(int(i) for i in patch_indices)), np.int32)
        if not pi.size:
            return 0
        m = np.isin(self.patch_index, pi) & (self.n_dirs >= 2)
        if not m.any():
            return 0
        self.vec2[:, m] = 0.0
        self.n_dirs[m] = 1
        self.labels[m] = LABEL_CODES["kissing"]
        return int(m.sum())

    def copy(self):
        return DenseField(self.vec1.copy(), self.vec2.copy(),
                          self.n_dirs.copy(), self.labels.copy(),
                          self.patch_index.copy(), self.shape)

    @property
    def n_multi(self):
        return int((self.n_dirs >= 2).sum())

    @property
    def n_valid(self):
        return int((self.n_dirs >= 1).sum())

    def summary(self):
        tot = int(np.prod(self.shape))
        counts = {CODE_LABELS.get(int(c), str(c)): int((self.labels == c).sum())
                  for c in np.unique(self.labels)}
        parts = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        return (f"{self.n_valid}/{tot} voxels oriented ({self.n_valid/tot:.0%}), "
                f"{self.n_multi} with two directions | {parts}")

    def as_export_dict(self):
        """The (z, y, x, 3) layout `field_export` writes to NIfTI."""
        return dict(vec1=np.moveaxis(self.vec1, 0, -1),
                    vec2=np.moveaxis(self.vec2, 0, -1),
                    labels=self.labels, n_dirs=self.n_dirs,
                    patch_index=self.patch_index)

    def save(self, path):
        np.savez_compressed(path, vec1=self.vec1, vec2=self.vec2,
                            n_dirs=self.n_dirs, labels=self.labels,
                            patch_index=self.patch_index,
                            shape=np.array(self.shape, int))
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["vec1"], z["vec2"], z["n_dirs"], z["labels"],
                   z["patch_index"], tuple(int(v) for v in z["shape"]))
