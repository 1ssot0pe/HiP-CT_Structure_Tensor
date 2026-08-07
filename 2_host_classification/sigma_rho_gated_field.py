"""
sigma_rho_gated_field.py -- the Stage 1 -> Stage 2 bridge.

Until this file, the two stages were independent: `1_sigma_rho_selection`
certified a (sigma, rho) per region and produced a dense e3 field, and
`2_host_classification` ignored it -- `adaptive_triage` re-derived its own
per-patch sigma_grad from the raw image and never saw Stage 1's output at all.
This wires them: the orientation field comes from Stage 1's adaptive (sigma,
rho) selection, and c_s > FA decides which of its patches go to HOST-FOD.

Output: `GatedField`, holding per patch
    centre, n_dirs, dirs, gated (the routing mask), label, c_s, FA, sigma, rho

Single-fibre patches carry one e3 from Stage 1's tensor; gated patches carry
HOST's multi-orientation answer INSTEAD, which is the point of gating.

Three scales, all measured -- and none of them written here
-----------------------------------------------------------
The literal reading of "gate the patches from Stage 1" -- compute c_s and FA at
Stage 1's own certified (sigma, rho) -- gates NOTHING, including a true
crossing. At Stage 1's sigma (~0.85-1.0 vox) the crossing and kissing c_s differ
by 0.9-1.2x: no separation at all. The reason is geometric. A single fibre's
gradients already lie in a plane, so lambda3 is small; a crossing only raises
lambda3 once the differentiation scale is coarse enough that gradients from BOTH
fibre families land in the same neighbourhood, and Stage 1's scale is by
construction the one at which a single fibre is best resolved.

So three scales are in play, each doing a different job:

    Stage 1 (sigma, rho)   ->  the single-fibre direction e3    select_scales.py
    gate sigma_grad        ->  c_s vs FA, the routing decision  certify_gate_
    direction sigma_grad   ->  extracting the peak directions   scales.py

All three are now MEASURED rather than written down. They used to be module
constants (GATE_HALF=14, GATE_SIGMA_GRAD=4.0, DIR_SIGMA_GRAD=2.5) justified by a
sweep table in this docstring -- the values were right and the evidence was real,
but it lived in a comment, so nothing re-derived it and nothing checked it still
held. `1_sigma_rho_selection/certify_gate_scales.py` runs that sweep against
known truth and writes `gate_scales.json`, which is read below.

Re-deriving them changed one of the three. The certification adds a SINGLE-FIBRE
control the old table lacked -- without it "maximise the crossing's c_s - FA" is
monotone in sigma_grad and just walks to the largest scale swept -- and against
all three cases the gate scale comes out at 5.0, not 4.0. half=14 and
dir_sigma_grad=2.5 are reproduced exactly, and half=14 turns out to be the only
window that both fires the gate and lets HOST decompose the crossing.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection")):
    sys.path.insert(0, _p)

from host_fod import analyse, classify
from tensor import tensor_field
from reconstruction import AIR_FRAC
from select_scales import select_scales, scale_at, distinct_scales
from certify_gate_scales import load_gate_scales

EPS = 1e-12
DEFAULT_STRIDE = 6
DEFAULT_BLOCK = 32

# The three buckets a routed patch can land in. This is the triage the pipeline
# always described and never performed: every routed patch used to be truncated
# to `dirs[:2]` regardless of what HOST said about it, so a patch the method had
# explicitly given up on arrived at Stage 3 carrying two confident directions and
# was scored, and possibly demoted, as though it were a crossing candidate.
#
# Measured on the i74 run this replaces: of 67 junctions handed to Stage 3, only
# 34 were patches HOST labelled "crossing". 28 were "disorganised" -- an ODF flat
# enough that no direction in it means much -- and 5 were "complex", i.e. three
# or more peaks, beyond this method's validated two-fibre ceiling.
BUCKET_SINGLE = "single"            # one trustworthy direction
BUCKET_CANDIDATE = "candidate"      # exactly two clean peaks -> kissing vs crossing
BUCKET_UNRESOLVABLE = "unresolvable"  # >=3 peaks, or too isotropic to call


def triage(m, label):
    """Which bucket a routed patch belongs in, from HOST's own verdict.

    The rule is by PEAK COUNT, with `classify`'s quality tests deciding whether
    a 2-peak patch is clean enough to be a candidate:

        >= 3 peaks          -> unresolvable  (past the validated ceiling)
        isotropic ODF       -> unresolvable  (`classify` says "disorganised")
        2 clean peaks       -> candidate     (`classify` says "crossing")
        2 peaks, not clean  -> single        (amp2 or separation failed, so the
                                              second peak is not trustworthy --
                                              keeping it would launder a
                                              rejected peak into a junction)
        1 peak              -> single
    """
    if m is None:
        return BUCKET_UNRESOLVABLE
    n = m.get("n_peaks")
    if n is not None and n >= 3:
        return BUCKET_UNRESOLVABLE
    if label == "disorganised":
        return BUCKET_UNRESOLVABLE
    if label == "crossing":
        return BUCKET_CANDIDATE
    return BUCKET_SINGLE


class GatedField:
    """Per-patch orientation field plus the boolean routing mask."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.patches = []

    def add(self, **kw):
        self.patches.append(kw)

    @property
    def n(self):
        return len(self.patches)

    def routing_mask(self):
        """The Stage-1 deliverable: True where the patch was routed to HOST."""
        return np.array([p["gated"] for p in self.patches], dtype=bool)

    def multi(self):
        return [p for p in self.patches if p["n_dirs"] >= 2]

    def save(self, path):
        """Serialise the unified orientation field -- the Stage 2 deliverable.

        Arrays rather than pickled objects so the artifact is readable without
        importing this module:
            centres  (N, 3) int      patch centres, (z, y, x)
            dirs     (N, K, 3) float directions, NaN-padded to K = max peaks
            n_dirs   (N,) int        how many are real for that patch
            routed   (N,) bool       THE ROUTING MASK (c_s > FA)
            c_s, FA  (N,) float      the two numbers the gate compared
            sigma,rho(N,) float      the scale Stage 1 certified for that patch
            labels   (N,) str        HOST's class where it ran
        """
        import numpy as _np
        n = self.n
        k = max([p["n_dirs"] for p in self.patches], default=1)
        dirs = _np.full((n, k, 3), _np.nan)
        for i, p in enumerate(self.patches):
            for j, d in enumerate(p["dirs"][:k]):
                dirs[i, j] = d
        _np.savez_compressed(
            path,
            centres=_np.array([p["centre"] for p in self.patches], dtype=int),
            dirs=dirs,
            n_dirs=_np.array([p["n_dirs"] for p in self.patches], dtype=int),
            routed=self.routing_mask(),
            c_s=_np.array([p["c_s"] for p in self.patches]),
            FA=_np.array([p["FA"] for p in self.patches]),
            sigma=_np.array([p["sigma"] for p in self.patches]),
            rho=_np.array([p["rho"] for p in self.patches]),
            labels=_np.array([p["label"] for p in self.patches]),
            buckets=_np.array([p.get("bucket", "") for p in self.patches]),
            n_peaks=_np.array([int(p.get("n_peaks", p["n_dirs"]))
                               for p in self.patches], dtype=int),
            shape=_np.array(self.shape, dtype=int))
        return path

    @classmethod
    def load(cls, path):
        """Rebuild a GatedField from what `save` wrote.

        The counterpart was missing, so `unified_field_*.npz` was written every
        run and never read: Stage 3 could only ever run behind a fresh Stage
        1+2. Reading it back is what lets the optimisation loop iterate on a
        cached field in seconds instead of re-deriving the scale map each time.

        `dirs` is NaN-padded to the widest patch on the way out, so the padding
        rows are dropped here using `n_dirs` rather than by testing for NaN --
        a real direction is never NaN, but relying on that would silently keep
        a partly-written row.
        """
        import numpy as _np
        z = _np.load(path, allow_pickle=False)
        f = cls(tuple(int(v) for v in z["shape"]))
        for i in range(len(z["centres"])):
            k = int(z["n_dirs"][i])
            f.add(centre=tuple(int(v) for v in z["centres"][i]),
                  dirs=[_np.asarray(d, float) for d in z["dirs"][i][:k]],
                  n_dirs=k,
                  gated=bool(z["routed"][i]),
                  c_s=float(z["c_s"][i]),
                  FA=float(z["FA"][i]),
                  sigma=float(z["sigma"][i]),
                  rho=float(z["rho"][i]),
                  label=str(z["labels"][i]),
                  # Older archives predate the triage, so a missing bucket is
                  # inferred from the direction count rather than defaulted to
                  # a value that would silently promote patches to candidates.
                  bucket=(str(z["buckets"][i]) if "buckets" in z.files
                          else (BUCKET_CANDIDATE if k >= 2 else BUCKET_SINGLE)),
                  n_peaks=int(z["n_peaks"][i]) if "n_peaks" in z.files else k)
        return f

    def copy(self):
        """Deep copy -- the snapshot the optimiser compares against.

        `patches` holds mutable dicts and `dirs` holds arrays, so a shallow
        copy would let the optimiser's demotions write through into the
        "initial" field and make the before/after export identical.
        """
        import copy as _copy
        out = GatedField(self.shape)
        out.patches = _copy.deepcopy(self.patches)
        return out

    def summary(self):
        if not self.patches:
            return "empty field"
        g = int(self.routing_mask().sum())
        x = len(self.multi())
        labs = {}
        for p in self.patches:
            labs[p["label"]] = labs.get(p["label"], 0) + 1
        parts = ", ".join(f"{k} {v}" for k, v in sorted(labs.items()))
        bk = {}
        for p in self.patches:
            if p["gated"]:
                bk[p.get("bucket", "?")] = bk.get(p.get("bucket", "?"), 0) + 1
        bparts = ", ".join(f"{k} {v}" for k, v in sorted(bk.items()))
        return (f"{self.n} patches | routed to HOST (c_s>FA): {g} ({g/self.n:.0%}) "
                f"| candidates: {x} | routed triage: {bparts} | {parts}")


def build_gated_field(vol, scales=None, mode="adaptive", block=DEFAULT_BLOCK,
                      half=None, stride=DEFAULT_STRIDE, gate_sigma_grad=None,
                      dir_sigma_grad=None, order=None, e3_xyz=None,
                      verbose=True):
    """The selected (sigma, rho), then the c_s > FA routing gate.

    scales : the dict from `select_scales`. Selected here if not supplied, in
             `mode` -- so a caller that has already certified the scales (the
             pipeline) passes them in and they are certified ONCE per run, while
             a caller that just wants a field (a test, a one-off) still works.
    e3_xyz : the dense orientation field in (x, y, z) components, if the caller
             already built it. Built here otherwise. Passing it in is what makes
             the pipeline compute ONE decomposition and share it between the
             gate, the density target and Stage 3's streamlines, instead of
             three measurements of the same volume.
    half, gate_sigma_grad, dir_sigma_grad, order
           : default to the CERTIFIED values from `gate_scales.json`, not to
             literals. Still overridable, for sweeps that deliberately vary them.
    """
    vol = np.asarray(vol, dtype=np.float32)
    field = GatedField(vol.shape)

    gs = load_gate_scales()
    half = int(gs["gate_half"]) if half is None else int(half)
    gate_sigma_grad = (float(gs["gate_sigma_grad"]) if gate_sigma_grad is None
                       else float(gate_sigma_grad))
    dir_sigma_grad = (float(gs["dir_sigma_grad"]) if dir_sigma_grad is None
                      else float(dir_sigma_grad))
    order = int(gs["host_order"]) if order is None else int(order)

    if scales is None:
        if verbose:
            print(f"  [stage 1] selecting scales ({mode}) ...")
        scales = select_scales(vol, mode=mode, block=block, verbose=verbose)
    elif verbose:
        print(f"  [stage 1] using scales selected upstream: "
              f"sigma={scales['sigma_global']:.3g}, rho={scales['rho_global']:.3g}")
    if verbose:
        print(f"  [stage 2] gate sigma_grad={gate_sigma_grad:g}, "
              f"dir sigma_grad={dir_sigma_grad:g}, half={half}, order={order} "
              f"(certified, not hardcoded)")

    # Grid ANCHORED ON THE VOLUME CENTRE, not on `half`. A grid that starts at
    # `half` and steps by `stride` silently misses the middle of the volume
    # unless (centre - half) happens to be divisible by stride -- which is
    # exactly where a phantom puts its junction, so the run reports "no
    # junction" while the gate is working perfectly.
    def _axis(n):
        c0 = n // 2
        ks = range(-((c0 - half) // stride), ((n - half - 1 - c0) // stride) + 1)
        return [c0 + k * stride for k in ks if half <= c0 + k * stride < n - half]
    centres = [(z, y, x) for z in _axis(vol.shape[0])
               for y in _axis(vol.shape[1]) for x in _axis(vol.shape[2])]
    if verbose:
        print(f"  [stage 1] {len(centres)} patch centres over "
              f"{len(distinct_scales(scales, centres))} distinct (sigma, rho)")

    # ONE dense field for the whole volume, built at the selected scale(s) --
    # per block in adaptive mode, in a single call in global mode. This used to
    # be a loop computing a whole-volume structure tensor per distinct scale and
    # reading only the centres assigned to it, which duplicated
    # `tensor_field_adaptive` with a subtly different treatment of block edges
    # (whole-volume support versus per-block context margins). One
    # implementation, one answer.
    if e3_xyz is None:
        from tensor import field_for
        e3_xyz = field_for(vol, scales, invariants=False)["e3"][::-1]
    # tensor_field returns e3 in (z, y, x) components; GatedField stores
    # directions in (x, y, z) to match host_fod, and Stage 3 reverses them back
    # on the way in. Keeping that contract is what test_frame_convention.py
    # checks, and the [::-1] above is where it is honoured.

    air = AIR_FRAC * float(vol.max())
    for c in centres:
        sigma, rho = scale_at(scales, c)
        patch = vol[c[0] - half:c[0] + half + 1,
                    c[1] - half:c[1] + half + 1,
                    c[2] - half:c[2] + half + 1]
        # MAX, not mean: a patch holding one thin bundle in an otherwise
        # empty crop has a low mean and was being discarded as air, which
        # emptied the field entirely on the kissing phantom.
        if patch.max() < air:
            continue
        # --- the routing gate, at ITS OWN scale --------------------------
        # peaks=False: the routing decision needs only c_s and FA, which
        # come from a 3x3 eigendecomposition of the gradient scatter. The
        # sphere evaluation is 99% of analyse()'s cost (600 ms of 604) and
        # is pure waste for a patch that is about to be routed anyway --
        # the peak-resolving call at the direction scale supplies its
        # directions.
        gm = analyse(patch, sigma_grad=gate_sigma_grad, order=order,
                     peaks=False)
        if gm is None:
            continue
        c_s, fa = float(gm["c_s"]), float(gm["FA"])
        # The n_peaks guard has to be HERE too, for the same reason it is in
        # host_fod.classify and adaptive_triage._is_single: this fast path
        # returns "single" WITHOUT calling classify(), so a >=3-peak patch
        # whose c_s <= FA would bypass classify's "complex" branch and hand
        # Stage 3 one confident direction for a patch the method has
        # actually given up on. Sending it down the HOST branch instead
        # lets classify() label it, which is what every other path does.
        # Measured on i74 (48^3 at the volume centre, half=14, stride=6):
        # 0 of 34 gate-passing patches had n_peaks >= 3, so this is a
        # latent inconsistency rather than an active fault -- consistent
        # with the ~0.5% rate recorded in this folder's README.
        # Only a patch that would be called "single" needs the n_peaks >= 3
        # guard, so the sphere is evaluated lazily, for those patches only.
        # Routed patches never consult it -- identical decisions, less work.
        if c_s <= fa:
            gp = analyse(patch, sigma_grad=gate_sigma_grad, order=order)
            n_pk = 99 if gp is None else int(gp["n_peaks"])
        else:
            n_pk = 99
        if c_s <= fa and n_pk < 3:
            e3 = e3_xyz[:, c[0], c[1], c[2]]
            if not np.all(np.isfinite(e3)):
                # Skielex returns a non-finite eigenvector at isotropic
                # voxels. Dropping the patch here emptied the field
                # entirely on the sparse kissing phantom -- the patch
                # clearly HAS structure (it passed the air test), the
                # tensor is just degenerate at that one voxel at Stage 1's
                # sigma. Fall back to HOST's own direction rather than
                # discarding the patch.
                fb = analyse(patch, sigma_grad=dir_sigma_grad, order=order)
                if fb is None or not len(fb["dirs"]):
                    continue
                e3 = np.asarray(fb["dirs"][0], float)
            field.add(centre=c, n_dirs=1, dirs=[np.asarray(e3, float)],
                      gated=False, label="single", bucket=BUCKET_SINGLE,
                      n_peaks=1, c_s=c_s, FA=fa, sigma=sigma, rho=rho)
            continue
        # --- routed to HOST: directions at the PEAK-RESOLVING scale ------
        dm = analyse(patch, sigma_grad=dir_sigma_grad, order=order)
        if dm is None or not len(dm["dirs"]):
            continue
        lab = classify(dm)
        bucket = triage(dm, lab)
        # Only a CANDIDATE keeps two directions. An unresolvable patch keeps its
        # strongest peak so the vector field still has an orientation there and
        # the label says how much to trust it -- but it makes no two-fibre
        # claim, so Stage 3 never scores it as a junction.
        k = 2 if bucket == BUCKET_CANDIDATE else 1
        dirs = [np.asarray(d, float) for d in dm["dirs"][:k]]
        field.add(centre=c, n_dirs=len(dirs), dirs=dirs, gated=True,
                  label=lab, bucket=bucket, n_peaks=int(dm["n_peaks"]),
                  c_s=c_s, FA=fa, sigma=sigma, rho=rho)

    if verbose:
        print(f"  [stage 2] {field.summary()}")
    return field


def run(block=32, stride=8):
    from localized_phantoms import (raw_localized_crossing, CROSS_SHAPE,
                                    CROSS_R1, CROSS_TAPER)
    print("=" * 78)
    print("GATED ORIENTATION FIELD -- Stage 1 adaptive sigma/rho, gated by c_s > FA")
    print("=" * 78)
    vol = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5, CROSS_R1, CROSS_TAPER)
    print(f"  localized-crossing phantom {vol.shape}")
    f = build_gated_field(vol, block=block, stride=stride)
    print("-" * 78)
    print(f"  routing mask: {int(f.routing_mask().sum())}/{f.n} patches to HOST")
    print("=" * 78)
    return f


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1 -> Stage 2 gated field")
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--stride", type=int, default=8)
    a = p.parse_args()
    run(block=a.block, stride=a.stride)
