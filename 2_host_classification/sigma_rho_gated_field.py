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

Three scales, all measured
--------------------------
The literal reading of "gate the patches from Stage 1" -- compute c_s and FA
from the structure tensor at Stage 1's own certified (sigma, rho) -- gates
NOTHING, including a true crossing:

    patch half | sigma_grad | crossing c_s vs FA | fires | kissing | fires
    -----------|------------|--------------------|-------|---------|------
        14     |    1.0     |   0.216 vs 0.664   |  no   |  0.244  |  no
         8     |    2.0     |   0.098 vs 0.719   |  no   |  0.070  |  no
        10     |    4.0     |   0.342 vs 0.723   |  no   |  0.037  |  no
        14     |    4.0     |   0.486 vs 0.454   | YES   |  0.148  |  no

At Stage 1's sigma (~0.85-1.0 vox) the crossing/kissing c_s ratio is 0.9-1.2x:
no separation at all. The reason is geometric -- a single fibre's gradients
already lie in a plane, so lambda3 is small; a crossing only raises lambda3
once the integration window is wide enough to hold BOTH fibre families, and
Stage 1's rho ~ 1 voxel is far too narrow to mix them.

A second measurement splits the scales again. HOST's second peak only exists in
a window that does NOT include the gate scale:

    sigma_grad :  1.0   1.5   2.0   2.5   3.0   4.0
    n_peaks    :   1     1     2     2     2     1     (true crossing)

So three different scales are in play, each doing its own job:

    Stage 1 sigma ~0.85-1.0  ->  the single-fibre direction e3
    GATE_SIGMA_GRAD = 4.0    ->  c_s vs FA, the routing decision
    DIR_SIGMA_GRAD  = 2.5    ->  extracting the peak directions

Using one scale for all three does not merely degrade the losers: at 1.0 the
gate never fires, and at 4.0 the peaks that justified gating are already
smoothed away.
"""

import argparse
import os
import sys

import numpy as np
from structure_tensor import structure_tensor_3d, eig_special_3d

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection")):
    sys.path.insert(0, _p)

from host_fod import analyse, classify
from sigma_rho_adaptive import compute_adaptive_map, SIGMA_FLOOR

EPS = 1e-12
GATE_HALF = 14           # = mass_conservation_validate.CROSS_HALF
GATE_SIGMA_GRAD = 4.0    # the only swept scale where c_s overtakes FA
DIR_SIGMA_GRAD = 2.5     # where HOST's second peak actually exists
# The brief specifies a FOURTH-order ODF. Measured on the validated crossing
# phantom at half=14, sigma_grad=2.5, order 4 returns n_peaks=1 and order 6
# returns n_peaks=2 -- the fourth-order expansion cannot resolve a 40 deg
# crossing here, and the whole repo's HOST validation is order 6. Using the
# order that actually resolves the structure; order=4 is still settable.
HOST_ORDER = 6
DEFAULT_STRIDE = 6


def tensor_fields(vol, sigma, rho, truncate=4.0):
    """Structure tensor eigenvalues (descending) and e3 at one (sigma, rho)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        S = structure_tensor_3d(vol.astype(np.float32), float(sigma), float(rho),
                                truncate=truncate)
        val, vec = eig_special_3d(S)
    del S
    return val, vec


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
            shape=_np.array(self.shape, dtype=int))
        return path

    def summary(self):
        if not self.patches:
            return "empty field"
        g = int(self.routing_mask().sum())
        x = len(self.multi())
        labs = {}
        for p in self.patches:
            labs[p["label"]] = labs.get(p["label"], 0) + 1
        parts = ", ".join(f"{k} {v}" for k, v in sorted(labs.items()))
        return (f"{self.n} patches | routed to HOST (c_s>FA): {g} ({g/self.n:.0%}) "
                f"| multi-orientation: {x} | {parts}")


def build_gated_field(vol, block=32, half=GATE_HALF, stride=DEFAULT_STRIDE,
                      gate_sigma_grad=GATE_SIGMA_GRAD,
                      dir_sigma_grad=DIR_SIGMA_GRAD, order=HOST_ORDER,
                      sigma_fallback=SIGMA_FLOOR, verbose=True):
    """Stage 1's adaptive (sigma, rho), then the c_s > FA routing gate."""
    vol = np.asarray(vol, dtype=np.float32)
    field = GatedField(vol.shape)

    if verbose:
        print(f"  [stage 1] adaptive sigma/rho on {block}^3 blocks ...")
    m = compute_adaptive_map(vol, block=block, verbose=False)
    sig_map, rho_map = m["sigma_map"], m["rho_map"]
    zs, ys, xs = m["zs"], m["ys"], m["xs"]
    if verbose:
        fin = sig_map[np.isfinite(sig_map)]
        print(f"           {fin.size}/{sig_map.size} blocks certified"
              + (f", sigma {fin.min():.2f}-{fin.max():.2f}" if fin.size else ""))

    def block_scale(c):
        bi = int(np.clip(np.searchsorted(zs, c[0], "right") - 1, 0, len(zs) - 1))
        bj = int(np.clip(np.searchsorted(ys, c[1], "right") - 1, 0, len(ys) - 1))
        bk = int(np.clip(np.searchsorted(xs, c[2], "right") - 1, 0, len(xs) - 1))
        s, r = sig_map[bi, bj, bk], rho_map[bi, bj, bk]
        if not np.isfinite(s) or s < sigma_fallback:
            return float(sigma_fallback), float(sigma_fallback)
        return float(s), float(r if np.isfinite(r) else s)

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
    scales = {}
    for c in centres:
        scales.setdefault(block_scale(c), []).append(c)
    if verbose:
        print(f"  [stage 1] {len(centres)} patch centres over "
              f"{len(scales)} distinct (sigma, rho)")

    air = 0.05 * float(vol.max())
    for (sigma, rho), cs in scales.items():
        val, vec = tensor_fields(vol, sigma, rho)
        for c in cs:
            patch = vol[c[0] - half:c[0] + half + 1,
                        c[1] - half:c[1] + half + 1,
                        c[2] - half:c[2] + half + 1]
            # MAX, not mean: a patch holding one thin bundle in an otherwise
            # empty crop has a low mean and was being discarded as air, which
            # emptied the field entirely on the kissing phantom.
            if patch.max() < air:
                continue
            # --- the routing gate, at ITS OWN scale --------------------------
            gm = analyse(patch, sigma_grad=gate_sigma_grad, order=order)
            if gm is None:
                continue
            c_s, fa = float(gm["c_s"]), float(gm["FA"])
            if c_s <= fa:
                e3 = vec[:, c[0], c[1], c[2]]
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
                          gated=False, label="single", c_s=c_s, FA=fa,
                          sigma=sigma, rho=rho)
                continue
            # --- routed to HOST: directions at the PEAK-RESOLVING scale ------
            dm = analyse(patch, sigma_grad=dir_sigma_grad, order=order)
            if dm is None or not len(dm["dirs"]):
                continue
            lab = classify(dm)
            dirs = [np.asarray(d, float) for d in dm["dirs"][:2]]
            field.add(centre=c, n_dirs=len(dirs), dirs=dirs, gated=True,
                      label=lab, c_s=c_s, FA=fa, sigma=sigma, rho=rho)
        del val, vec

    if verbose:
        print(f"  [stage 2] {field.summary()}")
    return field


def run(block=32, stride=8):
    sys.path.insert(0, os.path.join(_ROOT, "mass_conservation"))
    from mass_conservation_validate import (raw_localized_crossing, CROSS_SHAPE,
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
