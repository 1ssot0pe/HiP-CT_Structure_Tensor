"""
dense_gate.py -- the routing gate as a per-voxel field instead of a patch loop.

What the patch gate actually computes
-------------------------------------
`host_fod.analyse(patch, sigma_grad, peaks=False)` takes gradients at
`sigma_grad` over a (2*half+1)^3 patch, drops the voxels whose |grad| is below
10% of the patch maximum, and eigendecomposes G^T G. That IS a structure tensor:
the outer product of the gradient with itself, integrated over a neighbourhood.
The neighbourhood is a BOX of half-width `half`, and the integration is
unweighted.

So the dense equivalent is `tensor_field(vol, sigma_gate, rho_gate)`, whose
integration is a GAUSSIAN of width rho_gate. Two differences survive, and both
are real rather than cosmetic:

  1. BOX vs GAUSSIAN window. There is no rho that reproduces a box. Matching the
     support radius (4*rho ~ half) gives rho ~ 3.5; matching the second moment
     (h^2/3) gives rho ~ 8.1. They are different windows and they will not agree
     everywhere, so rho_gate is CALIBRATED against the patch decision below
     rather than derived from either identity.
  2. The mag_frac=0.1 CUTOFF has no dense counterpart. The patch version
     discards low-gradient voxels before the eigendecomposition; the structure
     tensor weights them by their (small) gradient energy instead of excluding
     them. Near a fibre both give the same answer because the kept voxels
     dominate; in near-flat tissue they need not.

Why bother
----------
Three reasons, in order of how much they matter:

  * the decision becomes PER VOXEL, which is what "the voxels that pass are
    handed to HOST" is supposed to mean. The patch version decides at patch
    centres on a stride-6 grid and nearest-fills the rest;
  * HOST can then be run only where the gate fired, instead of everywhere and
    discarded afterwards. On i74, 1331 patches were analysed to find 189;
  * the gradient work stops being repeated. At half=14 and stride 6 each voxel
    sits in roughly a hundred overlapping patches, and its gradient is
    recomputed for every one of them.

Nothing here routes anything on its own. `calibrate` measures how often the
dense decision matches the patch decision it would replace, and that number has
to be looked at before the swap is made.

RESULT: THE SWAP WAS NOT MADE. What the calibration found instead
------------------------------------------------------------------
No dense form reproduces the patch gate. On i74 (96^3 at the volume centre,
sigma_gate=5, half=14, against the 189 centres the patch gate routes):

    window                  agreement   fires   missed candidates
    gaussian rho=2..10       85.6-87.2%  0-1.8%   170-189 of 189
    box, global cutoff       86.2%       0.4%     184 of 189
    box, no cutoff          86.3%       0.6%     182 of 189

The agreement figures are meaningless -- only 14% of centres route, so a gate
that never fires already scores 86%. Every variant misses essentially every
candidate.

It is neither the window shape nor the cutoff. It is the PATCH BOUNDARY.
`patch_gradients` runs `gaussian_filter(..., mode="nearest")` on the ISOLATED
29^3 patch, and at sigma_grad=5 scipy's derivative kernel has radius
4*sigma = 20 > half = 14. The kernel is wider than the patch, so every gradient
inside it is computed substantially from replicated edge values. Holding the
box, the cutoff and sigma fixed and changing ONLY whether the gradients saw the
real surrounding tissue, on 120 sampled i74 centres:

                        patch-isolated      full-volume gradients
    mean c_s                0.291                  0.149
    mean FA                 0.770                  0.879
    fires (c_s > FA)        11 / 120               0 / 120

So the gate's firing is substantially an artifact of analysing patches in
isolation, and the dense version is not "wrong" -- it is the artifact-free
measurement, and it says the gate does not fire.

Worse, the phantom certification does not transfer. With full-volume gradients,
c_s vs sigma_grad moves in OPPOSITE directions on the two:

    sigma_grad          3.0     4.0     5.0     6.0
    i74 mean c_s       0.423   0.264   0.167   0.110    (falls)
    i74 fires /200       64       8       3       0
    phantom c_s - FA  -0.548  -0.489  +0.010  +0.707    (rises, crosses)

On the phantom, larger sigma finally mixes two fibre families 12 voxels apart
and lifts lambda3. On real tissue, larger sigma just smooths toward something
locally laminar and lambda3 falls. A scale certified on the phantom therefore
cannot be trusted on the tissue for this statistic, and sigma_grad ~ 3, where
i74 actually has candidates, is below where the phantom says a 40 deg crossing
is resolvable at all.

Until that is resolved this module is a diagnostic, not a component. The
pipeline still routes on the per-patch gate.

Run
---
    python dense_gate.py --i74 --field results/unified_field_i74.npz
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection")):
    sys.path.insert(0, _p)

from core.tensor import tensor_field                          # noqa: E402
from core.reconstruction import tissue_mask                   # noqa: E402
from certify_gate_scales import load_gate_scales         # noqa: E402

EPS = 1e-12

# Spanning both principled identifications of "a box of half-width 14":
# support-matched (4*rho = half -> 3.5) and second-moment-matched
# (rho = half/sqrt(3) -> 8.1), plus room either side of each.
GATE_RHO_GRID = (2.0, 3.0, 3.5, 5.0, 6.5, 8.0, 10.0)


def gate_fields(vol, sigma_gate=None, rho_gate=None):
    """Per-voxel (c_s, FA) at the gate scale, GAUSSIAN window.

    Kept because it is the obvious thing to try and it does not work -- see
    `box_gate_fields` and the measurement in this module's docstring footer.
    """
    gs = load_gate_scales()
    sigma_gate = float(gs["gate_sigma_grad"]) if sigma_gate is None else float(sigma_gate)
    if rho_gate is None:
        rho_gate = float(gs.get("gate_rho", gs["gate_half"] / 4.0))
    t = tensor_field(vol, sigma_gate, rho_gate, invariants=True)
    return t["c_s"], t["FA"]


def box_gate_fields(vol, sigma_gate=None, half=None, mag_frac=0.1,
                    cutoff="global"):
    """Per-voxel (c_s, FA) from a BOX-window gradient scatter -- the faithful
    dense form of what `host_fod.analyse(..., peaks=False)` computes per patch.

    The patch version sums grad (x) grad over an unweighted box of half-width
    `half`. A box sum at every voxel is exactly a uniform filter, which is
    separable and costs three passes per tensor component instead of one
    eigendecomposition per overlapping patch. c_s and FA are ratios of
    eigenvalues, so the 1/n^3 that `uniform_filter` divides by cancels and the
    mean can be used directly as the tensor.

    cutoff : how the |grad| > mag_frac * max rule is applied.
        "global"  threshold against the volume's own max gradient. Separable,
                  so it can be applied once before the box sums.
        "none"    keep every voxel.
      The patch version thresholds against each PATCH's max, which is not
      separable -- a voxel's inclusion then depends on which patch is asking.
      That difference is measured rather than assumed; see `calibrate`.
    """
    from scipy import ndimage

    gs = load_gate_scales()
    sigma_gate = float(gs["gate_sigma_grad"]) if sigma_gate is None else float(sigma_gate)
    half = int(gs["gate_half"]) if half is None else int(half)
    n = 2 * half + 1

    v = np.asarray(vol, np.float32)
    # Same derivative-of-Gaussian calls, and the same component order, as
    # two_orientation_test.patch_gradients -- computed ONCE for the volume
    # instead of once per patch.
    gz = ndimage.gaussian_filter(v, sigma_gate, order=(1, 0, 0), mode="nearest")
    gy = ndimage.gaussian_filter(v, sigma_gate, order=(0, 1, 0), mode="nearest")
    gx = ndimage.gaussian_filter(v, sigma_gate, order=(0, 0, 1), mode="nearest")
    G = [gx, gy, gz]

    if cutoff == "global":
        mag = np.sqrt(gx * gx + gy * gy + gz * gz)
        keep = (mag > float(mag_frac) * float(mag.max())).astype(np.float32)
        G = [g * keep for g in G]
    elif cutoff != "none":
        raise ValueError(f"cutoff must be 'global' or 'none', got {cutoff!r}")

    T = np.empty(v.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            b = ndimage.uniform_filter(G[i] * G[j], size=n, mode="nearest")
            T[..., i, j] = b
            if i != j:
                T[..., j, i] = b

    lam = np.linalg.eigvalsh(T)          # ASCENDING, matching _scatter_eigvals
    tot = lam.sum(-1)
    c_s = 3.0 * lam[..., 0] / (tot + EPS)
    fa = (np.sqrt(0.5) * np.sqrt((lam[..., 0] - lam[..., 1]) ** 2
                                 + (lam[..., 1] - lam[..., 2]) ** 2
                                 + (lam[..., 2] - lam[..., 0]) ** 2)
          / (np.sqrt((lam ** 2).sum(-1)) + EPS))
    return c_s, fa


def gate_mask(vol, sigma_gate=None, rho_gate=None, tissue=None, window="box",
              half=None, mag_frac=0.1, cutoff="global"):
    """The routing decision as a boolean volume: c_s > FA, inside tissue.

    window="box" is the default because it is the one that reproduces the patch
    gate; "gaussian" is kept only so the negative result stays reproducible.
    """
    if window == "box":
        c_s, FA = box_gate_fields(vol, sigma_gate, half=half,
                                  mag_frac=mag_frac, cutoff=cutoff)
    else:
        c_s, FA = gate_fields(vol, sigma_gate, rho_gate)
    m = np.asarray(c_s > FA)
    m &= np.isfinite(c_s) & np.isfinite(FA)
    if tissue is None:
        tissue = tissue_mask(vol)
    return m & np.asarray(tissue, bool)


def calibrate(vol, centres, patch_routed, sigma_gate=None,
              rhos=GATE_RHO_GRID, tissue=None, verbose=True):
    """How often does the dense decision match the patch decision?

    centres      : (N, 3) patch centres, (z, y, x)
    patch_routed : (N,) bool, what the patch gate decided at each

    Reported per rho_gate: agreement, and the two error types separately.
    They are not symmetric in cost. A dense gate that fires where the patch gate
    did not costs HOST calls on patches that will resolve to one direction; a
    dense gate that stays silent where the patch gate fired LOSES a candidate
    junction outright, and nothing downstream can recover it.
    """
    centres = np.asarray(centres, int)
    patch_routed = np.asarray(patch_routed, bool)
    if tissue is None:
        tissue = tissue_mask(vol)
    idx = (centres[:, 0], centres[:, 1], centres[:, 2])

    rows = []
    if verbose:
        print(f"  {'window':>18} {'agree':>7} {'dense%':>7} {'patch%':>7} "
              f"{'miss':>6} {'extra':>6}  (of {len(centres)} centres)")

    def _score(name, m):
        dense = m[idx]
        agree = float((dense == patch_routed).mean())
        miss = int((patch_routed & ~dense).sum())     # lost a candidate
        extra = int((~patch_routed & dense).sum())    # extra HOST work
        rows.append(dict(window=name, agreement=agree,
                         dense_frac=float(dense.mean()),
                         patch_frac=float(patch_routed.mean()),
                         missed=miss, extra=extra, mask_frac=float(m.mean())))
        if verbose:
            print(f"  {name:>18} {agree:7.1%} {dense.mean():7.1%} "
                  f"{patch_routed.mean():7.1%} {miss:6d} {extra:6d}")

    for r in rhos:
        _score(f"gaussian rho={r:g}",
               gate_mask(vol, sigma_gate=sigma_gate, rho_gate=r, tissue=tissue,
                         window="gaussian"))
    for co in ("global", "none"):
        _score(f"box cutoff={co}",
               gate_mask(vol, sigma_gate=sigma_gate, tissue=tissue,
                         window="box", cutoff=co))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="calibrate the dense routing gate")
    ap.add_argument("--i74", action="store_true")
    ap.add_argument("--centre", nargs=3, type=int, default=[384, 384, 384])
    ap.add_argument("--radius", type=int, default=48)
    ap.add_argument("--field", default=None,
                    help="unified_field_*.npz holding the patch decisions")
    a = ap.parse_args()

    from sigma_rho_gated_field import GatedField

    if a.i74:
        from core.reconstruction import load_volume
        lo = a.centre[0] - a.radius
        vol = load_volume(crop=(lo, a.centre[0] + a.radius),
                          dtype=np.uint8).astype(np.float32)
    else:
        from core.localized_phantoms import (raw_localized_crossing, CROSS_SHAPE,
                                        CROSS_R1, CROSS_TAPER)
        vol = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5,
                                     CROSS_R1, CROSS_TAPER)

    if a.field is None:
        raise SystemExit("pass --field <unified_field_*.npz> -- the patch "
                         "decisions are the reference this calibrates against")
    f = GatedField.load(a.field)
    gs = load_gate_scales()
    print("=" * 78)
    print(f"DENSE GATE CALIBRATION -- sigma_gate={gs['gate_sigma_grad']:g}, "
          f"volume {vol.shape}, {f.n} patch centres")
    print("=" * 78)
    rows = calibrate(vol, [p["centre"] for p in f.patches], f.routing_mask())
    # Ranked by how few CANDIDATES are lost, not by raw agreement: with only
    # 14% of centres routed, a gate that never fires already scores 86%.
    best = min(rows, key=lambda r: (r["missed"], r["extra"]))
    print("-" * 78)
    print(f"  fewest missed candidates: {best['window']} -- {best['missed']} "
          f"missed, {best['extra']} extra, agreement {best['agreement']:.1%}")
    print("=" * 78)
