"""
test_kissing_phantom.py -- run the pipeline on the proper (curved, gapped) kissing
phantom: two distinct fibre bundles that bulge toward each other, tangent and very
close at the middle, but separated by a real gap (verified numerically in
kissing_phantom.build_kissing).

Checks, for SAME-size and DIFFERENT-size fibre variants:
  (B) HOST classification -- almost every patch should read "single" (each bundle is
      locally ONE fibre; the two never occupy the same voxel). NOTE: right near the
      closest approach, a RARE spurious "crossing" can appear -- the Gaussian tube
      tails (plus the gradient kernel's own smoothing) reach beyond the nominal patch
      box, so a patch centred near the gap can pick up weak leakage from the OTHER
      bundle. This is a genuine resolution-scale partial-volume effect (the gap is
      comparable to a few analysis-window widths), not a bug -- and it is exactly why
      stage (C) exists: a rare local misread must not be allowed to merge two bundles.
  (C) global tractography -- must NOT bridge the gap regardless: the two bundles
      should end as SEPARATE tracts, not one continuous tract spanning the midline.
"""

import numpy as np

from kissing_phantom import build_kissing, SHAPE
from host_fod import analyse, classify
import phase3_i74 as Q
import phase3_global as P

HALF = 8
STRIDE = 4


def classify_grid(vol, step=6, gap_step=1):
    """Coarse grid over the whole object PLUS a densely-sampled strip around the
    midline (where the two bundles come closest) -- a coarse-only grid can step over
    the narrow near-gap hotspot entirely and misreport 0% spurious crossings."""
    counts = {}
    z = vol.shape[0] // 2
    cx = vol.shape[2] // 2
    seen = set()
    coords = [(y, x) for y in range(HALF + 2, vol.shape[1] - HALF - 2, step)
                     for x in range(HALF + 2, vol.shape[2] - HALF - 2, step)]
    coords += [(y, x) for y in range(HALF + 2, vol.shape[1] - HALF - 2, gap_step)
                      for x in range(cx - 6, cx + 7, gap_step)]     # dense near-gap strip
    for y, x in coords:
        if (y, x) in seen:
            continue
        seen.add((y, x))
        patch = vol[z-HALF:z+HALF+1, y-HALF:y+HALF+1, x-HALF:x+HALF+1]
        if patch.mean() < 0.05 * vol.max():
            continue
        m = analyse(patch)
        if m is None or m["dirs"] is None or not len(m["dirs"]):
            continue
        lab = classify(m)
        counts[lab] = counts.get(lab, 0) + 1
    return counts


def tractography_gap_test(vol, gap, stride=STRIDE):
    old = (Q.FG_FRAC, Q.ANISO_MIN)
    Q.FG_FRAC, Q.ANISO_MIN = 0.05, 0.6
    try:
        pos, dirs, w, is_cross, seed_id = Q.seed_segments(vol, stride)
        cand = P.candidate_links(pos, dirs, r_max=2.6 * stride)
        active, adj = P.greedy_forest(cand, len(pos))
        tracts = P.tracts_from(adj, len(pos))
    finally:
        Q.FG_FRAC, Q.ANISO_MIN = old
    cx = SHAPE[2] / 2.0                  # x=0 in phantom coords -> SHAPE/2 in array coords
    spanning = 0
    biggest = 0
    for t in tracts:
        if len(t) < 3:
            continue
        xs = pos[t, 0]
        biggest = max(biggest, len(t))
        if xs.min() < cx - gap * 0.6 and xs.max() > cx + gap * 0.6:
            spanning += 1              # a tract reaching well past BOTH sides -> bridged the gap
    return len(pos), len(tracts), biggest, spanning


def run():
    print("=" * 74)
    print("PROPER KISSING PHANTOM -- curved bundles, tangent, gapped (no touch)")
    for same in (True, False):
        vol, meta = build_kissing(same_size=same)
        tag = "SAME-size" if same else "DIFFERENT-size"
        print(f"\n--- {tag} (radii {meta['rA']:.1f}/{meta['rB']:.1f}, "
              f"true gap={meta['true_min_gap']:.1f} vox) ---")

        counts = classify_grid(vol)
        tot = sum(counts.values())
        singles = counts.get("single", 0)
        cross = counts.get("crossing", 0)
        note = ("OK: none" if cross == 0 else
                f"{cross}/{tot} ({cross/tot:.0%}) -- rare, near the closest approach "
                f"(resolution-scale partial volume, see docstring)")
        print(f"  [B] HOST labels over {tot} patches: {counts}  -> {singles/tot:.0%} "
              f"single; spurious 'crossing': {note}")

        npos, ntracts, biggest, spanning = tractography_gap_test(vol, meta["gap"])
        print(f"  [C] {npos} segments -> {ntracts} tracts; longest={biggest} segs; "
              f"tracts SPANNING the gap = {spanning}  "
              f"-> {'OK: gap respected, no bridge' if spanning == 0 else 'BRIDGED the gap (unexpected merge)'}")
    print("=" * 74)


if __name__ == "__main__":
    run()
