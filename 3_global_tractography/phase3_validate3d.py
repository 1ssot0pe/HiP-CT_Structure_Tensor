"""
phase3_validate3d.py -- validate the 3D global-tractography pipeline (phase3_i74's
seeding + linking + crossing verdict) on host_geometry_test 3D phantoms with KNOWN
truth, BEFORE trusting the ~76% demotion rate it reported on real i74.

The i74 verdict rests on an unchecked assumption: that the 3D pipeline actually
CONFIRMS a real crossing and DEMOTES a non-crossing. This checks it directly, the
same validate-first move that caught the local certifier in Phase 2b.

  TRUE CROSSING (make_true_crossing_same_angles): both fibres in every voxel ->
      HOST flags crossings everywhere; each peak is a real through-line -> the global
      model should CONFIRM most (both peaks link into extended tracts).
  KISSING (make_kissing_phantom): single-fibre except at the x=0 contact, where HOST
      partial-volumes two NON-crossing fibres into a false 'crossing'; the peak into
      the wrong half has no collinear continuation -> the model should DEMOTE most.

PASS = crossing confirm-rate high AND kissing confirm-rate low (i.e. the pipeline
separates real crossings from false ones). Only then is the i74 number trustworthy.

Run:  .venv/Scripts/python -u phase3_validate3d.py [--crop N] [--stride S]
"""

import os as _sr_os, sys as _sr_sys
_sr_sys.path.insert(0, _sr_os.path.dirname(_sr_os.path.dirname(__file__)))  # repo root (shared kernels)

import argparse
import numpy as np

from host_geometry_test import make_true_crossing_same_angles, make_kissing_phantom
import phase3_i74 as Q
import phase3_global as P


def pipeline(vol, stride):
    """Run seed -> link -> forest -> crossing verdict; return (conf, dem, longest)."""
    pos, dirs, w, is_cross, seed_id = Q.seed_segments(vol, stride)
    if len(pos) < 10:
        return 0, 0, 0, len(pos)
    cand = P.candidate_links(pos, dirs, r_max=2.6 * stride)
    active, adj = P.greedy_forest(cand, len(pos))
    tracts = P.tracts_from(adj, len(pos))
    tract_of = np.full(len(pos), -1, int)
    tract_len = np.zeros(len(tracts), int)
    for ti, c in enumerate(tracts):
        tract_len[ti] = len(c)
        for k in c:
            tract_of[k] = ti
    conf, dem = Q.crossing_verdicts(seed_id, is_cross, tract_of, tract_len)
    return conf, dem, int(tract_len.max()), len(pos)


def run(crop=64, stride=4):
    print("=" * 74)
    print(f"PHASE 3 VALIDATION -- 3D pipeline on known-truth phantoms "
          f"(shape={crop}^3, stride={stride})")
    shape = (crop, crop, crop)
    res = {}
    for name, vol in (("TRUE CROSSING", make_true_crossing_same_angles(shape=shape)[0]),
                      ("KISSING", make_kissing_phantom(shape=shape)[0])):
        conf, dem, longest, nseg = pipeline(vol.astype(np.float64), stride)
        tot = conf + dem
        rate = conf / tot if tot else float("nan")
        res[name] = (conf, dem, tot, rate)
        want = "confirm most" if "CROSS" in name else "demote most"
        print(f"  {name:>13}: {nseg} segs, longest tract {longest} segs; "
              f"HOST crossings={tot:3d} -> CONFIRMED {conf} ({rate:.0%}), "
              f"DEMOTED {dem} ({1-rate:.0%})  (expect {want})")
    print("-" * 74)
    cx = res["TRUE CROSSING"][3]
    ks = res["KISSING"][3]
    ok = (cx >= 0.5) and (ks <= 0.25) and (cx - ks >= 0.3)
    print(f"  crossing confirm-rate={cx:.0%}, kissing confirm-rate={ks:.0%}  => "
          f"{'VALIDATED: pipeline confirms real crossings and demotes false ones -> the i74 ~76%% demotion is trustworthy' if ok else 'NOT validated -- the i74 demotion number is suspect; investigate seeding/linking'}")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    run(crop=args.crop, stride=args.stride)
