"""
phase2b_validate.py -- validate a Phase-2 per-peak certifier on phantoms with known
truth, BEFORE trusting any demotion rate on real i74.

Two certifiers were tried and BOTH fail the kissing side (see BAYES_TRACTOGRAPHY.md):
  * STRAIGHTNESS |disp|/path  -> kissing demote only 27% (catches noise, not kissing);
  * this file: min(reach_fwd, reach_bwd) with a TIGHT cone (directional asymmetry) ->
    kissing demote only 13%. A one-sided kissing peak still reaches far enough on its
    real side, and boundary voxels are sampled at varied offsets, so min-reach stays
    high. Conclusion: a LOCAL per-peak certifier cannot separate kissing from crossing
    on realistic tissue -- it needs the full joint global model.

The certifier under test: at a voxel HOST calls a 2-way crossing, walk each peak as a
continuity streamline; a REAL through-fibre persists far in BOTH directions
(min-reach high), a one-sided/spurious peak should die on a side (min-reach low).

Two ground-truth cases:
  * TRUE CROSSING (make_true_crossing_same_angles, theta2=70, within the Phase-1
    wide-angle limit): both fibres are real everywhere -> BOTH peaks should be
    straight -> certifier should CONFIRM.
  * KISSING BOUNDARY (make_kissing_phantom): only near the x=0 contact does a patch
    straddle both regions and HOST report a 2-peak 'crossing' -- but the two fibres
    do NOT cross there (d1 lives only in x<0, d2 only in x>0). The peak pointing
    into the wrong half has no through-line -> should WANDER -> certifier DEMOTES.

Pass = confirms true crossings AND demotes kissing contacts. Only then is the i74
demotion number (phase2_i74_demote.py) measuring what we claim.
"""

import os as _sr_os, sys as _sr_sys
_sr_sys.path.insert(0, _sr_os.path.dirname(_sr_os.path.dirname(__file__)))  # repo root (shared kernels)

import numpy as np

from host_geometry_test import make_kissing_phantom, make_true_crossing_same_angles
from host_fod import analyse, classify
from phase2_i74_demote import _walk_support, HALF

BOUND_X = 48
TIGHT_CONE = 20.0        # a one-sided (kissing) peak cannot find its fibre off-axis
MINREACH_MIN = 4         # a peak is a through-fibre only if it persists BOTH ways


def peak_minreach(vol, pos, dir_xyz, cone=TIGHT_CONE):
    """min(reach_forward, reach_backward) with a TIGHT cone -- the directional
    ASYMMETRY a single straightness scalar throws away. A real through-fibre
    reaches far BOTH ways (min high); a kissing/terminating peak dies on one side
    (min low)."""
    rf, _ep, _l = _walk_support(vol, pos, dir_xyz, cone=cone)
    rb, _ep, _l = _walk_support(vol, pos, -np.asarray(dir_xyz, float), cone=cone)
    return min(rf, rb)


def certify_at(vol, pos, dirs):
    """(weaker_minreach, demoted?) over the two peaks at pos."""
    w = min(peak_minreach(vol, pos, dirs[0]), peak_minreach(vol, pos, dirs[1]))
    return w, (w < MINREACH_MIN)


def sample_crossings(vol, xrange, n_target=15, seed=0, max_attempts=1200):
    """Voxels in the given x-index range that HOST classifies as 'crossing'."""
    rng = np.random.default_rng(seed)
    lo_yz, hi_yz = HALF + 3, vol.shape[0] - HALF - 3
    out = []
    for _ in range(max_attempts):
        x = rng.integers(*xrange)
        y = rng.integers(lo_yz, hi_yz); z = rng.integers(lo_yz, hi_yz)
        pos = np.array([z, y, x], float)                 # (axis0,axis1,axis2)=(z,y,x)
        patch = vol[tuple(slice(int(pos[i]) - HALF, int(pos[i]) + HALF + 1)
                          for i in range(3))]
        m = analyse(patch)
        if m is not None and classify(m) == "crossing":
            out.append((pos, np.asarray(m["dirs"][:2])))
            if len(out) >= n_target:
                break
    return out


def run():
    print("=" * 74)
    print("PHASE 2b -- validate the straightness certifier on known-truth phantoms")
    cross, _ = make_true_crossing_same_angles(theta2=70.0)
    kiss, _ = make_kissing_phantom(theta2=70.0)

    # true crossing: sample the interior (away from edges); expect CONFIRM
    xc = sample_crossings(cross, (HALF + 6, cross.shape[2] - HALF - 6))
    # kissing: sample only near the x=0 contact; expect DEMOTE
    xk = sample_crossings(kiss, (BOUND_X - HALF + 2, BOUND_X + HALF - 2))

    res = {}
    for name, vol, cx, expect in (("TRUE CROSSING", cross, xc, "confirm"),
                                  ("KISSING BOUNDARY", kiss, xk, "demote")):
        if not cx:
            print(f"  {name}: no HOST 'crossing' voxels found (n=0)")
            res[name] = (np.nan, np.nan); continue
        certs = [certify_at(vol, p, d) for p, d in cx]
        ws = [w for w, _d in certs]; dem_rate = np.mean([d for _w, d in certs])
        res[name] = (np.median(ws), dem_rate)
        print(f"  {name:>16}: n={len(cx):2d}  weaker min-reach median="
              f"{np.median(ws):.1f} vox  demoted={dem_rate:.0%}  (expect {expect})")

    print("-" * 74)
    conf_true = 1 - res["TRUE CROSSING"][1]
    dem_kiss = res["KISSING BOUNDARY"][1]
    ok = (conf_true > 0.6) and (dem_kiss > 0.6)
    print(f"  confirm-rate on TRUE crossings = {conf_true:.0%} (want high); "
          f"demote-rate on KISSING = {dem_kiss:.0%} (want high)")
    print(f"  => certifier {'VALIDATED' if ok else 'NOT validated'}: it "
          f"{'separates real from spurious peaks' if ok else 'does not cleanly separate -- revisit metric/threshold'}.")
    print("=" * 74)


if __name__ == "__main__":
    run()
