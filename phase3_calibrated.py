"""
phase3_calibrated.py -- the full-build step: make the joint global model NOISE-AWARE
and re-run the i74 over-calling estimate end-to-end under calibration.

#1 showed realistic (correlated+anisotropic) noise drops true-crossing confirmation
89% -> 57% because it inflates the estimated segment orientations, so genuine
collinear links exceed the fixed curvature gate and real fibres FRAGMENT. The fix:
DERIVE the linking curvature budget from the noise instead of hard-coding it.

Calibration: on a single-fibre phantom under the SAME real noise, the candidate
continuation links incur a curvature cost purely from noise-perturbed orientations.
Their high percentile is the budget a true link needs at this noise level:
    gate* = P90( candidate-link curvature costs on single-fibre + real noise )
Then link the crossing/kissing phantoms with gate* (calibrated) vs the fixed 0.30
(un-calibrated), under the real noise, and recompute sensitivity / false-confirm.
Finally re-invert i74's observed 24% confirm with the calibrated rates.

This calibrates the PARSIMONY/linking budget from the measured noise (the design-doc
'derive the weight from sigma'); wiring the full FFT-whitened likelihood into per-link
inference (phase3_generative's term) is the remaining, heavier piece.
"""

import os
import numpy as np

from crossing_shape_3d import _tube_family, _coords
from host_geometry_test import _dir_inplane, _finalise
import phase3_i74 as Q
import phase3_global as P
from phase3_realnoise import real_noise, SIGMA_N

I74_OBS = 0.24          # observed HOST-crossing confirm fraction on i74 (96^3)


def build(kind, noise):
    c = _coords(noise.shape)
    t1 = _tube_family(c, _dir_inplane(0.0), 12.0, 1.5)
    t2 = _tube_family(c, _dir_inplane(70.0), 12.0, 1.5)
    if kind == "single":
        sig = t1
    elif kind == "crossing":
        sig = t1 + t2
    else:
        sig = np.where(c[0] < 0.0, t1, t2)
    return _finalise(sig + SIGMA_N * noise)


def seed(vol, stride):
    """HOST seed + candidate links once; returns what re-linking at any gate needs."""
    pos, dirs, w, is_cross, seed_id = Q.seed_segments(vol, stride)
    cand = P.candidate_links(pos, dirs, r_max=2.6 * stride)
    return len(pos), cand, seed_id, is_cross


def rate_at_gate(npos, cand, seed_id, is_cross, gate):
    active, adj = P.greedy_forest(cand, npos, gate=gate)
    tracts = P.tracts_from(adj, npos)
    tract_of = np.full(npos, -1, int)
    tract_len = np.zeros(len(tracts), int)
    for ti, comp in enumerate(tracts):
        tract_len[ti] = len(comp)
        for k in comp:
            tract_of[k] = ti
    conf, dem = Q.crossing_verdicts(seed_id, is_cross, tract_of, tract_len)
    tot = conf + dem
    return (conf / tot if tot else float("nan")), tot


def invert(obs, sens, fconf):
    return (obs - fconf) / (sens - fconf + 1e-9)


def run(crop=64, stride=4):
    print("=" * 74)
    print(f"PHASE 3 CALIBRATED -- noise-derived linking budget (shape={crop}^3, "
          f"stride={stride})")
    noise = real_noise((crop, crop, crop))

    # calibration: curvature budget a true link needs at this noise level
    _n, cand_sf, _s, _i = seed(build("single", noise), stride)
    gate_star = float(np.percentile(list(cand_sf.values()), 90))
    print(f"  calibrated gate* = P90 of single-fibre link costs under real noise "
          f"= {gate_star:.3f}  (default 0.30)")

    # seed crossing + kissing ONCE, re-link at each gate
    ncx, cand_cx, sid_cx, ic_cx = seed(build("crossing", noise), stride)
    nks, cand_ks, sid_ks, ic_ks = seed(build("kissing", noise), stride)
    print("-" * 74)
    print(f"  {'gate':>12} | {'sens(crossing)':>14} | {'fconf(kissing)':>14} | "
          f"{'i74 real-frac':>13}")
    best = None
    for gate in sorted({0.20, 0.30, 0.45, 0.60, 0.90, round(gate_star, 2)}):
        sens, _ = rate_at_gate(ncx, cand_cx, sid_cx, ic_cx, gate)
        fconf, _ = rate_at_gate(nks, cand_ks, sid_ks, ic_ks, gate)
        gap = sens - fconf                      # crossing/non-crossing separation
        p = invert(I74_OBS, sens, fconf)
        star = " *" if best is None or gap > best[0] else ""
        if best is None or gap > best[0]:
            best = (gap, gate, sens, fconf, p)
        tag = f"{gate:.2f}" + ("=gate*" if abs(gate - round(gate_star, 2)) < 1e-6 else "")
        print(f"  {tag:>12} | {sens:13.0%}  | {fconf:13.0%}  | "
              f"{(f'{p:.0%}' if 0 <= p <= 1 else 'n/a'):>12}   gap {gap:+.0%}{star}")
    print("-" * 74)
    g, gate, sens, fconf, p = best
    print(f"  BEST separation at gate={gate:.2f}: sens {sens:.0%}, fconf {fconf:.0%} "
          f"(gap {g:+.0%}) -> i74 real-frac {p:.0%}" if 0 <= p <= 1 else
          f"  BEST separation at gate={gate:.2f}: sens {sens:.0%}, fconf {fconf:.0%} "
          f"(gap {g:+.0%}); i74 confirms BELOW the noise floor -> essentially all over-called")
    print("  Under real noise the fixed gate fragments real crossings (low sens); the")
    print("  noise-derived gate* restores the linking budget. i74 real-frac = invert of")
    print("  the observed 24% with each row's (sens, fconf). Compare to the un-calibrated")
    print("  ~17% (real-noise) and the optimistic ~4% (white-noise) estimates.")
    print("=" * 74)


if __name__ == "__main__":
    run()
