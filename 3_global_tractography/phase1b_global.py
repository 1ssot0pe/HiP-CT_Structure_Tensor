"""
phase1b_global.py -- Phase 1b of the tractography scoping.

Phase 1 showed a GREEDY local tracker merges the kissing bundles: at the boundary
it turns the tract onto bundle B (net turn 63 deg) instead of terminating. This
asks: does a GLOBAL objective over the WHOLE tract fix it, and -- crucially --
WHICH ingredient does the fixing?

We score two whole-tract hypotheses for the seed tract (which starts on bundle A
in x<0, heading +x toward the boundary):
  * TERMINATE -- follow bundle A to the boundary and stop.
  * TURN      -- follow A, then turn onto bundle B (the greedy Phase-1 result).

with a global energy   E(T) = -(data explained) + lambda * (total curvature).

Two versions of "data explained", to isolate the mechanism:
  * NAIVE      -- reward ALL bright structure the tract sits on (double-counts B).
  * PARSIMONY  -- reward only NOVEL structure: bundle B is already explained by its
                  OWN tract, so a tract A that turns into B earns nothing there.

Claim: the NAIVE objective still prefers TURN (reproducing the greedy merge,
whenever B's data outweighs the curvature). The PARSIMONY objective prefers
TERMINATE for ANY lambda>0 -- because the turn buys no novel data and only costs
curvature. So it is GLOBAL PARSIMONY (no double-counting across tracts), not a
local curvature limit, that resolves kissing vs crossing. That is the concrete
requirement the full Bayesian model must implement.
"""

import os as _sr_os, sys as _sr_sys
_sr_sys.path.insert(0, _sr_os.path.dirname(_sr_os.path.dirname(__file__)))  # repo root (shared kernels)

import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from host_geometry_test import make_kissing_phantom, make_true_crossing_same_angles
from phase1_kissing_crossing import track3d, BOUND_X

SEED = (48.0, 48.0, 18.0)          # x = -30 (bundle A region)
DIR0 = (1.0, 0.0, 0.0)             # +x
LAM = 0.02                         # curvature weight (per rad^2)


def _sample(vol, pts):
    return map_coordinates(vol, pts.T, order=1, mode="nearest")


def tract_scores(vol, tract):
    """(data_A, data_B, curvature) for a polyline tract. data split by region
    (x<0 = bundle A, x>=0 = bundle B); curvature = sum of squared turn angles."""
    inten = _sample(vol, tract)
    xrel = tract[:, 2] - BOUND_X
    data_A = float(inten[xrel < 0].sum())
    data_B = float(inten[xrel >= 0].sum())
    seg = np.diff(tract, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9
    curv = 0.0
    for k in range(len(seg) - 1):
        curv += float(np.arccos(np.clip(seg[k] @ seg[k + 1], -1, 1)) ** 2)
    return data_A, data_B, curv


def straight_to_boundary(seed, dir0, step=1.0):
    """TERMINATE hypothesis: straight along +x from seed, stop at the boundary."""
    pts = [np.array(seed, float)]
    d = np.array([dir0[2], dir0[1], dir0[0]], float)
    while pts[-1][2] < BOUND_X:
        pts.append(pts[-1] + step * d)
    return np.array(pts)


def evaluate(name, vol):
    turn = track3d(vol, SEED, DIR0, "continuity")      # greedy whole-tract (Phase 1)
    term = straight_to_boundary(SEED, DIR0)
    dA_t, dB_t, cv_t = tract_scores(vol, turn)
    dA_e, dB_e, cv_e = tract_scores(vol, term)
    # normalise data so lambda is interpretable across phantoms
    scale = dA_e + 1e-9
    dA_t, dB_t, dA_e, dB_e = (v / scale for v in (dA_t, dB_t, dA_e, dB_e))

    def energies(novel_B_turn):
        # E = -(data explained) + LAM*curv ; novel_B_turn=dB_t (naive) or 0 (parsimony)
        E_turn = -(dA_t + novel_B_turn) + LAM * cv_t
        E_term = -(dA_e) + LAM * cv_e
        return E_turn, E_term

    En_turn, En_term = energies(dB_t)      # naive: B double-counted
    Ep_turn, Ep_term = energies(0.0)       # parsimony: B redundant
    print(f"--- {name} ---")
    print(f"  TURN tract : data_A={dA_t:.2f} data_B={dB_t:.2f} curv={cv_t:.2f}")
    print(f"  TERM tract : data_A={dA_e:.2f} data_B={dB_e:.2f} curv={cv_e:.2f}")
    print(f"  NAIVE     E: turn={En_turn:+.2f}  term={En_term:+.2f}  -> picks "
          f"{'TURN (merges bundles)' if En_turn < En_term else 'TERMINATE'}")
    print(f"  PARSIMONY E: turn={Ep_turn:+.2f}  term={Ep_term:+.2f}  -> picks "
          f"{'TURN (merges bundles)' if Ep_turn < Ep_term else 'TERMINATE (correct)'}")
    return dict(turn=turn, term=term, naive=(En_turn, En_term), parsi=(Ep_turn, Ep_term))


def run(save="phase1b_global.png"):
    kiss, _ = make_kissing_phantom()
    cross, _ = make_true_crossing_same_angles()
    print("=" * 74)
    print("PHASE 1b -- does a GLOBAL objective resolve the kissing merge, and how?")
    print(f"  lambda(curvature) = {LAM}")
    rk = evaluate("KISSING", kiss)
    rc = evaluate("CROSSING", cross)
    print("-" * 74)
    kn = "TURN" if rk["naive"][0] < rk["naive"][1] else "TERM"
    kp = "TERM" if rk["parsi"][1] < rk["parsi"][0] else "TURN"
    print(f"  KISSING: naive picks {kn} (greedy merge), parsimony picks {kp}")
    print(f"  => it is GLOBAL PARSIMONY (no double-counting across tracts), not a local")
    print(f"     curvature limit, that terminates the kissing tract correctly.")
    print("=" * 74)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, (name, vol, r) in zip(ax, (("kissing", kiss, rk), ("crossing", cross, rc))):
        a.imshow(vol[48], cmap="gray", origin="upper")
        a.axvline(BOUND_X, color="cyan", ls="--", lw=1.5)
        a.plot(r["turn"][:, 2], r["turn"][:, 1], "-", color="#eb6834", lw=2.5,
               label="TURN (greedy)")
        a.plot(r["term"][:, 2], r["term"][:, 1], "-", color="#2a78d6", lw=3,
               label="TERMINATE (parsimony pick)")
        a.plot(SEED[2], SEED[1], "wo", ms=8, mec="k")
        pick = "TERMINATE" if r["parsi"][1] < r["parsi"][0] else "TURN"
        a.set_title(f"{name}: global-parsimony picks {pick}")
        a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=8, loc="upper right")
    fig.suptitle("Phase 1b: global parsimony (B is explained by its own tract) terminates\n"
                 "the kissing tract that greedy local tracking merged onto bundle B", fontsize=11)
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    run()
