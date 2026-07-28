"""
phase1c_directional.py -- Phase 1c: the honest kissing/crossing SEPARATOR.

Phase 1b left a gap: with parsimony it picked TERMINATE for BOTH kissing and
crossing, so it never actually separated them -- it only ruled out the spurious
TURN merge. Two flaws caused that:
  (1) it never tested the TRAVERSE hypothesis (go straight ACROSS the whole
      volume), which is the CORRECT answer for a true crossing (bundle A really
      does continue past the boundary);
  (2) its data term sampled raw IMAGE INTENSITY, so a straight +x tract through
      the kissing phantom's x>0 half scored high on the bright d2 fibres even
      though those fibres point the WRONG way (+y, not +x).

Phase 1c fixes both. The data term is DIRECTIONAL: at each step it reads HOST
fibre-ness IN THE TRACT'S TANGENT DIRECTION (the same signal the greedy tracker
steers on), not brightness. And it scores THREE whole-tract hypotheses for a d1
seed heading +x from x<0:
  * TERMINATE -- straight +x, stop at the x=0 boundary.
  * TRAVERSE  -- straight +x ALL THE WAY across the volume.
  * TURN      -- follow A, then bend onto bundle B (the greedy Phase-1 merge).

Objective (to MINIMISE):  E(T) = -(novel directional data) + LAM*curv + MU*len
where PARSIMONY is a hard ANGULAR GATE: a segment earns NOTHING when its tangent
runs within ~20 deg of another tract's direction d2 (that structure is already
explained by bundle B), regardless of position. So:
  * TURN's past-boundary leg runs along d2 -> gated to 0 -> no credit, only
    curvature+length cost. Rejected in both phantoms (parsimony, as in 1b).
  * TRAVERSE's leg runs along +x (=d1), far from d2 -> ungated -> full normalised
    credit r. In a CROSSING d1 is really there (r~0.9) -> TRAVERSE wins. In KISSING
    there is no d1 past the boundary (r low) -> length cost dominates -> TERMINATE.

Result (the Phase-1 milestone): KISSING -> TERMINATE, CROSSING -> TRAVERSE. It
takes BOTH ingredients -- a directional data term AND global parsimony. The gate
(vs the earlier linear discount) is what lets this GENERALISE across crossing
angles instead of only at 70 deg -- see phase1d_generalize.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from host_geometry_test import make_kissing_phantom, make_true_crossing_same_angles
from phase1_kissing_crossing import local_fibreness, V, BOUND_X

SEED = (48.0, 48.0, 18.0)          # x = -30 (bundle A region)
DIR0 = (1.0, 0.0, 0.0)             # +x, toward the boundary
LAM = 0.02                         # curvature weight (per rad^2)
MU = 0.72                          # Occam price of tract length: a tract only
                                   # extends where its NORMALISED directional
                                   # support exceeds MU. With the angular gate the
                                   # scale is the raw discriminant (kissing 0.45,
                                   # crossing 0.92 at 70 deg), so MU sits between.


def _walk(start, dir_xyz, shape, stop_x=np.inf, step=1.0):
    """Straight polyline from `start` along +dir_xyz (in x,y,z) until x reaches
    stop_x or the volume edge. Returns points in (axis0,axis1,axis2)."""
    d = np.array([dir_xyz[2], dir_xyz[1], dir_xyz[0]], float)
    d /= np.linalg.norm(d) + 1e-9
    pts = [np.array(start, float)]
    while True:
        nxt = pts[-1] + step * d
        if nxt[2] >= min(stop_x, shape[2] - 2) or not (
                2 <= nxt[0] < shape[0] - 2 and 2 <= nxt[1] < shape[1] - 2):
            break
        pts.append(nxt)
    return np.array(pts)


def straight(seed, dir0, stop_x, shape, step=1.0):
    """TERMINATE (stop_x=BOUND_X) or TRAVERSE (stop_x=inf): straight +dir0."""
    return _walk(seed, dir0, shape, stop_x, step)


def turn_onto_B(seed, dir0, d2, shape, step=1.0):
    """TURN hypothesis as a FIXED geometry: straight A to the boundary, then bend
    onto bundle B's direction d2 and run to the edge. (A fixed shape, not the
    greedy tracker, so its curvature+parsimony cost is the same in both phantoms.)"""
    a = _walk(seed, dir0, shape, BOUND_X, step)
    b = _walk(a[-1], d2, shape, np.inf, step)
    return np.vstack([a, b[1:]]) if len(b) > 1 else a


def directional_scores(vol, tract, d_other, gate_deg=20.0):
    """(data, gated_out, curv, n_steps) with a NORMALISED, PARSIMONIOUS data term.
    Per step the credit is r = fibre-ness ALONG the tangent / local PEAK fibre-ness
    in [0,1] -- 'is the tract's own direction a genuine fibre here?',
    contrast-invariant. PARSIMONY is a hard angular GATE, not a linear discount: a
    step earns NOTHING when its tangent runs within `gate_deg` of another tract's
    direction `d_other` (that structure is already explained by that tract). This
    is position-independent -- it correctly zeroes a turn onto bundle B wherever it
    happens (fixing the old x<0/x>=0 split, which miscounted a +y leg at constant x)
    and it does NOT haircut genuinely distinct fibres far from d_other (fixing the
    old linear discount that penalised real d1 support at moderate angles)."""
    seg = np.diff(tract, axis=0)                          # (axis0,axis1,axis2) steps
    tang = seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9)
    txyz = np.stack([tang[:, 2], tang[:, 1], tang[:, 0]], axis=1)   # tangent in (x,y,z)
    d_other = np.asarray(d_other, float); d_other /= np.linalg.norm(d_other) + 1e-9
    cos_gate = np.cos(np.deg2rad(gate_deg))
    data = gated_out = 0.0
    n = 0
    for k in range(len(seg)):
        f = local_fibreness(vol, tract[k])
        if f is None:
            continue
        r = float(f[np.argmax(np.abs(V @ txyz[k]))]) / (float(f.max()) + 1e-9)
        owned_by_other = abs(float(txyz[k] @ d_other)) > cos_gate   # same fibre as B?
        n += 1
        if owned_by_other:
            gated_out += r
        else:
            data += r
    curv = sum(float(np.arccos(np.clip(tang[k] @ tang[k + 1], -1, 1)) ** 2)
               for k in range(len(tang) - 1))
    return data, gated_out, curv, n


def evaluate(name, vol, d2, shape):
    hyps = {
        "TERMINATE": straight(SEED, DIR0, BOUND_X, shape),
        "TRAVERSE":  straight(SEED, DIR0, np.inf, shape),
        "TURN":      turn_onto_B(SEED, DIR0, d2, shape),
    }
    print(f"--- {name} ---")
    E = {}
    for h in ("TERMINATE", "TRAVERSE", "TURN"):
        data, gated, cv, n = directional_scores(vol, hyps[h], d2)
        # E = -(novel directional data) + curvature + Occam length price (MU/step)
        E[h] = -data + LAM * cv + MU * n
        print(f"  {h:<9} data={data:5.2f} gated(B-owned)={gated:5.2f} "
              f"curv={cv:5.2f} steps={n:3d}  ->  E={E[h]:+.2f}")
    pick = min(E, key=E.get)
    print(f"  PICK: {pick}")
    return hyps, E, pick


def run(save="phase1c_directional.png"):
    kiss, mk = make_kissing_phantom()
    cross, mc = make_true_crossing_same_angles()
    shape = kiss.shape
    print("=" * 74)
    print("PHASE 1c -- directional data + parsimony: does it SEPARATE kissing/crossing?")
    print(f"  LAM(curv)={LAM}  MU(len)={MU}   d2={np.round(mk['d2'], 2)}")
    hk, Ek, pk = evaluate("KISSING", kiss, mk["d2"], shape)
    hc, Ec, pc = evaluate("CROSSING", cross, mc["d2"], shape)
    print("-" * 74)
    ok = (pk == "TERMINATE" and pc == "TRAVERSE")
    print(f"  KISSING -> {pk}   CROSSING -> {pc}   "
          f"=> {'SEPARATED (milestone met)' if ok else 'NOT separated'}")
    print(f"  It takes BOTH a directional data term (kills TRAVERSE on kissing) AND")
    print(f"  parsimony (kills TURN by zeroing B-owned support). Neither alone suffices.")
    # robustness: over what MU window does the separation hold? (not knife-edge)
    # precompute the MU-independent (data, curv, steps) once per hypothesis, then
    # sweep MU cheaply -- avoids recomputing HOST fibre-ness hundreds of times.
    comp = {("kiss", h): directional_scores(kiss, t, mk["d2"]) for h, t in hk.items()}
    comp.update({("cross", h): directional_scores(cross, t, mc["d2"]) for h, t in hc.items()})
    def pick_at(tag, mu):
        E = {h: -data + LAM * cv + mu * n
             for (t, h), (data, _g, cv, n) in comp.items() if t == tag}
        return min(E, key=E.get)
    lo = hi = None
    for mu in np.arange(0.20, 1.00, 0.01):
        if pick_at("kiss", mu) == "TERMINATE" and pick_at("cross", mu) == "TRAVERSE":
            lo = mu if lo is None else lo; hi = mu
    print(f"  Robust: separation holds for MU in [{lo:.2f}, {hi:.2f}] (default {MU}).")
    print("=" * 74)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    cols = {"TERMINATE": "#2a78d6", "TRAVERSE": "#17a047", "TURN": "#eb6834"}
    for a, (name, vol, hyps, pick) in zip(
            ax, (("kissing", kiss, hk, pk), ("crossing", cross, hc, pc))):
        a.imshow(vol[48], cmap="gray", origin="upper")
        a.axvline(BOUND_X, color="cyan", ls="--", lw=1.5)
        for h, t in hyps.items():
            lw = 3.5 if h == pick else 2.0
            a.plot(t[:, 2], t[:, 1], "-", color=cols[h], lw=lw,
                   label=f"{h}{'  <- PICK' if h == pick else ''}")
        a.plot(SEED[2], SEED[1], "wo", ms=8, mec="k")
        a.set_title(f"{name}: picks {pick}")
        a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=8, loc="upper right")
    fig.suptitle("Phase 1c: directional data + global parsimony separates kissing\n"
                 "(TERMINATE) from crossing (TRAVERSE) -- the Phase-1 milestone", fontsize=11)
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    run()
