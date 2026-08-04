"""
run_pipeline.py -- Stage 1 -> Stage 2 -> Stage 3, end to end.

This is the join. Before it, the three folders never exchanged data: each read
the raw volume and called the root engine independently. 

Here the flow is: 

    Stage 1  adaptive (sigma, rho) per block          [1_sigma_rho_selection]
       |     -> dense e3 field
       v
    Stage 2  route patches on <c_s> > <FA>            [2_host_classification]
       |     -> GatedField: routing mask + directions
       |        (single patches keep e3; routed patches carry HOST's peaks)
       v
    Stage 3  integrate -> D_rec -> L -> update theta   [this folder]
             -> converged theta, loss trace, per-junction S and verdict

Validation is on phantoms with known truth before any real data, per the brief:
a localized crossing (truth CROSSING) and a kissing pair (truth TANGENCY).
Both count towards the tally and the process exits non-zero on any FAIL, so
this doubles as a regression check.

What the kissing phantom does and does NOT test
-----------------------------------------------
It passes by producing NO junction: the routing gate never fires on it, so the
pipeline never invents a crossing. That is the right answer, and it is a real
pass -- but it means the superposition score S is exercised here only on the
CROSSING side. The gate's refusal was checked to be genuine, not an artifact of
Stage 1 failing to certify a scale on this sparse phantom: sweeping the
sigma fallback over 0.85-3.0 routes zero patches at every value.

S on a true tangency is covered elsewhere -- `phase3_validate3d.py` (kissing
scores 0 at every crossing angle) and `mass_conservation/
mass_conservation_validate.py` (catch 100%, false-positive 0%). Real i74 also
produces TANGENCY verdicts through this driver. So the coverage exists; it just
is not in this file.

Run
---
    python run_pipeline.py                # both phantoms
    python run_pipeline.py --iters 3
    python run_pipeline.py --i74          # real data, truth UNKNOWN
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "mass_conservation"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import self_optimising_tractography as S3
from sigma_rho_gated_field import build_gated_field
from mass_conservation_validate import (raw_localized_crossing, raw_kissing,
                                        CROSS_SHAPE, CROSS_R1, CROSS_TAPER)


UNKNOWN = "UNKNOWN"          # truth label for real data, where there is none


def _crop(vol, centre, r):
    """Cube of half-width r about `centre`, with the centre's new index."""
    sl, new = [], []
    for v, s in zip(centre, vol.shape):
        lo = max(v - r, 0)
        hi = min(v + r, s)
        sl.append(slice(lo, hi))
        new.append(v - lo)
    return vol[tuple(sl)], tuple(new)


def _status(truth, verdict):
    """(status, reason) for one case. `verdict` is None if nothing was scored.

    A case with no junction is NOT a skip, and treating it as one hid two
    things. For a kissing pair, finding no junction is the CORRECT answer --
    the pipeline declined to invent a crossing -- so it was a pass being
    reported as a non-result. For a true crossing it is a MISS. Both used to
    drop out of the tally entirely, which meant a total failure of Stage 2
    would print "0/0 phantoms classified correctly (2 skipped)" instead of
    failing: the validation went quiet exactly when it should have shouted.

    Real data carries truth=UNKNOWN, so no correctness claim is made at all --
    previously `majority == truth` compared against the string "UNKNOWN" and
    printed WRONG for every i74 run.
    """
    if truth == UNKNOWN:
        return "UNSCORED", "no ground truth"
    if verdict is None:
        if truth == "TANGENCY":
            return "PASS", "no junction -- no false crossing invented"
        return "FAIL", f"no junction -- a real {truth} was missed"
    if verdict == truth:
        return "PASS", f"verdict {verdict}"
    return "FAIL", f"verdict {verdict}, expected {truth}"


def run_case(name, vol, centre, truth, block=32, stride=8, iters=3, verbose=True):
    print(f"\n{'-' * 78}\n{name}   (truth: {truth})   volume {vol.shape}\n{'-' * 78}")

    # ---- Stage 1 + 2 --------------------------------------------------------
    gated = build_gated_field(vol, block=block, stride=stride, verbose=verbose)
    mask = gated.routing_mask()
    tag = name.split()[0].lower()
    gated.save(f"unified_field_{tag}.npz")
    print(f"  [stage 2] routing mask: {int(mask.sum())}/{gated.n} patches to HOST")

    # ---- Stage 3 ------------------------------------------------------------
    e3 = S3.orientation_field(vol)
    field, junctions, I0 = S3.field_from_gated(gated, vol, e3=e3)
    print(f"  [stage 3] field {len(field)} patches, {len(junctions)} junction(s), "
          f"I_0={I0:.3f}")
    if not junctions:
        status, reason = _status(truth, None)
        print("  [stage 3] no multi-orientation junction -- nothing to score")
        if status != "UNSCORED":
            print(f"  RESULT: {status} -- {reason}")
        return dict(name=name, truth=truth, skipped=True, status=status,
                    reason=reason, verdict=None,
                    correct=None if status == "UNSCORED" else status == "PASS",
                    n_patches=gated.n, n_routed=int(mask.sum()))

    for j in junctions:
        print(f"    junction {j['centre']}: I_c={j['I_c']:.2f} I_a={j['I_a']:.2f} "
              f"I_b={j['I_b']:.2f}  S={j['S']:.3f} -> {j['verdict']}")

    D_obs, tissue = S3.anisotropy_target(vol)
    theta, trace, info = S3.optimise(vol, field, junctions, D_obs, tissue, I0,
                                     n_iter=iters, verbose=verbose)

    verdicts = [j["verdict"] for j in junctions]
    majority = max(set(verdicts), key=verdicts.count)
    status, reason = _status(truth, majority)
    lead = (f"verdict {majority} (no ground truth)" if status == "UNSCORED"
            else f"{status} -- {reason}")
    print(f"  RESULT: {lead};  L {trace[0]['total']:.4f} -> "
          f"{trace[-1]['total']:.4f}")
    return dict(name=name, truth=truth, skipped=False, status=status,
                reason=reason,
                n_patches=gated.n, n_routed=int(mask.sum()),
                junctions=[{k: (v if not isinstance(v, tuple) else list(v))
                            for k, v in j.items() if k != "dirs"}
                           for j in junctions],
                verdict=majority,
                correct=None if status == "UNSCORED" else status == "PASS",
                theta=theta,
                L_initial=float(trace[0]["total"]),
                L_final=float(trace[-1]["total"]),
                trace=trace, junction_log=info["junctions"])


def run_i74(centre=(384, 384, 384), radius=32, block=32, stride=8, iters=3):
    """Real i74. No ground truth -- this is the case S exists for.

    On phantoms the gate already rejects a tangency upstream, so S never fires.

    On i74 the recorded Phase-2/3 result is that HOST-flagged crossings behave
    like the non-crossing phantom (24% confirm, at the 21% false-confirm floor),
    i.e. HOST over-calls. Those over-called crossings ARE routed by the gate, so
    this is where the superposition metric gets its first real test: it should
    demote a substantial fraction of them (S well below 0.75).
    """
    from reconstruction import load_volume
    c = tuple(centre)
    # load_volume's `crop` applies to ALL THREE axes, not just z -- slicing the
    # result again with absolute indices produces a zero-size array.
    vol = load_volume(crop=(c[0] - radius, c[0] + radius),
                      dtype=np.uint8).astype(np.float32)
    print("=" * 78)
    print(f"REAL i74 -- {vol.shape} at {c}, mean {vol.mean():.1f}")
    print("=" * 78)

    r = run_case("I74 REGION", vol, tuple(v // 2 for v in vol.shape), "UNKNOWN",
                 block=block, stride=stride, iters=iters)
    if not r["skipped"]:
        S = np.array([j["S"] for j in r["junctions"]], float)
        S = S[np.isfinite(S)]
        vs = [j["verdict"] for j in r["junctions"]]
        print(f"\n  {len(S)} scored junctions: S median {np.median(S):.3f}, "
              f"IQR [{np.percentile(S,25):.3f}, {np.percentile(S,75):.3f}]")
        for v in ("CROSSING", "TANGENCY", "AMBIGUOUS", "unscored"):
            k = vs.count(v)
            if k:
                print(f"    {v:10}: {k:4d}  ({k/len(vs):.0%})")
    with open("pipeline_i74_results.json", "w") as f:
        json.dump(r, f, indent=2, default=float)
    print("wrote pipeline_i74_results.json")
    return r


def main(iters=3, radius=24, stride=8, block=32):

    print("THREE-STAGE PIPELINE -- phantom validation (known truth)")


    X = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5, CROSS_R1, CROSS_TAPER)
    Xc, xc = _crop(X, tuple(s // 2 for s in CROSS_SHAPE), radius)
    K = raw_kissing(4.0, 0.028, 1.6, 1.6)
    Kc, kc = _crop(K, tuple(s // 2 for s in K.shape), radius)

    out = []
    out.append(run_case("TRUE CROSSING", Xc, xc, "CROSSING",
                        block=block, stride=stride, iters=iters))
    out.append(run_case("TANGENCY (kissing)", Kc, kc, "TANGENCY",
                        block=block, stride=stride, iters=iters))

    # Every phantom counts. A case with no junction is a pass or a fail
    # depending on its truth (see _status) -- never dropped from the tally.
    n_pass = sum(r["status"] == "PASS" for r in out)
    print("\n" + "=" * 78)
    print(f"END-TO-END: {n_pass}/{len(out)} phantoms correct")
    for r in out:
        if r["skipped"]:
            print(f"  {r['name']:22} {r['status']:4}  (truth {r['truth']})   "
                  f"{r['reason']}")
        else:
            s = r["junctions"][0]["S"]
            print(f"  {r['name']:22} {r['status']:4}  S={s:.3f} -> "
                  f"{r['verdict']:9} (truth {r['truth']})   "
                  f"L {r['L_initial']:.4f} -> {r['L_final']:.4f}")
    print("=" * 78)
    if n_pass < len(out):
        print("VALIDATION FAILED -- see the FAIL rows above")

    with open("pipeline_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("wrote pipeline_results.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 1->2->3 phantom validation")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--radius", type=int, default=24)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--i74", action="store_true", help="run on real i74 instead")
    ap.add_argument("--centre", nargs=3, type=int, default=[384, 384, 384])
    a = ap.parse_args()
    if a.i74:
        run_i74(centre=tuple(a.centre), radius=a.radius, block=a.block,
                stride=a.stride, iters=a.iters)
    else:
        res = main(iters=a.iters, radius=a.radius, stride=a.stride,
                   block=a.block)
        # Non-zero exit on any FAIL, so the phantom validation is usable as a
        # regression check rather than something you have to read carefully.
        sys.exit(1 if any(r["status"] == "FAIL" for r in res) else 0)
