"""
run_pipeline.py -- Stage 1 -> Stage 2 -> Stage 3, end to end.

This is the join. Before it, the three folders never exchanged data: each read
the raw volume and called the root engine independently. Here the flow is

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

Run
---
    python run_pipeline.py                # both phantoms
    python run_pipeline.py --iters 3
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


def _crop(vol, centre, r):
    """Cube of half-width r about `centre`, with the centre's new index."""
    sl, new = [], []
    for v, s in zip(centre, vol.shape):
        lo = max(v - r, 0)
        hi = min(v + r, s)
        sl.append(slice(lo, hi))
        new.append(v - lo)
    return vol[tuple(sl)], tuple(new)


def run_case(name, vol, centre, truth, block=32, stride=8, iters=3, verbose=True):
    print(f"\n{'-' * 78}\n{name}   (truth: {truth})   volume {vol.shape}\n{'-' * 78}")

    # ---- Stage 1 + 2 --------------------------------------------------------
    gated = build_gated_field(vol, block=block, stride=stride, verbose=verbose)
    mask = gated.routing_mask()
    print(f"  [stage 2] routing mask: {int(mask.sum())}/{gated.n} patches to HOST")

    # ---- Stage 3 ------------------------------------------------------------
    e3 = S3.orientation_field(vol)
    field, junctions, I0 = S3.field_from_gated(gated, vol, e3=e3)
    print(f"  [stage 3] field {len(field)} patches, {len(junctions)} junction(s), "
          f"I_0={I0:.3f}")
    if not junctions:
        print("  [stage 3] no multi-orientation junction -- nothing to score")
        return dict(name=name, truth=truth, skipped=True,
                    n_patches=gated.n, n_routed=int(mask.sum()))

    for j in junctions:
        print(f"    junction {j['centre']}: I_c={j['I_c']:.2f} I_a={j['I_a']:.2f} "
              f"I_b={j['I_b']:.2f}  S={j['S']:.3f} -> {j['verdict']}")

    D_obs, tissue = S3.anisotropy_target(vol)
    theta, trace, info = S3.optimise(vol, field, junctions, D_obs, tissue, I0,
                                     n_iter=iters, verbose=verbose)

    verdicts = [j["verdict"] for j in junctions]
    majority = max(set(verdicts), key=verdicts.count)
    ok = majority == truth
    print(f"  RESULT: verdict {majority} vs truth {truth}  "
          f"{'CORRECT' if ok else 'WRONG'};  L {trace[0]['total']:.4f} -> "
          f"{trace[-1]['total']:.4f}")
    return dict(name=name, truth=truth, skipped=False,
                n_patches=gated.n, n_routed=int(mask.sum()),
                junctions=[{k: (v if not isinstance(v, tuple) else list(v))
                            for k, v in j.items() if k != "dirs"}
                           for j in junctions],
                verdict=majority, correct=bool(ok), theta=theta,
                L_initial=float(trace[0]["total"]),
                L_final=float(trace[-1]["total"]),
                trace=trace, junction_log=info["junctions"])


def main(iters=3, radius=24, stride=8, block=32):
    print("=" * 78)
    print("THREE-STAGE PIPELINE -- phantom validation (known truth)")
    print("=" * 78)

    X = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5, CROSS_R1, CROSS_TAPER)
    Xc, xc = _crop(X, tuple(s // 2 for s in CROSS_SHAPE), radius)
    K = raw_kissing(4.0, 0.028, 1.6, 1.6)
    Kc, kc = _crop(K, tuple(s // 2 for s in K.shape), radius)

    out = []
    out.append(run_case("TRUE CROSSING", Xc, xc, "CROSSING",
                        block=block, stride=stride, iters=iters))
    out.append(run_case("TANGENCY (kissing)", Kc, kc, "TANGENCY",
                        block=block, stride=stride, iters=iters))

    scored = [r for r in out if not r["skipped"]]
    n_ok = sum(r["correct"] for r in scored)
    print("\n" + "=" * 78)
    print(f"END-TO-END: {n_ok}/{len(scored)} phantoms classified correctly "
          f"({len(out) - len(scored)} skipped)")
    for r in out:
        if r["skipped"]:
            print(f"  {r['name']:22} SKIPPED (no junction)")
        else:
            s = r["junctions"][0]["S"]
            print(f"  {r['name']:22} S={s:.3f} -> {r['verdict']:9} "
                  f"(truth {r['truth']})   L {r['L_initial']:.4f} -> "
                  f"{r['L_final']:.4f}")
    print("=" * 78)

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
    a = ap.parse_args()
    main(iters=a.iters, radius=a.radius, stride=a.stride, block=a.block)
