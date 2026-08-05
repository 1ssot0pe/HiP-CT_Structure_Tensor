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

S on a true tangency is covered elsewhere -- `phase3.py` (kissing
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
from field_refine import refine as refine_field
from render_field import render_field
from sigma_rho_gated_field import build_gated_field
from localized_phantoms import (raw_localized_crossing, raw_kissing,
                                CROSS_SHAPE, CROSS_R1, CROSS_TAPER)


UNKNOWN = "UNKNOWN"          # truth label for real data, where there is none

# Outputs used to be written CWD-relative, so `python run_pipeline.py` and
# `python 3_global_tractography/run_pipeline.py` dropped their artifacts in
# different places and a re-run from a different directory silently produced a
# second copy instead of updating the first. Everything now goes to one
# directory, `results/` at the repo root, overridable with --outdir.
DEFAULT_OUTDIR = os.path.join(_ROOT, "results")


def _out(outdir, name):
    """Absolute path for an output file, creating the directory on demand."""
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, name)


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


def run_case(name, vol, centre, truth, block=32, stride=8, iters=3, verbose=True,
             outdir=DEFAULT_OUTDIR, figures=True, refine=False):
    print(f"\n{'-' * 78}\n{name}   (truth: {truth})   volume {vol.shape}\n{'-' * 78}")

    # ---- Stage 1 + 2 --------------------------------------------------------
    gated = build_gated_field(vol, block=block, stride=stride, verbose=verbose)
    mask = gated.routing_mask()
    tag = name.split()[0].lower()
    gated.save(_out(outdir, f"unified_field_{tag}.npz"))
    print(f"  [stage 2] routing mask: {int(mask.sum())}/{gated.n} patches to HOST")

    # ---- Stage 3a: refine the field ----------------------------------------
    e3 = S3.orientation_field(vol)
    if refine:
        e3, rinfo = refine_field(e3)
        print(f"  [stage 3a] Frank-Oseen refine: {rinfo['iterations']} iters, "
              f"E {rinfo['E_initial']:.1f} -> {rinfo['E_final']:.1f}")

    # ---- Stage 3b: crossing vs kissing, then tracts -------------------------
    field, junctions, I0 = S3.field_from_gated(gated, vol, e3=e3)
    print(f"  [stage 3] field {len(field)} patches, {len(junctions)} junction(s), "
          f"I_0={I0:.3f}")

    # Drawn here, before the no-junction early return, because that case is the
    # one most worth looking at: it is a claim that nothing crosses, and the
    # figure is how you check the field still followed the fibres.
    def _figure(status_note=""):
        if not figures:
            return None
        p = render_field(
            vol, field, junctions, _out(outdir, f"field_{tag}.png"),
            title=f"{name}   (truth: {truth})",
            subtitle=(f"{gated.n} patches, {int(mask.sum())} routed to HOST, "
                      f"{len(junctions)} junction(s){status_note}"))
        print(f"  [figure] {p}")
        return p

    if not junctions:
        status, reason = _status(truth, None)
        print("  [stage 3] no multi-orientation junction -- nothing to score")
        if status != "UNSCORED":
            print(f"  RESULT: {status} -- {reason}")
        fig_path = _figure(" -- no crossing claimed")
        return dict(name=name, truth=truth, skipped=True, status=status,
                    reason=reason, verdict=None, figure=fig_path,
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
    fig_path = _figure()
    return dict(name=name, truth=truth, skipped=False, status=status,
                reason=reason, figure=fig_path,
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


def run_i74(centre=(384, 384, 384), radius=32, block=32, stride=8, iters=3,
            outdir=DEFAULT_OUTDIR, figures=True, refine=False):
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
                 block=block, stride=stride, iters=iters, outdir=outdir,
                 figures=figures, refine=refine)
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
    path = _out(outdir, "pipeline_i74_results.json")
    with open(path, "w") as f:
        json.dump(r, f, indent=2, default=float)
    print(f"wrote {path}")
    return r


def main(iters=3, radius=24, stride=8, block=32, outdir=DEFAULT_OUTDIR,
         figures=True, refine=False):

    print("THREE-STAGE PIPELINE -- phantom validation (known truth)")


    X = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5, CROSS_R1, CROSS_TAPER)
    Xc, xc = _crop(X, tuple(s // 2 for s in CROSS_SHAPE), radius)
    K = raw_kissing(4.0, 0.028, 1.6, 1.6)
    Kc, kc = _crop(K, tuple(s // 2 for s in K.shape), radius)

    out = []
    out.append(run_case("TRUE CROSSING", Xc, xc, "CROSSING",
                        block=block, stride=stride, iters=iters, outdir=outdir,
                        figures=figures, refine=refine))
    out.append(run_case("TANGENCY (kissing)", Kc, kc, "TANGENCY",
                        block=block, stride=stride, iters=iters, outdir=outdir,
                        figures=figures, refine=refine))

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

    path = _out(outdir, "pipeline_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 1->2->3 phantom validation")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--radius", type=int, default=24)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--i74", action="store_true", help="run on real i74 instead")
    ap.add_argument("--centre", nargs=3, type=int, default=[384, 384, 384])
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help="where to write results (default: <repo>/results)")
    ap.add_argument("--refine", action="store_true",
                    help="step 3a: Frank-Oseen smooth the director field first")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the orientation overlay PNGs")
    a = ap.parse_args()
    if a.i74:
        run_i74(centre=tuple(a.centre), radius=a.radius, block=a.block,
                stride=a.stride, iters=a.iters, outdir=a.outdir,
                figures=not a.no_figures, refine=a.refine)
    else:
        res = main(iters=a.iters, radius=a.radius, stride=a.stride,
                   block=a.block, outdir=a.outdir,
                   figures=not a.no_figures, refine=a.refine)
        # Non-zero exit on any FAIL, so the phantom validation is usable as a
        # regression check rather than something you have to read carefully.
        sys.exit(1 if any(r["status"] == "FAIL" for r in res) else 0)
