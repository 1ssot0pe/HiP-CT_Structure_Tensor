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
    Stage 2->3  DenseField: e3 at every voxel, HOST's peaks painted over it
       |        where the gate routed. ONE DIRECTION PER VOXEL, plus a second
       |        wherever Stage 2 resolved one. This is what Stage 3 tracks
       |        through -- see dense_field.py for why it is not the patch list.
       v
    Stage 3  cone-search tracts -> D_rec -> L -> update (lookahead, theta)
             -> converged parameters, loss trace, per-junction S and verdict

`--tracker patch` restores the old nearest-HOST-patch walker, so the two can be
compared on identical data rather than the change being asserted.

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
import dense_tracker as DT
import dense_optimise as DO
from dense_field import DenseField
# Step 3a's Frank-Oseen smoother is opt-in (--refine) and its module has been
# archived, so it is imported only if asked for rather than at start-up. An
# unconditional import here is what currently stops the pipeline from starting
# at all, for a feature that is off by default.
from field_export import export_field, export_difference
from render_field import render_field
from sigma_rho_gated_field import build_gated_field, DEFAULT_STRIDE
from reconstruction import VOXEL_SIZE_UM, tissue_mask
from nifti_io import nifti_affine, save_nifti, save_vector_field
from tensor import tensor_field, field_for
from localized_phantoms import (raw_localized_crossing, raw_kissing,
                                CROSS_SHAPE, CROSS_R1, CROSS_TAPER)

sys.path.insert(0, os.path.join(_ROOT, "1_sigma_rho_selection"))
from select_scales import select_scales, save_scales, paint_maps, summary as scale_summary


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


def steps_0_to_2(vol, tag, outdir, crop_origin=(0, 0, 0), voxel_um=VOXEL_SIZE_UM,
                 mode="adaptive", block=32, export_both=True, verbose=True):
    """Steps 0-2: mask, certify scales, ONE dense tensor, write the maps.

    Returns (scales, e3, lam, tissue) for the rest of the run to consume, so
    every later stage reads the same decomposition instead of making its own.

    Both e3 maps are written whenever `export_both`, not just the one the run
    uses. They are the deliverable that answers "what would the other choice
    have looked like" -- the adaptive-versus-global comparison was previously
    only reachable by running `sigma_rho_adaptive.py` by hand on a different
    region, so the two pictures were never of the same data.
    """
    A = nifti_affine(crop_origin, voxel_um)
    tissue = tissue_mask(vol)
    if verbose:
        print(f"  [step 0] tissue {int(tissue.sum())}/{tissue.size} voxels "
              f"({tissue.mean():.1%})")
    save_nifti(vol, _out(outdir, f"raw_{tag}.nii.gz"), A, dtype=np.float32)

    wanted = ("adaptive", "global") if export_both else (mode,)
    picked, fields = {}, {}
    for m in wanted:
        if verbose:
            print(f"  [step 1] certifying scales ({m}) ...")
        sc = select_scales(vol, mode=m, block=block, verbose=verbose)
        picked[m] = sc
        save_scales(sc, _out(outdir, f"scales_{tag}_{m}.json"))
        if verbose:
            print(f"  [step 2] dense tensor at the {m} scale ...")
        t = field_for(vol, sc, invariants=False)
        fields[m] = t
        # Name carries the scale, so a map on disk says what produced it.
        if m == "global":
            nm = (f"e3_{tag}_global_s{sc['sigma_global']:g}"
                  f"_r{sc['rho_global']:g}.nii.gz")
        else:
            nm = f"e3_{tag}_adaptive.nii.gz"
            sig, rho = paint_maps(sc, vol.shape)
            save_nifti(sig, _out(outdir, f"sigma_map_{tag}.nii.gz"), A,
                       dtype=np.float32)
            save_nifti(rho, _out(outdir, f"rho_map_{tag}.nii.gz"), A,
                       dtype=np.float32)
        p = save_vector_field(t["e3"], _out(outdir, nm), A)
        if verbose:
            print(f"           {scale_summary(sc)}")
            print(f"           wrote {os.path.basename(p)}")

    scales = picked[mode]
    t = fields[mode]
    return scales, t["e3"], t["lam"], tissue


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


def run_case(name, vol, centre, truth, mode="adaptive", block=32,
             stride=DEFAULT_STRIDE, iters=3,
             verbose=True, outdir=DEFAULT_OUTDIR, figures=True, refine=False,
             crop_origin=(0, 0, 0), voxel_um=VOXEL_SIZE_UM, export=True,
             export_both=True, tracker="cone", max_seeds=DO.MAX_SEEDS,
             alpha_check=False, demote_mode="none",
             align_weight=DO.ALIGN_WEIGHT):
    print(f"\n{'-' * 78}\n{name}   (truth: {truth})   volume {vol.shape}\n{'-' * 78}")
    tag = name.split()[0].lower()

    # ---- Steps 0-2: mask, scales, one dense tensor --------------------------
    scales, e3, lam, tissue = steps_0_to_2(
        vol, tag, outdir, crop_origin=crop_origin, voxel_um=voxel_um,
        mode=mode, block=block, export_both=export_both, verbose=verbose)

    # ---- Steps 3-5: the routing gate, then HOST -----------------------------
    # e3 is handed in, so the gate reads the SAME field Stage 3 will follow.
    gated = build_gated_field(vol, scales=scales, e3_xyz=e3[::-1], stride=stride,
                              verbose=verbose)
    mask = gated.routing_mask()
    gated.save(_out(outdir, f"unified_field_{tag}.npz"))
    print(f"  [stage 2] routing mask: {int(mask.sum())}/{gated.n} patches to HOST")

    # ---- Stage 3a: refine the field ----------------------------------------
    if refine:
        try:
            from field_refine import refine as refine_field
        except ImportError:
            sys.path.insert(0, os.path.join(_ROOT, "archive"))
            from field_refine import refine as refine_field
        e3, rinfo = refine_field(e3)
        print(f"  [stage 3a] Frank-Oseen refine: {rinfo['iterations']} iters, "
              f"E {rinfo['E_initial']:.1f} -> {rinfo['E_final']:.1f}")

    # ---- Stage 3b: crossing vs kissing, then tracts -------------------------
    # THE INPUT TO STAGE 3: one direction per VOXEL. Stage 1's e3 everywhere,
    # overwritten by HOST's peaks inside the patches the gate routed. `host_only`
    # no longer removes single-fibre tissue from the field -- e3 carries it --
    # it only decides who may overwrite e3. See dense_field.py.
    dense = DenseField.build(gated, e3, tissue=tissue, host_only=True)
    print(f"  [stage 3] dense field: {dense.summary()}")

    # The patch list is still what the superposition metric is defined on: S is
    # measured at a junction, from the two peak directions HOST resolved there.
    field, junctions, I0 = S3.field_from_gated(gated, vol, e3=e3, host_only=True)
    print(f"  [stage 3] {len(field)} HOST patches, {len(junctions)} "
          f"junction(s), I_0={I0:.3f}")

    # The INITIAL field, before any demotion. Snapshotted here so the exported
    # "before" is the state Stage 2 handed over, not a view onto an object the
    # demotion below mutates in place.
    gated_initial = gated.copy()
    F_before = None
    if export:
        _p, F_before = export_field(gated_initial, e3, outdir, f"{tag}_initial",
                                    crop_origin=crop_origin, voxel_um=voxel_um,
                                    tissue=tissue, host_only=True, dense=dense)

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
        # Export the optimised state anyway, so the before/after pair always
        # exists. With nothing to demote it is identical to the initial field,
        # which is the correct answer here rather than a missing file.
        if export:
            export_field(gated, e3, outdir, f"{tag}_optimised",
                         crop_origin=crop_origin, voxel_um=voxel_um,
                         tissue=tissue, host_only=True, dense=dense)
        return dict(name=name, truth=truth, skipped=True, status=status,
                    reason=reason, verdict=None, figure=fig_path, n_demoted=0,
                    correct=None if status == "UNSCORED" else status == "PASS",
                    n_patches=gated.n, n_routed=int(mask.sum()),
                    tracker=tracker, n_oriented=dense.n_valid,
                    n_two_direction=dense.n_multi)

    for j in junctions:
        print(f"    junction {j['centre']}: I_c={j['I_c']:.2f} I_a={j['I_a']:.2f} "
              f"I_b={j['I_b']:.2f}  S={j['S']:.3f} -> {j['verdict']}")

    # D_obs from the SAME eigenvalues that produced e3 -- not a second tensor at
    # a second scale, which is what `anisotropy_target(vol)` used to do.
    D_obs, _tissue_all = S3.anisotropy_target(vol, lam=lam, tissue=tissue)

    # Score only where a streamline can actually go. For the PATCH tracker that
    # is a neighbourhood of the HOST patches -- `integrate` stops when no patch
    # lies within assoc_radius, so most of `tissue` is unreachable by
    # construction, and including it measured the tracts' absence from forbidden
    # voxels: a term constant in theta that swamped the rest (loss_sensitivity).
    # The CONE tracker walks a dense field and has no forbidden region, so the
    # same restriction would now throw away most of the evidence instead of
    # correcting a bias.
    if tracker == "cone":
        support = tissue
    else:
        support = tissue & S3.host_support(field, vol.shape,
                                           float(S3.THETA0["assoc_radius"]))
    D_obs = S3.renormalise(D_obs, support)
    print(f"  [stage 3] scored support {int(support.sum())} voxels "
          f"({support.sum() / max(int(tissue.sum()), 1):.1%} of tissue)")

    dense_optimised, alpha_info, demote_info = dense, None, None
    evidence_info = None
    if tracker == "cone":
        seeds, sdirs = DT.seeds_from(dense, max_seeds=max_seeds, mask=support)
        print(f"  [stage 3] cone tracker: {len(seeds)} seeds "
              f"({int((dense.n_dirs[tuple(seeds.astype(int).T)] >= 2).sum())} "
              f"on two-direction voxels), alpha={DO.ALPHA:g} fixed")
        theta, trace, info = DO.optimise(vol, dense, D_obs, support, junctions,
                                         seeds, sdirs, n_iter=iters,
                                         verbose=verbose,
                                         align_weight=align_weight)
        if alpha_check:
            alpha_info = DO.alpha_convergence(vol, dense, D_obs, support, theta,
                                              junctions, seeds, sdirs)
        # The demotion changes the FIELD, so its effect is only visible by
        # tracking again at the same parameters over the same seeds.
        if demote_mode in ("s", "both"):
            dense_optimised, L_dem, info_dem, demote_info = \
                DO.demote_and_retrack(vol, dense, D_obs, support, theta,
                                      junctions, seeds, sdirs,
                                      align_weight=align_weight)
            demote_info["L_before"] = float(info["total"])
            demote_info["L_after"] = float(L_dem)

        # ...and the same question asked of the TRACTS rather than of S. Run
        # alongside rather than instead, so the two rules can be compared on
        # identical junctions instead of one replacing the other untested.
        if demote_mode in ("evidence", "both"):
            dense_ev, ev_rows, ev_info = DO.demote_by_evidence(
                vol, dense, D_obs, support, theta, junctions, seeds, sdirs)
            evidence_info = dict(ev_info, junctions=ev_rows)
            if demote_mode == "evidence":
                dense_optimised = dense_ev
    else:
        theta, trace, info = S3.optimise(vol, field, junctions, D_obs, support,
                                         I0, n_iter=iters, verbose=verbose)

    # ---- demotion: superposition says which flagged crossings are kisses ----
    # S below the decision point means the centre never carries two fibres'
    # worth of brightness, so the two peaks HOST found do not both pass
    # through. That patch is relabelled kissing and loses its second
    # direction. AMBIGUOUS is deliberately left alone -- demoting on a verdict
    # the metric itself will not commit to would launder uncertainty into a
    # hard label.
    gated_optimised = gated.copy()
    if demote_mode == "none":
        # Stage 2's routing is taken as given, OVER-CALLS INCLUDED. Deciding
        # kissing from crossing here meant reading a fibre count off the
        # tractogram, and that count is scale-free: junction density divided by
        # the tractogram's own single-fibre density, both of which scale
        # together. It could never have distinguished one fibre from two. So
        # the question is not asked, rather than answered badly.
        n_dem = 0
        print(f"  [demote] disabled -- keeping all {len(junctions)} HOST "
              f"crossing(s), over-calls included")
    else:
        overrides = {j["patch_index"]: "kissing"
                     for j in junctions if j["verdict"] == "TANGENCY"}
        for pi in overrides:
            p = gated_optimised.patches[pi]
            p["label"] = "kissing"
            p["dirs"] = list(p["dirs"])[:1]
            p["n_dirs"] = 1
        n_dem = len(overrides)
        print(f"  [demote] {n_dem}/{len(junctions)} junction(s) demoted "
              f"crossing -> kissing on S < {S3.S_DECISION - S3.S_BAND:g}")

    if export:
        _p, F_after = export_field(gated_optimised, e3, outdir,
                                   f"{tag}_optimised", crop_origin=crop_origin,
                                   voxel_um=voxel_um, tissue=tissue,
                                   host_only=True, dense=dense_optimised)
        if F_before is not None:
            export_difference(F_before, F_after, outdir, tag,
                              crop_origin=crop_origin, voxel_um=voxel_um)
        gated_optimised.save(_out(outdir, f"unified_field_{tag}_optimised.npz"))
        dense_optimised.save(_out(outdir, f"dense_field_{tag}_optimised.npz"))

    verdicts = [j["verdict"] for j in junctions]
    majority = max(set(verdicts), key=verdicts.count)
    status, reason = _status(truth, majority)
    lead = (f"verdict {majority} (no ground truth)" if status == "UNSCORED"
            else f"{status} -- {reason}")
    print(f"  RESULT: {lead};  L {trace[0]['total']:.4f} -> "
          f"{trace[-1]['total']:.4f}")
    fig_path = _figure()
    return dict(name=name, truth=truth, skipped=False, status=status,
                reason=reason, figure=fig_path, n_demoted=n_dem,
                n_patches=gated.n, n_routed=int(mask.sum()),
                junctions=[{k: (v if not isinstance(v, tuple) else list(v))
                            for k, v in j.items() if k != "dirs"}
                           for j in junctions],
                verdict=majority,
                correct=None if status == "UNSCORED" else status == "PASS",
                theta=theta, tracker=tracker,
                L_initial=float(trace[0]["total"]),
                L_final=float(trace[-1]["total"]),
                alpha_check=alpha_info, demotion=demote_info,
                demote_mode=demote_mode, evidence=evidence_info,
                trace=trace, junction_log=info["junctions"])


def run_i74(centre=(384, 384, 384), radius=48, block=32, stride=DEFAULT_STRIDE,
            iters=3, outdir=DEFAULT_OUTDIR, figures=True, refine=False,
            mode="adaptive", export_both=True, tracker="cone",
            max_seeds=DO.MAX_SEEDS, alpha_check=False, demote_mode="none",
            align_weight=DO.ALIGN_WEIGHT):
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
    lo = c[0] - radius
    vol = load_volume(crop=(lo, c[0] + radius),
                      dtype=np.uint8).astype(np.float32)
    # ...which is also why the crop origin is the SAME index on all three axes.
    # Every exported NIfTI carries this offset in its affine, so the maps land
    # on the real position in the 768^3 volume instead of at the world origin.
    crop_origin = (lo, lo, lo)
    print("=" * 78)
    print(f"REAL i74 -- {vol.shape} at {c}, mean {vol.mean():.1f}")
    print(f"  crop origin {crop_origin}, voxel {VOXEL_SIZE_UM:g} um, stride {stride}")
    print("=" * 78)

    r = run_case("I74 REGION", vol, tuple(v // 2 for v in vol.shape), "UNKNOWN",
                 mode=mode, block=block, stride=stride, iters=iters,
                 outdir=outdir, figures=figures, refine=refine,
                 crop_origin=crop_origin, voxel_um=VOXEL_SIZE_UM,
                 export_both=export_both, tracker=tracker,
                 max_seeds=max_seeds, alpha_check=alpha_check,
                 demote_mode=demote_mode, align_weight=align_weight)
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


def main(iters=3, radius=24, stride=DEFAULT_STRIDE, block=32,
         outdir=DEFAULT_OUTDIR, figures=True, refine=False, mode="adaptive",
         export_both=True, tracker="cone", max_seeds=DO.MAX_SEEDS,
         alpha_check=False, demote_mode="none",
         align_weight=DO.ALIGN_WEIGHT):

    print(f"THREE-STAGE PIPELINE -- phantom validation (known truth), "
          f"{tracker} tracker, align_weight={align_weight:g}")


    X = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5, CROSS_R1, CROSS_TAPER)
    Xc, xc = _crop(X, tuple(s // 2 for s in CROSS_SHAPE), radius)
    K = raw_kissing(4.0, 0.028, 1.6, 1.6)
    Kc, kc = _crop(K, tuple(s // 2 for s in K.shape), radius)

    kw = dict(mode=mode, block=block, stride=stride, iters=iters, outdir=outdir,
              figures=figures, refine=refine, export_both=export_both,
              tracker=tracker, max_seeds=max_seeds, alpha_check=alpha_check,
              demote_mode=demote_mode, align_weight=align_weight)
    out = []
    out.append(run_case("TRUE CROSSING", Xc, xc, "CROSSING", **kw))
    out.append(run_case("TANGENCY (kissing)", Kc, kc, "TANGENCY", **kw))

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
    # Default resolved per path below: the phantoms are small, the i74 crop is
    # sized for a map worth exporting. One shared default silently shrank the
    # i74 region back to the phantom's.
    ap.add_argument("--radius", type=int, default=None,
                    help="crop half-width (default: 24 phantom, 48 i74)")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                    help="patch grid spacing; smaller = denser HOST coverage")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--i74", action="store_true", help="run on real i74 instead")
    ap.add_argument("--centre", nargs=3, type=int, default=[384, 384, 384])
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help="where to write results (default: <repo>/results)")
    ap.add_argument("--mode", choices=("adaptive", "global"), default="adaptive",
                    help="scale selection: per-block, or one pair for the crop")
    ap.add_argument("--single-e3", action="store_true",
                    help="export only the e3 map the run uses, not both "
                         "(skips certifying the other mode's scales)")
    ap.add_argument("--refine", action="store_true",
                    help="step 3a: Frank-Oseen smooth the director field first")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the orientation overlay PNGs")
    ap.add_argument("--tracker", choices=("cone", "patch"), default="cone",
                    help="cone: walk the dense per-voxel field (default). "
                         "patch: the old nearest-HOST-patch walker, kept so the "
                         "two can be compared on the same data")
    ap.add_argument("--max-seeds", type=int, default=DO.MAX_SEEDS,
                    help="cone tracker seed budget; a compute cost, not a "
                         "parameter to optimise")
    ap.add_argument("--demote", dest="demote_mode",
                    choices=("none", "s", "evidence", "both"), default="none",
                    help="what retracts a second direction: s = the "
                         "superposition verdict (original), evidence = whichever "
                         "field explains the image better, both = run each and "
                         "compare (default)")
    ap.add_argument("--align-weight", type=float, default=DO.ALIGN_WEIGHT,
                    help="weight on the tract-vs-field alignment term. 0 (the "
                         "default) is the objective as it was; "
                         f"{DO.ALIGN_WEIGHT_VALIDATED:g} is the value validated "
                         "against phantom angular error -- see "
                         "objective_search.py")
    ap.add_argument("--alpha-check", action="store_true",
                    help="re-run the descent at half the cone's angular "
                         "sampling and report whether the optimum moves")
    a = ap.parse_args()
    if a.i74:
        run_i74(centre=tuple(a.centre), radius=a.radius or 48, block=a.block,
                stride=a.stride, iters=a.iters, outdir=a.outdir,
                figures=not a.no_figures, refine=a.refine, mode=a.mode,
                export_both=not a.single_e3, tracker=a.tracker,
                max_seeds=a.max_seeds, alpha_check=a.alpha_check,
                demote_mode=a.demote_mode, align_weight=a.align_weight)
    else:
        res = main(iters=a.iters, radius=a.radius or 24, stride=a.stride,
                   block=a.block, outdir=a.outdir, mode=a.mode,
                   figures=not a.no_figures, refine=a.refine,
                   export_both=not a.single_e3, tracker=a.tracker,
                   max_seeds=a.max_seeds, alpha_check=a.alpha_check,
                   demote_mode=a.demote_mode, align_weight=a.align_weight)
        # Non-zero exit on any FAIL, so the phantom validation is usable as a
        # regression check rather than something you have to read carefully.
        sys.exit(1 if any(r["status"] == "FAIL" for r in res) else 0)
