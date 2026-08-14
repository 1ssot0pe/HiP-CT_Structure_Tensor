r"""
sigma_rho_match.py -- Stage 4: which single (sigma, rho) best reproduces the
refined field?

The question
------------
Stage 3 now outputs a REFINED orientation field: the Stage-2 field relaxed by
the optimised search cone (`cone_field_smooth`). That field is the product of
the whole pipeline -- scale certification, HOST routing, tractography-driven
smoothing. This stage asks what it cost:

    is there a PLAIN structure tensor, at some single (sigma, rho), that gives
    the same directions?

If yes, the pipeline's answer is reproducible by one tensor call and the
machinery bought a scale choice rather than a better field. If no, the residual
angular gap is what the pipeline added. Either answer is worth having, and the
number that separates them is a mean angular error in degrees.

The search range comes from Stage 1, not from a guess
-----------------------------------------------------
The global Lindeberg pick (`sigma_global`, `rho_global`) is the START point.
The RANGE is bounded by the per-block picks Stage 1 already made: the min and
max of `sigma_map` / `rho_map` over blocks that certified. That is the interval
the tissue itself demanded somewhere in the volume, so a sweep outside it would
be testing scales Stage 1 had already rejected everywhere, and a sweep narrower
than it could not reach a scale some region needed.

Where the maps are absent (a `global` run), the range falls back to a factor
either side of the global pick, and that fallback is REPORTED rather than
silently substituted -- a range that came from a constant is a different claim
from one that came from the data.

Comparison is on DIRECTORS
--------------------------
e3 has no sign, so the angle between two fields is taken on |cos|. Without the
absolute value a field that merely re-signed itself would score ~90 deg, and
the eigensolver picks that sign arbitrarily per voxel. The error is summarised
by the MEDIAN as well as the mean: an angular error distribution over a real
volume has a tail from isotropic voxels where no direction is meaningful, and a
mean alone is dominated by it.

Run
---
    python sigma_rho_match.py --run ../results/loop_relax_r22 --n 7
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification"),
           os.path.join(_ROOT, "3_global_tractography")):
    sys.path.insert(0, _p)

from tensor import tensor_field                        # noqa: E402
from select_scales import select_scales                # noqa: E402
from reconstruction import tissue_mask, VOXEL_SIZE_UM  # noqa: E402
from nifti_io import nifti_affine, save_nifti          # noqa: E402

EPS = 1e-12
FALLBACK_FACTOR = 2.0      # range = pick / f .. pick * f when no maps exist


# ---------------------------------------------------------------------------

def scale_range(scales, factor=FALLBACK_FACTOR):
    """(sigma_lo, sigma_hi, rho_lo, rho_hi, provenance) from Stage 1.

    Bounded by the per-block picks where they exist. NaN blocks are the ones
    that did not certify, and they are dropped rather than treated as zeros --
    a block with no pick is not a block that picked 0.
    """
    sm, rm = scales.get("sigma_map"), scales.get("rho_map")
    if sm is not None and rm is not None:
        s, r = np.asarray(sm, float), np.asarray(rm, float)
        s, r = s[np.isfinite(s)], r[np.isfinite(r)]
        # A DEGENERATE spread is not a range. A 44^3 chunk tiled at block=32
        # holds ONE block per axis, so the maps certify a single pair and
        # min == max -- a "range" of zero width that would sweep the same
        # scale n^2 times and report it as the optimum by default. Needing
        # >= 2 distinct values is what makes the range a statement about
        # regional variation rather than an artefact of the tiling.
        if (s.size >= 2 and r.size >= 2
                and s.max() - s.min() > 1e-9 and r.max() - r.min() > 1e-9):
            return (float(s.min()), float(s.max()),
                    float(r.min()), float(r.max()),
                    f"per-block Lindeberg maps ({s.size} certified blocks)")
        if s.size:
            sgd, rgd = float(np.median(s)), float(np.median(r))
            return (sgd / factor, sgd * factor, rgd / factor, rgd * factor,
                    f"FALLBACK: only {s.size} certified block(s), so the "
                    f"per-block spread is degenerate; using the block median "
                    f"/{factor:g}..x{factor:g}")
    sg, rg = float(scales["sigma_global"]), float(scales["rho_global"])
    return (sg / factor, sg * factor, rg / factor, rg * factor,
            f"FALLBACK: no per-block maps, using global pick /{factor:g}..x{factor:g}")


def angular_error(a, b, mask=None):
    """Angle between two (3, Z, Y, X) DIRECTOR fields, in degrees.

    Returns (median, mean, p95, n) over `mask`. |cos| because e3 is a director:
    v and -v are the same orientation and the solver's sign is arbitrary.
    """
    A = np.asarray(a, float)
    B = np.asarray(b, float)
    A = A / (np.linalg.norm(A, axis=0) + EPS)
    B = B / (np.linalg.norm(B, axis=0) + EPS)
    c = np.abs((A * B).sum(axis=0))
    ok = np.isfinite(c)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    if not ok.any():
        return float("nan"), float("nan"), float("nan"), 0
    ang = np.degrees(np.arccos(np.clip(c[ok], 0.0, 1.0)))
    return (float(np.median(ang)), float(ang.mean()),
            float(np.percentile(ang, 95)), int(ok.sum()))


def sweep(vol, target, mask, sigmas, rhos, verbose=True):
    """Mean/median angular error of a plain e3 field against `target`.

    One `tensor_field` call per (sigma, rho) -- deliberately NOT the Stage 1-2-3
    pipeline. The comparison is against the simplest thing that could produce a
    direction field, which is what makes "the pipeline bought X degrees" a
    meaningful statement.
    """
    E = np.full((len(sigmas), len(rhos)), np.nan)
    M = np.full((len(sigmas), len(rhos)), np.nan)
    best, t0, done = None, time.time(), 0
    for i, sg in enumerate(sigmas):
        for j, rh in enumerate(rhos):
            t = tensor_field(vol, float(sg), float(rh), invariants=False)
            med, mean, p95, n = angular_error(t["e3"], target, mask)
            M[i, j], E[i, j] = med, mean
            if best is None or med < best["median"]:
                best = dict(sigma=float(sg), rho=float(rh), median=med,
                            mean=mean, p95=p95, n=n, e3=t["e3"])
            done += 1
            if verbose:
                el = time.time() - t0
                tot = len(sigmas) * len(rhos)
                print(f"    [{done:3d}/{tot}] sigma={sg:5.3f} rho={rh:5.3f}  "
                      f"median {med:6.3f} deg  mean {mean:6.3f}  "
                      f"({el / 60:.1f}m elapsed, "
                      f"~{(el / done) * (tot - done) / 60:.1f}m left)",
                      flush=True)
    return sigmas, rhos, M, E, best


# ---------------------------------------------------------------------------

def main(run_dir, variant="vec1", n=7, mode="adaptive", outdir=None,
         verbose=True):
    outdir = outdir or os.path.join(_ROOT, "results", "stage4")
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(run_dir, "descent_loop.json")) as f:
        run = json.load(f)
    d = run["runs"][variant]

    print("=" * 78)
    print("STAGE 4 -- the single (sigma, rho) closest to the refined field")
    print("=" * 78)

    # ---- rebuild the volume and the Stage-3 refined field ------------------
    import descent_loop as DL
    import cone_field_smooth as CFS
    from cone_descent import stage2

    vol, origin, label = DL.load_setup(run["dataset"], run["centre"],
                                       run["radius"], verbose=False)
    print(f"  {label} {vol.shape} at centre {run['centre']}, "
          f"radius {run['radius']}")
    dense, _D, tissue, _FA, scales = stage2(vol, mode=mode, verbose=False)

    L, th, al = d["L"], d["theta"], d["alpha"]
    # PREFER THE SAVED FIELD. Stage 3 accumulates -- each iteration relaxes the
    # PREVIOUS field -- so recomputing `smooth_field(field_0, L*, theta*, alpha*)`
    # would reproduce only the final step, not the accumulation, and would
    # silently compare against a field the run never produced. Recomputation is
    # the fallback for runs made before the field was saved, and it says so.
    npz = os.path.join(run_dir, f"refined_field_{variant.replace('+', '_')}.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        refined = type("F", (), {})()
        refined.vec1 = z["vec1"]
        # NO ZERO-FILL. A missing or all-zero vec2 means the field has no second
        # direction, and substituting zeros produced a full-size NIfTI of zeros
        # that reads on disk exactly like a computed result. `vec2` is left as
        # None and every consumer below has to handle that explicitly, so the
        # absence is reported instead of being rendered as an answer.
        refined.vec2 = z["vec2"] if "vec2" in z else None
        src = f"saved accumulated field ({os.path.basename(npz)})"
    else:
        refined = CFS.smooth_field(dense, L, th, al)
        src = ("RECOMPUTED as one smoothing from field_0 -- the run saved no "
               "field, so this is only the LAST step, not the accumulation")
    print(f"  reference field: {src}")

    # Why a second direction is absent matters, and the three causes are
    # distinguishable from what the run recorded:
    #   * the run tracked vec1 only, so vec2 was never carried;
    #   * the chunk had no `crossing` patches, so Stage 2 produced none;
    #   * it existed and was demoted.
    v2 = getattr(refined, "vec2", None)
    n_v2 = 0 if v2 is None else int((np.linalg.norm(np.asarray(v2, float),
                                                   axis=0) > EPS).sum())
    has_v2 = n_v2 > 0
    if not has_v2:
        why = ("the run used --variant vec1, so no second direction was carried"
               if variant == "vec1" else
               f"Stage 2 produced none on this chunk "
               f"(n_two_dir_voxels={run.get('n_two_dir_voxels', '?')})")
        print(f"  NO SECOND DIRECTION in the refined field -- {why}.")
        print(f"    vec2 is NOT written. A second direction requires HOST to "
              f"label a patch 'crossing'; superposition and demotion only "
              f"REMOVE one, they never create it.")
    else:
        print(f"  refined field carries a second direction at {n_v2} voxels")
    oriented = np.asarray(dense.n_dirs) >= 1
    print(f"  Stage 3 cone: L={L:.3f} vox, theta={np.degrees(th):.3f} deg, "
          f"alpha={DL.fmt_alpha(al, run.get('turn', 'sample'))}")
    mv = CFS.angular_change(refined.vec1, dense.vec1, mask=oriented)
    print(f"  refined field moved Stage 2 by median {mv['median']:.3f} deg "
          f"(mean {mv['mean']:.3f}, p95 {mv['p95']:.3f})")

    # ---- the search range, from Stage 1 ------------------------------------
    s_lo, s_hi, r_lo, r_hi, prov = scale_range(scales)
    sg, rg = float(scales["sigma_global"]), float(scales["rho_global"])
    print(f"\n  Stage 1 global pick: sigma={sg:.4f}  rho={rg:.4f}")
    print(f"  search range from {prov}")
    print(f"    sigma {s_lo:.4f} .. {s_hi:.4f}")
    print(f"    rho   {r_lo:.4f} .. {r_hi:.4f}")
    sigmas = np.exp(np.linspace(np.log(s_lo), np.log(s_hi), n))
    rhos = np.exp(np.linspace(np.log(r_lo), np.log(r_hi), n))

    # ---- sweep against the REFINED field (vec1 + vec2 product) -------------
    print(f"\n  [sweep] {n}x{n} plain structure tensors vs the REFINED field")
    _s, _r, M, E, best = sweep(vol, refined.vec1, oriented, sigmas, rhos)

    # ---- and against the raw Stage-2 field, as the control -----------------
    print(f"\n  [control] the same sweep vs the STAGE-2 field (vec1 only)")
    _s2, _r2, M2, E2, best2 = sweep(vol, dense.vec1, oriented, sigmas, rhos)

    A = nifti_affine(origin, VOXEL_SIZE_UM)
    paths = {
        "refined_vec1": save_nifti(np.moveaxis(np.asarray(refined.vec1), 0, -1),
                                   os.path.join(outdir, "refined_vec1.nii.gz"),
                                   A, vector=True, dtype=np.float32),
        "best_simple_e3": save_nifti(np.moveaxis(best["e3"], 0, -1),
                                     os.path.join(outdir, "best_simple_e3.nii.gz"),
                                     A, vector=True, dtype=np.float32),
    }
    # Written ONLY if there is one. A stale zero-volume from a previous run
    # would otherwise sit next to the new outputs looking current.
    stale = os.path.join(outdir, "refined_vec2.nii.gz")
    if has_v2:
        paths["refined_vec2"] = save_nifti(
            np.moveaxis(np.asarray(v2, float), 0, -1), stale, A, vector=True,
            dtype=np.float32)
    elif os.path.exists(stale):
        os.remove(stale)
        print(f"  removed stale {os.path.basename(stale)} from an earlier run")

    print("\n" + "=" * 78)
    print("  RESULT")
    print(f"    closest simple (sigma, rho) to the REFINED field: "
          f"sigma={best['sigma']:.4f}  rho={best['rho']:.4f}")
    print(f"      median angular error {best['median']:.3f} deg  "
          f"(mean {best['mean']:.3f}, p95 {best['p95']:.3f}, "
          f"n={best['n']})")
    print(f"    closest simple (sigma, rho) to the STAGE-2 field: "
          f"sigma={best2['sigma']:.4f}  rho={best2['rho']:.4f}")
    print(f"      median angular error {best2['median']:.3f} deg  "
          f"(mean {best2['mean']:.3f})")
    on_edge = [nm for nm, v, lo, hi in
               (("sigma", best["sigma"], sigmas[0], sigmas[-1]),
                ("rho", best["rho"], rhos[0], rhos[-1]))
               if abs(v - lo) < 1e-9 or abs(v - hi) < 1e-9]
    if on_edge:
        print(f"    NOTE the best pair is ON THE RANGE EDGE in "
              f"{', '.join(on_edge)} -- Stage 1's per-block spread did not "
              f"bracket it, so the true optimum may lie outside.")
    print("=" * 78)

    out = dict(run_dir=run_dir, variant=variant, label=label,
               shape=list(vol.shape), centre=run["centre"],
               radius=run["radius"], n=n, mode=mode,
               stage3=dict(L=L, theta=th, alpha=al,
                           turn=run.get("turn", "sample"),
                           metric=run.get("metric", "mse"),
                           moved_median_deg=mv["median"],
                           moved_mean_deg=mv["mean"]),
               stage1=dict(sigma_global=sg, rho_global=rg, provenance=prov,
                           sigma_lo=s_lo, sigma_hi=s_hi,
                           rho_lo=r_lo, rho_hi=r_hi),
               sigmas=[float(v) for v in sigmas],
               rhos=[float(v) for v in rhos],
               median_deg=M.tolist(), mean_deg=E.tolist(),
               median_deg_vs_stage2=M2.tolist(),
               best_vs_refined={k: v for k, v in best.items() if k != "e3"},
               best_vs_stage2={k: v for k, v in best2.items() if k != "e3"},
               best_on_range_edge=on_edge,
               # Explicit, so a consumer reading the JSON cannot infer "vec2
               # exists and is zero" from a missing file.
               has_vec2=bool(has_v2), n_vec2_voxels=int(n_v2),
               paths={k: os.path.basename(v) for k, v in paths.items()})
    p = os.path.join(outdir, "stage4_sigma_rho.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote {p}")
    for k, v in paths.items():
        print(f"    {k:16} {v}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", default=os.path.join(_ROOT, "results",
                                                  "loop_relax_r22"),
                    help="a completed descent_loop output directory")
    ap.add_argument("--variant", default="vec1")
    ap.add_argument("--n", type=int, default=7, help="grid points per axis")
    ap.add_argument("--mode", choices=("adaptive", "global"), default="adaptive")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    main(a.run, variant=a.variant, n=a.n, mode=a.mode, outdir=a.outdir)
