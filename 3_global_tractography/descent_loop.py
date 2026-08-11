r"""
descent_loop.py -- the Stage 3 optimisation loop, end to end.

    1. gradient descent on the search cone (L, theta, alpha)
    2. scored by MSE between the PREPROCESSED IMAGE and the NORMALISED TDI of
       the tractogram of the post-Stage-2 per-voxel field
    3. everything written out in a form a viewer can open

Both field variants, always
---------------------------
The loop is run twice on one setup: once on a field carrying only `vec1`, once
on the full `vec1 + vec2` field HOST produces. That is not a sensitivity check
bolted on the side -- the second direction is the entire product of Stage 2, and
a loop that only ever ran on the full field could not say whether it was worth
anything. Two descents on identical seeds, identical image, identical objective,
differing only in whether the tracker has a second direction available.

The seeds are SHARED between the two, drawn once from the full field. That is
deliberate and it is the fair comparison: `cone_descent._dirs_at` seeds along
vec2 wherever Stage 2 resolved one, so letting each variant draw its own would
give them different seed COUNTS at different places, and the difference between
the two descents would then be partly a difference in seeding. Sharing them
means the only thing that changes is what the tracker is allowed to follow. A
walker launched along vec2 into a vec1-only field is simply dragged onto vec1,
which is precisely what dropping the second direction costs and is worth seeing.

What this loop is expected to find, stated before it runs
---------------------------------------------------------
The standing result in `cone_objective` is that a density-vs-image term is
minimised by the DEGENERATE no-turn cone, because the HipCT image is bright over
almost all of a tissue chunk and straight tracts cover more of it than turning
ones do. Normalising both sides onto [0, 1] fixes the scale mismatch that made
the previous raw number meaningless (0.327 against a mean(I^2) of 0.314); it
does not fix that. `tdi_mse.decompose` is carried through every report for
exactly this reason, so the run says how much of its own loss is the sparsity
offset rather than leaving it to be inferred from where the descent ended.

Run
---
    python descent_loop.py                        # radius 24, both variants
    python descent_loop.py --radius 20 --seeds 100 --max-iters 8
    python descent_loop.py --sweep-n 5            # + the (L,theta,alpha) volume
    python descent_loop.py --smooth 1.5           # scale-matched variant
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
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import smooth_tracker as ST                                   # noqa: E402
import tdi_mse as TM                                          # noqa: E402
import viz_export as VE                                       # noqa: E402
from cone_descent import stage2, seed, FLOOR_FRAC             # noqa: E402
from gradient_optimise import Problem, descend                # noqa: E402
from param_volume import load_chunk, L_RANGE, THETA_RANGE, ALPHA_RANGE  # noqa: E402
from reconstruction import VOXEL_SIZE_UM                      # noqa: E402

VARIANTS = ("vec1", "vec1+vec2")

# The localized crossing phantom, at the parameters `cone_descent` validates on.
# Separation is wide because that is where Phase 1 is closed; a narrow crossing
# is a stated open limit and would confound "the loop cannot see the second
# direction" with "HOST could not resolve it".
PHANTOM_SEP_DEG = 70.0
PHANTOM_SPACING = 9.0
PHANTOM_WIDTH = 2.0


def load_setup(dataset, centre, radius, verbose=True):
    """(volume, crop_origin, label) for whichever data the loop is run on.

    Two datasets, because on real i74 the two field variants are provably
    IDENTICAL and the comparison the loop exists to make cannot happen there --
    see `main`. The phantom is not a substitute for the real run; it is the only
    place the vec1 / vec1+vec2 machinery can be exercised at all, so it is what
    shows the loop does what it claims when a second direction exists.

    The phantom is passed through the SAME preprocessing the real volume gets
    (`tdi_mse.prepare_image` detects it is not already on [0, 1] and applies the
    clip + rescale), so the objective sees one kind of input in both cases.
    """
    if dataset == "i74":
        vol, origin = load_chunk(centre, radius, verbose=verbose)
        return vol, origin, "i74"

    from localized_phantoms import raw_localized_crossing
    n = 2 * int(radius)
    vol = raw_localized_crossing((n, n, n), PHANTOM_SEP_DEG, PHANTOM_SPACING,
                                 PHANTOM_WIDTH).astype(np.float32)
    if verbose:
        print(f"  localized crossing phantom {vol.shape}, "
              f"separation {PHANTOM_SEP_DEG:g} deg, spacing {PHANTOM_SPACING:g},"
              f" width {PHANTOM_WIDTH:g}")
    return vol, (0, 0, 0), "crossing phantom"


def _bar(done, total, t0, width=26):
    frac = done / max(total, 1)
    n = int(round(frac * width))
    el = time.time() - t0
    eta = (el / max(done, 1)) * (total - done)
    return (f"[{'#' * n}{'-' * (width - n)}] {frac:5.1%} {done}/{total}  "
            f"{el / 60:.1f}m elapsed, ~{eta / 60:.1f}m left")


def field_for_variant(dense, variant):
    """The field the tracker is handed.

    `dense_tracker._fields` returns (vec1, None) for a bare (3, Z, Y, X) array
    and (vec1, vec2) for a DenseField, so the variant is chosen by WHAT IS
    PASSED, not by a flag threaded through the tracker. Nothing inside the
    tracking path has to know which run it is in, which is what stops the two
    from differing anywhere except where they are meant to.
    """
    if variant == "vec1":
        return np.asarray(dense.vec1, float)
    return dense


def sweep(prob, n, verbose=True):
    """The loss on a log-spaced (L, theta, alpha) grid -- the descent's terrain."""
    Ls = np.exp(np.linspace(*np.log(L_RANGE), n))
    ths = np.exp(np.linspace(*np.log(THETA_RANGE), n))
    als = np.exp(np.linspace(*np.log(ALPHA_RANGE), n))
    Z = np.zeros((n, n, n), np.float32)
    tot, t0, done = n ** 3, time.time(), 0
    for i, L in enumerate(Ls):
        for j, th in enumerate(ths):
            for k, al in enumerate(als):
                Z[i, j, k] = prob.f(np.log([L, th, al]))
                done += 1
                if verbose and (done % max(n, 1) == 0 or done == tot):
                    print("      " + _bar(done, tot, t0), flush=True)
    return Ls, ths, als, Z


# ---------------------------------------------------------------------------

def run_variant(variant, vol, dense, s, sd, floor, tissue, outdir, origin,
                L0, theta0, alpha0, max_iters, smooth, sweep_n, tag,
                voxel_um=VOXEL_SIZE_UM, label="i74"):
    """One descent, plus everything written out for it."""
    print("\n" + "-" * 78)
    print(f"  VARIANT: {variant}")
    print("-" * 78)

    field = field_for_variant(dense, variant)
    obj = TM.TdiMse(vol, tissue, smooth=smooth)
    prob = Problem(vol, field, s, sd, floor, objective=obj)

    t0 = time.time()
    f_start = prob.f(np.log([L0, theta0, alpha0]))
    per = time.time() - t0
    print(f"    image already preprocessed: {obj.was_preprocessed}; "
          f"support {int(tissue.sum())} voxels")
    print(f"    one pass {per:.2f} s; start loss {f_start:.6f}; "
          f"descent cap {max_iters} iters (~{8.7 * max_iters * per / 60:.1f} min)"
          + (f" + sweep {sweep_n ** 3} passes "
             f"(~{sweep_n ** 3 * per / 60:.1f} min)" if sweep_n else ""))

    L, th, al, trace = descend(prob, L0=L0, theta0=theta0, alpha0=alpha0,
                               max_iters=max_iters)

    # ---- the tractogram the loss actually scored, at the converged point ---
    pos, dirs, w, ids, n_seed = ST.track_soft(
        vol, field, s, sd, lookahead=L, theta=th, alpha=al, floor=floor,
        quad=prob.quad, return_ids=True)
    lines, wts = ST.as_streamlines(pos, dirs, w, ids, n_seed, n_keep=obj.n_keep)
    rep_end = obj.report(pos, dirs, w)
    tdi_norm = obj.normalised(pos, dirs, w)

    # the same, at the START, so the pictures show what the descent changed
    p0, d0, w0, i0, n0 = ST.track_soft(
        vol, field, s, sd, lookahead=L0, theta=theta0, alpha=alpha0,
        floor=floor, quad=prob.quad, return_ids=True)
    rep_start = obj.report(p0, d0, w0)
    lines0, wts0 = ST.as_streamlines(p0, d0, w0, i0, n0, n_keep=obj.n_keep)

    print(f"\n    start   loss {rep_start['mse']:.6f}  "
          f"offset {rep_start['offset_frac']:.1%}  r {rep_start['pearson_r']:+.4f}  "
          f"coverage {rep_start['coverage']:.2%}  {len(lines0)} tracts")
    print(f"    end     loss {rep_end['mse']:.6f}  "
          f"offset {rep_end['offset_frac']:.1%}  r {rep_end['pearson_r']:+.4f}  "
          f"coverage {rep_end['coverage']:.2%}  {len(lines)} tracts")

    # ---- exports -----------------------------------------------------------
    print("\n    [export]")
    paths = VE.export_volumes(obj.image, tdi_norm, outdir, tag,
                              crop_origin=origin, voxel_um=voxel_um,
                              support=tissue)
    paths["trk"] = VE.save_trk(lines, vol.shape,
                               os.path.join(outdir, f"tracts_{tag}.trk"),
                               crop_origin=origin, voxel_um=voxel_um,
                               weights=wts)
    paths["trk_start"] = VE.save_trk(
        lines0, vol.shape, os.path.join(outdir, f"tracts_{tag}_start.trk"),
        crop_origin=origin, voxel_um=voxel_um, weights=wts0)
    paths["figure"] = VE.comparison_figure(
        obj.image, tdi_norm, tissue,
        os.path.join(outdir, f"compare_{tag}.png"),
        title=f"{label} -- {variant} field, converged cone "
              f"(L={L:.2f} vox, theta={np.degrees(th):.1f} deg, "
              f"alpha={np.degrees(al):.1f} deg)",
        info=rep_end)
    print(f"      figure    {paths['figure']}")
    paths["html"] = VE.descent_3d_html(
        trace, os.path.join(outdir, f"descent_{tag}.html"),
        title=f"{label} {variant}: descent on MSE(preprocessed image, "
              f"normalised TDI)")
    print(f"      html      {paths['html']}")

    out = dict(variant=variant, L=L, theta=th, alpha=al,
               theta_deg=float(np.degrees(th)),
               alpha_deg=float(np.degrees(al)),
               L0=L0, theta0=theta0, alpha0=alpha0,
               trace=trace, n_passes=prob.n_eval, seconds_per_pass=per,
               start=rep_start, end=rep_end,
               n_tracts=len(lines), n_tracts_start=len(lines0),
               paths={k: os.path.basename(v) for k, v in paths.items()})

    # ---- the terrain, optionally -------------------------------------------
    if sweep_n:
        print(f"\n    [sweep] {sweep_n}^3 over (L, theta, alpha)")
        Ls, ths, als, Z = sweep(prob, sweep_n)
        k = np.unravel_index(int(np.argmin(Z)), Z.shape)
        pv = VE.save_param_volume(
            Ls, ths, als, Z, os.path.join(outdir, f"lossvol_{tag}.nii.gz"),
            descrip=b"MSE(preprocessed image, normalised TDI); "
                    b"x=log10(L vox) y=log10(theta deg) z=log10(alpha deg)")
        print(f"      lossvol   {pv}")
        print(f"      grid min {Z[k]:.6f} at L={Ls[k[0]]:.3f} "
              f"theta={np.degrees(ths[k[1]]):.2f}d "
              f"alpha={np.degrees(als[k[2]]):.2f}d")
        edge = (k[0] in (0, sweep_n - 1) or k[1] in (0, sweep_n - 1)
                or k[2] in (0, sweep_n - 1))
        if edge:
            print("      -> the grid minimum is ON A BOUNDARY: the loss has no "
                  "interior optimum in at least one parameter here.")
        out["sweep"] = dict(
            n=sweep_n, L=[float(v) for v in Ls],
            theta_deg=[float(v) for v in np.degrees(ths)],
            alpha_deg=[float(v) for v in np.degrees(als)],
            grid_min=dict(loss=float(Z[k]), L=float(Ls[k[0]]),
                          theta_deg=float(np.degrees(ths[k[1]])),
                          alpha_deg=float(np.degrees(als[k[2]])),
                          on_boundary=bool(edge)),
            loss_min=float(Z.min()), loss_max=float(Z.max()))
        out["paths"]["lossvol"] = os.path.basename(pv)

    return out


# ---------------------------------------------------------------------------

def main(centre=384, radius=24, n_seeds=150, seeding="fa", L0=4.0,
         theta0=0.15, alpha0=0.15, max_iters=12, smooth=0.0, sweep_n=0,
         variants=VARIANTS, dataset="i74", outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results",
                                    "loop" if dataset == "i74"
                                    else f"loop_{dataset}")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 78)
    print("STAGE 3 LOOP -- descend the cone on MSE(preprocessed image, "
          "normalised TDI)")
    print("=" * 78)

    t_setup = time.time()
    vol, origin, label = load_setup(dataset, centre, radius)
    print(f"  [setup] volume loaded in {time.time() - t_setup:.1f} s; "
          f"range [{vol.min():.3f}, {vol.max():.3f}]", flush=True)
    t0 = time.time()
    dense, _D_obs, tissue, FA, _sc = stage2(vol, verbose=False)
    print(f"  [setup] stage 1-2 in {(time.time() - t0) / 60:.1f} min; "
          f"{dense.summary()}", flush=True)

    # ONE seed set, from the FULL field, shared by both variants -- see the
    # module docstring for why they are not drawn per variant.
    s, sd = seed(dense, tissue, FA, seeding, n=n_seeds)
    floor = FLOOR_FRAC * float(vol.max())
    n2 = int((dense.n_dirs >= 2).sum())
    print(f"  {len(s)} shared seeds ({seeding}); {int(tissue.sum())} tissue "
          f"voxels; {n2} two-direction voxels; floor {floor:.4f}")
    print(f"  setup total {(time.time() - t_setup) / 60:.1f} min")

    # If Stage 2 resolved no second direction anywhere, the two variants are the
    # SAME FIELD and running both is not a comparison. Say so up front rather
    # than letting two identical result rows be read as agreement between them.
    if n2 == 0 and len(variants) > 1:
        print("\n  NOTE no voxel carries a second direction, so 'vec1' and "
              "'vec1+vec2' are the\n       identical field here and the two "
              "descents must return identical numbers --\n       which reads "
              "exactly like agreement between them, hence this warning.\n"
              "       `DenseField.build` writes vec2 only for "
              "bucket=='candidate' patches and\n       `triage` grants that only "
              "to the label 'crossing'. Such patches are RARE and\n       "
              "LOCATION-specific on i74: measured zero at centres 300/450/520/"
              "600 (radius 24),\n       450 (radius 32) and 300 (radius 40), but "
              "8332 at centre 384 radius 32 --\n       a BIGGER chunk does not "
              "help. Use --centre 384 --radius 32 for a real\n       comparison, "
              "or --dataset crossing.")

    runs = {}
    for v in variants:
        tag = v.replace("+", "_")
        runs[v] = run_variant(v, vol, dense, s, sd, floor, tissue, outdir,
                              origin, L0, theta0, alpha0, max_iters, smooth,
                              sweep_n, tag, label=label)

    # ---- the comparison ----------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  {'variant':>12} {'L':>7} {'theta':>9} {'alpha':>9} "
          f"{'loss':>10} {'offset':>8} {'r':>9} {'cover':>7} {'tracts':>7}")
    for v, r in runs.items():
        e = r["end"]
        print(f"  {v:>12} {r['L']:7.2f} {r['theta_deg']:8.2f}d "
              f"{r['alpha_deg']:8.2f}d {e['mse']:10.6f} "
              f"{e['offset_frac']:7.1%} {e['pearson_r']:+9.4f} "
              f"{e['coverage']:6.2%} {r['n_tracts']:7d}")

    # The verdict the loop owes the reader, from the loop's own numbers.
    worst = max(r["end"]["offset_frac"] for r in runs.values())
    moved = {v: 100.0 * abs(r["L"] - r["L0"]) / r["L0"] for v, r in runs.items()}
    print(f"\n  the descent moved L by "
          + ", ".join(f"{v} {m:.1f}%" for v, m in moved.items()))
    if worst > 0.5:
        print(f"  WARNING offset is {worst:.0%} of the loss at the optimum: this "
              f"objective is\n          dominated by how much of the tissue the "
              f"tracts COVER, not by where they\n          went. See "
              f"tdi_mse.decompose and the standing result in cone_objective.")
    print("=" * 78)

    out = dict(dataset=dataset, label=label, centre=centre, radius=radius,
               shape=list(vol.shape),
               crop_origin=list(origin), voxel_um=VOXEL_SIZE_UM,
               n_seeds=int(len(s)), seeding=seeding, smooth=smooth,
               floor=floor, n_two_dir_voxels=n2,
               tissue_voxels=int(tissue.sum()),
               start=dict(L=L0, theta=theta0, alpha=alpha0),
               runs=runs)
    p = os.path.join(outdir, "descent_loop.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote {p}")
    print(f"  all outputs in {outdir}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--centre", type=int, default=384)
    ap.add_argument("--radius", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--seeding", choices=("fa", "random"), default="fa")
    ap.add_argument("--L0", type=float, default=4.0)
    ap.add_argument("--theta0", type=float, default=0.15)
    ap.add_argument("--alpha0", type=float, default=0.15)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="Gaussian on the TDI before log10, in voxels "
                         "(scale matching; 0 = off)")
    ap.add_argument("--sweep-n", type=int, default=0,
                    help="grid points per axis for the loss volume; "
                         "0 = skip (costs n^3 tracking passes)")
    ap.add_argument("--variant", choices=VARIANTS + ("both",), default="both")
    ap.add_argument("--dataset", choices=("i74", "crossing"), default="i74",
                    help="'crossing' is the localized crossing phantom, the "
                         "only setup where vec2 exists at all")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    main(centre=a.centre, radius=a.radius, n_seeds=a.seeds, seeding=a.seeding,
         L0=a.L0, theta0=a.theta0, alpha0=a.alpha0, max_iters=a.max_iters,
         smooth=a.smooth, sweep_n=a.sweep_n,
         variants=VARIANTS if a.variant == "both" else (a.variant,),
         dataset=a.dataset, outdir=a.outdir)
