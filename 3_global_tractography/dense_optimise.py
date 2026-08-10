"""
dense_optimise.py -- Stage 3's loop over the dense field.

    track (lookahead, theta) -> D_rec -> L = mse(D_rec, D_obs) + verdict penalty
      -> coordinate descent on (lookahead, theta) -> demote the tangencies
      -> re-track

What is different from `self_optimising_tractography.optimise`
--------------------------------------------------------------
The loss is the same quantity and the weight fit is the same code. Two things
changed, both consequences of the field being dense:

  * THE SUPPORT IS THE TISSUE MASK. `host_support` existed because streamlines
    could only occupy `assoc_radius` bubbles around the 344 HOST patches, so
    scoring against all of tissue measured, almost entirely, their absence from
    voxels they were forbidden to enter. A dense field has no such forbidden
    region, so that correction is not just unnecessary -- keeping it would now
    discard most of the evidence.

  * THE SEARCHED PARAMETERS ARE THE CONE'S. `step`, `angle_deg`, `seed_density`
    and `assoc_radius` described the nearest-patch walker and have no meaning
    here. What is left is the cone: how far ahead evidence is gathered
    (`lookahead`) and how sharply it may turn (`theta`).

alpha is NOT searched
---------------------
It is the cone's angular sampling, and cost scales as (theta/alpha)^2. It has no
minimum to find: finer is always at least as good and always more expensive, so
a descent on it walks to the edge of whatever ladder it is given and reports
that as a result. It is fixed, and `alpha_convergence` below measures what the
current value costs in loss -- which is the honest version of the question
"should alpha be smaller".

Nor is the seed budget: `weight_fit`'s docstring records why more streamlines
can essentially only lower the residual. Both are compute budgets, set as high
as the runtime allows and held fixed while the shape parameters are searched.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    sys.path.insert(0, _p)

import dense_tracker as DT
from tract_density import rasterise
from weight_fit import design_matrix, fit_weights

EPS = 1e-12

# The cone's shape. 
# Defaults are the operating point `cone_search` found:
# largest traversal of a true crossing at zero bridging of a kiss.
PARAMS0 = dict(lookahead=4.0, theta=0.30)
LADDER = dict(lookahead=[2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0],
              theta=[0.10, 0.15, 0.20, 0.25, 0.30, 0.45, 0.60])

ALPHA = 0.20             # cone angular sampling -- a budget, see the docstring
MAX_SEEDS = 2000         # likewise
FLOOR_FRAC = 0.05        # support below this fraction of vol.max() ends a walk
L1_WEIGHT = 1e-3         # price per unit of tract weight; see fit_weights

# Weight on the alignment term -- how much the tracts are required to agree with
# the field they are following. DEFAULT 0 = the objective as it was, so nothing
# in the pipeline changes until this is set deliberately.
#
# It is here because the data term alone is not merely uninformative about the
# cone, it is ANTI-correlated with tract accuracy. Measured on a curved phantom
# with analytic truth (`theta_validation`, `objective_search`): descending on
# mse(TDI, D_obs) selects a cone that cannot turn and lands 10.6 deg from the
# true tangent, where the best setting on the same grid reaches 3.2 deg and the
# field itself sits at 0.88 deg. Every mse variant tried -- raw, scale-matched,
# and the full NNLS weight fit -- has Spearman rho between -0.37 and -0.55
# against the true angular error. The alignment term alone reaches the oracle
# (regret +0.00 deg).
#
# 4.0 is the smallest weight tested that overcomes the data term: at 0.25 and
# 1.0 the mse still drags the optimum back to a no-turn cone. It is NOT tuned
# against the truth -- three values an order of magnitude apart were tried and
# this is the reported sensitivity, not a fitted number.
#
# NOW THE DEFAULT. The earlier reservation was that this was validated on a
# single-fibre phantom which cannot exercise the density term's job -- counting
# fibres at a junction. That reservation is withdrawn because the fibre-count
# job has been removed from this loop entirely (see VERDICT_PENALTY below), so
# there is no longer a competing claim for the objective to damage. Safety on
# the existing suite was measured separately: phantom validation stays 2/2 and
# the descent lands on the same cone with the term on or off.
ALIGN_WEIGHT = 4.0
ALIGN_WEIGHT_VALIDATED = 4.0

# The junction fibre-count penalty, OFF by default.
#
# It was never able to do what it claimed. n_fibres is junction density divided
# by the single-fibre density measured FROM THE SAME TRACTOGRAM, so numerator
# and denominator scale together and the term is exactly invariant under
# D_rec -> c * D_rec (measured: range 0.000e+00 over c in [0.25, 4]). It can say
# "twice as dense as a single fibre in this tractogram"; it can never say "two
# fibres". Kissing-versus-crossing is not decidable from a scale-free quantity,
# so the question is not asked here any more -- Stage 2's HOST routing is taken
# as given, over-calls included.
VERDICT_PENALTY = 0.0


def _smooth(D):
    """3^3 box mean -- `tract_density.density_at(..., half=1)` for every voxel
    at once. A point read of a splatted density is noisy at the sub-voxel scale,
    and the junction centres are only known to a voxel or two."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(np.asarray(D, float), size=3, mode="nearest")


def loss(vol, field, D_obs, support, params, junctions, seeds, seed_dirs,
         alpha=ALPHA, floor=None, l1=L1_WEIGHT,
         verdict_penalty=VERDICT_PENALTY, align_weight=ALIGN_WEIGHT):
    """L = mse(fitted D_rec, D_obs) + verdict penalty + align_weight * (1 - align).

    The third term is off by default (see ALIGN_WEIGHT). With it at zero this is
    exactly the loss it always was.

    `seeds` and `seed_dirs` are passed IN, fixed for the whole descent. Drawing
    them inside would make two evaluations of the same parameters differ, and a
    coordinate descent cannot distinguish that noise from a real improvement.
    """
    floor = FLOOR_FRAC * float(vol.max()) if floor is None else float(floor)
    paths = DT.track(vol, field, seeds, seed_dirs, alpha=alpha,
                     lookahead=float(params["lookahead"]),
                     theta=float(params["theta"]), floor=floor)

    d_obs = D_obs[support]
    Phi = design_matrix(paths, vol.shape, support, step=DT.STEP)
    w, fit = fit_weights(Phi, d_obs, l1=l1)
    data_term = float(np.mean((Phi @ w - d_obs) ** 2)) if Phi.shape[1] else float(
        np.mean(d_obs ** 2))
    D_rec = rasterise(paths, vol.shape, step=DT.STEP, weights=w)

    # Junction targets as a RATIO to the single-fibre density, not the absolute
    # 1 and 2: once the weights are fitted, D_rec is in the units of D_obs
    # (unit-mean anisotropy over the support), in which "one fibre" is not 1.0.
    # `unit` is measured where the field carries exactly one direction.
    Dsm = _smooth(D_rec)
    single = (field.n_dirs == 1) & support
    unit = float(np.median(Dsm[single])) if single.any() else 0.0
    if not np.isfinite(unit) or unit <= EPS:
        unit = 1.0

    
    ### See failure for TDI to record number of fibres at junctions ###
    contribs, pen = [], 0.0
    for j in junctions:
        c = [int(np.clip(round(v), 0, s - 1))
             for v, s in zip(j["centre"], vol.shape)]
        d_here = float(Dsm[c[0], c[1], c[2]])
        n_fib = d_here / unit
        if j["verdict"] == "TANGENCY":
            pc = max(n_fib - 1.0, 0.0) ** 2      # should carry ~1 fibre, not 2
        elif j["verdict"] == "CROSSING":
            pc = (n_fib - 2.0) ** 2              # should carry ~2
        else:
            pc = 0.0                             # AMBIGUOUS commits to nothing
        pen += pc
        contribs.append(dict(centre=c, S=float(j["S"]), verdict=j["verdict"],
                             density=d_here, n_fibres=float(n_fib),
                             loss_contribution=float(pc)))
    pen = verdict_penalty * pen / max(len(junctions), 1)

    # The tracts measured against the field they were meant to follow. Uses e3
    # only -- no ground truth -- so it is available on real data, which is the
    # whole point of preferring it to the angular error it stands in for.
    align = 0.0
    if align_weight:
        from cone_objective import Context, alignment_error
        align = alignment_error(paths, Context(vol, field, D_obs, support,
                                               rho=1.0, step=DT.STEP, l1=l1))

    total = ((0.0 if not np.isfinite(data_term) else data_term) + pen
             + float(align_weight) * align)
    return total, dict(mse=float(data_term), penalty=float(pen),
                       align=float(align), align_weight=float(align_weight),
                       total=float(total), n_streams=len(paths),
                       single_fibre_unit=float(unit),
                       n_active=int(fit.get("n_active", 0)),
                       coverage=float((D_rec[support] > 0).mean()),
                       junctions=contribs)


def optimise(vol, field, D_obs, support, junctions, seeds, seed_dirs,
             params=None, ladder=None, alpha=ALPHA, n_iter=4, verbose=True,
             align_weight=ALIGN_WEIGHT):
    """Coordinate descent on (lookahead, theta). Returns (params, trace, info).

    Coordinate descent, not a gradient. L is piecewise constant in these
    parameters -- the tracker takes an argmax over a discrete cone and stops on
    a hard support floor, so a change too small to alter any of those decisions
    changes nothing at all, and one that does alter them changes L in a jump. No
    derivative exists to descend. Sweeping one axis at a time over a ladder is
    cheap, cannot diverge, and reports exactly which move bought what.
    """
    params = dict(params or PARAMS0)
    ladder = ladder or LADDER
    kw = dict(alpha=alpha, align_weight=align_weight)
    best, info = loss(vol, field, D_obs, support, params, junctions,
                      seeds, seed_dirs, **kw)
    keys = ("mse", "penalty", "align", "total", "n_streams", "coverage")
    trace = [dict(iteration=-1, params=dict(params),
                  **{k: info[k] for k in keys})]
    if verbose:
        extra = (f" + {align_weight:g}*align {info['align']:.4f}"
                 if align_weight else "")
        print(f"    init  L={best:.4f} (mse {info['mse']:.4f} + pen "
              f"{info['penalty']:.4f}{extra}), {info['n_streams']} tracts, "
              f"coverage {info['coverage']:.1%}")

    for it in range(n_iter):
        improved = False
        for key, values in ladder.items():
            cur = params[key]
            for v in values:
                if v == cur:
                    continue
                trial = dict(params)
                trial[key] = v
                L, inf2 = loss(vol, field, D_obs, support, trial, junctions,
                               seeds, seed_dirs, **kw)
                if np.isfinite(L) and L < best - 1e-9:
                    best, params, info, improved = L, trial, inf2, True
        trace.append(dict(iteration=it, params=dict(params),
                          **{k: info[k] for k in keys}))
        if verbose:
            extra = (f" + {align_weight:g}*align {info['align']:.4f}"
                     if align_weight else "")
            print(f"    iter {it}: L={best:.4f} (mse {info['mse']:.4f} + pen "
                  f"{info['penalty']:.4f}{extra})  "
                  f"lookahead {params['lookahead']:g}, "
                  f"theta {params['theta']:g}")
        if not improved:
            if verbose:
                print("    converged (no coordinate move reduces L)")
            break
    return params, trace, info


def alpha_convergence(vol, field, D_obs, support, params, junctions, seeds,
                      seed_dirs, alpha=ALPHA, redescend=True, verbose=True):
    """What the fixed angular sampling costs, measured rather than assumed.

    Halving alpha quadruples the cone's cost for a strictly finer sampling of
    the same cone. The question that matters is not whether L changes -- it
    always does, because a max over a finer set of candidate directions is
    biased upward -- but whether the OPTIMUM changes. So by default this runs
    the descent again at alpha/2 and compares the two optima. Re-reading L at
    the coarse optimum instead would report a large change that is mostly the
    coarse parameters no longer being optimal, which answers nothing.

    theta is quantised in units of alpha: the cone carries
    round(theta / alpha) tilt rings, so on the crossing phantom at alpha = 0.2
    the values 0.15, 0.20 and 0.30 give bit-identical tracts. A reported theta
    is therefore only meaningful to within alpha, and the two parameters cannot
    be interpreted independently.
    """
    a2 = alpha / 2.0
    L0, _ = loss(vol, field, D_obs, support, params, junctions, seeds,
                 seed_dirs, alpha=alpha)
    if not redescend:
        L1_, _ = loss(vol, field, D_obs, support, params, junctions, seeds,
                      seed_dirs, alpha=a2)
        p2 = dict(params)
    else:
        p2, _tr, inf2 = optimise(vol, field, D_obs, support, junctions, seeds,
                                 seed_dirs, alpha=a2, n_iter=2, verbose=False)
        L1_ = float(inf2["total"])
    rel = abs(L1_ - L0) / max(abs(L0), EPS)
    moved = any(p2[k] != params[k] for k in params)
    if verbose:
        print(f"    [alpha] L={L0:.4f} at alpha={alpha:g} "
              f"{ {k: params[k] for k in params} }")
        print(f"            L={L1_:.4f} at alpha={a2:g} { {k: p2[k] for k in p2} }"
              f"  -> optimum {'MOVED' if moved else 'unchanged'}, "
              f"L differs {rel:.1%}")
    return dict(alpha=float(alpha), L=float(L0), alpha_half=float(a2),
                L_half=float(L1_), rel_change=float(rel),
                params_half={k: float(v) for k, v in p2.items()},
                optimum_moved=bool(moved))


def demote_by_evidence(vol, field, D_obs, support, params, junctions, seeds,
                       seed_dirs, alpha=ALPHA, tol=0.0, verbose=True):
    """Let the TRACTS decide each junction, instead of S deciding for them.

    For every junction, the field is tried both ways -- second direction kept,
    second direction retracted -- and the one that explains the image better
    wins. Retracting is chosen where it LOWERS the data term by more than `tol`.

    Why the verdict penalty is switched off here
    --------------------------------------------
    The penalty asks whether a junction carries the number of fibres its own S
    verdict claims, so scoring this comparison with it would let S vote on the
    question S is being tested against, and the tractography would agree with it
    by construction. The decision therefore uses the DATA TERM ALONE --
    mse(D_rec, D_obs) -- which is the only part of the loss that answers to the
    image rather than to a prior decision.

    Seeds are held IDENTICAL across the two arms, including seeds launched along
    the vec2 being retracted. Dropping them would change the tractogram's size
    at the same moment as its shape, and the difference in mse could then be
    read either way. A seed walking a direction the field no longer supports
    simply dies at the floor, which is the honest outcome and not a special case.

    Returns (field, rows, info). `rows` is one record per junction with its S
    verdict, its evidence verdict, and the delta that decided it -- so the two
    can be compared rather than one silently overwriting the other.
    """
    kw = dict(alpha=alpha, verdict_penalty=0.0)
    _L, base = loss(vol, field, D_obs, support, params, junctions, seeds,
                    seed_dirs, **kw)
    base_mse = float(base["mse"])
    if verbose:
        print(f"    [evidence] baseline mse {base_mse:.6f} over "
              f"{len(junctions)} junction(s), {base['n_streams']} tracts")

    rows = []
    for j in junctions:
        pi = j.get("patch_index")
        trial = field.copy()
        n_vox = trial.demote([pi]) if pi is not None else 0
        if not n_vox:
            rows.append(dict(patch_index=pi, centre=list(j["centre"]),
                             S=float(j["S"]), verdict_S=j["verdict"],
                             delta_mse=None, verdict_evidence=None,
                             note="no vec2 painted for this patch"))
            continue
        _l, inf = loss(vol, trial, D_obs, support, params, junctions, seeds,
                       seed_dirs, **kw)
        d = float(inf["mse"]) - base_mse
        rows.append(dict(patch_index=int(pi), centre=list(j["centre"]),
                         S=float(j["S"]), verdict_S=j["verdict"],
                         delta_mse=d, n_voxels=int(n_vox),
                         verdict_evidence="TANGENCY" if d < -tol else "CROSSING"))

    chosen = [r["patch_index"] for r in rows
              if r.get("verdict_evidence") == "TANGENCY"]
    out = field.copy()
    n_vox = out.demote(chosen) if chosen else 0
    _l2, final = loss(vol, out, D_obs, support, params, junctions, seeds,
                      seed_dirs, **kw)

    scored = [r for r in rows if r.get("delta_mse") is not None]
    agree = sum(r["verdict_S"] == r["verdict_evidence"] for r in scored)
    info = dict(base_mse=base_mse, final_mse=float(final["mse"]),
                n_demoted=len(chosen), n_voxels=int(n_vox),
                n_scored=len(scored), n_agree=int(agree),
                patch_indices=[int(c) for c in chosen])
    if verbose:
        print(f"    [evidence] demoted {len(chosen)}/{len(scored)} on the data "
              f"term alone; mse {base_mse:.6f} -> {float(final['mse']):.6f}")
        if scored:
            print(f"    [evidence] agrees with S on {agree}/{len(scored)} "
                  f"junction(s) ({agree / len(scored):.0%})")
    return out, rows, info


def demote_and_retrack(vol, field, D_obs, support, params, junctions, seeds,
                       seed_dirs, alpha=ALPHA, verbose=True,
                       align_weight=ALIGN_WEIGHT):
    """Retract the second direction where S says TANGENCY, then re-track.

    The demotion is a change to the FIELD, not to the loss, so its effect has to
    be measured by tracking again. The parameters are held at the descent's
    optimum, so the before/after difference is the demotion and not a reoptimisation.

    Seeds are NOT quite identical, and the exception is deliberate: a seed
    launched along a vec2 that has just been retracted describes a fibre the
    field no longer claims exists, so it goes too. `n_seeds_dropped` records how
    many, because that is a change to the tractogram's size as well as its
    shape and the two should not be confused when reading the loss.

    AMBIGUOUS is left alone: demoting on a verdict the metric itself will not
    commit to would launder uncertainty into a hard label.
    """
    targets = sorted({int(j["patch_index"]) for j in junctions
                      if j["verdict"] == "TANGENCY"
                      and j.get("patch_index") is not None})
    out = field.copy()
    n_vox = out.demote(targets) if targets else 0
    # Seeds along a retracted vec2 no longer describe the field, so they go too.
    keep = np.ones(len(seeds), bool)
    if n_vox:
        idx = np.clip(np.round(seeds).astype(int), 0,
                      np.array(field.shape) - 1)
        gone = ((field.n_dirs[idx[:, 0], idx[:, 1], idx[:, 2]] >= 2)
                & (out.n_dirs[idx[:, 0], idx[:, 1], idx[:, 2]] < 2))
        along2 = np.abs(np.einsum(
            "ij,ij->i", seed_dirs,
            field.vec2[:, idx[:, 0], idx[:, 1], idx[:, 2]].T)) > 0.9
        keep = ~(gone & along2)
    L, info = loss(vol, out, D_obs, support, params, junctions,
                   seeds[keep], seed_dirs[keep], alpha=alpha,
                   align_weight=align_weight)
    if verbose:
        print(f"    [demote] {len(targets)} junction(s), {n_vox} voxels lost "
              f"vec2, {int((~keep).sum())} seeds dropped -> L={L:.4f}")
    return out, L, info, dict(n_junctions=len(targets), n_voxels=int(n_vox),
                              n_seeds_dropped=int((~keep).sum()),
                              patch_indices=targets)
