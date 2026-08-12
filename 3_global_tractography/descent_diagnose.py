r"""
descent_diagnose.py -- why does the descent stop after 5 iterations of 12?

The descent reports "line search exhausted -- converged". Those are two claims
and only the first is measured: the loop halts because `lr_try` fell below
`LR_MIN` without any step clearing `TOL`, which is a statement about the
OPTIMISER's floors, not about the surface having a minimum there.

This module separates the two possibilities, because the fix is different for
each and guessing between them is how a tuning parameter gets blamed for a
modelling problem (or the reverse).

  A. the point really is a local minimum -- no step in the descent direction
     improves the loss, at any length. Then the early stop is CORRECT and the
     thing to change is the objective, not the optimiser.

  B. an improving step exists but the line search cannot reach it -- either it
     sits below LR_MIN, or its improvement is real but smaller than TOL, or the
     direction being searched is wrong because the finite difference measured a
     neighbourhood far larger than the step.

Two measurements, both on the converged point of a completed run.

LINE SCAN. Walk the loss along -g over step lengths spanning four decades,
INCLUDING lengths below LR_MIN and negative ones. If the minimum along that
line sits inside the searched range, the line search should have found it and
the acceptance test is what blocked it. If it sits below LR_MIN, the floor is
the cause. If nothing improves in either direction, the direction is wrong.

H-CONVERGENCE. Re-measure the finite-difference gradient at a ladder of h. A
derivative is the limit as h -> 0, so a gradient worth following must stop
moving as h shrinks. `gradient_optimise.main` already runs exactly this test on
the HARD tracker, where a non-halving difference proves no derivative exists;
it has never been run on the SMOOTH tracker during a descent. If the direction
swings as h falls, then FD_STEP = 0.05 was measuring a different neighbourhood
from the one the 0.0008-length steps move in, and the descent has been
following a slope that is not local to it.

Run
---
    python descent_diagnose.py --outdir ../results/loop_r32 --variant vec1
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

import tdi_mse as TM                                          # noqa: E402
from cone_descent import stage2, seed, FLOOR_FRAC             # noqa: E402
from gradient_optimise import Problem, LR_MIN, TOL, FD_STEP   # noqa: E402
from descent_loop import load_setup, field_for_variant        # noqa: E402

H_LADDER = (0.05, 0.025, 0.0125, 0.00625, 0.003125)
SCAN = np.concatenate([-np.geomspace(0.3, 1e-4, 9),
                       np.geomspace(1e-4, 0.3, 25)])


def rebuild(run, variant, verbose=True):
    """The exact Problem the run descended on, from its saved settings.

    THE SEED COUNT MUST BE THE REQUESTED ONE, NOT THE RESULTING ONE. `seed(n=k)`
    samples k VOXELS, then `_dirs_at` adds a second seed at every voxel carrying
    a vec2 -- so a run asking for 150 reports 153, and rebuilding with 153 makes
    `seed` draw a DIFFERENT 153 voxels. The loss then fails to reproduce (it came
    out +3.8e-4 off on the r32 run) and every measurement below describes a
    neighbouring problem rather than the one that was descended.

    `n_seeds_requested` is written by `descent_loop`; older JSONs only have
    `n_seeds`, and for those the count is inverted from the two-direction voxel
    count when it was recorded. The reproduction check in `main` is what
    actually decides whether the rebuild succeeded, so a wrong guess here is
    reported rather than hidden.
    """
    vol, origin, label = load_setup(run.get("dataset", "i74"), run["centre"],
                                    run["radius"], verbose=verbose)
    dense, _D, tissue, FA, _sc = stage2(vol, verbose=False)
    n = run.get("n_seeds_requested")
    if n is None:
        # Older run: recover the request by SEARCH rather than by arithmetic.
        # The obvious estimate -- divide out the two-direction fraction over all
        # tissue -- is biased, because `seeding="fa"` samples only FA > 0.7
        # voxels and vec2 is not distributed the same way there: it returned 148
        # where the truth was 150 on the r32 run. `seed` does no tracking, so
        # simply trying candidates until the resulting count matches is both
        # exact and cheap.
        target = int(run["n_seeds"])
        n = target
        for cand in range(target, max(target // 2, 1) - 1, -1):
            if len(seed(dense, tissue, FA, run["seeding"], n=cand)[0]) == target:
                n = cand          # smallest request giving this count
        if verbose:
            print(f"  recovered seed request {n} -> {target} seeds "
                  f"(run recorded only the resulting count)")
    s, sd = seed(dense, tissue, FA, run["seeding"], n=int(n))
    floor = FLOOR_FRAC * float(vol.max())
    obj = TM.TdiMse(vol, tissue, smooth=run.get("smooth", 0.0))
    if verbose:
        print(f"  rebuilt with n={n} requested -> {len(s)} seeds "
              f"(run recorded {run['n_seeds']})")
    return Problem(vol, field_for_variant(dense, variant), s, sd, floor,
                   objective=obj, turn=run.get("turn", "sample")), label


def main(outdir, variant="vec1", json_name="descent_loop.json", verbose=True):
    with open(os.path.join(outdir, json_name)) as f:
        run = json.load(f)
    d = run["runs"][variant]
    u = np.log([d["L"], d["theta"], d["alpha"]])

    print("=" * 78)
    print(f"DESCENT DIAGNOSIS -- {outdir}  variant {variant}")
    print(f"  converged at L={d['L']:.4f} theta={d['theta_deg']:.3f}d "
          f"alpha={d['alpha_deg']:.3f}d after {len(d['trace']) - 1} of 12 "
          f"iterations")
    print(f"  optimiser floors: LR_MIN={LR_MIN:g}  TOL={TOL:g}  "
          f"FD_STEP={FD_STEP:g}")
    print("=" * 78)

    t0 = time.time()
    prob, label = rebuild(run, variant, verbose=verbose)

    # ---- does the loss reproduce? ------------------------------------------
    f_here = prob.f(u)
    saved = d["trace"][-1]["loss"]
    ok = abs(f_here - saved) < 1e-12
    print(f"\n  reproduce converged loss: {f_here:.9f} vs saved {saved:.9f}  "
          f"-> {'EXACT' if ok else f'DIFFERS by {f_here - saved:+.3e}'}")
    if not ok:
        print("  (a mismatch means the setup here is not the run's; everything"
              " below describes a different problem)")

    # ---- H-CONVERGENCE ------------------------------------------------------
    print(f"\n  [h-convergence] is the finite difference a derivative?")
    print(f"      {'h':>9} {'d/dlogL':>11} {'d/dlogth':>11} {'d/dlogal':>11} "
          f"{'|g|':>9} {'angle vs h=0.05':>16}")
    g_ref, rows = None, []
    for h in H_LADDER:
        g, _f = prob.grad(u, h=h)
        gn = float(np.linalg.norm(g))
        if g_ref is None:
            g_ref, ang = g.copy(), 0.0
        else:
            c = float(np.dot(g, g_ref) / max(np.linalg.norm(g) *
                                             np.linalg.norm(g_ref), 1e-30))
            ang = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        rows.append(dict(h=float(h), g=[float(x) for x in g], norm=gn,
                         angle_deg=ang))
        print(f"      {h:9.5f} {g[0]:+11.5f} {g[1]:+11.5f} {g[2]:+11.5f} "
              f"{gn:9.5f} {ang:15.1f}d")
    angs = [r["angle_deg"] for r in rows[1:]]
    norms = [r["norm"] for r in rows]
    spread = (max(norms) - min(norms)) / max(np.mean(norms), 1e-30)
    stable = max(angs) < 20.0 and spread < 0.5
    print(f"      -> direction swings up to {max(angs):.1f} deg, |g| spreads "
          f"{spread:.0%} across the ladder")
    print(f"      -> the gradient is {'STABLE in h -- it is a derivative' if stable else 'NOT CONVERGED in h: FD_STEP is measuring a different neighbourhood from the step'}")

    # ---- LINE SCAN ----------------------------------------------------------
    g0 = np.array(rows[0]["g"], float)
    step = g0 / max(np.linalg.norm(g0), 1e-30)
    print(f"\n  [line scan] loss along -g, {len(SCAN)} lengths from "
          f"{abs(SCAN).min():.0e} to {abs(SCAN).max():.1f} "
          f"(LR_MIN = {LR_MIN:g})")
    scan = []
    for t in SCAN:
        ft = prob.f(u - t * step)
        scan.append(dict(t=float(t), loss=float(ft), delta=float(ft - f_here)))
    best = min(scan, key=lambda r: r["loss"])
    improving = [r for r in scan if r["delta"] < -TOL]
    reachable = [r for r in improving if 0 < r["t"] <= 0.30
                 and r["t"] >= LR_MIN]
    below_floor = [r for r in improving if 0 < r["t"] < LR_MIN]
    backwards = [r for r in improving if r["t"] < 0]

    print(f"      {'t':>10} {'loss':>12} {'delta':>12}   note")
    for r in scan:
        note = []
        if abs(r["t"]) < LR_MIN:
            note.append("below LR_MIN")
        if r["t"] < 0:
            note.append("UPHILL direction")
        if r is best:
            note.append("<-- best on this line")
        if r["delta"] < -TOL and not note:
            note.append("improves")
        print(f"      {r['t']:+10.5f} {r['loss']:12.8f} {r['delta']:+12.3e}   "
              f"{', '.join(note)}")

    print(f"\n  best on the line: t = {best['t']:+.5f}, "
          f"delta = {best['delta']:+.3e}")
    print(f"  steps that improve by more than TOL: {len(improving)} of "
          f"{len(scan)}  "
          f"({len(reachable)} reachable by the line search, "
          f"{len(below_floor)} below LR_MIN, {len(backwards)} in the UPHILL "
          f"direction)")

    if not improving:
        verdict = ("A: a genuine local minimum along this direction -- no step "
                   "improves. The early stop is correct; the objective is what "
                   "needs changing.")
    elif reachable:
        verdict = ("B: improving steps EXIST inside the searched range, so the "
                   "line search should have found them. The acceptance test "
                   "(f < f_here - TOL, absolute) is what rejected them.")
    elif below_floor:
        verdict = ("B: the only improving steps are BELOW LR_MIN. Lowering the "
                   "floor is the direct fix.")
    else:
        verdict = ("B: the loss improves only in the UPHILL direction -- the "
                   "measured gradient points the wrong way, which is what an "
                   "h-unconverged finite difference does.")
    print(f"\n  VERDICT  {verdict}")
    print(f"\n  ({prob.n_eval} tracking passes, {(time.time() - t0) / 60:.1f} "
          f"min)")
    print("=" * 78)

    out = dict(outdir=outdir, variant=variant, label=label,
               converged=dict(L=d["L"], theta=d["theta"], alpha=d["alpha"]),
               loss=f_here, reproduced=bool(ok),
               floors=dict(LR_MIN=LR_MIN, TOL=TOL, FD_STEP=FD_STEP),
               h_ladder=rows, h_stable=bool(stable),
               h_angle_max=float(max(angs)), h_norm_spread=float(spread),
               scan=scan, best=best, n_improving=len(improving),
               n_reachable=len(reachable), n_below_floor=len(below_floor),
               n_backwards=len(backwards), verdict=verdict,
               n_passes=int(prob.n_eval))
    p = os.path.join(outdir, f"diagnose_{variant.replace('+', '_')}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote {p}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="why the descent stopped early")
    ap.add_argument("--outdir", default=os.path.join(_ROOT, "results",
                                                     "loop_r32"))
    ap.add_argument("--variant", default="vec1")
    a = ap.parse_args()
    main(a.outdir, a.variant)
