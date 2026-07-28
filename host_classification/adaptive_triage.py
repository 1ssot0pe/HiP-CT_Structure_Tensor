"""
adaptive_triage.py

Steps 1+2 towards applying HOST-FOD to real i74 data:

STEP 1 -- scale-adaptive sigma_grad. Phantom 1 (host_geometry_test.py) showed
HOST-FOD order-6 only works when sigma_grad matches the LOCAL fibre width; a
single global setting misfires across regions of different scale. Fix: before
running HOST-FOD on a patch, run a LOCAL Lindeberg sigma sweep on that same
patch (reusing lindeberg_response_3d + first_local_max, the exact code
validated on clean single-width phantoms in lindeberg_sigma_phantoms.py) and
use its peak as sigma_grad for that patch. This is a per-patch/per-region
scale pick, not a single volume-wide constant.

STEP 1b (v2 fix) -- the naive version above regressed a real crossing: Lindeberg
picks the sigma of STRONGEST SINGLE-STRUCTURE energy, which is not the same
question as "which sigma best SEPARATES two crossing orientations" -- it has no
way to know a crossing is present. Fix: only trust the cheap Lindeberg+fast-path
result when it's confidently single (low c_s). Otherwise, SWEEP sigma_grad
through HOST-FOD itself and pick whichever value gives the best-separated,
most-balanced 2-peak result (highest amp2, the same balance measure classify()
already gates on) -- and if no swept sigma clears that bar, fall back to the
honest single-scale characterisation rather than reporting an unresolved pick.

STEP 2 -- triage instead of universal resolution. The messy 4-family phantom
showed a real ceiling: HOST-FOD cannot resolve arbitrary multi-way crossings.
Rather than forcing a direction everywhere, triage each patch:
  - LOW sphericity (c_s <= cutoff, the "clean single fibre" ladder rung from
    model_ladder.py) -> cheap fast path, trust e3 directly, skip HOST-FOD.
  - otherwise -> run HOST-FOD at the locally-picked sigma_grad and trust ITS
    OWN classify() output (single / crossing / dispersed / disorganised) --
    that function already encodes the aniso/c_s gates needed to say "give up"
    on genuinely unresolvable voxels, so no new ad hoc threshold is invented
    for the hard cases; only the easy majority gets a shortcut.

Reuses lindeberg_response_3d + first_local_max (pick_sigma_rho.py,
validated real-data scale selector), analyse + classify (host_fod.py, the
validated order-6 HOST), and the phantom suite (host_geometry_test.py). Only
new code is the two-step orchestration (pick_local_sigma_grad, triage_patch)
and its test against the SAME phantoms that exposed the original problems.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sigma_rho_selection"))
from pick_sigma_rho import lindeberg_response_3d, first_local_max
from host_fod import analyse, classify
from host_geometry_test import (
    make_two_region_phantom, make_messy_crossing, make_kissing_phantom,
    make_true_crossing_same_angles, _patch_at, _ang_err, SHAPE,
)

SIGMAS_LINDEBERG = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
RHO_LINDEBERG = 1.0
CS_SINGLE_CUTOFF = 0.15    # from the model_ladder.py c_s ladder: single-fibre rung


# ---------------------------------------------------------------------------
# STEP 1 -- local scale pick
# ---------------------------------------------------------------------------

def pick_local_sigma_grad(patch, sigmas=SIGMAS_LINDEBERG, rho=RHO_LINDEBERG, verbose=False):
    """Local Lindeberg sigma sweep on THIS patch only. Returns (sigma_grad, sigmas, response)."""
    _s, response = lindeberg_response_3d(patch, sigmas, rho=rho, verbose=verbose)
    i = first_local_max(sigmas, response)
    if i is None:
        i = int(np.nanargmax(response))
    return float(sigmas[i]), sigmas, response


# ---------------------------------------------------------------------------
# STEP 2 -- triage: cheap fast-path for clean single fibre, else defer to HOST
# ---------------------------------------------------------------------------

def triage_patch(patch, cs_single_cutoff=CS_SINGLE_CUTOFF):
    """Returns dict(sigma_grad, tier, label, m) where m is the HOST-FOD
    metrics dict (always computed once, at the locally-picked sigma_grad --
    cheap, and needed anyway to read off c_s for the tier decision)."""
    sigma_grad, sg_sigmas, sg_resp = pick_local_sigma_grad(patch)
    m = analyse(patch, sigma_grad=sigma_grad, order=6)
    if m is None:
        return dict(sigma_grad=sigma_grad, tier="no-data", label="n/a", m=None)
    if m["c_s"] <= cs_single_cutoff:
        tier, label = "fast-path", "single"
    else:
        tier, label = "host-fod", classify(m)
    return dict(sigma_grad=sigma_grad, tier=tier, label=label, m=m)


# ---------------------------------------------------------------------------
# STEP 1b (v2) -- sweep sigma WITHIN HOST-FOD for the best multi-peak
# resolution, only after the cheap single-structure check has failed to
# confirm a clean single fibre. Falls back to the honest single-scale
# characterisation if no swept sigma clears the amp2 balance gate.
# ---------------------------------------------------------------------------

SIGMA_SWEEP_HOST = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])


def triage_patch_v2(patch, sigmas_lindeberg=SIGMAS_LINDEBERG, rho_lindeberg=RHO_LINDEBERG,
                    sigma_sweep=SIGMA_SWEEP_HOST, cs_single_cutoff=CS_SINGLE_CUTOFF,
                    amp2_min=0.4):
    sigma_lind, _sg_sigmas, _sg_resp = pick_local_sigma_grad(patch, sigmas_lindeberg, rho_lindeberg)
    m_lind = analyse(patch, sigma_grad=sigma_lind, order=6)

    if m_lind is not None and m_lind["c_s"] <= cs_single_cutoff:
        return dict(sigma_grad=sigma_lind, tier="fast-path", label="single",
                    m=m_lind, sweep=None)

    sweep = []
    for sg in sigma_sweep:
        m = analyse(patch, sigma_grad=sg, order=6)
        # only EXACTLY-2-peak results count as resolution quality -- n_peaks>=3
        # is spurious splitting / unresolvable, and must not be rewarded (that
        # is what let the old amp2 metric select large-sigma split artefacts).
        quality = (m["amp2"] if (m is not None and m["n_peaks"] == 2
                                 and np.isfinite(m["amp2"])) else -np.inf)
        sweep.append((sg, m, quality))
    best_sg, best_m, best_q = max(sweep, key=lambda t: t[2])

    if best_q >= amp2_min:
        return dict(sigma_grad=best_sg, tier="host-fod-multi", label=classify(best_m),
                    m=best_m, sweep=sweep)
    return dict(sigma_grad=sigma_lind, tier="fallback-single-scale", label=classify(m_lind),
               m=m_lind, sweep=sweep)


# ---------------------------------------------------------------------------
# Test against the same phantoms that exposed the original problems
# ---------------------------------------------------------------------------

def _err_report(m, truths):
    if not truths or m is None or m["dirs"] is None or len(m["dirs"]) == 0:
        return "-"
    if len(truths) == 1:
        return f"{_ang_err(m['dirs'][0], truths[0]):.1f}"
    if len(m["dirs"]) == 1:
        e = min(_ang_err(m["dirs"][0], t) for t in truths)
        return f"{e:.1f}(1pk)"
    e1 = min(_ang_err(m["dirs"][0], t) for t in truths)
    e2 = min(_ang_err(m["dirs"][1], t) for t in truths)
    return f"{e1:.0f}/{e2:.0f}"


def _row(r, truths):
    cs = "n/a" if r["m"] is None else f"{r['m']['c_s']:.2f}"
    err = _err_report(r["m"], truths)
    return f"sg*={r['sigma_grad']:5.2f} | c_s={cs:>5} | tier={r['tier']:18s} | label={r['label']:12s} | err={err}"


if __name__ == "__main__":
    HALF = 20   # patch half-size: must safely fit the sigma=5 crop margin (14)
    h = SHAPE[0] // 2
    q, tq = SHAPE[0] // 4, 3 * SHAPE[0] // 4

    print("=" * 100)
    print("PHANTOM 1 -- perpendicular regions, different widths (1.2 vs 3.5)")
    vol1, tr1 = make_two_region_phantom()
    cases1 = [("region A (w=1.2)", (h, h, q), [tr1["d1"]]),
             ("region B (w=3.5)", (h, h, tq), [tr1["d2"]]),
             ("BOUNDARY (mixed)", (h, h, h), [tr1["d1"], tr1["d2"]])]
    for name, center, truths in cases1:
        patch = _patch_at(vol1, center, half=HALF)
        print(f"  {name:>18} | v1: {_row(triage_patch(patch), truths)}")
        print(f"  {'':>18} | v2: {_row(triage_patch_v2(patch), truths)}")

    print("-" * 100)
    print("PHANTOM 2 -- messy crossing, 4 random-angle/width families (true count=4)")
    vol2, tr2 = make_messy_crossing()
    print(f"  true widths: {np.round(tr2['widths'], 2)}")
    patch2 = _patch_at(vol2, (h, h, h), half=HALF)
    print(f"  {'messy x4':>18} | v1: {_row(triage_patch(patch2), [])}")
    print(f"  {'':>18} | v2: {_row(triage_patch_v2(patch2), [])}")

    print("-" * 100)
    print("PHANTOM 3 -- kissing vs true crossing (same 2 angles)  <-- v1 regressed HERE")
    kiss, trk = make_kissing_phantom()
    cross, trc = make_true_crossing_same_angles()
    for name, vol, tr in [("true crossing", cross, trc), ("kissing", kiss, trk)]:
        patch = _patch_at(vol, (h, h, h), half=HALF)
        truths = [tr["d1"], tr["d2"]]
        print(f"  {name:>18} | v1: {_row(triage_patch(patch), truths)}")
        print(f"  {'':>18} | v2: {_row(triage_patch_v2(patch), truths)}")
    print("=" * 100)
