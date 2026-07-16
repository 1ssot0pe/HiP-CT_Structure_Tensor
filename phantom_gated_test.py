"""
phantom_gated_test.py

Validate the lambda3 HOST-gate on the "+" phantom (make_perp_bundles), where
ground-truth regions are known: does lambda3 fire HOST on the crossing centre
and SKIP it (cheap single-e3) on the single-fibre arms and empty corners?

Bigger phantom (160^3, crossing half-width 40) so 41^3 patches fall cleanly
inside one region. A 5x5 grid of patch centres samples crossing / arm / empty.

For each patch we compute the cheap gate (lambda3, e3) AND run HOST anyway (for
truth), so we can score, per region: mean lambda3, HOST-fired fraction at the
threshold, and what HOST actually said. Reuses make_perp_bundles
(crossing_trigger_test) and the pipeline pieces from refined_field.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from crossing_trigger_test import make_perp_bundles
from refined_field import (cheap_gate, analyse_all_scales, persistent_peaks,
                           classify_persistent, spatial_regularize, _panel,
                           C_SINGLE, C_CROSS, C_DISP, INK)

HALF = 20
SHAPE = (160, 160, 160)
CROSS_HALF = 40
Z = SHAPE[0] // 2


def region_of(masks, cy, cx):
    for name in ("crossing", "arm_h", "arm_v", "empty"):
        if masks[name][Z, cy, cx]:
            return name
    return "empty"


if __name__ == "__main__":
    import time
    from reconstruction import iter_progress

    vol, masks = make_perp_bundles(shape=SHAPE, half=CROSS_HALF, spacing=10.0,
                                   spacing2=13.0, noise_sigma=0.2)
    centers = [30, 55, 80, 105, 130]
    grid = [(gi, cy, gj, cx) for gi, cy in enumerate(centers)
            for gj, cx in enumerate(centers)]
    print(f"phantom {SHAPE}, crossing half={CROSS_HALF}; {len(grid)} patches (half={HALF})")

    rows = []
    t0 = time.time()
    for gi, cy, gj, cx in iter_progress(grid, desc="phantom gate patches"):
        patch = vol[Z - HALF:Z + HALF + 1, cy - HALF:cy + HALF + 1, cx - HALF:cx + HALF + 1]
        lam3, e3 = cheap_gate(patch)
        hlabel, hdirs = classify_persistent(persistent_peaks(analyse_all_scales(patch)))
        rows.append(dict(gi=gi, gj=gj, cy=cy, cx=cx, lam3=lam3, e3=e3,
                         region=region_of(masks, cy, cx), hlabel=hlabel, hdirs=hdirs))

    lam3s = np.array([r["lam3"] for r in rows])
    # threshold: midway (log) between the single-fibre bulk and the high tail;
    # use a percentile that leaves the crossing patches above it.
    thr = float(np.percentile(lam3s, 55))

    # ---- validation table by ground-truth region ----
    print("=" * 70)
    print(f"lambda3 gate on phantom (threshold={thr:.4g}, pct=55)")
    print(f"  {'region':>10} | {'n':>2} | {'mean lam3':>10} | {'HOST-fired':>10} | HOST labels")
    from collections import Counter
    for region in ("crossing", "arm_h", "arm_v", "empty"):
        rs = [r for r in rows if r["region"] == region]
        if not rs:
            continue
        fired = np.mean([r["lam3"] > thr for r in rs])
        labs = Counter(r["hlabel"] for r in rs)
        print(f"  {region:>10} | {len(rs):>2} | {np.mean([r['lam3'] for r in rs]):10.4g} | "
              f"{fired:10.0%} | {dict(labs)}")
    cross = [r for r in rows if r["region"] == "crossing"]
    noncross = [r for r in rows if r["region"] != "crossing"]
    recall = np.mean([r["lam3"] > thr for r in cross])
    waste = np.mean([r["lam3"] > thr for r in noncross])
    print("-" * 70)
    print(f"  crossing patches that FIRE HOST (recall)          = {recall:.0%}")
    print(f"  non-crossing patches that fire HOST (wasted HOST) = {waste:.0%}")
    print(f"  -> HOST skipped on {np.mean(lam3s <= thr):.0%} of patches")
    print("=" * 70)

    # ---- build gated field + figure ----
    gated = {}
    for r in rows:
        key = (r["gi"], r["gj"])
        if r["lam3"] > thr:
            gated[key] = dict(label=r["hlabel"], dirs=r["hdirs"], cy=r["cy"], cx=r["cx"])
        else:
            gated[key] = dict(label="single", dirs=([r["e3"]] if r["e3"] is not None else []),
                              cy=r["cy"], cx=r["cx"])
    gated = spatial_regularize(gated)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
    # panel 1: which patches fired HOST (lambda3 map view)
    ax[0].imshow(vol[Z], cmap="gray")
    for r in rows:
        fired = r["lam3"] > thr
        ax[0].plot(r["cx"], r["cy"], "s", ms=13, mfc="none",
                   mec=(C_CROSS if fired else C_SINGLE), mew=2.5)
    ax[0].add_patch(plt.Rectangle((Z - CROSS_HALF, Z - CROSS_HALF), 2 * CROSS_HALF,
                                  2 * CROSS_HALF, fill=False, ec="cyan", lw=2, ls="--"))
    ax[0].set_title("lambda3 gate decision per patch\n"
                    "orange=run HOST, blue=skip (single-e3); cyan=true crossing zone",
                    fontsize=10)
    _panel(ax[1], vol[Z], gated, "lambda3-gated orientation field")
    ax[1].add_patch(plt.Rectangle((Z - CROSS_HALF, Z - CROSS_HALF), 2 * CROSS_HALF,
                                  2 * CROSS_HALF, fill=False, ec="cyan", lw=2, ls="--"))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.legend(handles=[
        Line2D([], [], color=C_SINGLE, lw=3, label="single (1 direction)"),
        Line2D([], [], color=C_CROSS, lw=3, label="crossing (2 directions)"),
        Line2D([], [], color=C_DISP, marker="o", ls="", label="unresolvable")],
        loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("lambda3 HOST-gate on the '+' phantom (validate before real data)",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig("phantom_gated_test.png", dpi=130)
    print(f"({time.time()-t0:.0f}s)  Saved phantom_gated_test.png")
