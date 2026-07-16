"""
apply_i74.py

First application of the scale-adaptive + triage pipeline (adaptive_triage.
triage_patch_v2) to REAL i74 data. There is NO ground truth, so we cannot
report angular error; instead we report the TRIAGE BREAKDOWN over a
representative 3D sample of patches -- what fraction of real tissue the pipeline
calls confidently-single (fast-path), genuinely resolvable-crossing
(host-fod-multi), or unresolvable/complex (fallback-single-scale) -- plus
whether the locally-picked sigma_grad actually VARIES across the tissue (if it
never moves, scale-adaptivity buys nothing on real data).

Reuses load_interior (selfcalib_threshold, the validated <1%-air tissue box) and
triage_patch_v2 (adaptive_triage). Sigma grids are trimmed to 5 values for speed
-- this is a quick representative first look, not a full-volume run.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from selfcalib_threshold import load_interior
from adaptive_triage import triage_patch_v2

SIGMAS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # trimmed grid for speed
HALF = 20                                       # 41^3 patches (fits sigma=5 crop margin 14)


if __name__ == "__main__":
    import time

    vol = load_interior()
    print(f"i74 interior tissue box {vol.shape}, mean={vol.mean():.1f}, "
          f"air-frac(<25)={(vol < 25).mean():.3f}")

    # representative 3D grid of patch centres, all >= HALF from the box edges
    centers_xy = [45, 100, 155]
    centers_z = [60, 140]
    patches = [(z, y, x) for z in centers_z for y in centers_xy for x in centers_xy]
    print(f"sampling {len(patches)} patches (half={HALF}), sigma grid={SIGMAS.tolist()}\n")

    rows = []
    t0 = time.time()
    for i, (z, y, x) in enumerate(patches):
        p = vol[z - HALF:z + HALF + 1, y - HALF:y + HALF + 1, x - HALF:x + HALF + 1]
        r = triage_patch_v2(p, sigmas_lindeberg=SIGMAS, sigma_sweep=SIGMAS)
        m = r["m"]
        sep = "-" if (m is None or not np.isfinite(m["sep"])) else f"{m['sep']:.0f}"
        rows.append(dict(center=(z, y, x), mean=float(p.mean()), sg=r["sigma_grad"],
                        cs=(np.nan if m is None else m["c_s"]), tier=r["tier"],
                        label=r["label"], npk=(0 if m is None else m["n_peaks"]), sep=sep))
        print(f"  [{i+1:2d}/{len(patches)}] c={str((z,y,x)):>16} mean={p.mean():5.1f} | "
              f"sg*={r['sigma_grad']:.1f} c_s={rows[-1]['cs']:.2f} | "
              f"{r['tier']:22s} {r['label']:12s} npk={rows[-1]['npk']} sep={sep}")
    print(f"\n  ({time.time()-t0:.0f}s total)")

    # ---- summary distribution ----
    print("=" * 70)
    print("TRIAGE BREAKDOWN over the real i74 sample")
    from collections import Counter
    tier_counts = Counter(r["tier"] for r in rows)
    label_counts = Counter(r["label"] for r in rows)
    n = len(rows)
    print("  tiers:")
    for t, c in tier_counts.most_common():
        print(f"    {t:24s} {c:2d}/{n}  ({c/n:.0%})")
    print("  labels:")
    for l, c in label_counts.most_common():
        print(f"    {l:24s} {c:2d}/{n}  ({c/n:.0%})")
    sgs = np.array([r["sg"] for r in rows])
    css = np.array([r["cs"] for r in rows])
    print(f"  picked sigma_grad: range [{sgs.min():.1f}, {sgs.max():.1f}], "
          f"unique {sorted(set(sgs.tolist()))}  -> scale-adaptivity "
          f"{'IS' if len(set(sgs.tolist())) > 1 else 'is NOT'} varying across tissue")
    print(f"  c_s: median={np.nanmedian(css):.2f}, range [{np.nanmin(css):.2f}, {np.nanmax(css):.2f}]")
    n_multi = sum(1 for r in rows if r["tier"] == "host-fod-multi")
    print(f"  genuinely resolved as a crossing (host-fod-multi): {n_multi}/{n}")
    print("=" * 70)
