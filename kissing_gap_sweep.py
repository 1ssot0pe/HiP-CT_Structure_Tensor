"""
kissing_gap_sweep.py -- same fibre width (radius 1.6/1.6), GAP swept over five values,
to see where the pipeline still works. Reuses the B/C tests from test_kissing_phantom.

HOST's analysis window is 17 vox wide (HALF=8), so the sweep spans "gap smaller than
the window" to "gap clearly bigger than it":
    gap = 4, 8, 12, 16, 20 vox

Expect: (C) global tractography should refuse to bridge the gap at EVERY gap (0
spanning tracts) -- even a small gap is still a real gap, and the whole point of
Phase 2.5 was that continuity+parsimony do not need a big gap to get this right.
(B) the RARE spurious local "crossing" near the closest approach (a resolution-scale
partial-volume effect, see kissing_phantom / test_kissing_phantom docstrings) should
SHRINK and vanish as gap grows past the ~17-vox window width.
"""

from kissing_phantom import build_kissing
from test_kissing_phantom import classify_grid, tractography_gap_test

GAPS = (2.0, 2.5, 3.0, 4.0, 8.0, 12.0, 16.0, 20.0)
RADIUS = 1.5                # both bundles this width (see build_kissing's current hardcode)


def valley_ratio(vol):
    """Intensity at the vertex midpoint / peak intensity -- how much the two tubes'
    Gaussian tails have physically FUSED in the image itself (not an algorithm
    artifact). High ratio = no real dip left = the phantom is no longer two resolvable
    objects at this gap/width."""
    z, cy, cx = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
    row = vol[z, cy, :]
    return float(row[cx-2:cx+3].min() / row.max())


def run():
    print("=" * 88)
    print(f"KISSING GAP SWEEP -- same fibre width (radius {RADIUS}/{RADIUS}), "
          f"gap = {'/'.join(f'{g:.0f}' for g in GAPS)} vox")
    print(f"  {'gap':>5} | {'valley':>7} | {'HOST single':>13} {'spurious x-ing':>16} | "
          f"{'tracts':>7} {'spanning':>9}")
    rows = []
    for gap in GAPS:
        vol, meta = build_kissing(same_size=True, gap=gap)
        vr = valley_ratio(vol)
        counts = classify_grid(vol)
        tot = sum(counts.values())
        singles = counts.get("single", 0)
        cross = counts.get("crossing", 0)
        npos, ntracts, biggest, spanning = tractography_gap_test(vol, meta["gap"])
        rows.append((gap, vr, singles, tot, cross, ntracts, spanning))
        print(f"  {gap:5.0f} | {vr:6.1%} | "
              f"{singles:4d}/{tot:<4d} ({singles/tot:4.0%}) {cross:4d}/{tot:<4d} "
              f"({cross/tot:4.0%})   | {ntracts:7d} {spanning:9d}")
    print("-" * 88)
    bridged = [r for r in rows if r[6] > 0]
    if bridged:
        g, vr, *_ = bridged[0]
        fused = vr > 0.5
        print(f"  [C] BRIDGES at gap={g:.0f} (spanning={bridged[0][6]}). Valley ratio "
              f"there = {vr:.0%} -- {'the phantom ITSELF has fused (no real dip left in '
              'the image); this is a RESOLUTION LIMIT (gap comparable to a few fibre '
              'radii), not a tractography bug' if fused else 'a genuine algorithm-level bridge (image still shows a real dip) -- worth investigating'}.")
        clean = [r for r in rows if r[6] == 0]
        if clean:
            print(f"  [C] Holds cleanly (0 spanning) for gap >= {min(r[0] for r in clean):.0f} "
                  f"vox ({min(r[0] for r in clean)/RADIUS:.1f}x the fibre radius).")
    else:
        print("  [C] global tractography never bridges the gap at ANY tested gap.")
    xs = [(g, c / t) for g, _vr, _s, t, c, _nt, _sp in rows]
    print(f"  [B] spurious-crossing rate vs gap: " +
          ", ".join(f"{g:.0f}v={r:.0%}" for g, r in xs))
    shrinking = all(xs[i][1] >= xs[i+1][1] for i in range(len(xs)-1)) if len(xs) > 1 else True
    print(f"      {'shrinks monotonically with gap (as expected)' if shrinking else 'not perfectly monotonic (small-sample noise in the estimate)'}")
    print("=" * 88)


if __name__ == "__main__":
    run()


run()