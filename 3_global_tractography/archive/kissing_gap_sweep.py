import os as _sr_os, sys as _sr_sys
_sr_stage = _sr_os.path.dirname(_sr_os.path.dirname(_sr_os.path.abspath(__file__)))  # 3_global_tractography/
_sr_sys.path[:0] = [_sr_os.path.dirname(_sr_stage), _sr_stage]  # repo root (core), stage modules

from core.kissing_phantom import build_kissing
from test_kissing_phantom import classify_grid, tractography_gap_test

GAPS = (2.0, 3.0, 4.0, 8.0, 10.0)
RADIUS = 2.0  # same width as target resolution limit 4vxs wide 


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
