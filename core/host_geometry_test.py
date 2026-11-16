"""
host_geometry_test.py

Test the validated higher-order structure tensor (host_fod.py, order=6 beat
five other methods on synthetic crossings) against THREE geometries beyond the
simple overlapping-crossing phantoms it was tuned on, to see what different
orders buy you -- or where the method's honest limits are -- before real data.

  1. TWO PERPENDICULAR REGIONS, DIFFERENT WIDTHS -- spatially SEPARATE, locally
     single-fibre regions (not a per-voxel crossing) at 90 deg to each other,
     with different fibre widths. Tests scale sensitivity: does one
     (sigma_grad, order) setting recover the right direction deep in each
     region, and what happens to a patch straddling the boundary (two
     different-width single-fibre populations meeting, not interpenetrating)?

  2. MESSY MULTI-ANGLE / MULTI-WIDTH CROSSING -- 3-4 fibre families at random
     3D angles and random widths, ALL overlapping in the same voxels (true
     per-voxel multi-orientation). Tests whether higher order (8 vs 6 vs 4)
     recovers more of the true directions as complexity grows, and where it
     saturates.

  3. KISSING BUNDLES -- two SEPARATE bundles that touch at a boundary plane
     (segregated, not interpenetrating) at a crossing-like angle, compared
     against a TRUE overlapping crossing at the SAME two angles. This is the
     classic dMRI "kissing vs crossing" ambiguity: local per-voxel gradient
     evidence is expected to look similar for both, because a patch straddling
     the kissing boundary contains gradients from both orientations just like
     an overlapping crossing does -- if HOST can't tell them apart, that is an
     honest limitation to report, not a bug to fix.

Reuses _tube_family / _coords (crossing_shape_3d, the tube/coordinate
primitives), patch_gradients (two_orientation_test), and analyse / classify
(host_fod, the validated order-6 HOST). Only new code is the phantom
COMPOSITIONS (spatial arrangement of regions) and a small greedy direction
matcher for the multi-family case.
"""

import numpy as np

from core.crossing_shape_3d import _tube_family, _coords
from core.host_fod import analyse, classify

EPS = 1e-12
SHAPE = (96, 96, 96)


def _dir_inplane(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([np.cos(t), np.sin(t), 0.0])


def _random_dirs(n, min_sep_deg=25.0, seed=0):
    rng = np.random.default_rng(seed)
    dirs, tries = [], 0
    while len(dirs) < n and tries < 4000:
        v = rng.standard_normal(3)
        v = v / np.linalg.norm(v)
        if all(np.degrees(np.arccos(np.clip(abs(v @ d), 0, 1))) > min_sep_deg for d in dirs):
            dirs.append(v)
        tries += 1
    return dirs


def _patch_at(vol, center, half=14):
    sl = tuple(slice(c - half, c + half + 1) for c in center)
    return vol[sl]


def _add_noise(vol, noise_sigma, seed):
    if noise_sigma <= 0:
        return vol
    rng = np.random.default_rng(seed)
    return vol + noise_sigma * rng.standard_normal(vol.shape).astype(np.float32)


def _finalise(vol):
    vol = vol - vol.min()
    return (vol / (vol.max() + EPS)).astype(np.float32)


def _ang_err(d, t):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(d, t)), 0, 1))))


def _match_dirs(rec_dirs, true_dirs):
    """Greedy one-to-one nearest-angle matching. Returns list of
    (true_idx, rec_idx_or_None, angle_deg_or_None)."""
    pairs = []
    for ti, t in enumerate(true_dirs):
        for ri, r in enumerate(rec_dirs):
            pairs.append((_ang_err(r, t), ti, ri))
    pairs.sort(key=lambda p: p[0])
    used_t, used_r, matched = set(), set(), {}
    for ang, ti, ri in pairs:
        if ti in used_t or ri in used_r:
            continue
        matched[ti] = (ri, ang)
        used_t.add(ti); used_r.add(ri)
    return [(ti, *matched[ti]) if ti in matched else (ti, None, None)
            for ti in range(len(true_dirs))]


# ---------------------------------------------------------------------------
# PHANTOM 1 -- two perpendicular regions, different fibre widths
# ---------------------------------------------------------------------------
# Split along the x-coordinate (coords[0]); region A = x<0 (first quarter of
# the volume along that axis), region B = x>=0. NOT a per-voxel crossing --
# each half is locally single-fibre, just at a different angle AND scale.

def make_two_region_phantom(shape=SHAPE, theta1=0.0, theta2=90.0, width1=1.2,
                            width2=3.5, noise_sigma=0.3, seed=0):
    coords = _coords(shape)
    d1, d2 = _dir_inplane(theta1), _dir_inplane(theta2)
    volA = _tube_family(coords, d1, 6.0 * width1, width1)
    volB = _tube_family(coords, d2, 6.0 * width2, width2)
    region = coords[0] < 0.0                  # True = region A (x<0)
    vol = _add_noise(np.where(region, volA, volB), noise_sigma, seed)
    return _finalise(vol), dict(d1=d1, d2=d2, width1=width1, width2=width2)


def test_phantom1(save="crossing_shape_layer4_geom1.png"):
    print("=" * 78)
    print("PHANTOM 1 -- two perpendicular regions, different widths (1.2 vs 3.5)")
    print("  (NOT a per-voxel crossing: spatially separate single-fibre regions)")
    vol, tr = make_two_region_phantom()
    h = SHAPE[0] // 2
    q, tq = SHAPE[0] // 4, 3 * SHAPE[0] // 4       # quarter / three-quarter index
    regions = [("deep region A (width=1.2)", (h, h, q), tr["d1"]),
              ("deep region B (width=3.5)", (h, h, tq), tr["d2"]),
              ("BOUNDARY (mixed)", (h, h, h), None)]
    print(f"  {'region':>26} | {'sigma_grad':>10} | {'order':>5} | label        | err(deg)")
    for name, center, truth in regions:
        patch = _patch_at(vol, center)
        for sg in (1.0, 2.0, 4.0):
            for order in (2, 6):
                m = analyse(patch, sigma_grad=sg, order=order)
                lab = classify(m)
                if truth is not None and m and m["dirs"] is not None and len(m["dirs"]):
                    err = f"{_ang_err(m['dirs'][0], truth):6.1f}"
                elif truth is None and m and m["n_peaks"] >= 2:
                    e1 = min(_ang_err(m["dirs"][0], tr["d1"]), _ang_err(m["dirs"][0], tr["d2"]))
                    e2 = min(_ang_err(m["dirs"][1], tr["d1"]), _ang_err(m["dirs"][1], tr["d2"]))
                    err = f"{e1:.0f}/{e2:.0f}"
                else:
                    err = "  -  "
                print(f"  {name:>26} | {sg:10.1f} | {order:5d} | {lab:12s} | {err}")
    print("=" * 78)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(vol[h], cmap="gray")
    ax.axvline(h, color="r", ls="--", lw=1)
    ax.set_title("Phantom 1: perpendicular regions, widths 1.2 / 3.5\n(z-mid slice, boundary marked)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")


# ---------------------------------------------------------------------------
# PHANTOM 2 -- messy multi-angle / multi-width crossing (true overlap)
# ---------------------------------------------------------------------------

def make_messy_crossing(shape=SHAPE, n_families=4, width_range=(1.0, 3.0),
                        noise_sigma=0.3, seed=0, angle_seed=0):
    dirs = _random_dirs(n_families, min_sep_deg=25.0, seed=angle_seed)
    rng = np.random.default_rng(seed)
    widths = rng.uniform(*width_range, len(dirs))
    coords = _coords(shape)
    vol = np.zeros(shape, np.float32)
    for i, (d, w) in enumerate(zip(dirs, widths)):
        spacing = 6.0 * w * (1.0 + 0.15 * i)      # stagger to avoid shared periods
        vol = vol + _tube_family(coords, d, spacing, w)
    vol = _add_noise(vol, noise_sigma, seed)
    return _finalise(vol), dict(dirs=dirs, widths=widths)


def test_phantom2(save="crossing_shape_layer4_geom2.png"):
    print("=" * 78)
    print("PHANTOM 2 -- messy crossing: 4 random-angle, random-width families")
    vol, tr = make_messy_crossing()
    print(f"  true widths: {np.round(tr['widths'], 2)}")
    h = SHAPE[0] // 2
    patch = _patch_at(vol, (h, h, h))
    for order in (4, 6, 8):
        m = analyse(patch, sigma_grad=2.0, order=order)
        lab = classify(m)
        n_found = 0 if m is None else m["n_peaks"]
        matches = _match_dirs(m["dirs"], tr["dirs"]) if m and m["dirs"] is not None else []
        match_str = ", ".join(
            f"T{ti}->{'R'+str(ri) if ri is not None else 'none'}"
            f"({ang:.0f}deg)" if ri is not None else f"T{ti}->none"
            for ti, ri, ang in matches)
        print(f"  order={order}: n_peaks_found={n_found} (true n_families={len(tr['dirs'])}) "
              f"label={lab}")
        print(f"    matches: {match_str}")
    print("=" * 78)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(vol[h], cmap="gray")
    ax.set_title(f"Phantom 2: {len(tr['dirs'])} overlapping random-angle families\n(z-mid slice)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")


# ---------------------------------------------------------------------------
# PHANTOM 3 -- kissing bundles vs true crossing (same two angles)
# ---------------------------------------------------------------------------

def make_kissing_phantom(shape=SHAPE, theta1=0.0, theta2=70.0, width=1.5,
                         spacing=12.0, noise_sigma=0.3, seed=0):
    """Two SEPARATE bundles, segregated at the x=0 boundary (touching, not
    overlapping): each voxel sees only ONE orientation."""
    coords = _coords(shape)
    d1, d2 = _dir_inplane(theta1), _dir_inplane(theta2)
    volA = _tube_family(coords, d1, spacing, width)
    volB = _tube_family(coords, d2, spacing, width)
    region = coords[0] < 0.0
    vol = _add_noise(np.where(region, volA, volB), noise_sigma, seed)
    return _finalise(vol), dict(d1=d1, d2=d2)


def make_true_crossing_same_angles(shape=SHAPE, theta1=0.0, theta2=70.0, width=1.5,
                                   spacing=12.0, spacing2=12.0, noise_sigma=0.3, seed=0):
    """The SAME two angles/width, but summed everywhere (true per-voxel
    overlap). Uses EQUAL spacing on both families (a fair, symmetric-strength
    baseline) -- an unequal spacing2 was found to itself suppress the 2nd
    peak (the known asymmetric-crossing weakness), which would confound the
    kissing-vs-crossing comparison this phantom exists to make."""
    coords = _coords(shape)
    d1, d2 = _dir_inplane(theta1), _dir_inplane(theta2)
    vol = _tube_family(coords, d1, spacing, width) + _tube_family(coords, d2, spacing2, width)
    vol = _add_noise(vol, noise_sigma, seed)
    return _finalise(vol), dict(d1=d1, d2=d2)


def test_phantom3(save="crossing_shape_layer4_geom3.png"):
    print("=" * 78)
    print("PHANTOM 3 -- kissing bundles vs true overlapping crossing (same angles)")
    print("  known dMRI ambiguity: can per-voxel gradient evidence tell these apart?")
    h = SHAPE[0] // 2
    kiss, trk = make_kissing_phantom()
    cross, trc = make_true_crossing_same_angles()
    print(f"  {'case':>14} | {'order':>5} | {'n_peaks':>7} | {'sep':>5} | {'amp2':>5} | "
          f"{'aniso':>6} | {'cs':>5} | label        | peak errs (deg)")
    for name, vol, tr in [("kissing", kiss, trk), ("true crossing", cross, trc)]:
        patch = _patch_at(vol, (h, h, h))
        for order in (4, 6):
            m = analyse(patch, sigma_grad=2.0, order=order)
            lab = classify(m)
            if m and m["dirs"] is not None and len(m["dirs"]) >= 2:
                e1 = min(_ang_err(m["dirs"][0], tr["d1"]), _ang_err(m["dirs"][0], tr["d2"]))
                e2 = min(_ang_err(m["dirs"][1], tr["d1"]), _ang_err(m["dirs"][1], tr["d2"]))
                errs = f"{e1:.0f}/{e2:.0f}"
            else:
                errs = "-"
            npk = 0 if m is None else m["n_peaks"]
            sep = np.nan if m is None else m["sep"]
            amp2 = np.nan if m is None else m["amp2"]
            aniso = np.nan if m is None else m["aniso"]
            cs = np.nan if m is None else m["c_s"]
            print(f"  {name:>14} | {order:5d} | {npk:7d} | {sep:5.0f} | {amp2:5.2f} | "
                  f"{aniso:6.2f} | {cs:5.2f} | {lab:12s} | {errs}")
    print("  VERDICT: compare the two 'true crossing' rows against the two 'kissing'")
    print("  rows above -- if the metrics are close, per-voxel gradient evidence")
    print("  cannot distinguish touching-but-separate bundles from a real crossing")
    print("  (the expected, honest limitation), and spatial/tractographic context")
    print("  beyond a single voxel would be needed to tell them apart.")
    print("=" * 78)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(kiss[h], cmap="gray"); axes[0].axvline(h, color="r", ls="--", lw=1)
    axes[0].set_title("kissing (segregated)"); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(cross[h], cmap="gray")
    axes[1].set_title("true crossing (overlapping)"); axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.suptitle("Phantom 3: kissing vs true crossing, same two angles (z-mid slice)")
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("1", "all"):
        test_phantom1()
    if stage in ("2", "all"):
        test_phantom2()
    if stage in ("3", "all"):
        test_phantom3()
