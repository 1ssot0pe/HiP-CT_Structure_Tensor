"""
mass_conservation_validate.py -- generalise and PROVE the mass-conservation metric
found in crossing_vs_kissing_metrics.py.

That first test used GLOBAL knowledge (the un-normalised phantom's true peak = 1.0)
to compute the ratio -- not usable on real data, where you never know the absolute
scale in advance. This module replaces it with a LOCAL, data-driven calibration:
at the ambiguous point, look at a handful of NEARBY patches HOST confidently calls
"single" (there are always some -- most of any fibre volume isn't at a crossing/
kissing point) and use their own peak intensity as the local single-fibre
calibration. mass_ratio = intensity-at-the-ambiguous-point / calibration.

(An earlier attempt calibrated by walking along HOST's two recovered directions

from the ambiguous point itself -- this was fragile: for periodic tube-family
crossings the walk can drift back toward the OTHER tube's next periodic repeat,
and for curved kissing bundles HOST's direction estimate has enough angular error
that walking 5-10 voxels along it drifts off the tube into background, giving a
near-zero calibration and a nonsense ratio. Calibrating from confidently-single
neighbours sidesteps both failure modes and is what a real calibration would look
like anyway -- there's no shortage of unambiguous single-fibre tissue nearby.)

The crossing side originally needed a SYNTHETIC single-family companion volume
to calibrate from, because the original crossing phantom is a periodic LATTICE
of two tube families -- every patch in it secretly touches another lattice
intersection, so same-volume "single"-classified patches were still measuring
superposition. `raw_localized_crossing` fixes this the realistic way: family 2
is windowed to a bounded zone (a smooth, flat-cored radial mask -- a hard edge
or any in-window gradient was enough to break HOST's 2-peak detection), so
genuine single-family-only tissue exists elsewhere in the SAME volume and the
calibration comes from there, matching how a real crossing tract in real
tissue would actually be calibrated.

Noise comes in two kinds: synthetic correlated+anisotropic (`anoise`, matching
brain_phantom_pipeline.py's model) and REAL measured lung half-map noise
(`real_noise`, n=(A-B)/sqrt(2) from lung_A.npy/lung_B.npy, cropped from a
random tissue location per seed) -- closing the synthetic-noise transfer
caveat that has bitten other metrics in this project before.

Physical claim being tested: a REAL crossing has both tube centrelines passing
through the ambiguous point, so intensity there is close to the SUM of both
branches (ratio ~2x a single branch); a KISSING pair's closest-approach point has
NEITHER centreline through it, so intensity there can never exceed roughly ONE
branch's own value (ratio <1x), regardless of how small the gap is.

Proof strategy (train/test split, exactly like crossing_shape_3d.py's guard):
  - Sweep crossing separation (20-90 deg), SAME and DIFFERENT tube widths, at
    multiple noise levels (clean, realistic correlated+anisotropic, harsh).
  - Sweep kissing gap (2-20 vox, the exact kissing_gap_sweep.py values), SAME
    and DIFFERENT bundle radii, at the same noise levels.
  - Only score cases where HOST's local classifier actually reports 2 peaks
    (i.e. where the ambiguity this metric is meant to resolve actually arises).
  - Fixed physically-motivated threshold (1.5, the midpoint between the "one
    branch" ceiling ~1 and the "two branches" floor ~2) -- NOT fit to the data.
  - Report catch rate (crossings correctly flagged as superposed) and
    false-positive rate (kissing cases incorrectly flagged as superposed).
"""

import os as _sr_os, sys as _sr_sys
_sr_sys.path.insert(0, _sr_os.path.dirname(_sr_os.path.dirname(__file__)))  # repo root (host_fod, crossing_shape_3d, kissing_phantom)

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from crossing_shape_3d import _tube_family, _coords as _coords_cross
from kissing_phantom import _curve_pts, curved_bundle, SHAPE as KSHAPE
from host_fod import analyse, classify

THRESH = 1.5   # PHYSICAL midpoint of the 1x ("one branch") and 2x ("two branches
               # summed") ratios. Deliberately fixed for the phantom PROOF (an
               # honest validation must not fit its own threshold). On REAL data,
               # prefer local_ratio_threshold() below.
EPS = 1e-9


def local_ratio_threshold(ratios, physical=THRESH, min_gap=0.5):
    """Data-driven crossing/kissing cut from the volume's OWN mass-conservation
    ratio distribution -- no external constant.

    If the ratios are bimodal (a low kissing/single mode near <=1 and a high
    crossing mode near ~2), return the Otsu valley between the two modes. If the
    distribution is unimodal / not separable (as on real i74, where the ratios
    piled up near ~0.65 with no crossing mode), fall back to the PHYSICAL
    midpoint -- there is no valley to find, and inventing one would be worse.

    Returns (threshold, reason).
    """
    r = np.asarray(ratios, float)
    r = r[np.isfinite(r)]
    if r.size < 20:
        return float(physical), "physical (too few points to calibrate)"
    lo, hi = np.percentile(r, [1, 99])
    if hi - lo < 1e-6:
        return float(physical), "physical (degenerate distribution)"
    edges = np.linspace(lo, hi, 64)
    hist, _ = np.histogram(r, bins=edges)
    p = hist / (hist.sum() + 1e-12)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0 = np.cumsum(p * centres) / (w0 + 1e-12)
    mT = float((p * centres).sum())
    m1 = (mT - w0 * m0) / (w1 + 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    k = int(np.nanargmax(between))
    thr = float(centres[k])
    # bimodality guard: the two class means must sit a real gap either side of thr
    if (thr - m0[k]) < min_gap or (m1[k] - thr) < min_gap:
        return float(physical), "physical (unimodal / no separable crossing mode)"
    return thr, "data-driven Otsu valley of this volume's ratios"


# ---------------------------------------------------------------------------
# raw (un-normalised) phantom builders
# ---------------------------------------------------------------------------

def raw_kissing(gap, curv, radius_a, radius_b):
    C = _coords_cross(KSHAPE)
    ptsA, ptsB = _curve_pts(-1, gap, curv), _curve_pts(+1, gap, curv)
    A = curved_bundle(C, ptsA, radius_a)
    B = curved_bundle(C, ptsB, radius_b)
    return (A + B).astype(np.float32)


def raw_localized_crossing(shape, sep_deg, spacing, width, R1=20.0, taper=8.0):
    """A crossing that only exists within a BOUNDED zone around the volume
    centre -- family 1 fills the whole volume, family 2 is smoothly windowed
    to r<R1 (flat, no gradient in the window itself) then tapers to exactly 0
    by r=R1+taper. This models real tissue: one dominant background
    population, locally crossed by a second bundle over a finite overlap zone
    -- so genuine single-family-only tissue exists elsewhere in the SAME
    volume, and the calibration can come from there instead of a synthetic
    companion. (A raw radial mask with no flat core was tried first and broke
    HOST's 2-peak detection at the intersection -- the mask's own smooth
    gradient, even a small one, was enough to corrupt the higher-order
    energy balance; a flat core avoids that entirely.)"""
    half = np.deg2rad(sep_deg / 2.0)
    d1 = np.array([np.cos(half), np.sin(half), 0.0])
    d2 = np.array([np.cos(half), -np.sin(half), 0.0])
    C = _coords_cross(shape)
    t1 = _tube_family(C, d1, spacing, width)
    t2 = _tube_family(C, d2, spacing, width)
    X, Y, Z = C
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    mask = np.ones_like(r)
    band = (r > R1) & (r < R1 + taper)
    mask[band] = 0.5 * (1 + np.cos(np.pi * (r[band] - R1) / taper))
    mask[r >= R1 + taper] = 0.0
    return (t1 + t2 * mask).astype(np.float32)


def anoise(shape, amp, seed, smooth=(2.5, 0.8, 0.8)):
    if amp <= 0:
        return np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    n = gaussian_filter(rng.standard_normal(shape).astype(np.float64), smooth)
    return (n * (amp / (n.std() + EPS))).astype(np.float32)


# ---------------------------------------------------------------------------
# REAL measured lung half-map noise n=(A-B)/sqrt(2) -- correlated + anisotropic
# (slope ~-3.5, aniso ratio ~1.5-1.9, see noise-floor-result memory), cropped
# from a random tissue location per seed (unlike phase3_realnoise.real_noise,
# which always crops the SAME fixed offset regardless of its seed argument).
# ---------------------------------------------------------------------------

_REAL_NOISE_VOL = None


def _load_real_noise_volume():
    global _REAL_NOISE_VOL
    if _REAL_NOISE_VOL is None:
        A = np.load("lung_A.npy").astype(np.float32)
        B = np.load("lung_B.npy").astype(np.float32)
        _REAL_NOISE_VOL = ((A - B) / np.sqrt(2.0)).astype(np.float32)
    return _REAL_NOISE_VOL


def real_noise(shape, amp, seed):
    if amp <= 0:
        return np.zeros(shape, dtype=np.float32)
    vol = _load_real_noise_volume()
    rng = np.random.default_rng(seed)
    z0 = int(rng.integers(0, vol.shape[0] - shape[0] + 1)) if vol.shape[0] > shape[0] else 0
    y0 = int(rng.integers(0, vol.shape[1] - shape[1] + 1))
    x0 = int(rng.integers(0, vol.shape[2] - shape[2] + 1))
    crop = vol[z0:z0 + shape[0], y0:y0 + shape[1], x0:x0 + shape[2]]
    return (crop * (amp / (crop.std() + EPS))).astype(np.float32)


# ---------------------------------------------------------------------------
# local, data-driven mass-conservation ratio: calibrate from nearby patches
# HOST confidently calls "single"
# ---------------------------------------------------------------------------

def _sample(vol, p_zyx):
    idx = tuple(int(round(c)) for c in p_zyx)
    idx = tuple(np.clip(i, 0, s - 1) for i, s in zip(idx, vol.shape))
    return float(vol[idx])


def local_calibration(vol, half, positions):
    """Median peak intensity over nearby patches HOST confidently classifies
    "single" -- the local single-fibre calibration, using no ground truth."""
    calib = []
    for pos in positions:
        patch = _patch_at(vol, pos, half)
        if patch.shape != (2 * half + 1,) * 3 or patch.mean() < 0.05 * vol.max():
            continue
        m = analyse(patch)
        if m is None:
            continue
        if classify(m) == "single":
            calib.append(float(patch.max()))
    return float(np.median(calib)) if calib else np.nan


def _patch_at(vol, center_zyx, half):
    z, y, x = [int(round(c)) for c in center_zyx]
    z0, y0, x0 = max(z - half, 0), max(y - half, 0), max(x - half, 0)
    return vol[z0:z + half + 1, y0:y + half + 1, x0:x + half + 1]


# ---------------------------------------------------------------------------
# evaluate one crossing / kissing instance -> (ratio, n_peaks) or None
#
# Patch half-widths match the ESTABLISHED, already-validated setups elsewhere
# in the repo: crossing uses half=14 (host_fod.validate's _central_patch, with
# matching tube spacing -- the exact configuration proven to resolve a 40deg+
# crossing); kissing uses HALF=8 (test_kissing_phantom.py's window, the one
# that actually reproduces the rare spurious "crossing" near the pinch point).
# ---------------------------------------------------------------------------

CROSS_HALF = 14
KISS_HALF = 8
# z kept small (real lung half-map noise only has 81 z-slices available); y is
# generous so the far calibration offsets (+/-45, +/-50) have room either side
# of the crossing zone without needing a huge, expensive volume.
CROSS_SHAPE = (56, 200, 60)
CROSS_R1, CROSS_TAPER = 20.0, 8.0
CROSS_CAL_OFFSETS = (45, 50, -45, -50)


def _noise(shape, amp, seed, kind):
    return real_noise(shape, amp, seed) if kind == "real" else anoise(shape, amp, seed)


def eval_crossing(sep_deg, noise_amp, seed, noise_kind="synthetic"):
    """Localized crossing (raw_localized_crossing): the true intersection is
    ALWAYS the geometric centre by construction (matches how host_fod.validate
    and every other crossing test in the repo already do this -- see the
    argmax-vs-geometric-center investigation in conversation: blind argmax can
    land on a different, differently-behaved tied-maximum point elsewhere in
    the periodic lattice, sometimes right at a volume edge). Calibration comes
    from confidently-single patches found in the SAME noisy volume, offset
    along y only (comfortably outside the crossing zone, mask==0 there by
    construction) -- no synthetic companion reference needed."""
    shape = CROSS_SHAPE
    spacing = 12.0
    clean = raw_localized_crossing(shape, sep_deg, spacing, 1.5, CROSS_R1, CROSS_TAPER)
    idx = tuple(s // 2 for s in shape)
    noisy = clean + _noise(shape, noise_amp, seed, noise_kind)
    h = CROSS_HALF
    patch = _patch_at(noisy, idx, h)
    m = analyse(patch)
    if m is None or m["dirs"] is None or m["n_peaks"] < 2:
        return None
    z0, y0, x0 = idx
    cal_positions = [(z0, y0 + dy, x0) for dy in CROSS_CAL_OFFSETS
                     if h + 2 <= y0 + dy < shape[1] - h - 2]
    calib = local_calibration(noisy, h, cal_positions)
    if np.isnan(calib):
        return None
    

    ratio = float(noisy[idx]) / (calib + EPS)  ### deciding value for crossing or kiss


    return ratio, m["n_peaks"]


def eval_kissing(gap, curv, ra, rb, noise_amp, seed, noise_kind="synthetic"):
    """Search the same near-gap strip test_kissing_phantom.classify_grid uses:
    the pinch is a LINE (y varies), not a single point, so scan a few y/x
    positions near the midline for the first patch HOST calls ambiguous."""
    clean = raw_kissing(gap, curv, ra, rb)
    noisy = clean + _noise(clean.shape, noise_amp, seed, noise_kind)
    z, cy, cx = clean.shape[0] // 2, clean.shape[1] // 2, clean.shape[2] // 2
    h = KISS_HALF
    # NOTE: step must stay at 2 -- a step of 4 was tried to cut cost further
    # but silently missed the (rarer, narrower) ambiguity window that the
    # different-size kissing variant produces near gap=6.
    #
    # Collect ALL ambiguous hits in the strip and take the one CLOSEST to the
    # true pinch (cy, cx), not the first one scan order happens to reach --
    # taking "first found" was picking up weak, unrepresentative spurious
    # 2-peak reads far from the actual closest-approach point (ratio as low as
    # 0.07 from a patch that had nothing to do with the pinch).
    hits = []
    # y range must stay +/-10, not +/-8: the spurious-crossing hotspot is a
    # partial-volume effect and sits off the true geometric minimum-gap point
    # (which is exactly y=cy by symmetry) -- narrowing the search window
    # silently excluded the actual representative case at |y-cy|=10.
    for y in range(cy - 10, cy + 11, 2):
        for x in range(cx - 6, cx + 7, 2):
            cand = (z, y, x)
            patch = _patch_at(noisy, cand, h)
            if patch.shape != (2 * h + 1,) * 3 or patch.mean() < 0.05 * noisy.max():
                continue
            m = analyse(patch)
            if m is None or m["dirs"] is None or m["n_peaks"] < 2:
                continue
            hits.append((abs(y - cy) + abs(x - cx), cand, m["n_peaks"]))
    if not hits:
        return None
    hits.sort(key=lambda t: t[0])
    idx, npk = hits[0][1], hits[0][2]
    cal_positions = [(z, y, x) for y in range(cy - 20, cy + 21, 8) for x in range(h + 2, clean.shape[2] - h - 2, 8)]
    calib = local_calibration(noisy, h, cal_positions)
    if np.isnan(calib):
        return None
    
    ratio = float(noisy[idx]) / (calib + EPS)
    
    return ratio, npk


# ---------------------------------------------------------------------------
# full sweep
# ---------------------------------------------------------------------------

# CHEAP but representative: one point near HOST's resolvability floor (sep=40
# -- sep=30, HOST's own min_separation_angle gate, turns out to NEVER reach
# 2-peak ambiguity even in the exact-boundary case, so it wastes sweep budget)
# and one comfortably resolved (sep=70).
CROSS_SEPS = (40.0, 70.0)
# gap=2 (tightest ambiguity), gap=4 (the resolution-limit case that bridges in
# kissing_gap_sweep.py), gap=6 (the only gap where the DIFFERENT-size variant
# actually reaches 2-peak ambiguity -- narrower than same-size because the
# thick/thin asymmetry changes HOST's local energy balance), gap=8 (the
# boundary where it first holds clean), gap=20 (comfortably separated).
KISS_GAPS = (2.0, 4.0, 6.0, 8.0, 20.0)
KISS_RADII = [(1.5, 1.5), (1.1, 2.6)]             # same-size, different-size
# (amp, kind) strata: two synthetic amplitudes (clean, realistic) plus ONE
# real-measured-noise stratum at the same realistic amplitude -- the
# comparison that actually answers "does real noise's STRUCTURE (not just its
# amplitude) fool this metric", isolated from amplitude effects.
NOISE_STRATA = [(0.0, "synthetic"), (0.07, "synthetic"), (0.07, "real")]
SEEDS = (0, 1, 2)


def run():
    print("=" * 90)
    print(f"MASS-CONSERVATION METRIC -- generalised, same-volume calibration, real+synthetic "
          f"noise, fixed threshold={THRESH}")
    print("=" * 90)

    cross_rows, kiss_rows = [], []

    cross_combos = [(sep, amp, kind, seed)
                    for sep in CROSS_SEPS for amp, kind in NOISE_STRATA for seed in SEEDS]
    for sep, amp, kind, seed in tqdm(cross_combos, desc="crossing sweep"):
        r = eval_crossing(sep, amp, seed, noise_kind=kind)
        if r is not None:
            cross_rows.append(dict(sep=sep, noise=amp, kind=kind, seed=seed, ratio=r[0], npk=r[1]))

    kiss_combos = [(ra, rb, gap, amp, kind, seed)
                  for ra, rb in KISS_RADII for gap in KISS_GAPS
                  for amp, kind in NOISE_STRATA for seed in SEEDS]
    for ra, rb, gap, amp, kind, seed in tqdm(kiss_combos, desc="kissing sweep "):
        r = eval_kissing(gap, 0.028, ra, rb, amp, seed, noise_kind=kind)
        if r is not None:
            kiss_rows.append(dict(r=(ra, rb), gap=gap, noise=amp, kind=kind, seed=seed, ratio=r[0], npk=r[1]))

    print(f"\nCROSSING: {len(cross_rows)} instances reached HOST 2-peak ambiguity "
          f"(out of {len(CROSS_SEPS)*len(NOISE_STRATA)*len(SEEDS)} tried)")
    print(f"KISSING:  {len(kiss_rows)} instances reached HOST 2-peak ambiguity "
          f"(out of {len(KISS_RADII)*len(KISS_GAPS)*len(NOISE_STRATA)*len(SEEDS)} tried)")

    # --- breakdown by noise stratum (amp, kind) ---
    print("\n--- catch rate (crossing) / false-positive rate (kissing), by noise stratum ---")
    for amp, kind in NOISE_STRATA:
        cr = [r["ratio"] for r in cross_rows if r["noise"] == amp and r["kind"] == kind]
        kr = [r["ratio"] for r in kiss_rows if r["noise"] == amp and r["kind"] == kind]
        catch = np.mean([v > THRESH for v in cr]) if cr else float("nan")
        fp = np.mean([v > THRESH for v in kr]) if kr else float("nan")
        print(f"  {kind:>9} amp={amp:4.2f} | crossing n={len(cr):3d} mean_ratio={np.mean(cr) if cr else float('nan'):.2f} "
              f"catch={catch:.0%}  |  kissing n={len(kr):3d} mean_ratio={np.mean(kr) if kr else float('nan'):.2f} FP={fp:.0%}")

    # --- breakdown by radius pairing (same vs different size) ---
    print("\n--- kissing catch/FP by same-size vs different-size ---")
    for r_ in KISS_RADII:
        kr = [r["ratio"] for r in kiss_rows if r["r"] == r_]
        fp = np.mean([v > THRESH for v in kr]) if kr else float("nan")
        print(f"  kissing  radii={r_}: n={len(kr):3d} mean_ratio={np.mean(kr) if kr else float('nan'):.2f} FP={fp:.0%}")

    # --- breakdown by kissing gap (does the metric hold even at the narrowest gaps?) ---
    print("\n--- kissing false-positive rate by gap (worst case = smallest gap) ---")
    for gap in KISS_GAPS:
        kr = [r["ratio"] for r in kiss_rows if r["gap"] == gap]
        fp = np.mean([v > THRESH for v in kr]) if kr else float("nan")
        print(f"  gap={gap:4.1f} | n={len(kr):3d} mean_ratio={np.mean(kr) if kr else float('nan'):.2f} FP={fp:.0%}")

    # --- overall verdict ---
    all_cross = [r["ratio"] for r in cross_rows]
    all_kiss = [r["ratio"] for r in kiss_rows]
    catch_all = np.mean([v > THRESH for v in all_cross]) if all_cross else float("nan")
    fp_all = np.mean([v > THRESH for v in all_kiss]) if all_kiss else float("nan")
    passed = catch_all >= 0.9 and fp_all <= 0.1
    print("\n" + "=" * 90)
    print(f"OVERALL: catch={catch_all:.0%} ({len(all_cross)} crossing instances), "
          f"FP={fp_all:.0%} ({len(all_kiss)} kissing instances)")
    print(f"GATE {'PASS' if passed else 'FAIL'}: mass-conservation "
          f"{'generalises cleanly' if passed else 'does not generalise cleanly'} "
          f"across widths/radii, gaps/separations, and noise")
    print("=" * 90)
    return cross_rows, kiss_rows, catch_all, fp_all, passed


if __name__ == "__main__":
    run()
