"""
s_calibration.py -- can the superposition test return CROSSING on i74-like data?

The question
------------
A full i74 run scored 67 junctions and demoted every one of them: S ranged
0.421-0.611 with a median of 0.488, and not one reached the 0.85 the CROSSING
verdict needs. Before treating that as a statement about the tissue, it has to
be shown that the OTHER answer was reachable at all.

It is not, arithmetically. With I_0 = 115 and a median arm-excess sum of 198.5,
a genuine crossing needs

    I_c = I_0 + (I_a - I_0) + (I_b - I_0) = 115 + 198.5 = 313.5

in a uint8 volume that stops at 255. The ceiling on S is (255-115)/198.5 =
0.705, below the 0.85 threshold. So 67/67 TANGENCY is a property of the
container, not the sample -- and a test that cannot return the other answer is
not evidence.

What this script separates
--------------------------
Three candidate causes, on a phantom whose truth is KNOWN, mapped onto i74's
measured intensity statistics (background ~115-183, single fibre ~214, ceiling
255):

    IDEAL      float, no clipping, exact I_0        does the estimator work?
    SCALED     + i74 offset/gain, still float       does the mapping alone break it?
    CLIPPED    + uint8 saturation at 255            how much is the container?
    EST_I0     + I_0 from background_level()        how much is I_0 misestimation?
    REALISTIC  clipped AND estimated I_0            what i74 actually gets

and, crossed with those, the ESTIMATOR ASYMMETRY: the arms are re-centred by
`_snap_to_ridge` before being read and the junction centre is not, so only the
denominator is optimised upward. `snap=False` turns that off.

A row is USABLE if the known crossing scores >= 0.85 and the known tangency
scores <= 0.65 -- i.e. both verdicts are reachable.

Run
---
    python s_calibration.py
    python s_calibration.py --quick
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    sys.path.insert(0, _p)

import self_optimising_tractography as S3                        # noqa: E402
from localized_phantoms import (raw_localized_crossing, raw_kissing,  # noqa: E402
                                CROSS_SHAPE, CROSS_R1, CROSS_TAPER)

# i74's measured intensity statistics, from the 96^3 crop at (384, 384, 384).
I74_BG = 115.0        # background_level(air_frac=0.05) on that crop
I74_TISSUE = 183.0    # median voxel -- the level a fibre actually sits ON
I74_FIBRE = 214.0     # median measured arm intensity
UINT8_MAX = 255.0

SEP_DEG = 60.0
KISS_GAP, KISS_CURV = 6.0, 0.028


def synth(phantom, bg, gain, clip=None, quantise=False):
    """Map a 0/1/2-valued phantom onto an intensity scale like i74's.

    bg   : intensity where there is no fibre
    gain : intensity ADDED by one fibre, so a crossing adds 2*gain
    clip : saturate here (255 for uint8); None leaves it unbounded
    """
    I = bg + gain * np.asarray(phantom, np.float64)
    if clip is not None:
        I = np.minimum(I, clip)
    if quantise:
        I = np.round(I)
    return I.astype(np.float32)


def _crossing_dirs():
    """The two true fibre directions, in (z, y, x) to match superposition_S."""
    half = np.deg2rad(SEP_DEG / 2.0)
    d1 = np.array([np.cos(half), np.sin(half), 0.0])     # (x, y, z)
    d2 = np.array([np.cos(half), -np.sin(half), 0.0])
    return d1[::-1], d2[::-1]


def _pick_scale(vol):
    """(sigma, rho) certified ONCE, on the un-remapped phantom.

    Selected on the base geometry rather than per condition, and that is a
    design choice not a shortcut. Every variant below is the SAME phantom under
    a different intensity mapping (pedestal, gain, uint8 clipping), and the
    point of the script is to isolate what that mapping does to S. Re-selecting
    the scale per variant would let the mapping move the scale too, so a change
    in S could no longer be attributed to the mapping alone -- the confound the
    whole design exists to remove.
    """
    from select_scales import select_scales
    sc = select_scales(vol, mode="global", verbose=False)
    return sc["sigma_global"], sc["rho_global"]


def _score(vol, centre, d1, d2, I0, snap, scale):
    e3 = S3.orientation_field(vol, *scale)
    S, Ic, Ia, Ib, _ = S3.superposition_S(vol, centre, d1, d2, I0=I0, e3=e3,
                                          snap=snap)
    return S, Ic, Ia, Ib


def conditions(bg, gain, tissue_floor):
    """(name, clip, I0_mode) for each thing being isolated."""
    return [
        ("IDEAL",     None,      "true"),
        ("SCALED",    None,      "true"),
        ("CLIPPED",   UINT8_MAX, "true"),
        ("EST_I0",    None,      "estimated"),
        ("REALISTIC", UINT8_MAX, "estimated"),
        ("TISSUE_I0", UINT8_MAX, "tissue"),
    ]


# ---------------------------------------------------------------------------
# part 2: which I_0 estimator, under the intensity regime i74 actually has
# ---------------------------------------------------------------------------
#
# Part 1 shows the CROSSING verdict is unreachable once uint8 clips. But that
# clipping is itself a consequence of referencing I_0 to AIR: it makes one
# fibre worth 99 grey levels, so a crossing is predicted at 313. Referenced to
# the tissue a fibre is embedded in, the same fibre is worth 31 and the
# crossing lands at 245 -- inside the container. Those are two different models
# of the same volume, and only one of them fits.
#
# So the regimes are swept explicitly, and the I_0 ESTIMATOR is chosen against
# known truth rather than by whichever spreads the real i74 scores widest.

REGIMES = [
    # name              pedestal      fibre contrast              surrounded by air?
    ("air-referenced",  I74_BG,       I74_FIBRE - I74_BG,         False),   # 115 + 99
    ("tissue-pedestal", I74_TISSUE,   I74_FIBRE - I74_TISSUE,     False),   # 183 + 31
    # The first two cannot distinguish the I_0 ESTIMATORS from each other,
    # because a crop that is uniformly pedestal-plus-fibres has no air in it:
    # background_level's 5% quantile lands on the pedestal, so it agrees with
    # every local quantile by accident. i74 is not like that -- its histogram
    # is trimodal (air 115 / tissue 183 / fibre 214) and the two references
    # disagree by 68 grey levels. This regime reproduces that, and it is the
    # only one in which the choice of estimator is actually being tested.
    ("air+tissue",      I74_TISSUE,   I74_FIBRE - I74_TISSUE,     True),
]


def _tissue_block(shape, frac=0.62):
    """Central cube of tissue, air outside -- i74's air/tissue/fibre structure."""
    m = np.zeros(shape, bool)
    sl = tuple(slice(int(s * (1 - frac) / 2), int(s * (1 + frac) / 2))
               for s in shape)
    m[sl] = True
    return m

ESTIMATORS = ["air", "local_q10", "local_q25", "local_q50", "true"]


def estimate_I0(kind, vol, centre, true_bg):
    if kind == "true":
        return float(true_bg)
    if kind == "air":
        return float(S3.background_level(vol))
    q = {"local_q10": 0.10, "local_q25": 0.25, "local_q50": 0.50}[kind]
    return S3.local_baseline(vol, centre, half=S3.BASELINE_HALF, q=q)


def run_estimators(radius=20, verbose=True):
    """Known crossing vs known tangency, per regime, per I_0 estimator."""
    X = raw_localized_crossing(CROSS_SHAPE, SEP_DEG, 12.0, 1.5,
                               CROSS_R1, CROSS_TAPER)
    K = raw_kissing(KISS_GAP, KISS_CURV, 1.6, 1.6)
    xv = X[tuple(slice(c - radius, c + radius)
                 for c in (s // 2 for s in X.shape))]
    kv = K[tuple(slice(c - radius, c + radius)
                 for c in (s // 2 for s in K.shape))]
    scale = _pick_scale(xv)
    xc = tuple(s // 2 for s in xv.shape)
    kc = tuple(s // 2 for s in kv.shape)
    d1, d2 = _crossing_dirs()

    rows = []
    if verbose:
        print("\n" + "=" * 92)
        print("PART 2 -- which I_0 estimator, under each intensity regime "
              "(uint8 clipped throughout)")
        print("=" * 92)
        print(f"\n  {'regime':<16} {'I_0 est':<10} | {'I_0':>6} "
              f"{'CROSSING':>8} | {'KISS':>7} | {'verdict':>9}")

    for rname, ped, gain, in_air in REGIMES:
        xs = synth(xv, ped, gain, clip=UINT8_MAX, quantise=True)
        ks = synth(kv, ped, gain, clip=UINT8_MAX, quantise=True)
        if in_air:
            for arr in (xs, ks):
                arr[~_tissue_block(arr.shape)] = I74_BG
        ex = S3.orientation_field(xs, *scale)
        ek = S3.orientation_field(ks, *scale)
        for est in ESTIMATORS:
            I0x = estimate_I0(est, xs, xc, ped)
            I0k = estimate_I0(est, ks, kc, ped)
            Sx = S3.superposition_S(xs, xc, d1, d2, I0=I0x, e3=ex)[0]
            Sk = S3.superposition_S(ks, kc, d1, d2, I0=I0k, e3=ek)[0]
            ok = (np.isfinite(Sx) and Sx >= 0.85
                  and np.isfinite(Sk) and Sk <= 0.65)
            rows.append(dict(part=2, regime=rname, estimator=est,
                             I0=float(I0x), S_crossing=float(Sx),
                             S_kissing=float(Sk), usable=bool(ok)))
            if verbose:
                print(f"  {rname:<16} {est:<10} | {I0x:6.1f} {Sx:8.3f} | "
                      f"{Sk:7.3f} | {'USABLE' if ok else '-':>9}")
    return rows


def run(radius=20, verbose=True):
    X = raw_localized_crossing(CROSS_SHAPE, SEP_DEG, 12.0, 1.5,
                               CROSS_R1, CROSS_TAPER)
    K = raw_kissing(KISS_GAP, KISS_CURV, 1.6, 1.6)
    xv = X[tuple(slice(c - radius, c + radius)
                 for c in (s // 2 for s in X.shape))]
    kv = K[tuple(slice(c - radius, c + radius)
                 for c in (s // 2 for s in K.shape))]
    scale = _pick_scale(xv)
    xc = tuple(s // 2 for s in xv.shape)
    kc = tuple(s // 2 for s in kv.shape)
    d1, d2 = _crossing_dirs()

    gain = I74_FIBRE - I74_BG          # 99: what one fibre adds above background
    rows = []

    if verbose:
        print("=" * 92)
        print("S CALIBRATION -- is the CROSSING verdict reachable on i74-like "
              "intensities?")
        print(f"  bg={I74_BG:g}  fibre={I74_FIBRE:g}  gain={gain:g}  "
              f"ceiling={UINT8_MAX:g}  need S>=0.85 for CROSSING")
        print("=" * 92)
        print(f"\n  {'condition':<11} {'snap':>5} | {'CROSSING':>8} {'I_c':>7} "
              f"{'I_a':>7} | {'KISS':>6} | {'verdict':>9}")

    for name, clip, i0_mode in conditions(I74_BG, gain, I74_TISSUE):
        # IDEAL keeps the phantom's own 0/1/2 scale; everything else is i74-like
        bg, g = (0.0, 1.0) if name == "IDEAL" else (I74_BG, gain)
        xs = synth(xv, bg, g, clip=clip, quantise=(clip is not None))
        ks = synth(kv, bg, g, clip=clip, quantise=(clip is not None))

        for snap in (True, False):
            if i0_mode == "true":
                I0x = I0k = bg
            elif i0_mode == "tissue":
                # the level a fibre sits ON, not the air level
                I0x, I0k = I74_TISSUE, I74_TISSUE
            else:
                I0x = S3.background_level(xs)
                I0k = S3.background_level(ks)

            Sx, Ic, Ia, _Ib = _score(xs, xc, d1, d2, I0x, snap, scale)
            # the kissing pair's two bundles run near-parallel at closest
            # approach, so both contributors are read along the same tangent
            Sk, _kc, _ka, _kb = _score(ks, kc, d1, d2, I0k, snap, scale)

            ok = np.isfinite(Sx) and Sx >= 0.85 and np.isfinite(Sk) and Sk <= 0.65
            rows.append(dict(condition=name, snap=bool(snap), clip=clip,
                             I0_mode=i0_mode, S_crossing=float(Sx),
                             S_kissing=float(Sk), I_c=float(Ic), I_a=float(Ia),
                             I0=float(I0x), usable=bool(ok)))
            if verbose:
                print(f"  {name:<11} {str(snap):>5} | {Sx:8.3f} {Ic:7.1f} "
                      f"{Ia:7.1f} | {Sk:6.3f} | "
                      f"{'USABLE' if ok else '-':>9}")

    if verbose:
        usable = [r for r in rows if r["usable"]]
        print("\n" + "-" * 92)
        if usable:
            print(f"  {len(usable)}/{len(rows)} configurations can return BOTH "
                  f"verdicts.")
            for r in usable:
                print(f"    {r['condition']} snap={r['snap']}: "
                      f"crossing {r['S_crossing']:.3f}, kissing "
                      f"{r['S_kissing']:.3f}")
        else:
            print("  NO configuration can return both verdicts -- the CROSSING "
                  "side is unreachable")
        print("=" * 92)
    return rows


def main(outdir=None, radius=20):
    outdir = outdir or os.path.join(_ROOT, "results")
    rows = run(radius=radius)
    rows += run_estimators(radius=radius)

    part2 = [r for r in rows if r.get("part") == 2 and r["usable"]]
    print("\n" + "-" * 92)
    if part2:
        print("  I_0 estimators that recover BOTH verdicts under uint8:")
        for r in part2:
            print(f"    {r['regime']:<16} {r['estimator']:<10} "
                  f"crossing {r['S_crossing']:.3f}, kissing {r['S_kissing']:.3f}")
    else:
        print("  NO estimator recovers both verdicts under uint8.")
    print("=" * 92)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "s_calibration.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {path}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="superposition-S calibration")
    ap.add_argument("--radius", type=int, default=20)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    main(outdir=a.outdir, radius=a.radius)
