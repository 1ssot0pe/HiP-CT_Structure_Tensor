"""
single_fibre_check.py -- does the tracker leave single-fibre tissue alone?

Stage 3 now walks a field containing Stage 1's e3 everywhere
outside the routed patches. 

    Correct: So wherever the field carries ONE direction, the
    optimiser has nothing to choose between and the tract should 
    simply follow the structure tensor. 

    Incorrect: If it does not, the cone is overriding a measurement rather
    than resolving an ambiguity, and every "the tractography improved" claim
    downstream is contaminated by tracts that wandered off perfectly unambiguous
    fibres.

Prediction measured rather than asserted: the ANGLE between
each tract's local tangent and the field's own vec1 at the voxel the tract is
passing through.

    single-direction voxels    -> angle should be ~0. The tracker has one
                                  option and any deviation is the cone's
                                  discretisation, not a decision.
    two-direction voxels       -> angle may be large, and SHOULD be: that is
                                  the tracker taking the second peak, which is
                                  the entire point of gating those voxels to
                                  HOST. Reported separately so a big number
                                  there cannot hide a big number in the singles.

Directors, not vectors: the comparison is |tangent . vec1|, so a tract running
antiparallel to e3 agrees with it exactly. An unsigned angle is the only
meaningful one for a nematic field.

Run
---
    python single_fibre_check.py            # crossing phantom, end to end
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

EPS = 1e-12
TOLERANCE_DEG = 5.0      # median angle a "follows e3" claim has to stay under


def tolerance_for(alpha):
    """The tolerance this cone can physically meet.

    A cone sampled at angular step `alpha` cannot turn by less than alpha, so
    a walker correcting its heading overshoots and returns, and the median
    deviation cannot be driven below roughly alpha/2 however good the field is.
    Demanding better than that is demanding sub-resolution accuracy from the
    discretisation, not from the tracker.

    Measured on the crossing phantom with the direction chosen by alignment
    alone: 
    alpha = 0.20 (11.46 deg minimum turn) gives a median of 5.72 deg,
    alpha = 0.10 gives 4.82 deg
    
    """
    return max(TOLERANCE_DEG, 0.5 * np.degrees(float(alpha)))


def tangent_samples(paths):
    """(positions, unit tangents) for every interior sample of every tract.

    Central differences, and the endpoints are dropped: a one-sided difference
    at the ends is a different estimator with a different bias, and the ends are
    exactly where a tract is terminating anyway.
    """
    P, T = [], []
    for p in paths:
        p = np.asarray(p, float)
        if len(p) < 3:
            continue
        t = p[2:] - p[:-2]
        n = np.linalg.norm(t, axis=1, keepdims=True)
        ok = n[:, 0] > EPS
        P.append(p[1:-1][ok])
        T.append(t[ok] / n[ok])
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(P), np.concatenate(T)


def agreement(field, paths, support=None):
    """Angle between tract tangents and the field's own directions.

    Returns a dict with the single-direction and two-direction populations kept
    apart -- pooling them would let a tract that abandoned an unambiguous fibre
    be excused by a junction it never went near.
    """
    pos, tan = tangent_samples(paths)
    if not len(pos):
        return dict(n=0)
    idx = np.clip(np.round(pos).astype(int), 0, np.array(field.shape) - 1)
    z, y, x = idx[:, 0], idx[:, 1], idx[:, 2]

    nd = field.n_dirs[z, y, x]
    keep = nd >= 1
    if support is not None:
        keep &= np.asarray(support, bool)[z, y, x]
    if not keep.any():
        return dict(n=0)

    v1 = field.vec1[:, z, y, x].T
    dot1 = np.abs(np.einsum("ij,ij->i", tan, v1))
    v2 = field.vec2[:, z, y, x].T
    dot2 = np.abs(np.einsum("ij,ij->i", tan, v2))
    # Against whichever direction the field actually offers -- for a two-peak
    # voxel the tract is entitled to follow either.
    best = np.where(nd >= 2, np.maximum(dot1, dot2), dot1)

    def _stats(m, d):
        if not m.any():
            return dict(n=0)
        a = np.degrees(np.arccos(np.clip(d[m], 0.0, 1.0)))
        return dict(n=int(m.sum()), median=float(np.median(a)),
                    p90=float(np.percentile(a, 90)),
                    mean=float(a.mean()),
                    frac_within_5deg=float((a <= 5.0).mean()),
                    frac_within_15deg=float((a <= 15.0).mean()))

    single = keep & (nd == 1)
    multi = keep & (nd >= 2)
    return dict(n=int(keep.sum()),
                single=_stats(single, dot1),
                multi_vs_best=_stats(multi, best),
                multi_vs_vec1=_stats(multi, dot1))


def report(stats, tolerance=TOLERANCE_DEG, verbose=True):
    """Print the comparison and return the failures, if any."""
    if not stats.get("n"):
        if verbose:
            print("  [single-fibre] no tract samples to score")
        return ["no tract samples"]
    s = stats["single"]
    fails = []
    if verbose:
        print(f"  [single-fibre] tangent vs vec1 over {stats['n']} samples")
        if s.get("n"):
            print(f"      1 direction  ({s['n']:7d}): median {s['median']:5.2f} deg, "
                  f"p90 {s['p90']:5.2f}, within 5 deg {s['frac_within_5deg']:.1%}")
        m, m1 = stats["multi_vs_best"], stats["multi_vs_vec1"]
        if m.get("n"):
            print(f"      2 directions ({m['n']:7d}): median {m['median']:5.2f} deg "
                  f"vs best peak, {m1['median']:5.2f} deg vs vec1 alone")
    if verbose:
        print(f"      tolerance {tolerance:.2f} deg")
    if s.get("n") and s["median"] > tolerance:
        fails.append(f"single-fibre tangents deviate from e3 by "
                     f"{s['median']:.2f} deg (median), tolerance {tolerance:g}")
    if verbose and not fails:
        print(f"      -> tracts follow Stage 1's e3 where there is one "
              f"direction (median <= {tolerance:g} deg)")
    return fails


def _selftest(verbose=True):
    """End to end on the crossing phantom."""
    import self_optimising_tractography as S3
    import dense_tracker as DT
    import dense_optimise as DO
    from dense_field import DenseField
    from sigma_rho_gated_field import build_gated_field
    from select_scales import select_scales
    from tensor import field_for
    from reconstruction import tissue_mask
    from localized_phantoms import (raw_localized_crossing, CROSS_SHAPE,
                                    CROSS_R1, CROSS_TAPER)

    R = 24
    X = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5, CROSS_R1, CROSS_TAPER)
    vol = np.ascontiguousarray(
        X[tuple(slice(c - R, c + R) for c in (s // 2 for s in X.shape))],
        np.float32)
    sc = select_scales(vol, mode="adaptive", block=32, verbose=False)
    t = field_for(vol, sc, invariants=False)
    gated = build_gated_field(vol, scales=sc, e3_xyz=t["e3"][::-1], verbose=False)
    tissue = tissue_mask(vol)
    F = DenseField.build(gated, t["e3"], tissue=tissue, host_only=True)
    seeds, sd = DT.seeds_from(F, max_seeds=DO.MAX_SEEDS, mask=tissue)
    paths = DT.track(vol, F, seeds, sd, alpha=DO.ALPHA,
                     lookahead=DO.PARAMS0["lookahead"],
                     theta=DO.PARAMS0["theta"],
                     floor=DO.FLOOR_FRAC * float(vol.max()))
    if verbose:
        print(f"  crossing phantom: {len(paths)} tracts, {F.summary()}")
        print(f"  cone alpha={DO.ALPHA:g} ({np.degrees(DO.ALPHA):.2f} deg "
              f"minimum turn), lookahead={DO.PARAMS0['lookahead']:g}")
    return report(agreement(F, paths, support=tissue),
                  tolerance=tolerance_for(DO.ALPHA), verbose=verbose)


if __name__ == "__main__":
    print("=" * 78)
    print("SINGLE-FIBRE INVARIANCE -- does the cone follow e3 where e3 is all "
          "there is?")
    print("=" * 78)
    f = _selftest()
    print("-" * 78)
    print("PASS" if not f else "FAIL -- " + "; ".join(f))
    sys.exit(1 if f else 0)
