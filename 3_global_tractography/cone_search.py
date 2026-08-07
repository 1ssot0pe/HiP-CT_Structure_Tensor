"""
cone_search.py -- a 3-parameter cone tracker, scored on kissing vs crossing.

    L      cone LENGTH   (voxels)   how far ahead evidence is gathered
    theta  cone WIDTH    (radians)  the widest turn allowed in one step
    alpha  ANGULAR STEP  (radians)  how finely the cone is sampled

At each step, look ahead inside a cone about the current heading and take the
best-supported continuation. Support is image brightness times |d . e3| -- the
absolute value throughout, because the field is NEMATIC: e3 and -e3 are the same
fibre, so a candidate agreeing with -e3 agrees exactly as much as one agreeing
with +e3.

What is being measured, and why it is not the TDI
-------------------------------------------------
A tract-density loss CANNOT separate a kiss from a crossing: both put fibre mass
in the same voxels, which is the whole difficulty. Scoring this tracker by TDI
error would measure something real (does the field explain the image) but not
this.

What a tracker uniquely answers is CONTINUITY -- not "is it bright here" but
"does anything actually continue through". So the two phantoms are scored by
what the tracts DO:

    TRAVERSAL  (crossing phantom)  fraction of tracts that enter the junction
                                   and come out the FAR side. Want HIGH: a real
                                   crossing should be traversable.
    BRIDGE     (kissing phantom)   fraction of tracts starting on bundle A that
                                   end on bundle B. Want ZERO: the bundles never
                                   touch, so any bridge is a false merge.

These pull against each other and (theta, L) is the dial between them. A wide
cone with a long lookahead traverses everything -- including the gap it must not
cross. A narrow cone bridges nothing but also traverses nothing. The useful
operating point is the largest traversal at zero bridging, and whether such a
point exists at all is the question this sweep answers.

Run
---
    python cone_search.py                 # default sweep, both phantoms
    python cone_search.py --quick
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    sys.path.insert(0, _p)

import self_optimising_tractography as S3               # noqa: E402
from field_refine import fill_invalid                   # noqa: E402
from localized_phantoms import (raw_localized_crossing, raw_kissing,  # noqa: E402
                                CROSS_SHAPE, CROSS_R1, CROSS_TAPER)

EPS = 1e-12
STEP = 1.0            # voxels advanced per iteration; L is LOOKAHEAD, not step
MAX_STEPS = 90
JUNCTION_R = 5.0      # voxels: "inside the junction"
KISS_GAP = 6.0        # gap between the two kissing bundles
KISS_CURV = 0.028


# ---------------------------------------------------------------------------
# the tracker
# ---------------------------------------------------------------------------

def _basis(v):
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, v)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    a = a - np.dot(a, v) * v
    a /= np.linalg.norm(a) + EPS
    return a, np.cross(v, a)


def cone_directions(v, theta, alpha):
    """Directions within `theta` of v, sampled at angular resolution `alpha`.

    Work scales as (theta/alpha)^2, so the two cannot be swept independently
    without watching the clock -- that coupling is why alpha is a discretisation
    knob rather than a model parameter.
    """
    a, b = _basis(v)
    out = [v]
    for i in range(1, max(int(round(theta / alpha)), 0) + 1):
        tilt = i * alpha
        n_phi = max(int(round(2 * np.pi * np.sin(tilt) / alpha)), 1)
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            d = (np.cos(tilt) * v
                 + np.sin(tilt) * (np.cos(phi) * a + np.sin(phi) * b))
            out.append(d / (np.linalg.norm(d) + EPS))
    return np.asarray(out)


def _sample(vol, pts):
    return map_coordinates(vol, pts.T, order=1, mode="nearest")


def support(vol, e3, x, dirs, L, n=6):
    """Image brightness x |d . e3|, averaged along a ray of length L."""
    t = np.linspace(STEP, L, n)[:, None]
    pts = x[None, None, :] + t[None, :, :] * dirs[:, None, :]
    flat = pts.reshape(-1, 3)
    inten = _sample(vol, flat).reshape(len(dirs), n)
    ez = np.stack([_sample(e3[k], flat) for k in range(3)], -1)
    ez = ez.reshape(len(dirs), n, 3)
    return (inten * np.abs(np.einsum("dnc,dc->dn", ez, dirs))).mean(axis=1)


def track(vol, e3, seeds, alpha, L, theta, floor):
    """Cone-search streamlines from `seeds`, walked both ways (nematic)."""
    paths, shape = [], vol.shape
    for s in seeds:
        v0 = e3[:, int(s[0]), int(s[1]), int(s[2])]
        if not np.all(np.isfinite(v0)) or np.linalg.norm(v0) < EPS:
            continue
        v0 = v0 / np.linalg.norm(v0)
        for sgn in (1.0, -1.0):
            x, v, path = s.astype(float).copy(), sgn * v0, [s.astype(float).copy()]
            for _ in range(MAX_STEPS):
                dirs = cone_directions(v, theta, alpha)
                sc = support(vol, e3, x, dirs, L)
                k = int(np.argmax(sc))
                if sc[k] < floor:
                    break
                v = dirs[k]
                x = x + STEP * v
                if not all(0 <= x[i] < shape[i] for i in range(3)):
                    break
                path.append(x.copy())
            if len(path) >= 2:
                paths.append(np.asarray(path))
    return paths


# ---------------------------------------------------------------------------
# the two scores
# ---------------------------------------------------------------------------

def traversal(paths, centre, r=JUNCTION_R):
    """Fraction of tracts that enter the junction and leave the FAR side.

    "Far side" is decided by the sign of the displacement from the centre at
    entry versus exit -- a tract that comes in and turns back has both on the
    same side and does not count.
    """
    c = np.asarray(centre, float)
    hit = through = 0
    for p in paths:
        d = np.linalg.norm(p - c, axis=1)
        inside = np.where(d <= r)[0]
        if inside.size == 0:
            continue
        hit += 1
        i0, i1 = inside[0], inside[-1]
        if i0 == 0 or i1 == len(p) - 1:
            continue                      # started or ended inside: no verdict
        if float(np.dot(p[i0 - 1] - c, p[i1 + 1] - c)) < 0:
            through += 1
    return (through / hit) if hit else 0.0, hit


def bridged(paths, centre_x, margin=1.0):
    """Fraction of tracts that start on bundle A and end on bundle B.

    The kissing bundles sit either side of x = centre_x (they are the curves
    x = +/- (gap/2 + curv*y^2)), so which bundle a point is on is simply the
    sign of x - centre_x. Any tract that changes that sign has merged two
    fibres that never touch.
    """
    n = cross = 0
    for p in paths:
        x0, x1 = p[0][2] - centre_x, p[-1][2] - centre_x
        if abs(x0) < margin or abs(x1) < margin:
            continue                      # too close to the midline to attribute
        n += 1
        if x0 * x1 < 0:
            cross += 1
    return (cross / n) if n else 0.0, n


# ---------------------------------------------------------------------------

def _prepare(vol, seed_stride):
    # The scale is selected on this volume, not assumed. The tracker follows e3,
    # so a default (sigma, rho) here would make the whole sweep a statement
    # about a field the pipeline never builds.
    _sc, e3, lam = S3.fields_for(vol)
    e3 = fill_invalid(e3)
    _D, tissue = S3.anisotropy_target(vol, lam=lam)
    zz, yy, xx = np.where(tissue)
    seeds = np.stack([zz, yy, xx], 1)[::seed_stride].astype(float)
    return e3, seeds, 0.05 * float(vol.max())


def sweep(alphas, Ls, thetas, radius=20, seed_stride=24, verbose=True):
    X = raw_localized_crossing(CROSS_SHAPE, 60.0, 12.0, 1.5, CROSS_R1, CROSS_TAPER)
    K = raw_kissing(KISS_GAP, KISS_CURV, 1.6, 1.6)
    xv = X[tuple(slice(c - radius, c + radius)
                 for c in (s // 2 for s in X.shape))].astype(np.float32)
    kv = K[tuple(slice(c - radius, c + radius)
                 for c in (s // 2 for s in K.shape))].astype(np.float32)
    xc = tuple(s // 2 for s in xv.shape)
    kcx = kv.shape[2] // 2

    xe, xs, xf = _prepare(xv, seed_stride)
    ke, ks, kf = _prepare(kv, seed_stride)
    if verbose:
        print(f"  crossing {xv.shape} {len(xs)} seeds | "
              f"kissing {kv.shape} {len(ks)} seeds, gap={KISS_GAP:g}")
        print(f"\n  {'alpha':>6} {'L':>5} {'theta':>6} | {'TRAVERSAL':>10} "
              f"{'(n)':>5} | {'BRIDGE':>8} {'(n)':>5} | {'verdict':>9} {'s':>5}")

    rows = []
    for a, L, th in itertools.product(alphas, Ls, thetas):
        t0 = time.time()
        tr, nt = traversal(track(xv, xe, xs, a, L, th, xf), xc)
        br, nb = bridged(track(kv, ke, ks, a, L, th, kf), kcx)
        dt = time.time() - t0
        good = tr > 0.0 and br == 0.0
        rows.append(dict(alpha=a, L=L, theta=th, traversal=tr, n_traversal=nt,
                         bridge=br, n_bridge=nb, usable=bool(good), seconds=dt))
        if verbose:
            print(f"  {a:6.2f} {L:5.1f} {th:6.2f} | {tr:9.1%} {nt:5d} | "
                  f"{br:7.1%} {nb:5d} | {'USABLE' if good else '-':>9} {dt:5.1f}")
    return rows


def main(quick=False, radius=20, outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results")
    if quick:
        alphas, Ls, thetas = [0.20], [3.0, 8.0], [0.20, 0.50]
    else:
        alphas, Ls, thetas = [0.10, 0.20], [2.0, 4.0, 8.0], [0.15, 0.30, 0.50, 0.70]

    print("=" * 88)
    print("CONE SEARCH -- can (alpha, L, theta) traverse a crossing WITHOUT "
          "bridging a kiss?")
    print("=" * 88)
    rows = sweep(alphas, Ls, thetas, radius=radius)

    usable = [r for r in rows if r["usable"]]
    print("\n" + "-" * 88)
    if usable:
        best = max(usable, key=lambda r: r["traversal"])
        print(f"  {len(usable)}/{len(rows)} settings separate the two phantoms.")
        print(f"  best: alpha={best['alpha']:.2f} L={best['L']:.1f} "
              f"theta={best['theta']:.2f} -> traversal {best['traversal']:.1%}, "
              f"bridge {best['bridge']:.1%}")
    else:
        anyt = [r for r in rows if r["traversal"] > 0]
        print("  NO setting traverses the crossing while bridging nothing.")
        print(f"  ({len(anyt)}/{len(rows)} traverse at all; every one of those "
              f"also bridges the kiss)")
        print("  -> continuity alone cannot separate them at this gap; the "
              "verdict must come from superposition.")
    print("=" * 88)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "cone_search.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="cone-search kissing vs crossing")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--radius", type=int, default=20)
    a = ap.parse_args()
    main(quick=a.quick, radius=a.radius)
