"""
loss_sensitivity.py -- does L actually respond to theta?

Why
---
Before choosing a descent algorithm it has to be shown that the loss depends on
what is being descended. Measured on i74 at the converged theta, it did not:

    tissue voxels            884,736
    streamlines              396
    D_rec mean over tissue   0.0057      (D_obs is 1.0 by construction)
    D_rec nonzero            1.27% of tissue
    MSE as-is                1.4368
    MSE with NO streamlines  1.4367      <- the same to four decimals

Deleting every tract moved the loss by 0.0001, and the verdict penalty (which
needs a junction density above 1) saw a median of 0.019. The optimiser was
searching a flat landscape, so a better search would have bought nothing.

Three causes, all fixed in `self_optimising_tractography`:

  1. SEED COUNT was capped at the patch count -- `rng.choice(..., replace=False)`
     over 344 HOST patches. `seed_density` may now exceed 1 and repeated seeds
     are jittered within their patch.
  2. REACHABILITY -- `integrate` stops when no patch lies within `assoc_radius`,
     so tracts live only near patch centres.
  3. SUPPORT MISMATCH -- D_rec was scored against D_obs over all tissue, most of
     which no tract may enter. `host_support` restricts the comparison, and
     `renormalise` puts the target on that same support.

This script sweeps seed density and reports, for each, the coverage, the loss,
and the loss with the streamlines deleted. The gap between those last two is
the sensitivity: while it is ~0 no optimiser can work.

Run
---
    python loss_sensitivity.py            # phantom
    python loss_sensitivity.py --i74
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import self_optimising_tractography as S3                       # noqa: E402
from tract_density import rasterise                             # noqa: E402
from sigma_rho_gated_field import GatedField, build_gated_field  # noqa: E402

DENSITIES = [0.5, 2.0, 8.0, 32.0]
ASSOC = 8.0


def measure(vol, gated, densities=DENSITIES, assoc=ASSOC, seed=0, verbose=True):
    _sc, e3, lam = S3.fields_for(vol)
    field, junctions, _I0 = S3.field_from_gated(gated, vol, e3=e3, host_only=True)
    if not field:
        print("  no HOST patches -- nothing to measure")
        return []

    # Same decomposition as the field the streamlines follow.
    D_obs, tissue = S3.anisotropy_target(vol, lam=lam)
    support = tissue & S3.host_support(field, vol.shape, assoc)
    D_obs_s = S3.renormalise(D_obs, support)

    if verbose:
        print(f"  {len(field)} HOST patches, {len(junctions)} junctions")
        print(f"  tissue {int(tissue.sum())} voxels | reachable support "
              f"{int(support.sum())} ({support.mean():.1%} of volume, "
              f"{support.sum()/max(tissue.sum(),1):.1%} of tissue)")
        print(f"\n  {'seed_d':>7} {'streams':>8} {'cover':>7} {'D_rec':>8} "
              f"{'L':>9} {'L(no tracts)':>13} {'sensitivity':>12} {'s':>6}")

    rows = []
    zero = float(np.mean(D_obs_s[support] ** 2))       # L with no streamlines
    for sd in densities:
        theta = dict(S3.THETA0)
        theta.update(seed_density=float(sd), assoc_radius=assoc)
        t0 = time.time()
        streams, w = S3.integrate(field, vol.shape, theta,
                                  rng=np.random.default_rng(seed))
        D_rec = rasterise(streams, vol.shape, step=float(theta["step"]),
                          weights=w)
        L = float(np.mean((D_rec[support] - D_obs_s[support]) ** 2))
        dt = time.time() - t0
        cover = float((D_rec[support] > 0).mean())
        dens = [S3.density_at(D_rec, j["centre"], half=1) for j in junctions]
        rows.append(dict(seed_density=sd, n_streams=len(streams),
                         coverage=cover, D_rec_mean=float(D_rec[support].mean()),
                         L=L, L_no_tracts=zero, sensitivity=zero - L,
                         junction_density_median=float(np.median(dens)) if dens else 0.0,
                         seconds=dt))
        if verbose:
            print(f"  {sd:7.1f} {len(streams):8d} {cover:6.1%} "
                  f"{D_rec[support].mean():8.4f} {L:9.4f} {zero:13.4f} "
                  f"{zero - L:12.4f} {dt:6.1f}")
    if verbose and rows:
        jd = rows[-1]["junction_density_median"]
        print(f"\n  junction TDI density at the densest setting: median {jd:.3f} "
              f"(the verdict penalty needs > 1 to fire)")
    return rows


def main(i74=False, radius=48, stride=6, block=32, outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results")
    print("=" * 96)
    print("LOSS SENSITIVITY -- does L respond to the streamlines at all?")
    print("=" * 96)

    if i74:
        from reconstruction import load_volume
        lo = 384 - radius
        vol = load_volume(crop=(lo, 384 + radius), dtype=np.uint8).astype(np.float32)
        cache = os.path.join(outdir, "unified_field_i74.npz")
        gated = (GatedField.load(cache) if os.path.exists(cache)
                 else build_gated_field(vol, block=block, stride=stride))
        print(f"  i74 {vol.shape}")
    else:
        from localized_phantoms import (raw_localized_crossing, CROSS_SHAPE,
                                        CROSS_R1, CROSS_TAPER)
        X = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5,
                                   CROSS_R1, CROSS_TAPER)
        r = 24
        vol = X[tuple(slice(c - r, c + r)
                      for c in (s // 2 for s in CROSS_SHAPE))].astype(np.float32)
        gated = build_gated_field(vol, block=block, stride=stride, verbose=False)
        print(f"  phantom {vol.shape}")

    rows = measure(vol, gated)
    os.makedirs(outdir, exist_ok=True)
    tag = "i74" if i74 else "phantom"
    path = os.path.join(outdir, f"loss_sensitivity_{tag}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {path}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="loss sensitivity to streamlines")
    ap.add_argument("--i74", action="store_true")
    ap.add_argument("--radius", type=int, default=48)
    ap.add_argument("--stride", type=int, default=6)
    a = ap.parse_args()
    main(i74=a.i74, radius=a.radius, stride=a.stride)
