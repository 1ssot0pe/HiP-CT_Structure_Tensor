"""
Why: sigma_rho_robustness.py's multi-box consistency check found the pick
drifts badly across real i74 tissue
    - box A (mean intensity 200) picked sigma_L=0.90
    - box C (mean 160) picked sigma_L=0.88
    - box B (mean 105, sparser tissue) picked sigma_L=6.00
Verdict: A single global (sigma, rho) for whole volume not defensible
    - Similiar to per patch sigma_grad for HOST selection!

Process:
  (1) compute_adaptive_map(region, block, ...): tile a (loaded) 3D region into
      non-overlapping blocks, run pick_sigma_rho independently on each
      (skipping background/air blocks by a mean-intensity check), and return
      per-block (sigma, rho) plus validity/ramp/floor flags -- the map.

  (2) fibre_direction_adaptive(region, block, sigma_map, rho_map, valid, ...):
      consume that map to build the actual whole-region e3 (fibre direction)
      field: each block is recomputed with an EXTRA context margin pulled
      from its neighbours in `region` (so the structure-tensor kernel has
      real support near block edges, not a hard zero-padded boundary), then
      cropped back to the block's own interior and stitched into the output.
      
      Blocks use their OWN locally-certified (sigma, rho), not one global
      value -- piecewise-constant across blocks (not smoothly blended; see
      module docstring note below on why that's an intentional, simpler
      choice for now).

Kept at real-but-modest scale (a demonstration region + a handful of blocks,
not the full 768^3 volume) so the whole thing finishes in single-digit
minutes -- per-block certification costs the same ~30-60s it always has;
scaling to the full volume is a matter of budget, not a different algorithm.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root
from reconstruction import load_volume, VOXEL_SIZE_UM
from orthogonality_residual import fibre_direction, crop_margin

from pick_sigma_rho import pick_sigma_rho, first_local_max

SIGMA_FLOOR = 0.85
TRUNCATE = 4.0

SIG_LIND = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
SIG_GRID = np.array([1.0, 2.0, 3.0, 4.0])
RHO_GRID = np.array([1.0, 2.0, 3.0])

# Demonstration region: one 96-thick z-slab, 3x3 blocks in-plane (288x288),
# inside the same general tissue area as sigma_rho_robustness.BOXES.
REGION = (slice(100, 196), slice(150, 438), slice(150, 438))
BLOCK = 96


# ---------------------------------------------------------------------------
# (1) The adaptive (sigma, rho) map
# ---------------------------------------------------------------------------

def block_starts(n, block):
    """Non-overlapping block start indices covering [0, n); the last block is
    clipped to end exactly at n (so it may overlap its neighbour slightly
    rather than running off the edge or leaving a ragged remainder)."""
    starts = list(range(0, n - block + 1, block))
    if not starts:
        return [0]
    if starts[-1] + block < n:
        starts.append(n - block)
    return starts


def compute_adaptive_map(region, block=BLOCK, sigmas_lind=SIG_LIND,
                          sigmas_grid=SIG_GRID, rhos_grid=RHO_GRID,
                          min_mean_frac=0.05, verbose=True):
    """Tile `region` into `block`-sized cells and certify (sigma, rho)
    independently in each via pick_sigma_rho. Returns a dict with per-axis
    block start-index lists and (nz, ny, nx) arrays: sigma_map, rho_map,
    valid (bool), ramp (bool, no interior Lindeberg peak in that block)."""
    zs = block_starts(region.shape[0], block)
    ys = block_starts(region.shape[1], block)
    xs = block_starts(region.shape[2], block)
    shape = (len(zs), len(ys), len(xs))
    sigma_map = np.full(shape, np.nan)
    rho_map = np.full(shape, np.nan)
    valid = np.zeros(shape, dtype=bool)
    ramp = np.zeros(shape, dtype=bool)
    thresh = min_mean_frac * region.max()

    n_total = shape[0] * shape[1] * shape[2]
    k = 0
    for bi, z0 in enumerate(zs):
        for bj, y0 in enumerate(ys):
            for bk, x0 in enumerate(xs):
                k += 1
                blk = region[z0:z0 + block, y0:y0 + block, x0:x0 + block]
                if verbose:
                    print(f"  block {k}/{n_total} (z0={z0},y0={y0},x0={x0}) "
                          f"mean={blk.mean():.0f} ...", end="")
                if blk.mean() < thresh:
                    if verbose:
                        print(" background, skipped")
                    continue
                sigma_L, rho_L, diag = pick_sigma_rho(
                    blk.astype(np.float32), sigmas_lind, sigmas_grid, rhos_grid,
                    rho_ref=1.0, verbose=False)
                is_ramp = first_local_max(sigmas_lind, diag["response"]) is None
                sigma_map[bi, bj, bk] = sigma_L
                rho_map[bi, bj, bk] = rho_L
                valid[bi, bj, bk] = sigma_L >= SIGMA_FLOOR
                ramp[bi, bj, bk] = is_ramp
                if verbose:
                    print(f" sigma={sigma_L:.2f}, rho={rho_L:.2f}"
                          f"{'  [BELOW FLOOR]' if sigma_L < SIGMA_FLOOR else ''}"
                          f"{'  [RAMP, no interior peak]' if is_ramp else ''}")

    return dict(zs=zs, ys=ys, xs=xs, block=block, sigma_map=sigma_map,
                rho_map=rho_map, valid=valid, ramp=ramp)


def summarize_map(m):
    sv = m["sigma_map"][np.isfinite(m["sigma_map"])]
    rv = m["rho_map"][np.isfinite(m["rho_map"])]
    n = sv.size
    print("-" * 88)
    print(f"  {n} tissue blocks certified (of {m['sigma_map'].size} total)")
    if n:
        print(f"  sigma: min={sv.min():.2f} max={sv.max():.2f} range={np.ptp(sv):.2f} "
              f"median={np.median(sv):.2f}")
        print(f"  rho  : min={rv.min():.2f} max={rv.max():.2f} range={np.ptp(rv):.2f} "
              f"median={np.median(rv):.2f}")
        drifted = np.ptp(sv) > 1.0 or np.ptp(rv) > 1.0
        print("  " + ("PICK DRIFTS across blocks within this single region -- confirms a "
                       "spatially-adaptive map is needed even WITHIN one contiguous area, "
                       "not only across widely-separated boxes."
                       if drifted else
                       "pick is fairly stable across blocks in this region."))
    print("-" * 88)


# ---------------------------------------------------------------------------
# (2) Consuming the map: whole-region fibre-direction field, per-block scale
# ---------------------------------------------------------------------------
#
# NOTE (scope): this stitches PIECEWISE-CONSTANT per-block results (each
# block computed at its own certified sigma,rho, then cropped to its own
# interior and placed edge-to-edge) rather than smoothly blending scales
# across block boundaries. 
# 
# A voxel-continuous blend would need structure
# tensors at MULTIPLE candidate scales per voxel and a soft combination rule
# 
# Piecewise-constant is simple,
# correct within each block's interior (real context margin, not zero-padded
# edges), and already directly answers the finding that drove this: different
# regions genuinely need different scales.

def fibre_direction_adaptive(region, m, truncate=TRUNCATE, verbose=True):
    zs, ys, xs, block = m["zs"], m["ys"], m["xs"], m["block"]
    out = np.full((3,) + region.shape, np.nan, dtype=np.float32)
    n_total = len(zs) * len(ys) * len(xs)
    k = 0
    for bi, z0 in enumerate(zs):
        for bj, y0 in enumerate(ys):
            for bk, x0 in enumerate(xs):
                k += 1
                sigma_L, rho_L = m["sigma_map"][bi, bj, bk], m["rho_map"][bi, bj, bk]
                if not np.isfinite(sigma_L):
                    continue
                pad = crop_margin(sigma_L, rho_L, truncate)
                z0p, z1p = max(z0 - pad, 0), min(z0 + block + pad, region.shape[0])
                y0p, y1p = max(y0 - pad, 0), min(y0 + block + pad, region.shape[1])
                x0p, x1p = max(x0 - pad, 0), min(x0 + block + pad, region.shape[2])
                ctx = region[z0p:z1p, y0p:y1p, x0p:x1p].astype(np.float32)
                if verbose:
                    print(f"  block ({bi},{bj},{bk}): sigma={sigma_L:.2f} rho={rho_L:.2f} "
                          f"context {ctx.shape} ...", end="")
                e3 = fibre_direction(ctx, sigma_L, rho_L, truncate)
                # slice out just this block's own interior from the padded context
                sl = (slice(z0 - z0p, z0 - z0p + block),
                      slice(y0 - y0p, y0 - y0p + block),
                      slice(x0 - x0p, x0 - x0p + block))
                out[:, z0:z0 + block, y0:y0 + block, x0:x0 + block] = e3[(slice(None),) + sl]
                if verbose:
                    print(" done")
    return out


def fibre_direction_single_global(region, sigma, rho, truncate=TRUNCATE):
    """Baseline for comparison: one global (sigma, rho) for the whole region."""
    return fibre_direction(region.astype(np.float32), sigma, rho, truncate)


# ---------------------------------------------------------------------------
# Demonstration on real i74
# ---------------------------------------------------------------------------

def run(region_slices=REGION, block=BLOCK, global_sigma=0.90, global_rho=3.46,
        save="sigma_rho_adaptive_map.png"):
    
    print("Whole-region demonstration on real i74")

    full = load_volume(crop=None, dtype=np.uint8)
    region = full[region_slices].astype(np.float32)
    del full
    print(f"  region {region.shape}, mean={region.mean():.0f} ({VOXEL_SIZE_UM:g} um/vox), "
          f"block={block}")

    m = compute_adaptive_map(region, block=block)
    summarize_map(m)

    print("\n  building adaptive-scale fibre-direction field ...")
    e3_adaptive = fibre_direction_adaptive(region, m, verbose=True)
    print(f"  finite fraction (adaptive): {np.isfinite(e3_adaptive).all(axis=0).mean():.1%}")

    print(f"\n  building SINGLE-global-scale field for comparison "
          f"(sigma={global_sigma}, rho={global_rho}, the sigma_rho_global.py box-A pick) ...")
    e3_global = fibre_direction_single_global(region, global_sigma, global_rho)
    print(f"  finite fraction (single global): {np.isfinite(e3_global).all(axis=0).mean():.1%}")
    print("=" * 88)

    _figure(m, e3_adaptive, e3_global, save)
    return m, e3_adaptive, e3_global


def _figure(m, e3_adaptive, e3_global, save):
    zmid = e3_adaptive.shape[1] // 2
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    im0 = axes[0, 0].imshow(m["sigma_map"][0], cmap="viridis")
    axes[0, 0].set_title("adaptive sigma_L per block (z-layer 0)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(m["rho_map"][0], cmap="magma")
    axes[0, 1].set_title("adaptive rho_L per block (z-layer 0)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    rgb_a = np.abs(e3_adaptive[:, zmid]).transpose(1, 2, 0)
    axes[1, 0].imshow(np.nan_to_num(rgb_a))
    axes[1, 0].set_title("|e3| RGB, ADAPTIVE (sigma,rho), mid z-slice")

    rgb_g = np.abs(e3_global[:, zmid]).transpose(1, 2, 0)
    axes[1, 1].imshow(np.nan_to_num(rgb_g))
    axes[1, 1].set_title("|e3| RGB, SINGLE GLOBAL (sigma,rho), same slice")

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Whole-region adaptive sigma/rho vs a single global pick", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save, dpi=130)
    print(f"Saved {save}")


if __name__ == "__main__":
    run()
