"""
select_scales.py -- the single entry point for "what scale does this run use?"

Every (sigma, rho) the pipeline touches is chosen here and written to one JSON,
so a run's scales can be read off an artifact instead of recovered by grepping
for float literals. Before this, scale selection was split three ways and only
one third of it was reachable from the pipeline:

    per-block adaptive   `sigma_rho_adaptive.compute_adaptive_map`   ran, then
                         its answer was recorded and never read again
    one global pick      `sigma_rho_global.py`                       reachable
                         only by hand, never from `run_pipeline`
    everything else      hardcoded                                   sigma=1.0
                         rho=2.0 for the dense e3, rho=3.0 for D_obs

The two modes
-------------
    mode="global"    one (sigma, rho) certified on the whole crop.
    mode="adaptive"  one (sigma, rho) per block, plus a consensus pair.

Both modes always return a `sigma_global` / `rho_global`, because the pipeline
owes a single recommended pair for the volume as a deliverable regardless of how
the field itself was built. In global mode it is the pick; in adaptive mode it
is the MEDIAN over certified blocks -- a consensus, not a re-derivation, so the
two numbers cannot disagree about what the same data said.

Why the median and not the mean: block picks come off a discrete grid and a
handful of blocks land on the grid edge (`sigma_rho_robustness` recorded one box
picking 6.0 against its neighbours' 0.9). A mean carries those outliers into the
consensus; the median does not.

Uncertified blocks
------------------
A block with no pick is not an error -- it falls back to SIGMA_FLOOR and the run
continues -- but when MOST blocks fail there is no adaptive selection happening
at all, just one hardcoded fallback wearing the word "adaptive". `certified_frac`
is reported for exactly that reason, and the two failure causes are counted
separately because they mean different things: a background block means "nothing
here", a below-floor pick means "the pick itself is not trustworthy".

Run
---
    python select_scales.py --i74                  # adaptive, real data
    python select_scales.py --i74 --mode global
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

from pick_sigma_rho import (pick_sigma_rho, first_local_max,      # noqa: E402
                            lindeberg_response_3d)
from sigma_rho_adaptive import (compute_adaptive_map, SIGMA_FLOOR,  # noqa: E402
                                SIG_LIND, SIG_GRID, RHO_GRID)

# Warn when fewer than this fraction of blocks certify a (sigma, rho).
FALLBACK_WARN_FRAC = 0.5
DEFAULT_BLOCK = 32


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def select_scales(vol, mode="adaptive", block=DEFAULT_BLOCK,
                  sigmas_lind=SIG_LIND, sigmas_grid=SIG_GRID,
                  rhos_grid=RHO_GRID, verbose=True):
    """Certify the scales this run will use. Returns a `scales` dict.

    Keys:
        mode                    "global" | "adaptive"
        sigma_global, rho_global    the pair for the whole volume (always set)
        sigma_map, rho_map      (nz, ny, nx) per-block picks, or None in global
                                mode; NaN where a block did not certify
        zs, ys, xs, block       the block tiling those maps index
        certified_frac          fraction of blocks with a pick
        n_background, n_below_floor     why the rest have none
        evidence                what was swept, for the record
    """
    vol = np.asarray(vol, dtype=np.float32)
    if mode not in ("global", "adaptive"):
        raise ValueError(f"mode must be 'global' or 'adaptive', got {mode!r}")

    base = dict(mode=mode, block=int(block),
                evidence=dict(sigmas_lindeberg=[float(s) for s in sigmas_lind],
                              sigmas_grid=[float(s) for s in sigmas_grid],
                              rhos_grid=[float(r) for r in rhos_grid],
                              sigma_floor=float(SIGMA_FLOOR),
                              volume_shape=[int(s) for s in vol.shape]))

    if mode == "global":
        if verbose:
            print(f"  [scales] global pick on {vol.shape} ...")
        rho_certified = True
        try:
            sigma, rho, diag = pick_sigma_rho(vol, sigmas_lind, sigmas_grid,
                                              rhos_grid, rho_ref=1.0,
                                              verbose=False)
        except ValueError:
            # `sweep_rho_stability_3d` takes nanargmin over its sensitivity
            # surface, which is all-NaN when no voxel has a finite e3 at BOTH
            # rhos of a pair. That happens on a sparse volume -- a single thin
            # fibre in air, or a crop that is mostly background -- where the
            # tensor is degenerate almost everywhere, and it is a real
            # condition, not a bug: there is nothing there whose stability
            # could be measured.
            #
            # Falling back rather than propagating, because sigma is still
            # certifiable from the Lindeberg response alone and a usable sigma
            # with an uncertified rho beats no scale at all. rho = sigma is the
            # minimal choice (the same rule `scale_at` already applies when a
            # block has sigma but no rho), and `rho_certified` records that it
            # was NOT measured, so a caller can tell the two cases apart.
            _s, response = lindeberg_response_3d(vol, sigmas_lind, rho=1.0,
                                                 verbose=False)
            i = first_local_max(np.asarray(sigmas_lind, float), response)
            i = int(np.argmax(response)) if i is None else i
            sigma = float(np.asarray(sigmas_grid, float)[
                int(np.argmin(np.abs(np.asarray(sigmas_grid, float)
                                     - sigmas_lind[i])))])
            rho = sigma
            rho_certified = False
            diag = dict(response=response)
            if verbose:
                print(f"  [scales] rho could not be certified (the stability "
                      f"sweep found no voxel with a finite e3 at consecutive "
                      f"rhos -- the volume is too sparse). Using rho = sigma = "
                      f"{sigma:g}, recorded as uncertified.")
        # A pick under the discretisation floor is not a pick: below ~0.85 vox
        # the discrete Gaussian derivative stops being a faithful derivative
        # (noise_floor.py T1). Say so rather than silently using it.
        certified = sigma >= SIGMA_FLOOR
        if not certified:
            if verbose:
                print(f"  [scales] WARNING: global sigma={sigma:.2f} is below the "
                      f"discretisation floor {SIGMA_FLOOR:g}; clamping.")
            sigma = float(SIGMA_FLOOR)
        base["evidence"]["lindeberg_response"] = [float(v) for v in diag["response"]]
        base["evidence"]["lindeberg_is_ramp"] = bool(
            first_local_max(np.asarray(sigmas_lind, float), diag["response"]) is None)
        base.update(sigma_global=float(sigma), rho_global=float(rho),
                    sigma_map=None, rho_map=None, zs=None, ys=None, xs=None,
                    certified_frac=1.0 if certified else 0.0,
                    rho_certified=bool(rho_certified),
                    n_background=0, n_below_floor=0 if certified else 1)
        if verbose:
            print(f"  [scales] global pick: sigma={sigma:.3g}, rho={rho:.3g}")
        return base

    # --- adaptive ----------------------------------------------------------
    if verbose:
        print(f"  [scales] adaptive pick on {block}^3 blocks ...")
    m = compute_adaptive_map(vol, block=block, sigmas_lind=sigmas_lind,
                             sigmas_grid=sigmas_grid, rhos_grid=rhos_grid,
                             verbose=False)
    sig_map, rho_map = m["sigma_map"], m["rho_map"]
    fin = np.isfinite(sig_map)
    n_blocks = int(sig_map.size)
    n_cert = int(fin.sum())
    frac = n_cert / n_blocks if n_blocks else 0.0
    n_background = n_blocks - n_cert
    n_below_floor = int((fin & ~m["valid"]).sum())

    # Consensus over the blocks that certified AND cleared the floor -- the
    # deliverable pair for the volume. Blocks below the floor are excluded
    # rather than clamped: including them would drag the consensus toward a
    # value the method itself rejects.
    good = fin & m["valid"]
    if good.any():
        sigma_g = float(np.median(sig_map[good]))
        rho_ok = rho_map[good][np.isfinite(rho_map[good])]
        # Every certified block can still have failed to pin a rho. Falling back
        # to the sigma consensus keeps the pair self-consistent instead of
        # emitting a NaN that would surface later as a silent kernel of size nan.
        rho_g = float(np.median(rho_ok)) if rho_ok.size else sigma_g
    else:
        sigma_g, rho_g = float(SIGMA_FLOOR), float(SIGMA_FLOOR)

    if verbose:
        if n_cert:
            print(f"  [scales] {n_cert}/{n_blocks} blocks certified, "
                  f"sigma {np.nanmin(sig_map):.2f}-{np.nanmax(sig_map):.2f}")
        if frac < FALLBACK_WARN_FRAC:
            why = []
            if n_background:
                why.append(f"{n_background} background (block mean < 5% of max)")
            if n_below_floor:
                why.append(f"{n_below_floor} below the discretisation floor")
            print(f"  [scales] WARNING: only {n_cert}/{n_blocks} blocks produced a "
                  f"(sigma, rho)" + (" -- " + ", ".join(why) if why else "") + ".")
            print(f"           The rest fall back to sigma={SIGMA_FLOOR:g}; "
                  f"adaptive selection did not really run for this volume.")
        print(f"  [scales] consensus for the volume: sigma={sigma_g:.3g}, "
              f"rho={rho_g:.3g}  (median over {int(good.sum())} certified blocks)")

    base.update(sigma_global=sigma_g, rho_global=rho_g, rho_certified=True,
                sigma_map=sig_map, rho_map=rho_map,
                zs=list(m["zs"]), ys=list(m["ys"]), xs=list(m["xs"]),
                certified_frac=float(frac), n_background=int(n_background),
                n_below_floor=int(n_below_floor))
    return base


def scale_at(scales, centre):
    """The (sigma, rho) that applies at a voxel centre, in either mode.

    This lookup used to be a closure inside `build_gated_field`, which meant
    Stage 2 was the only thing that could ask the question and the rule could
    not be tested on its own. In global mode every centre gets the same pair,
    so a caller never has to branch on the mode.
    """
    if scales["mode"] == "global" or scales.get("sigma_map") is None:
        return float(scales["sigma_global"]), float(scales["rho_global"])

    sig_map = np.asarray(scales["sigma_map"], float)
    rho_map = np.asarray(scales["rho_map"], float)
    zs, ys, xs = scales["zs"], scales["ys"], scales["xs"]

    def _bin(starts, v):
        return int(np.clip(np.searchsorted(starts, v, "right") - 1,
                           0, len(starts) - 1))

    bi, bj, bk = (_bin(zs, centre[0]), _bin(ys, centre[1]), _bin(xs, centre[2]))
    s, r = sig_map[bi, bj, bk], rho_map[bi, bj, bk]
    if not np.isfinite(s) or s < SIGMA_FLOOR:
        # Fall back to the volume CONSENSUS, not to the bare floor. The floor is
        # a lower bound on what is measurable, not an estimate of this volume's
        # fibre scale; using it as a fallback quietly analysed uncertified
        # blocks at a scale nothing had chosen.
        return float(scales["sigma_global"]), float(scales["rho_global"])
    return float(s), float(r if np.isfinite(r) else s)


def paint_maps(scales, shape):
    """(sigma_map, rho_map) as dense per-voxel volumes, for export.

    The picks are per BLOCK, so this is an upsampling of block-resolution
    evidence, not a per-voxel measurement -- the same honesty `field_export`
    applies to the patch-resolution labels. It is worth exporting anyway: it is
    the only way to see, in a viewer and against the raw data, WHERE the scale
    selection changed its mind, which a (nz, ny, nx) array of numbers does not
    show.

    In global mode both volumes are constant, which is the correct picture of
    what global mode did.
    """
    shape = tuple(int(s) for s in shape)
    sig = np.full(shape, float(scales["sigma_global"]), np.float32)
    rho = np.full(shape, float(scales["rho_global"]), np.float32)
    if scales.get("sigma_map") is None:
        return sig, rho

    sig_map = np.asarray(scales["sigma_map"], float)
    rho_map = np.asarray(scales["rho_map"], float)
    zs, ys, xs, block = scales["zs"], scales["ys"], scales["xs"], scales["block"]
    for bi, z0 in enumerate(zs):
        for bj, y0 in enumerate(ys):
            for bk, x0 in enumerate(xs):
                s, r = sig_map[bi, bj, bk], rho_map[bi, bj, bk]
                if not np.isfinite(s):
                    continue                      # leave the consensus fallback
                sl = (slice(z0, min(z0 + block, shape[0])),
                      slice(y0, min(y0 + block, shape[1])),
                      slice(x0, min(x0 + block, shape[2])))
                sig[sl] = s
                rho[sl] = r if np.isfinite(r) else s
    return sig, rho


def distinct_scales(scales, centres):
    """{(sigma, rho): [centres...]} -- the grouping that makes the dense tensor
    affordable. One structure-tensor computation per DISTINCT pair, not per
    patch; in global mode that collapses to a single computation."""
    out = {}
    for c in centres:
        out.setdefault(scale_at(scales, c), []).append(c)
    return out


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_scales(scales, path):
    """Write scales.json. Arrays become nested lists with NaN as null."""
    def _clean(a):
        if a is None:
            return None
        arr = np.asarray(a, float)
        return [[[None if not np.isfinite(v) else float(v) for v in row]
                 for row in plane] for plane in arr]

    out = dict(scales)
    out["sigma_map"] = _clean(scales.get("sigma_map"))
    out["rho_map"] = _clean(scales.get("rho_map"))
    for k in ("zs", "ys", "xs"):
        if out.get(k) is not None:
            out[k] = [int(v) for v in out[k]]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def load_scales(path):
    """Read back what `save_scales` wrote, with null restored to NaN."""
    with open(path) as f:
        s = json.load(f)
    for k in ("sigma_map", "rho_map"):
        if s.get(k) is not None:
            s[k] = np.array([[[np.nan if v is None else float(v) for v in row]
                              for row in plane] for plane in s[k]], float)
    return s


def summary(scales):
    """One-line description, for logs and figure subtitles."""
    if scales["mode"] == "global":
        return (f"global sigma={scales['sigma_global']:.3g}, "
                f"rho={scales['rho_global']:.3g}")
    n = np.asarray(scales["sigma_map"], float)
    return (f"adaptive {scales['block']}^3 blocks, "
            f"{scales['certified_frac']:.0%} certified, consensus "
            f"sigma={scales['sigma_global']:.3g}, rho={scales['rho_global']:.3g} "
            f"({int(np.isfinite(n).sum())} picks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="certify (sigma, rho) for a run")
    ap.add_argument("--mode", choices=("adaptive", "global"), default="adaptive")
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    ap.add_argument("--i74", action="store_true", help="real volume")
    ap.add_argument("--centre", nargs=3, type=int, default=[384, 384, 384])
    ap.add_argument("--radius", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "scales.json"))
    a = ap.parse_args()

    if a.i74:
        from core.reconstruction import load_volume
        lo = a.centre[0] - a.radius
        vol = load_volume(crop=(lo, a.centre[0] + a.radius),
                          dtype=np.uint8).astype(np.float32)
    else:
        from core.localized_phantoms import (raw_localized_crossing, CROSS_SHAPE,
                                        CROSS_R1, CROSS_TAPER)
        vol = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5,
                                     CROSS_R1, CROSS_TAPER)

    print("=" * 78)
    print(f"SCALE SELECTION -- mode={a.mode}, volume {vol.shape}")
    print("=" * 78)
    s = select_scales(vol, mode=a.mode, block=a.block)
    print("-" * 78)
    print("  " + summary(s))
    print(f"  wrote {save_scales(s, a.out)}")
    print("=" * 78)
