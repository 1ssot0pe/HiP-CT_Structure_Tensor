"""
phase3_i74.py -- scale the Phase-2.5 joint global model from 2D phantoms to REAL 3D
i74 tissue.

The 2D model (phase3_global.py) is dimension-agnostic in its two workhorses --
candidate_links (vector dots) and greedy_forest (a graph) -- so this file only adds
(i) a 3D MULTI-ORIENTATION seeder that reuses HOST FODs (one segment per HOST peak,
the design-doc's L1 observation) and (ii) an i74-appropriate readout.

The honest i74 question (no ground truth here, per the host-gate work): at each voxel
the PER-VOXEL HOST pipeline calls a 2-way CROSSING, does the GLOBAL model link BOTH
peaks into extended tracts (-> a real crossing, both fibres continue) or leave one
peak unlinked/short (-> DEMOTED: a spurious lobe with no collinear continuation)?
This is the Phase-2 question, now answered by the joint model that PASSED phantom
validation (kissing=0 at every angle) -- not the local certifier that failed it.

Caveat carried forward: dense tissue is where Phase-2's reach saturated. The global
model is stricter (it needs a DIRECTION-COLLINEAR continuation, not just any in-cone
fibre), so it should over-connect less -- but with no i74 ground truth we report what
it does, we do not claim an accuracy.

Run:  .venv/Scripts/python -u phase3_i74.py [--crop N] [--stride S]
"""

import os as _sr_os, sys as _sr_sys
_sr_sys.path.insert(0, _sr_os.path.dirname(_sr_os.path.dirname(__file__)))  # repo root (shared kernels)

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reconstruction import load_volume, VOXEL_SIZE_UM
from host_fod import analyse, classify
import phase3_global as P                # reuse candidate_links, greedy_forest, tracts_from

HALF = 8                 # HOST patch half-width (17^3), matches phase 1/2
ANISO_MIN = 0.9          # PHYSICAL FALLBACK for the aniso gate (see calibrate_aniso)
FG_FRAC = 0.25           # skip air/background (patch-mean below this * global max) -- already local
ANISO_Q = 15.0           # skip the most-isotropic q% of foreground patches (data-driven gate)


def calibrate_aniso(vol, q=ANISO_Q, n_sample=250, seed=0):
    """Data-driven anisotropy gate from the volume's OWN distribution -- not 0.9.

    Sample foreground patches, read HOST anisotropy, and set the gate at the
    q-th percentile: skip the most-isotropic (noise-like) tail relative to THIS
    volume's tissue. Falls back to the physical ANISO_MIN if too few foreground
    patches are found. Returns (aniso_min, n_used)."""
    rng = np.random.default_rng(seed)
    thr = FG_FRAC * vol.max()
    los = [HALF] * 3
    his = [s - HALF for s in vol.shape]
    anis = []
    for _ in range(n_sample * 12):
        z = int(rng.integers(los[0], his[0]))
        y = int(rng.integers(los[1], his[1]))
        x = int(rng.integers(los[2], his[2]))
        patch = vol[z - HALF:z + HALF + 1, y - HALF:y + HALF + 1, x - HALF:x + HALF + 1]
        if patch.mean() < thr:
            continue
        m = analyse(patch)
        if m is None:
            continue
        anis.append(float(m["aniso"]))
        if len(anis) >= n_sample:
            break
    if len(anis) < 20:
        return ANISO_MIN, len(anis)
    return float(np.percentile(anis, q)), len(anis)


def seed_segments(vol, stride, max_peaks=2, aniso_min=ANISO_MIN):
    """3D multi-orientation seeding via HOST. At each strided, foreground, anisotropic
    voxel, place ONE segment per HOST peak (up to max_peaks). Returns pos [M,3] as
    (x,y,z)=(axis2,axis1,axis0) -- the same frame HOST dirs live in -- unit dirs
    [M,3], weights, and `is_cross` (did HOST classify that seed a 2-way crossing?).

    aniso_min: the anisotropy gate; pass calibrate_aniso(vol)[0] for the
    data-driven value (defaults to the physical ANISO_MIN)."""
    from reconstruction import iter_progress
    Z, Y, X = vol.shape
    thr = FG_FRAC * vol.max()
    pos, dirs, w, is_cross, seed_id = [], [], [], [], []
    seeds = [(z, y, x)
             for z in range(HALF, Z - HALF, stride)
             for y in range(HALF, Y - HALF, stride)
             for x in range(HALF, X - HALF, stride)]
    for sid, (z, y, x) in enumerate(iter_progress(seeds, desc="HOST seeding", enable=True)):
        patch = vol[z - HALF:z + HALF + 1, y - HALF:y + HALF + 1, x - HALF:x + HALF + 1]
        if patch.mean() < thr:
            continue
        m = analyse(patch)
        if m is None or m["aniso"] < aniso_min or m["dirs"] is None:
            continue
        cross = (classify(m) == "crossing")
        for d in m["dirs"][:max_peaks]:
            pos.append([float(x), float(y), float(z)])     # (x,y,z) = (axis2,axis1,axis0)
            dirs.append(np.asarray(d, float))
            w.append(float(m["aniso"])); is_cross.append(cross); seed_id.append(sid)
    return (np.array(pos), np.array(dirs), np.array(w),
            np.array(is_cross, bool), np.array(seed_id, int))


def crossing_verdicts(seed_id, is_cross, tract_of, tract_len, min_len=3):
    """For each HOST 'crossing' seed, is EACH of its (<=2) peaks in an extended tract
    (len>=min_len)? Both -> CONFIRMED; at least one short/unlinked -> DEMOTED."""
    from collections import defaultdict
    peaks = defaultdict(list)
    for k, sid in enumerate(seed_id):
        peaks[sid].append(k)
    conf = dem = 0
    for sid, ks in peaks.items():
        if not is_cross[ks[0]] or len(ks) < 2:
            continue
        both = all(tract_len[tract_of[k]] >= min_len for k in ks)
        conf += both; dem += (not both)
    return conf, dem


def run(crop=96, stride=5, save="phase3_i74.png"):
    print("=" * 74)
    print(f"PHASE 3 on real i74 -- joint global model, crop={crop}^3, stride={stride}")
    c0 = (768 - crop) // 2
    vol = load_volume(crop=(c0, c0 + crop)).astype(np.float64)
    print(f"  volume {vol.shape}, mean={vol.mean():.1f}, {(vol == 0).mean():.0%} clipped air")

    aniso_min, n_a = calibrate_aniso(vol)
    a_src = "volume anisotropy null" if n_a >= 20 else "PHYSICAL FALLBACK"
    print(f"  aniso gate (data-driven, {a_src}, n={n_a}): ANISO_MIN={aniso_min:.2f}")
    pos, dirs, w, is_cross, seed_id = seed_segments(vol, stride, aniso_min=aniso_min)
    n_cross_seeds = len(set(seed_id[is_cross]))
    print(f"  seeded {len(pos)} segments from {len(set(seed_id))} voxels "
          f"({n_cross_seeds} HOST-'crossing')")
    if len(pos) < 10:
        print("  too few segments; raise crop or lower gates."); return

    r_max = 2.6 * stride
    cand = P.candidate_links(pos, dirs, r_max=r_max)
    active, adj = P.greedy_forest(cand, len(pos))
    tracts = P.tracts_from(adj, len(pos))
    print(f"  {len(cand)} candidate links (r_max={r_max:.0f}) -> {len(active)} linked")

    # tract bookkeeping
    tract_of = np.full(len(pos), -1, int)
    tract_len = np.zeros(len(tracts), int)
    for ti, c in enumerate(tracts):
        tract_len[ti] = len(c)
        for k in c:
            tract_of[k] = ti
    big = tract_len[tract_len >= 3]
    print(f"  {len(tracts)} tracts; {len(big)} with len>=3, "
          f"longest={tract_len.max()} segs (~{tract_len.max()*stride*VOXEL_SIZE_UM:.0f} um)")

    conf, dem = crossing_verdicts(seed_id, is_cross, tract_of, tract_len)
    tot = conf + dem
    print("-" * 74)
    if tot:
        print(f"  GLOBAL verdict on HOST crossings: CONFIRMED {conf}/{tot} = {conf/tot:.0%} "
              f"(both peaks continue), DEMOTED {dem}/{tot} = {dem/tot:.0%} (a peak has no "
              f"collinear continuation).")
    else:
        print("  no HOST-crossing seeds with 2 peaks in this crop.")
    print(f"  (Descriptive: no i74 ground truth. The model that produced this passed the "
          f"phantom test kissing=0; dense-tissue over-connection remains a caveat.)")
    print("=" * 74)

    # figure: z-MIP with tracts projected to (x,y)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(vol.max(0), cmap="gray", origin="upper")   # MIP along axis0(z): [y,x]
    rng = np.random.default_rng(0)
    for c in tracts:
        if len(c) < 3:
            continue
        Pp = pos[P._order_chain(adj, c)]
        ax[0].plot(Pp[:, 0], Pp[:, 1], "-", lw=1.3, color=plt.cm.tab20(rng.integers(20)))
    ax[0].set_title(f"i74 z-MIP + {len(big)} global tracts (len>=3)")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].hist(tract_len[tract_len >= 1], bins=range(1, tract_len.max() + 2),
               color="tab:blue", alpha=0.8)
    ax[1].axvline(3, color="k", ls=":")
    ax[1].set_xlabel("tract length (segments)"); ax[1].set_ylabel("count")
    ax[1].set_title(f"tract lengths (CONFIRMED {conf}/{tot} HOST crossings)" if tot
                    else "tract lengths")
    fig.suptitle("Phase 3 on real i74: joint global tractography from HOST FOD seeds", fontsize=12)
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()
    run(crop=args.crop, stride=args.stride)
