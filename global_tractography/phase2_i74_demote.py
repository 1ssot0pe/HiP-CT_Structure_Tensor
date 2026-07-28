"""
phase2_i74_demote.py -- Phase 2 of the global-tractography scoping, on REAL i74.

Phase 1 (phantoms) established a generalising signal: FORWARD SUPPORT -- walk a
streamline along a candidate direction and ask whether it stays a genuine fibre
(normalised fibre-ness r = f_along/f_max) as you move along it. A real fibre
persists; a spurious one (noise / partial-volume peak) dies quickly.

Phase 2 applies that as a per-crossing CERTIFIER on real tissue, to answer the
design-doc question: does global continuity DEMOTE spurious crossings the per-voxel
pipeline flags?  For every voxel HOST `classify` calls a 2-way CROSSING, we walk
BOTH peaks (both directions) and record each peak's REACH -- how many voxels of
continuous support (r >= R_STOP) it sustains before the fibre is lost.

  * CONFIRMED crossing : BOTH peaks reach >= REACH_MIN (two real tracts cross).
  * DEMOTED crossing   : the weaker peak dies fast -> not a real 2-way crossing
                         (single fibre + a spurious lobe, or a kissing contact).

Honest scope: (i) no ground truth on i74 (per the host-gate work) -- we report the
DISTRIBUTION of weaker-peak reach and the demotion rate at a provisional threshold,
not a fabricated accuracy; (ii) Phase-1's wide-angle limit carries over -- a genuine
narrow crossing is unresolvable and may itself look 'demoted'. Both are stated, not
hidden.

Run:  .venv/Scripts/python -u phase2_i74_demote.py [--crop N] [--n-cross N]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reconstruction import load_volume, DEFAULT_VOLUME_DIR, VOXEL_SIZE_UM
from host_fod import analyse, classify
from phase1_kissing_crossing import local_fibreness, V

HALF = 8                 # local_fibreness patch half-width (matches phase 1)
CONE_DEG = 35.0          # curvature cone for the continuity walk
R_STOP = 0.55            # a step still 'on a fibre' if r=f_along/f_max >= R_STOP
MAX_STEPS = 25           # cap each half-walk (25 vox ~ 500 um at 20 um/vox)
REACH_MIN = 6            # a peak must persist at least this far to be certifiable
STRAIGHT_MIN = 0.70      # provisional: a real through-fibre keeps straightness >=0.70
EPS = 1e-12


def _walk_support(vol, pos0, dir_xyz, cone=CONE_DEG, max_steps=MAX_STEPS):
    """Walk a continuity streamline from pos0 along +dir_xyz; stop when no in-cone
    direction keeps r=f_along/f_max above R_STOP. Return (reach_steps, end_pos,
    path_length). Reach saturates in dense tissue -- straightness (below) is the
    discriminating quantity, not reach."""
    pos = np.array(pos0, float)
    prev = np.asarray(dir_xyz, float); prev /= np.linalg.norm(prev) + EPS
    coscone = np.cos(np.deg2rad(cone))
    steps = 0
    for _ in range(max_steps):
        if not all(HALF + 2 <= pos[i] < vol.shape[i] - HALF - 2 for i in range(3)):
            break
        f = local_fibreness(vol, pos, half=HALF)
        if f is None:
            break
        dots = V @ prev
        incone = np.abs(dots) >= coscone
        if not incone.any():
            break
        idx = np.where(incone)[0]
        b = idx[np.argmax(f[idx])]
        if float(f[b]) / (float(f.max()) + EPS) < R_STOP:     # fibre lost -> stop
            break
        u = V[b] * (1.0 if dots[b] >= 0 else -1.0)
        pos = pos + np.array([u[2], u[1], u[0]])              # (x,y,z)->(axis order)
        prev = u
        steps += 1
    return steps, pos, float(steps)      # unit steps -> path length == steps


def peak_reach(vol, pos, dir_xyz):
    """Certify a peak by the full streamline through it (backward + forward).
    Returns (reach, straightness) where straightness = |end-to-end displacement| /
    path length in [0,1]: ~1 for a real through-going fibre, low for a walk that
    wanders through ambient structure (a spurious peak that does not persist)."""
    rf, ep_f, lf = _walk_support(vol, pos, dir_xyz)
    rb, ep_b, lb = _walk_support(vol, pos, -np.asarray(dir_xyz, float))
    reach = rf + rb
    path = lf + lb
    disp = float(np.linalg.norm(ep_f - ep_b))
    straight = disp / (path + EPS) if path > 0 else 0.0
    return reach, straight


def collect_crossings(vol, n_target, stride=6, seed=0, max_attempts=1500):
    """Sample interior voxels; return up to n_target that HOST calls a 'crossing',
    as (pos, dirs). Random order so we don't bias to one region."""
    rng = np.random.default_rng(seed)
    lo, hi = HALF + 3, np.array(vol.shape) - HALF - 3
    out = []
    for _ in range(max_attempts):
        pos = np.array([rng.integers(lo, hi[i]) for i in range(3)])
        patch = vol[tuple(slice(pos[i] - HALF, pos[i] + HALF + 1) for i in range(3))]
        m = analyse(patch)
        if m is not None and classify(m) == "crossing":
            out.append((pos.astype(float), np.asarray(m["dirs"][:2])))
            if len(out) >= n_target:
                break
    return out


def run(crop=160, n_cross=40, save="phase2_i74_demote.png"):
    print("=" * 74)
    print(f"PHASE 2 -- forward-support certifier on real i74 ({DEFAULT_VOLUME_DIR.name})")
    c0 = (768 - crop) // 2
    vol = load_volume(crop=(c0, c0 + crop)).astype(np.float64)
    print(f"  volume {vol.shape}, mean={vol.mean():.1f}  (uint8 source; "
          f"{(vol == 0).mean():.0%} exactly 0 = clipped air)")
    print(f"  seeking {n_cross} HOST-'crossing' voxels ...")
    cx = collect_crossings(vol, n_cross)
    print(f"  found {len(cx)} crossings; certifying each peak by STRAIGHTNESS of its "
          f"streamline (reach saturates in dense tissue).")

    weak, strong, seps, reaches = [], [], [], []
    demoted = 0
    for i, (pos, dirs) in enumerate(cx):
        print(f"    certifying {i+1}/{len(cx)}", end="\r", flush=True)
        rc0, s0 = peak_reach(vol, pos, dirs[0])
        rc1, s1 = peak_reach(vol, pos, dirs[1])
        w, s = min(s0, s1), max(s0, s1)
        weak.append(w); strong.append(s)
        reaches.append(min(rc0, rc1))
        seps.append(float(np.degrees(np.arccos(np.clip(abs(dirs[0] @ dirs[1]), 0, 1)))))
        # demote if the weaker peer wanders (low straightness) or dies too early
        if w < STRAIGHT_MIN or min(rc0, rc1) < REACH_MIN:
            demoted += 1
    weak, strong, seps = np.array(weak), np.array(strong), np.array(seps)
    n = len(weak)
    print("-" * 74)
    if n:
        frac = demoted / n
        print(f"  weaker-peak straightness: median={np.median(weak):.2f}, "
              f"IQR[{np.percentile(weak,25):.2f},{np.percentile(weak,75):.2f}]; "
              f"stronger median={np.median(strong):.2f}")
        print(f"  DEMOTED {demoted}/{n} = {frac:.0%}: weaker peak WANDERS "
              f"(straightness<{STRAIGHT_MIN}) or dies <{REACH_MIN} vox "
              f"-> not a coherent 2nd fibre.")
        print(f"  CONFIRMED {n-demoted}/{n} = {1-frac:.0%}: both peaks trace straight, "
              f"persistent lines (real crossings).")
        print(f"  (Caveat: no i74 ground truth; narrow crossings <~70 deg may be "
              f"demoted despite being real -- Phase-1 limit. Median crossing "
              f"sep={np.median(seps):.0f} deg.)")
    print("=" * 74)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 21)
    ax[0].hist(weak, bins=bins, color="tab:red", alpha=0.7, label="weaker peak")
    ax[0].hist(strong, bins=bins, color="tab:green", alpha=0.5, label="stronger peak")
    ax[0].axvline(STRAIGHT_MIN, color="k", ls="--", label=f"STRAIGHT_MIN={STRAIGHT_MIN}")
    ax[0].set_xlabel("peak straightness  |displacement| / path length")
    ax[0].set_ylabel("crossings"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Phase 2: {demoted}/{n} crossings DEMOTED\n(weaker peak wanders "
                    f"or dies)")
    ax[1].scatter(seps, weak, c=(weak < STRAIGHT_MIN), cmap="RdYlGn_r", s=28,
                  edgecolor="0.3", lw=0.4)
    ax[1].axhline(STRAIGHT_MIN, color="k", ls="--")
    ax[1].set_xlabel("HOST crossing separation (deg)")
    ax[1].set_ylabel("weaker-peak straightness")
    ax[1].set_title("weaker-peak coherence vs crossing angle\n(narrow crossings: real or spurious?)")
    fig.suptitle("Phase 2 -- forward-support (straightness) demotes non-coherent i74 crossings", fontsize=12)
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--n-cross", type=int, default=40)
    args = ap.parse_args()
    run(crop=args.crop, n_cross=args.n_cross)
