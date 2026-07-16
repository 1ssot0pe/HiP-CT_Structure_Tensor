"""
phase1_kissing_crossing.py -- Phase 1 of the global-tractography scoping: does a
continuity tracker correctly DISTINGUISH a kissing junction from a true crossing
in 3D, on the host_geometry_test phantoms with known truth?

Geometry (host_geometry_test):
  * KISSING  -- two SEPARATE single-orientation regions split at the x=0 plane:
                d1=[1,0,0] (along +x) for x<0, d2=~[0.34,0.94,0] (~+y) for x>0.
                A d1 fibre in x<0 ENDS at the boundary; it does NOT connect to the
                d2 bundle.
  * CROSSING -- d1 and d2 summed EVERYWHERE (both orientations in every voxel).
                A d1 fibre runs across the WHOLE volume.

Test: seed a streamline on a d1 fibre in x<0, heading +x (toward the boundary).
  * TRUE behaviour: crossing -> traverse the whole volume; kissing -> terminate
    at/near the boundary (no d1 support beyond it).
  * The OVER-EAGER prior failure: a pure-inertia tracker plows +x through BOTH,
    merging the kissing bundles -> cannot tell them apart.

The tracker's data term is HOST directional fibre-ness (host_fod), evaluated in a
CURVATURE CONE around the incoming direction; it TERMINATES when no fibre near the
incoming direction has enough support. That data+continuity balance is what
separates kissing from crossing -- the key design claim for the full method.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from host_geometry_test import make_kissing_phantom, make_true_crossing_same_angles
from host_fod import patch_gradients, directional_energy
from fod_dipy import SPHERE

V = SPHERE.vertices                     # (Nverts, 3) in (x, y, z) = (axis2,axis1,axis0)
BOUND_X = 48                            # x=0 plane in axis2 index (SHAPE//2)


def local_fibreness(vol, pos, half=8, sigma_grad=2.0, order=6):
    """HOST fibre-ness f(u) over the sphere at `pos` (axis0,axis1,axis2). High f =
    a fibre points that way. None if the patch is too small/flat."""
    ctr = [int(round(p)) for p in pos]
    sl = tuple(slice(max(0, c - half), c + half + 1) for c in ctr)
    patch = vol[sl]
    if min(patch.shape) < 6:
        return None
    G = patch_gradients(patch, sigma_grad)
    if G.shape[0] < 40:
        return None
    E = directional_energy(G, order)
    return E.max() - E                  # fibre-ness, indexed by V


def track3d(vol, seed, dir0, mode, steps=80, step=1.0, cone_deg=30, term_ratio=0.75):
    """mode='continuity' = HOST fibre-ness within a curvature cone + terminate on
    weak support; mode='inertia' = keep the initial direction (the over-eager
    prior). Directions live in (x,y,z); positions in (axis0,axis1,axis2)."""
    pos = np.array(seed, float); prev = np.array(dir0, float); prev /= np.linalg.norm(prev)
    path = [pos.copy()]; coscone = np.cos(np.deg2rad(cone_deg))
    for _ in range(steps):
        if not all(2 <= pos[i] < vol.shape[i] - 2 for i in range(3)):
            break
        if mode == "inertia":
            u = prev
        else:
            f = local_fibreness(vol, pos)
            if f is None:
                break
            dots = V @ prev
            incone = np.abs(dots) >= coscone
            if not incone.any():
                break
            idx = np.where(incone)[0]
            b = idx[np.argmax(f[idx])]
            if f[b] < term_ratio * f.max():        # no fibre near incoming -> stop
                break
            u = V[b] * (1.0 if dots[b] >= 0 else -1.0)
        pos = pos + step * np.array([u[2], u[1], u[0]])   # (x,y,z)->(axis0,axis1,axis2)
        prev = u
        path.append(pos.copy())
    return np.array(path)


def run(save="phase1_kissing_crossing.png"):
    kiss, _ = make_kissing_phantom()
    cross, _ = make_true_crossing_same_angles()
    seed = (48.0, 48.0, 18.0)               # x = 18-48 = -30 (left/x<0 region)
    dir0 = (1.0, 0.0, 0.0)                   # +x, toward the boundary
    print("=" * 74)
    print("PHASE 1 -- kissing vs crossing: does continuity+data tell them apart?")
    print(f"  seed x={seed[2]-BOUND_X:.0f} (left region), heading +x toward the x=0 boundary")
    def net_turn(p):
        if len(p) < 6:
            return float("nan")
        d = p[-1] - p[-6]; d = d / (np.linalg.norm(d) + 1e-9)
        dv = np.array([dir0[2], dir0[1], dir0[0]])       # dir0 (x,y,z)->axis order
        return float(np.degrees(np.arccos(np.clip(abs(d @ dv), 0, 1))))

    results = {}
    print(f"  {'phantom':>10} | {'tracker':>11} | {'x-depth':>7} | {'net turn deg':>12}")
    for name, vol in (("crossing", cross), ("kissing", kiss)):
        for mode in ("continuity", "inertia"):
            p = track3d(vol, seed, dir0, mode)
            depth = float(p[:, 2].max() - BOUND_X)
            turn = net_turn(p)
            results[(name, mode)] = (p, depth, turn)
            print(f"  {name:>10} | {mode:>11} | {depth:7.1f} | {turn:12.0f}")
    print("-" * 74)
    tc = results[("crossing", "continuity")][2]
    tk = results[("kissing", "continuity")][2]
    print(f"  DATA-continuity carries a SIGNAL: crossing keeps its direction "
          f"(net turn {tc:.0f} deg, traverses straight) while kissing is forced to "
          f"turn onto bundle B (net turn {tk:.0f} deg) because bundle A ended.")
    print(f"  INERTIA (over-eager) is identical on both (net turn ~0) -> blind to the difference.")
    print("  HONEST LIMIT: neither LOCAL greedy tracker cleanly TERMINATES kissing --")
    print("  data-continuity turns onto B via the partial-volume boundary 'bridge'.")
    print("  => correct kissing/crossing separation needs GLOBAL (whole-tract) inference,")
    print("     not local steps. This is the concrete motivation for the full Bayesian model.")
    print("=" * 74)

    # figure: x-y projection (axis2 horizontal, axis1 vertical) at z=48
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, (name, vol) in zip(ax, (("crossing", cross), ("kissing", kiss))):
        a.imshow(vol[48], cmap="gray", origin="upper")     # slice z=48: rows=y, cols=x
        a.axvline(BOUND_X, color="cyan", ls="--", lw=1.5, label="x=0 boundary")
        for mode, col in (("continuity", "#2a78d6"), ("inertia", "#eb6834")):
            p = results[(name, mode)][0]
            a.plot(p[:, 2], p[:, 1], "-", color=col, lw=2.5,
                   label=f"{mode}")
        a.plot(seed[2], seed[1], "wo", ms=8, mec="k")
        a.set_title(f"{name}: continuity {'FORCED to turn onto bundle B' if name=='kissing' else 'traverses straight'}")
        a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=8, loc="upper right")
    fig.suptitle("Phase 1: data-continuity behaves differently on kissing (turns) vs crossing\n"
                 "(traverses) -- but local greedy tracking does not cleanly TERMINATE kissing\n"
                 "-> motivates GLOBAL inference", fontsize=11)
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    run()
