"""
toy_demo.py -- the whole point of Stage 3, on one picture.

Two junctions sit in one volume and look IDENTICAL to a per-voxel orientation
analysis: at both, two fibre directions meet. HOST flags both as
multi-orientation and hands Stage 3 two vectors per site.

    site A -- TRUE CROSSING. Both centrelines pass through. The two fibres
              superpose, so the intensity at the centre is the SUM: 2 fibres'
              worth over the tissue pedestal.
    site B -- TANGENCY (kissing). The two fibres touch and turn away. Neither
              centreline goes through, so the centre carries the MAX, not the
              sum: 1 fibre's worth.

Geometry cannot separate them -- locally the pictures are the same, which is
the whole difficulty and why the tracker/continuity route stalls. BRIGHTNESS
can, because superposition is additive and tangency is not:

    S = (I_c - I_0) / ((I_a - I_0) + (I_b - I_0))     ~1.0 crossing, ~0.5 kiss

So the demo builds the two sites with `sum` and `max` respectively -- that one
operator IS the difference between a crossing and a kiss -- measures S at each,
and renders the vector field before and after the verdict is applied.

I_0 is the LOCAL TISSUE level, not air. Referencing to air makes one fibre
worth ~99 grey levels, predicts a crossing at 313 in a uint8 volume, and caps S
below the decision threshold so that CROSSING becomes unreachable -- see
`s_calibration.py`. The toy is built at i74's real intensities (tissue 183,
fibre 214, uint8) so that trap is reproduced rather than avoided.

Run
---
    python toy_demo.py
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE = os.path.dirname(_HERE)             # 3_global_tractography/
_ROOT = os.path.dirname(_STAGE)             # repository root
for _p in (_ROOT, _STAGE, _HERE):
    sys.path.insert(0, _p)

import self_optimising_tractography as S3          # noqa: E402

# i74's measured intensities, so the toy inherits the real dynamic range
TISSUE, FIBRE, UINT8_MAX = 183.0, 214.0, 255.0
# z is 40, not the 24 a picture needs: GATE_HALF is 14, so patch centres must
# satisfy 14 <= c < n-14 on EVERY axis, and a 24-thick slab admits none at all.
# The toy is the only substrate here with two labelled junctions in one volume,
# so it has to be thick enough for Stage 1/2 to run on it.
SHAPE = (40, 64, 112)          # (z, y, x): a slab, visualised at mid-z
SITE_A = (20, 32, 32)          # true crossing
SITE_B = (20, 32, 80)          # tangency
SEP_DEG = 50.0
RADIUS = 3.0


def _tube(shape, centre, direction, radius):
    """Unit-height tube of the given radius through `centre` along `direction`."""
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    p = np.stack([zz - centre[0], yy - centre[1], xx - centre[2]], -1).astype(float)
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    along = p @ d
    perp2 = (p ** 2).sum(-1) - along ** 2
    r = np.sqrt(np.maximum(perp2, 0.0))
    # soft edge, so the structure tensor sees a gradient rather than a step
    return np.clip((radius + 1.0 - r) / 1.5, 0.0, 1.0)


def build_toy(noise=0.0, seed=0):
    """Volume plus the ground truth for both sites.

    noise : Gaussian sigma in GREY LEVELS, added before the uint8 round. Not a
            cosmetic option -- a noiseless toy breaks two things. Stage 1's
            rho-stability sweep returns an all-NaN sensitivity surface on a
            perfectly smooth volume and raises, so Stage 1/2 cannot run on it
            at all; and with no noise there is nothing for an L1 penalty to
            protect against, so its optimum is trivially zero. For scale, the
            fibre contrast here is FIBRE - TISSUE = 31 grey levels.
    """
    half = np.deg2rad(SEP_DEG / 2.0)
    # directions in (z, y, x) -- the frame superposition_S and positions use
    dA1 = np.array([0.0, np.sin(half), np.cos(half)])
    dA2 = np.array([0.0, -np.sin(half), np.cos(half)])

    # --- site A: two tubes through the same point, SUMMED = superposition ---
    a1 = _tube(SHAPE, SITE_A, dA1, RADIUS)
    a2 = _tube(SHAPE, SITE_A, dA2, RADIUS)
    siteA = a1 + a2

    # --- site B: two tubes offset so they TOUCH, combined by MAX = no
    # superposition. The offset is 2*radius, so the surfaces meet at exactly
    # one point and neither centreline passes through it.
    offB1 = (SITE_B[0], SITE_B[1] + RADIUS, SITE_B[2])
    offB2 = (SITE_B[0], SITE_B[1] - RADIUS, SITE_B[2])
    b1 = _tube(SHAPE, offB1, dA1, RADIUS)
    b2 = _tube(SHAPE, offB2, dA2, RADIUS)
    siteB = np.maximum(b1, b2)

    occupancy = np.maximum(siteA, 0) + 0.0
    occupancy[:, :, SHAPE[2] // 2:] = siteB[:, :, SHAPE[2] // 2:]

    vol = TISSUE + (FIBRE - TISSUE) * occupancy
    if noise > 0:
        vol = vol + np.random.default_rng(seed).normal(scale=float(noise),
                                                       size=vol.shape)
    vol = np.clip(vol, 0.0, UINT8_MAX)
    vol = np.round(vol).astype(np.float32)

    truth = [dict(name="A: true crossing", centre=SITE_A, truth="CROSSING",
                  dirs=[dA1, dA2]),
             dict(name="B: tangency (kiss)", centre=SITE_B, truth="TANGENCY",
                  dirs=[dA1, dA2])]
    return vol, truth


def measure(vol, truth, verbose=True):
    """Superposition S at both sites, with the local-tissue baseline."""
    e3 = S3.orientation_field(vol)
    out = []
    if verbose:
        print(f"  {'site':<22} {'I_0':>6} {'I_c':>6} {'I_a':>6} {'I_b':>6} "
              f"{'S':>7} {'verdict':>10} {'truth':>10}")
    for t in truth:
        I0 = S3.local_baseline(vol, t["centre"], half=S3.BASELINE_HALF)
        S, Ic, Ia, Ib, _ = S3.superposition_S(vol, t["centre"], t["dirs"][0],
                                              t["dirs"][1], I0=I0, e3=e3)
        v = S3.verdict_from_S(S)
        r = dict(name=t["name"], centre=t["centre"], truth=t["truth"],
                 S=float(S), verdict=v, I_0=float(I0), I_c=float(Ic),
                 I_a=float(Ia), I_b=float(Ib),
                 correct=bool(v == t["truth"]),
                 n_dirs_before=2, n_dirs_after=2 if v == "CROSSING" else 1,
                 dirs=[d.tolist() for d in t["dirs"]])
        out.append(r)
        if verbose:
            mark = "OK" if r["correct"] else "WRONG"
            print(f"  {t['name']:<22} {I0:6.1f} {Ic:6.1f} {Ia:6.1f} {Ib:6.1f} "
                  f"{S:7.3f} {v:>10} {t['truth']:>10}  {mark}")
    return out


def plot(vol, results, path):
    """Before/after vector field at the mid-z slice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = SHAPE[0] // 2
    sl = vol[z]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, sharey=True)

    for ax, stage in zip(axes, ("before", "after")):
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal",
                  vmin=TISSUE - 10, vmax=FIBRE + 5)
        for r in results:
            cy, cx = r["centre"][1], r["centre"][2]
            n = r["n_dirs_before"] if stage == "before" else r["n_dirs_after"]
            dirs = [np.array(d) for d in r["dirs"][:n]]
            for k, d in enumerate(dirs):
                # (z, y, x) -> plot (x, y); nematic, so draw both ways
                dy, dx = d[1], d[2]
                col = "#ff4d4d" if k == 0 else "#4dd2ff"
                ax.quiver([cx, cx], [cy, cy], [dx * 11, -dx * 11],
                          [dy * 11, -dy * 11], color=col, angles="xy",
                          scale_units="xy", scale=1, width=0.004,
                          headwidth=4, zorder=3)
            lab = (f"{n} orientation" + ("s" if n > 1 else "")
                   + (f"\nS={r['S']:.2f} -> {r['verdict']}"
                      if stage == "after" else ""))
            ax.text(cx, cy - 17, lab, color="w", ha="center", va="top",
                    fontsize=8, zorder=4,
                    bbox=dict(fc="k", alpha=0.55, ec="none", pad=1.8))
        ax.set_ylabel("y (voxels)")
        ax.set_title(
            "BEFORE -- HOST flags both sites as two-orientation; they are "
            "indistinguishable" if stage == "before" else
            "AFTER -- superposition demotes the tangency; the crossing survives",
            fontsize=10, loc="left")
    axes[-1].set_xlabel("x (voxels)")
    fig.suptitle("Stage 3: telling a crossing from a kiss  "
                 "(left = true crossing, right = tangency)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results")
    print("=" * 84)
    print("TOY DEMO -- crossing vs kiss, before and after Stage 3")
    print("=" * 84)
    vol, truth = build_toy()
    print(f"  volume {vol.shape}, tissue {TISSUE:g}, fibre {FIBRE:g}, "
          f"uint8 ceiling {UINT8_MAX:g}")
    print(f"  intensity at A (sum) = {vol[SITE_A]:.0f}   "
          f"at B (max) = {vol[SITE_B]:.0f}\n")
    res = measure(vol, truth)
    n_ok = sum(r["correct"] for r in res)
    print(f"\n  {n_ok}/{len(res)} sites classified correctly")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "toy_demo.json"), "w") as f:
        json.dump(res, f, indent=2)
    p = plot(vol, res, os.path.join(outdir, "toy_demo.png"))
    print(f"  wrote {p}")
    print("=" * 84)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="crossing-vs-kiss toy demo")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    r = main(outdir=a.outdir)
    sys.exit(0 if all(x["correct"] for x in r) else 1)
