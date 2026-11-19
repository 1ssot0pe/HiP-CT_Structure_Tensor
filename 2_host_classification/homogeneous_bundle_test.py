"""homogeneous_bundle_test.py -- does HOST-FOD work inside a homogeneous bundle?

The question (from a collaborator): HOST looks good where there are clear
striations, but what does it do in a bundle whose interior has little or no
visible texture?

HOST and the plain structure tensor both read orientation from intensity
GRADIENTS. A fibre is the direction along which intensity stays constant, so it
can only be seen if something varies ACROSS it. The prediction is:

  * striations resolved at sigma_grad -> correct peak, label single;
  * striations fading                 -> ODF flattens (aniso falls), peak error grows;
  * no texture (homogeneous interior) -> gradients are pure noise, ODF ~isotropic,
                                         label `disorganised` -- an honest refusal,
                                         not a confident wrong direction;
  * bundle EDGE                       -> the boundary alone still gives the axis,
                                         because the boundary normals all lie
                                         perpendicular to it.

PART A (synthetic, no data needed): a cylindrical bundle, radius 30 voxels, with
striations along its axis (periodic Gaussian tubes) whose contrast is swept from
1 (strong striations) to 0 (perfectly homogeneous), plus Gaussian noise. Patches
are taken at the bundle CENTRE and on its EDGE, at random bundle orientations.
Also swept: the striation period (6 voxels: near the resolution limit; 12:
clearly resolved) and sigma_grad (2.0, host_fod's default; 4.0, what Stage 2
runs at).

PART B (real data, optional, --i74): HiP-CT has no labelled "homogeneous
bundle", so tissue patches on a grid are binned by TEXTURE STRENGTH (mean
gradient magnitude at sigma_grad) and the label mix, aniso, and HOST-vs-tensor
agreement are reported per quintile. Hand-picked centres can be given with
--centres "z,y,x;z,y,x" (coordinates inside the loaded crop).

For every patch the plain structure tensor's e3 (from the SAME gradients) is
reported next to HOST's first peak, so the comparison is HOST vs EV on equal
terms.

    python 2_host_classification/homogeneous_bundle_test.py            # Part A
    python 2_host_classification/homogeneous_bundle_test.py --i74      # A + B
    python 2_host_classification/homogeneous_bundle_test.py --i74 --centres "128,90,140;60,60,60"
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from core.crossing_shape_3d import _coords, _random_rotation, _tube_family  # noqa: E402
from core.two_orientation_test import patch_gradients                       # noqa: E402

HALF = 14                       # host_fod / Stage 2 patch half-width
SHAPE = (96, 96, 96)
RADIUS = 30.0                   # bundle radius, voxels
EDGE_WIDTH = 1.0                # tanh edge softness, voxels
CONTRASTS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.0)
PERIODS = (12.0, 6.0)
SIGMAS = (2.0, 4.0)
NOISE = 0.05                    # Gaussian, relative to bundle intensity 1
SEEDS = tuple(range(8))         # 3 left the no-texture medians at chance level


# ---------------------------------------------------------------------------
# helpers that need no dipy (smoke-testable anywhere)
# ---------------------------------------------------------------------------

def make_bundle(contrast, period, seed, noise=NOISE, shape=SHAPE, radius=RADIUS):
    """Cylindrical bundle through the volume centre along a random axis.

    Intensity inside = (1 - contrast) + contrast * striation, where striation is
    a periodic tube family along the axis scaled to [0, 1]. contrast = 0 is a
    perfectly homogeneous interior. Returns (vol, axis, u) with axis and u in
    the (x, y, z) basis host_fod reports in; u is a unit vector perpendicular to
    the axis, used to place the edge patch.
    """
    R = _random_rotation(seed)
    d = R @ np.array([1.0, 0.0, 0.0])
    u = R @ np.array([0.0, 1.0, 0.0])
    x, y, z = _coords(shape)
    along = x * d[0] + y * d[1] + z * d[2]
    r = np.sqrt(np.maximum(x ** 2 + y ** 2 + z ** 2 - along ** 2, 0.0))
    inside = 0.5 * (1.0 - np.tanh((r - radius) / EDGE_WIDTH))
    tex = _tube_family((x, y, z), d, period, 1.5)
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-12)
    vol = inside * ((1.0 - contrast) + contrast * tex)
    rng = np.random.default_rng(1000 + seed)
    vol = vol + noise * rng.standard_normal(shape)
    return vol.astype(np.float32), d, u


def crop(vol, centre_zyx, half=HALF):
    c = [int(round(v)) for v in centre_zyx]
    return vol[c[0] - half:c[0] + half + 1,
               c[1] - half:c[1] + half + 1,
               c[2] - half:c[2] + half + 1]


def tensor_e3(patch, sigma_grad):
    """Plain structure tensor's fibre direction from the same gradients HOST
    uses: the eigenvector of the gradient scatter with the SMALLEST eigenvalue."""
    G = patch_gradients(patch, sigma_grad)
    if G.shape[0] < 40:
        return None
    w, V = np.linalg.eigh(G.T @ G)
    return V[:, 0]


def ang(a, b):
    """Axial angle in degrees (sign of a fibre direction is meaningless)."""
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def patch_centres(d_xyz_unused, u_xyz, shape=SHAPE, radius=RADIUS):
    """Centre and edge patch centres in (z, y, x) voxel coordinates."""
    mid = np.array(shape, float) / 2.0
    u_zyx = np.asarray(u_xyz)[::-1]          # (x,y,z) -> (z,y,x)
    return {"centre": mid, "edge": mid + radius * u_zyx}


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def part_a(seeds=SEEDS, contrasts=CONTRASTS, periods=PERIODS, sigmas=SIGMAS,
           noise=NOISE, smoke=False):
    if not smoke:
        from core.host_fod import analyse, classify
    rows = []
    t0 = time.time()
    for period in periods:
        for c in contrasts:
            for seed in seeds:
                vol, d, u = make_bundle(c, period, seed, noise=noise)
                for where, ctr in patch_centres(d, u).items():
                    p = crop(vol, ctr)
                    for sg in sigmas:
                        e3 = tensor_e3(p, sg)
                        row = dict(period=period, contrast=c, seed=seed,
                                   where=where, sigma_grad=sg,
                                   ev_err=ang(e3, d))
                        if not smoke:
                            m = analyse(p, sigma_grad=sg)
                            lab = classify(m)
                            pk = m["dirs"][0] if m and m["n_peaks"] else None
                            row.update(label=lab, aniso=m["aniso"] if m else np.nan,
                                       n_peaks=m["n_peaks"] if m else 0,
                                       host_err=ang(pk, d),
                                       c_s=m["c_s"] if m else np.nan,
                                       FA=m["FA"] if m else np.nan)
                        rows.append(row)
    print(f"  [part A] {len(rows)} patches in {time.time() - t0:.0f} s")
    return rows


def _fmt_labels(labs):
    order = ("single", "dispersed", "crossing", "complex", "disorganised", "n/a")
    out = []
    for k in order:
        n = sum(1 for x in labs if x == k)
        if n:
            out.append(f"{k[:5]}x{n}")
    return " ".join(out)


def report_a(rows, smoke=False):
    print("=" * 100)
    print("PART A -- synthetic bundle: striation contrast swept to 0 (homogeneous)")
    print(f"  noise sigma {NOISE} (bundle = 1), patch {2 * HALF + 1}^3, "
          f"{len(set(r['seed'] for r in rows))} random orientations each")
    for period in sorted(set(r["period"] for r in rows), reverse=True):
        for sg in sorted(set(r["sigma_grad"] for r in rows)):
            print("-" * 100)
            print(f"  striation period {period:g} vox, sigma_grad {sg:g}")
            hdr = f"  {'where':>6} {'contrast':>8} | {'EV err':>7}"
            if not smoke:
                hdr += f" {'HOST err':>8} {'aniso':>6} {'npk':>4} | labels"
            print(hdr)
            for where in ("centre", "edge"):
                for c in sorted(set(r["contrast"] for r in rows), reverse=True):
                    sel = [r for r in rows if r["period"] == period
                           and r["sigma_grad"] == sg and r["where"] == where
                           and r["contrast"] == c]
                    if not sel:
                        continue
                    line = (f"  {where:>6} {c:8.2f} | "
                            f"{np.nanmedian([r['ev_err'] for r in sel]):7.1f}")
                    if not smoke:
                        line += (f" {np.nanmedian([r['host_err'] for r in sel]):8.1f}"
                                 f" {np.nanmedian([r['aniso'] for r in sel]):6.2f}"
                                 f" {np.nanmedian([r['n_peaks'] for r in sel]):4.0f} | "
                                 f"{_fmt_labels([r['label'] for r in sel])}")
                    print(line)
    print("  (errors are medians over orientations, degrees; 90 = random-ish, "
          "a uniform random axis has median 60)")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

def part_b(sigma_grad=4.0, stride=24, centres=None):
    from core.host_fod import analyse, classify
    from core.reconstruction import load_volume, tissue_mask
    from scipy import ndimage

    vol = load_volume().astype(np.float32)
    mask = tissue_mask(vol)
    print(f"  [part B] volume {vol.shape}, tissue {mask.mean():.0%}")
    gmag = np.sqrt(sum(ndimage.gaussian_filter(vol, sigma_grad, order=o) ** 2
                       for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))))
    if centres is None:
        lo, hi = HALF, min(vol.shape) - HALF - 1
        grid = range(lo, hi, stride)
        centres = [(z, y, x) for z in grid for y in grid for x in grid
                   if crop(mask, (z, y, x)).mean() > 0.95]
    rows = []
    t0 = time.time()
    for ctr in centres:
        p = crop(vol, ctr)
        m = analyse(p, sigma_grad=sigma_grad)
        e3 = tensor_e3(p, sigma_grad)
        pk = m["dirs"][0] if m and m["n_peaks"] else None
        rows.append(dict(centre=[int(v) for v in ctr],
                         texture=float(crop(gmag, ctr).mean()),
                         label=classify(m), aniso=m["aniso"] if m else np.nan,
                         n_peaks=m["n_peaks"] if m else 0,
                         host_vs_ev=ang(pk, e3)))
    print(f"  [part B] {len(rows)} patches in {time.time() - t0:.0f} s")
    return rows


def report_b(rows, hand_picked):
    print("=" * 100)
    print("PART B -- i74 tissue patches" + (" (hand-picked)" if hand_picked
                                            else ", binned by texture strength"))
    if hand_picked:
        for r in rows:
            print(f"  {str(r['centre']):>16} texture {r['texture']:.4f} "
                  f"aniso {r['aniso']:.2f} npk {r['n_peaks']} "
                  f"HOST-vs-EV {r['host_vs_ev']:5.1f}  {r['label']}")
    else:
        tex = np.array([r["texture"] for r in rows])
        q = np.quantile(tex, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        print(f"  {'quintile':>9} {'texture':>17} {'n':>4} | {'aniso':>6} "
              f"{'HOST-vs-EV':>10} | labels")
        for i in range(5):
            sel = [r for r in rows if q[i] <= r["texture"] <= q[i + 1]
                   and (i == 4 or r["texture"] < q[i + 1])]
            print(f"  {i + 1:>9} {q[i]:8.4f}-{q[i + 1]:<8.4f} {len(sel):4d} | "
                  f"{np.nanmedian([r['aniso'] for r in sel]):6.2f} "
                  f"{np.nanmedian([r['host_vs_ev'] for r in sel]):10.1f} | "
                  f"{_fmt_labels([r['label'] for r in sel])}")
        print("  quintile 1 = least texture (the closest proxy for a homogeneous "
              "bundle without labels)")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--i74", action="store_true", help="also run Part B on real data")
    ap.add_argument("--centres", default=None,
                    help='hand-picked Part B centres "z,y,x;z,y,x" in crop coords')
    ap.add_argument("--stride", type=int, default=24, help="Part B grid stride")
    ap.add_argument("--sigma-grad", type=float, default=4.0, help="Part B sigma_grad")
    ap.add_argument("--smoke", action="store_true",
                    help="Part A without HOST (no dipy): phantom + EV only")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results",
                                                  "homogeneous_bundle_test.json"))
    a = ap.parse_args()

    out = {}
    rows = part_a(smoke=a.smoke)
    report_a(rows, smoke=a.smoke)
    out["part_a"] = rows
    if a.i74:
        cs = None
        if a.centres:
            cs = [tuple(int(v) for v in c.split(",")) for c in a.centres.split(";")]
        rb = part_b(sigma_grad=a.sigma_grad, stride=a.stride, centres=cs)
        report_b(rb, hand_picked=cs is not None)
        out["part_b"] = rb
    if not a.smoke:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
