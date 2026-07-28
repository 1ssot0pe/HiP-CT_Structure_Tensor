"""
phase3_realnoise.py -- #1: re-validate the 3D global-tractography pipeline under the
REAL MEASURED noise instead of idealised white noise, to close the synthetic->real
transfer caveat behind the i74 over-calling numbers.

The 3D validation (phase3_validate3d) confirmed true crossings 89% / demoted kissing
79% -- but under WHITE gaussian noise. We now have the actual lung half-map noise
n=(A-B)/sqrt2, which is CORRELATED + ANISOTROPIC. We inject THAT (a real crop of it,
not a synthetic match) into clean phantoms at the SAME amplitude the white baseline
used (noise_sigma=0.3), and re-run the identical seed->link->forest->verdict pipeline.

If the confirm/demote rates hold, the white-noise validation transfers. If they
degrade (crossing confirm drops, kissing false-confirm rises), realistic directional
noise really does erode the pipeline -- and the i74 numbers were optimistic. Either
way it is measured, not assumed.
"""

import os
import numpy as np

from crossing_shape_3d import _tube_family, _coords
from host_geometry_test import _dir_inplane, _finalise
import phase3_validate3d as V

SLAB = os.environ.get("LUNG_SLAB_DIR", ".")
SIGMA_N = 0.30            # same amplitude the white baseline used
WHITE_BASELINE = (0.89, 0.21)   # (crossing confirm, kissing confirm) at 64^3 stride 4


def real_noise(shape, seed=0):
    """A unit-std crop of the REAL lung half-map noise n=(A-B)/sqrt2, from a tissue
    region (so it carries the measured correlation + anisotropy)."""
    A = np.load(os.path.join(SLAB, "lung_A.npy")).astype(np.float64)
    B = np.load(os.path.join(SLAB, "lung_B.npy")).astype(np.float64)
    n = (A - B) / np.sqrt(2.0)
    z0 = (n.shape[0] - shape[0]) // 2
    y0 = min(432, n.shape[1] - shape[1]); x0 = min(208, n.shape[2] - shape[2])
    crop = n[z0:z0 + shape[0], y0:y0 + shape[1], x0:x0 + shape[2]]
    return crop / (crop.std() + 1e-12)


def build(kind, noise, shape):
    """Clean phantom signal + real noise at SIGMA_N, then finalise -- matching how
    make_*_phantom builds the WHITE version, but with real noise substituted."""
    c = _coords(shape)
    d1, d2 = _dir_inplane(0.0), _dir_inplane(70.0)
    t1 = _tube_family(c, d1, 12.0, 1.5)
    t2 = _tube_family(c, d2, 12.0, 1.5)
    if kind == "crossing":
        sig = t1 + t2
    else:                                    # kissing: d1 for x<0, d2 for x>=0
        sig = np.where(c[0] < 0.0, t1, t2)
    return _finalise(sig + SIGMA_N * noise)


def run(crop=64, stride=4):
    print("=" * 74)
    print(f"PHASE 3 under REAL measured noise (shape={crop}^3, stride={stride}, "
          f"sigma_n={SIGMA_N})")
    shape = (crop, crop, crop)
    noise = real_noise(shape)
    # sanity: is the injected noise actually anisotropic/correlated?
    import noise_floor as nf
    w = np.linalg.eigvalsh(nf.N_matrix(noise, 4.0))
    print(f"  injected noise anisotropy lambda1/lambda3 = {w[-1]/(w[0]+1e-12):.2f} "
          f"(white would be ~1)")
    res = {}
    for kind in ("crossing", "kissing"):
        vol = build(kind, noise, shape)
        conf, dem, longest, nseg = V.pipeline(vol, stride)
        tot = conf + dem
        rate = conf / tot if tot else float("nan")
        res[kind] = rate
        print(f"  {kind:>9}: {nseg} segs; HOST crossings={tot:3d} -> "
              f"CONFIRMED {conf} ({rate:.0%}), DEMOTED {dem} ({1-rate:.0%})")
    print("-" * 74)
    cx, ks = res["crossing"], res["kissing"]
    bcx, bks = WHITE_BASELINE
    print(f"  crossing confirm: real-noise {cx:.0%} vs white-baseline {bcx:.0%}")
    print(f"  kissing  confirm: real-noise {ks:.0%} vs white-baseline {bks:.0%} "
          f"(lower is better)")
    held = (cx >= bcx - 0.15) and (ks <= bks + 0.15)
    print(f"  => {'HOLDS: white-noise validation transfers to real noise' if held else 'DEGRADES: realistic directional noise erodes the pipeline -> i74 numbers were optimistic'}")
    print("=" * 74)


if __name__ == "__main__":
    run()
