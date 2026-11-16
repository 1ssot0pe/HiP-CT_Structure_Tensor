"""
build_lung_slab.py - one-time reader for the LADAF lung half-maps (16.547 um).

The half-recon TIFFs are 8881x8881 float32, not tiled (one chunk/slice), so a windowed
read is impossible -- every slice costs a full 315 MB read. Even/odd folders share ~81
z-slices, matched by the z-position token in the filename (e.g. ..._0154p9558). This
reads each matched slice ONCE, crops a tissue-centred in-plane window, stacks the two
half-maps, and caches them to .npy so the actual analysis is fast and repeatable.

Also prints the sanity check that must pass before trusting anything: are A and B
REGISTERED (high corr), and is n=(A-B)/sqrt(2) small vs the signal (a real noise floor)?
"""

import glob
import os
import numpy as np
import tifffile

# Folders of the two half-reconstructions (Human Organ Atlas, LADAF-2022-13 left
# lung, 16.547 um overview, "full-volume_2-0" and "full-volume_2-1").
A_DIR = os.environ.get("LUNG_HALF_A_DIR", "full-volume_2-0_16.547um_LADAF-2022-13_lung-left_overview__0001")
B_DIR = os.environ.get("LUNG_HALF_B_DIR", "full-volume_2-1_16.547um_LADAF-2022-13_lung-left_overview__0001")
VOX_UM = 16.547
W = 512  # in-plane crop half-window is W//2; read cost is fixed anyway
OUT = os.environ.get("LUNG_SLAB_DIR", ".")

_tok = lambda f: os.path.basename(f).split("_")[-1].split(".tif")[0]
_zpos = lambda t: float(t.replace("p", "."))


def main():
    a = {_tok(f): f for f in glob.glob(A_DIR + "/*.tif*")}
    b = {_tok(f): f for f in glob.glob(B_DIR + "/*.tif*")}
    if not a or not b:
        raise SystemExit(f"no TIFF slices found (A_DIR={A_DIR!r}, B_DIR={B_DIR!r}); "
                         "set LUNG_HALF_A_DIR and LUNG_HALF_B_DIR to the two half-map folders")
    common = sorted(set(a) & set(b), key=_zpos)
    print(f"matched z-slices: {len(common)}  (A={len(a)}, B={len(b)}); "
          f"z {_zpos(common[0]):.3f}..{_zpos(common[-1]):.3f}")

    # tissue-centred crop from the middle matched slice
    mid = tifffile.imread(a[common[len(common) // 2]]).astype(np.float32)
    thr = np.percentile(mid, 85)
    ys, xs = np.where(mid > thr)
    h = W // 2
    cy = int(np.clip(ys.mean(), h, mid.shape[0] - h))
    cx = int(np.clip(xs.mean(), h, mid.shape[1] - h))
    print(f"crop centre (row,col)=({cy},{cx}), window {W}x{W}, {len(common)} slices")

    def read_stack(fdict, tag):
        out = np.empty((len(common), W, W), np.float32)
        for i, t in enumerate(common):
            out[i] = tifffile.imread(fdict[t])[cy - h:cy + h, cx - h:cx + h]
            if i % 10 == 0:
                print(f"  {tag} {i+1}/{len(common)}", flush=True)
        return out

    A = read_stack(a, "A(even)")
    B = read_stack(b, "B(odd)")
    np.save(os.path.join(OUT, "lung_A.npy"), A)
    np.save(os.path.join(OUT, "lung_B.npy"), B)

    Ad, Bd = A.astype(np.float64), B.astype(np.float64)
    n = (Ad - Bd) / np.sqrt(2.0); avg = (Ad + Bd) / 2.0
    print("-" * 60)
    print(f"A mean {Ad.mean():.4f} std {Ad.std():.4f} | B mean {Bd.mean():.4f} std {Bd.std():.4f}")
    print(f"corr(A,B) = {np.corrcoef(Ad.ravel(), Bd.ravel())[0,1]:.4f}  (want high = registered)")
    print(f"noise std={n.std():.5f}  signal(avg) std={avg.std():.5f}  "
          f"SNR~{avg.std()/(n.std()+1e-12):.2f}")
    print(f"cached lung_A.npy / lung_B.npy shape {A.shape} in {OUT}")


if __name__ == "__main__":
    main()
