"""
run_lung_halfmap.py -- the half-map analysis (T1-T4) on the cached LADAF lung slab.

Loads the lung_A/lung_B .npy slabs from build_lung_slab.py, takes a cubic block (the
analysis assumes a cube), and runs the four tasks from half_map_analysis:
  T1 NOISE      measure the real noise floor N(sigma) on n=(A-B)/sqrt2 -- WHITE
                (slope ~ -5) or CORRELATED (ring/cupping)? anisotropic?  <-- the
                question that was untestable on i74 (clipped, no half-recon)
  T2 PREMISE    correct c_s on the averaged map with the MEASURED N -- does the
                pervasive high c_s COLLAPSE (noise) or SURVIVE (real tissue)?
  T3 CERTIFY    orientation-FSC (sigma,rho) certifier between the two half-maps
  T4 RESOLUTION plain FSC(A,B) -> the reconstruction's own resolution

This is the definitive version of the noise thread: N is MEASURED here, not bounded.
"""

import os
import numpy as np

import half_map_analysis as H

VOX_UM = 16.547
K = int(os.environ.get("LUNG_K", "80"))
SLAB = os.environ.get("LUNG_SLAB_DIR", ".")


def best_block(A, B, k, stride=16):
    """Pick the k^3 sub-block with the highest SNR = std(signal)/std(noise). The
    geometric centre of a lung crop is mostly air (signal < noise); T2/T3/T4 need a
    tissue-rich block or they trivially read 'noise'. z is capped at ~81 so the cube
    uses the central k in z and slides in-plane."""
    Z, Y, X = A.shape
    z0 = (Z - k) // 2
    noise = ((A - B) / np.sqrt(2.0)).std() + 1e-12
    best, best_snr = (z0, (Y - k)//2, (X - k)//2), -1.0
    for y0 in range(0, Y - k + 1, stride):
        for x0 in range(0, X - k + 1, stride):
            avg = 0.5 * (A[z0:z0+k, y0:y0+k, x0:x0+k] + B[z0:z0+k, y0:y0+k, x0:x0+k])
            snr = avg.std() / noise
            if snr > best_snr:
                best_snr, best = snr, (z0, y0, x0)
    z, y, x = best
    return (A[z:z+k, y:y+k, x:x+k].copy(), B[z:z+k, y:y+k, x:x+k].copy(),
            best, best_snr)


def main():
    Aslab = np.load(os.path.join(SLAB, "lung_A.npy")).astype(np.float64)
    Bslab = np.load(os.path.join(SLAB, "lung_B.npy")).astype(np.float64)
    A, B, off, snr = best_block(Aslab, Bslab, K)
    print(f"selected tissue block at (z,y,x)={off}, SNR={snr:.2f}", flush=True)
    LOG = []
    def log(m): print(m, flush=True); LOG.append(m)

    log(f"=== LUNG HALF-MAP ANALYSIS (LADAF-2022-13 lung-left, {VOX_UM} um) ===")
    log(f"cubic block {A.shape}; A mean={A.mean():.4f} B mean={B.mean():.4f}")
    n = (A - B) / np.sqrt(2.0)
    log(f"noise n std={n.std():.5f}; averaged-map std={((A+B)/2).std():.5f}; "
        f"frac avg==0 {((A+B)/2 == 0).mean():.1%}")
    log("-" * 66)
    N_full = H.task1_noise(A, B, log)
    log("-" * 66); H.task2_premise(A, B, N_full, log)
    log("-" * 66); H.task3_certify(A, B, log)
    log("-" * 66); H.task4_resolution(A, B, log, vox_um=VOX_UM)
    log("=" * 66)
    open("LUNG_HALFMAP_RESULTS.md", "w").write(
        "# Lung half-map analysis (LADAF-2022-13, 16.547 um)\n\n```\n"
        + "\n".join(LOG) + "\n```\n")
    log("Wrote LUNG_HALFMAP_RESULTS.md")


if __name__ == "__main__":
    main()
