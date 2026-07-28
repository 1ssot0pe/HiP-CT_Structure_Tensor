"""
run_lung_multiblock.py -- robustness of the half-map noise verdict across MANY tissue
blocks (a single 80^3 block is a weak basis for a definitive claim; T1's slope was
block-dependent). For each high-SNR block we measure the two headline quantities:
  * T1 noise character: high-sigma tr N slope (WHITE ~ -5 vs CORRELATED shallower)
    and anisotropy lambda1/lambda3 of N at sigma=4;
  * T2 premise: fraction of c_s>thr on the averaged map, UNCORRECTED vs corrected by
    the MEASURED N, and the PSD-clip fraction (over-subtraction guard).
Report the distribution over blocks, so the verdict rests on the spread, not one cube.
"""

import os
import numpy as np

import noise_floor as nf

VOX_UM = 16.547
K = 80
THR = 0.15
SIGMAS_N = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.7, 8.0])
SLAB = os.environ.get("LUNG_SLAB_DIR", ".")


def blocks(A, B, k, stride=80, min_snr=4.0):
    Z, Y, X = A.shape
    z0 = (Z - k) // 2
    noise = ((A - B) / np.sqrt(2.0)).std() + 1e-12
    out = []
    for y0 in range(0, Y - k + 1, stride):
        for x0 in range(0, X - k + 1, stride):
            a = A[z0:z0+k, y0:y0+k, x0:x0+k]; b = B[z0:z0+k, y0:y0+k, x0:x0+k]
            snr = (0.5*(a+b)).std() / noise
            if snr >= min_snr:
                out.append((snr, a.copy(), b.copy(), (z0, y0, x0)))
    out.sort(key=lambda t: -t[0])
    return out


def analyse_block(a, b):
    n = (a - b) / np.sqrt(2.0)
    avg = 0.5 * (a + b)
    # T1: tr N slope (sigma>=2) and anisotropy at sigma=4
    trN = np.array([np.trace(nf.N_matrix(n, sg)) for sg in SIGMAS_N])
    slope = nf.loglog_slope(SIGMAS_N[SIGMAS_N >= 2], trN[SIGMAS_N >= 2])
    aniso = nf.westin(nf.N_matrix(n, 4.0))["aniso_ratio"]
    # T2: c_s frac uncorrected vs measured-N corrected (full-recon-equiv N = /2)
    N4 = nf.N_matrix(n, 4.0) / 2.0
    Su = nf.structure_tensor_field(avg, 4.0, 4.0)
    csu, _ = nf.cs_field(Su)
    Sc = nf.structure_tensor_field(avg, 4.0, 4.0, subtract_N=N4)
    csc, clip = nf.cs_field(Sc, psd_clip=True)
    m = int(np.ceil(4 * (4.0 + 4.0)))
    csu = nf.crop_border(csu, m).ravel(); csc = nf.crop_border(csc, m).ravel()
    return slope, aniso, float((csu > THR).mean()), float((csc > THR).mean()), clip


def main():
    A = np.load(os.path.join(SLAB, "lung_A.npy")).astype(np.float64)
    B = np.load(os.path.join(SLAB, "lung_B.npy")).astype(np.float64)
    bl = blocks(A, B, K)[:12]
    print(f"=== LUNG half-map robustness over {len(bl)} tissue blocks "
          f"(SNR>=4, {VOX_UM} um) ===")
    print(f"  {'SNR':>5} {'slope':>6} {'aniso':>6} | {'cs%unc':>7} {'cs%cor':>7} "
          f"{'clip%':>6}")
    rows = []
    for snr, a, b, off in bl:
        sl, an, fu, fc, cl = analyse_block(a, b)
        rows.append((snr, sl, an, fu, fc, cl))
        print(f"  {snr:5.1f} {sl:6.2f} {an:6.2f} | {fu:7.0%} {fc:7.0%} {cl:6.0%}")
    R = np.array(rows)
    print("-" * 58)
    med = np.median(R, axis=0)
    print(f"  MEDIAN slope={med[1]:.2f} (white=-5), aniso={med[2]:.2f} "
          f"(iso=1); c_s {med[3]:.0%}->{med[4]:.0%} corrected, clip {med[5]:.0%}")
    survives = np.median(R[:, 4]) >= 0.6 * np.median(R[:, 3]) and np.median(R[:, 5]) < 0.3
    corr = med[1] > -4.0
    print(f"  => NOISE: {'CORRELATED/low-frequency' if corr else 'approx WHITE'}, "
          f"{'ANISOTROPIC' if med[2] > 1.2 else 'isotropic'}")
    print(f"  => PREMISE: c_s complexity {'SURVIVES measured-N -> REAL tissue' if survives else 'COLLAPSES -> largely noise'} "
          f"(clip {med[5]:.0%})")


if __name__ == "__main__":
    main()
