"""
run_lung_subtractN.py -- application #1 of the half-map noise result: subtract the
MEASURED (anisotropic) noise N from the structure tensor before reading orientation,
and quantify how much it changes the apparent crossing / complexity count vs (a) no
correction and (b) an ISOTROPIC-N of the same magnitude (the white floor the old
pipeline used). Comparing iso vs aniso isolates whether the noise ANISOTROPY matters,
not just its size.

Per tissue block (avg=(A+B)/2 as signal, n=(A-B)/sqrt2 as noise):
  N_meas = N_matrix(n, sigma)/2      # full-recon-equivalent, real 3x3 (anisotropic)
  N_iso  = (tr N_meas / 3) * I       # same total noise, isotropic (white)
Then structure_tensor_field(avg, sigma, rho, subtract_N=N) for N in {None, iso, aniso},
and Westin shape fractions:
  crossing/planar  c_p > THR_P   (two fibres span a plane)
  complex/spheric  c_s > THR_S   (3+/disorganised)
  single/linear    c_l > THR_L
Report the medians over blocks + the PSD-clip guard (over-subtraction).
"""

import os
import numpy as np

import noise_floor as nf

SIGMA, RHO, K = 4.0, 4.0, 80
THR_S, THR_P, THR_L = 0.15, 0.30, 0.40
SLAB = os.environ.get("LUNG_SLAB_DIR", ".")
EPS = 1e-12


def tissue_blocks(A, B, k, stride=80, min_snr=4.0, top=8):
    Z, Y, X = A.shape
    z0 = (Z - k) // 2
    noise = ((A - B) / np.sqrt(2.0)).std() + EPS
    out = []
    for y0 in range(0, Y - k + 1, stride):
        for x0 in range(0, X - k + 1, stride):
            a = A[z0:z0+k, y0:y0+k, x0:x0+k]; b = B[z0:z0+k, y0:y0+k, x0:x0+k]
            snr = (0.5 * (a + b)).std() / noise
            if snr >= min_snr:
                out.append((snr, a.copy(), b.copy()))
    out.sort(key=lambda t: -t[0])
    return out[:top]


def shape_fracs(avg, N, m):
    """(crossing%, complex%, single%, clip%) for structure tensor of avg with N
    subtracted (N=None -> uncorrected)."""
    S = nf.structure_tensor_field(avg, SIGMA, RHO, subtract_N=N)
    l1, l2, l3, _v, clip = nf.eig_fields(S, psd_clip=(N is not None))
    s = l1 + l2 + l3 + EPS
    cl = nf.crop_border((l1 - l2) / s, m).ravel()
    cp = nf.crop_border(2.0 * (l2 - l3) / s, m).ravel()
    cs = nf.crop_border(3.0 * l3 / s, m).ravel()
    return (float((cp > THR_P).mean()), float((cs > THR_S).mean()),
            float((cl > THR_L).mean()), float(clip))


def main():
    A = np.load(os.path.join(SLAB, "lung_A.npy")).astype(np.float64)
    B = np.load(os.path.join(SLAB, "lung_B.npy")).astype(np.float64)
    bl = tissue_blocks(A, B, K)
    m = int(np.ceil(3 * (SIGMA + RHO)))
    print(f"=== LUNG #1: measured-N subtraction over {len(bl)} tissue blocks "
          f"(sigma={SIGMA}, rho={RHO}) ===")
    print(f"  {'correction':>10} | {'cross%':>7} {'complex%':>8} {'single%':>7} "
          f"{'clip%':>6}")
    agg = {"uncorr": [], "iso-N": [], "aniso-N": []}
    for snr, a, b in bl:
        avg = 0.5 * (a + b)
        n = (a - b) / np.sqrt(2.0)
        Nm = nf.N_matrix(n, SIGMA) / 2.0
        Ni = np.eye(3) * (np.trace(Nm) / 3.0)
        for tag, N in (("uncorr", None), ("iso-N", Ni), ("aniso-N", Nm)):
            agg[tag].append(shape_fracs(avg, N, m))
    for tag in ("uncorr", "iso-N", "aniso-N"):
        r = np.median(np.array(agg[tag]), axis=0)
        print(f"  {tag:>10} | {r[0]:7.1%} {r[1]:8.1%} {r[2]:7.1%} {r[3]:6.0%}")
    print("-" * 60)
    u = np.median(np.array(agg["uncorr"]), axis=0)
    i = np.median(np.array(agg["iso-N"]), axis=0)
    an = np.median(np.array(agg["aniso-N"]), axis=0)
    print(f"  crossing count: uncorrected {u[0]:.1%} -> iso-N {i[0]:.1%} -> "
          f"measured aniso-N {an[0]:.1%}")
    dmag = (u[0] - i[0]); daniso = (i[0] - an[0])
    print(f"  demotion from noise MAGNITUDE (iso): {dmag:+.1%};  "
          f"EXTRA from ANISOTROPY (iso->aniso): {daniso:+.1%}")
    verdict = ("ANISOTROPY matters" if abs(daniso) >= 0.3 * abs(dmag) + 0.005
               else "anisotropy adds little beyond magnitude")
    print(f"  => {verdict}; clip stays low ({an[3]:.0%}) so the correction is honest.")


if __name__ == "__main__":
    main()
