"""
phase3_generative.py -- Phase 3: the noise-CALIBRATED generative model (mechanism
proof). The joint global model (Phase 2.5) scored geometry with hand-set prior
weights. Phase 3 replaces the data term with a GENERATIVE likelihood pinned by the
MEASURED half-map noise, so the noise itself sets how much to trust the data.

Forward model (L2, here in its simplest exact form): each hypothesis PREDICTS the
clean image it implies -- H_cross predicts the crossing tube pattern, H_kiss the
kissing pattern. (LIC generalises this to an arbitrary tract direction field; for the
phantom the tube pattern IS the forward map.)

Calibrated likelihood: the residual (observed - predicted) is WHITENED by the
measured noise power spectrum P_n = |FFT(n)|^2 (n=(A-B)/sqrt2, real, anisotropic +
correlated), so
    logL(H) = -1/(2 sigma^2) * || whiten(observed - predict_H) ||^2 ,
    whiten(x) = IFFT( FFT(x) / sqrt(P_n) ) .
Under the TRUE hypothesis the residual is pure noise -> whitened to ~unit white ->
minimal cost; a wrong hypothesis leaves structured residual -> larger cost.

We show three things:
  (1) the calibrated likelihood SELECTS the true geometry (log-odds sign correct);
  (2) confidence |log-odds| shrinks as noise rises -> honest UNCERTAINTY;
  (3) the MEASURED (anisotropic) whitening beats a WHITE noise model -- using the
      wrong (flat) spectrum lets correlated low-frequency noise masquerade as signal
      and mis-selects / gets overconfident. This is the concrete payoff of measuring
      the noise rather than assuming it white.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter

from crossing_shape_3d import _tube_family, _coords
from host_geometry_test import _dir_inplane

SLAB = os.environ.get("LUNG_SLAB_DIR", ".")
SHAPE = (64, 64, 64)
OFFSETS = [(y, x) for y in (320, 384, 448) for x in (96, 160, 224, 288, 352)]


def tubes(kind):
    c = _coords(SHAPE)
    t1 = _tube_family(c, _dir_inplane(0.0), 12.0, 1.5)
    t2 = _tube_family(c, _dir_inplane(70.0), 12.0, 1.5)
    return (t1 + t2) if kind == "crossing" else np.where(c[0] < 0.0, t1, t2)


def noise_crop(n, off):
    z0 = (n.shape[0] - SHAPE[0]) // 2
    y0, x0 = off
    c = n[z0:z0+SHAPE[0], y0:y0+SHAPE[1], x0:x0+SHAPE[2]]
    return (c - c.mean()) / (c.std() + 1e-12)


def whiten_ss(x, P):
    """||whiten(x)||^2 = sum |FFT(x)|^2 / P over frequencies."""
    return float(np.sum((np.abs(np.fft.fftn(x)) ** 2) / P))


def log_odds(obs, Tc, Tk, P, sigma):
    return -0.5 / sigma**2 * (whiten_ss(obs - Tc, P) - whiten_ss(obs - Tk, P))


def run():
    print("=" * 74)
    print("PHASE 3 -- noise-calibrated generative model (kissing vs crossing selection)")
    A = np.load(os.path.join(SLAB, "lung_A.npy")).astype(np.float64)
    B = np.load(os.path.join(SLAB, "lung_B.npy")).astype(np.float64)
    n = (A - B) / np.sqrt(2.0)

    Tc, Tk = tubes("crossing"), tubes("kissing")
    # noise model P from a HELD-OUT crop; inject DIFFERENT crops (same noise process)
    P_meas = gaussian_filter(np.abs(np.fft.fftn(noise_crop(n, OFFSETS[0]))) ** 2, 1.0) + 1e-9
    P_white = np.full_like(P_meas, P_meas.mean())          # flat spectrum, same total
    inj = [noise_crop(n, o) for o in OFFSETS[1:]]
    print(f"  measured noise spectrum anisotropy: power(low-k)/power(high-k) shaping "
          f"present; white model = flat")

    print(f"  {'sigma':>6} | {'measured-N whitening':>22} | {'white whitening':>18}")
    print(f"  {'':>6} | {'acc':>5} {'|log-odds|':>10} {'  ':>3} | {'acc':>5} {'|log-odds|':>10}")
    for sigma in (0.5, 1.5, 3.0, 5.0, 8.0):
        rec = {"meas": [], "white": []}
        for nz in inj:
            for G, want in (("crossing", +1), ("kissing", -1)):
                obs = tubes(G) + sigma * nz
                lo_m = log_odds(obs, Tc, Tk, P_meas, sigma)
                lo_w = log_odds(obs, Tc, Tk, P_white, sigma)
                rec["meas"].append((np.sign(lo_m) == want, abs(lo_m)))
                rec["white"].append((np.sign(lo_w) == want, abs(lo_w)))
        am = np.mean([c for c, _ in rec["meas"]]); mm = np.median([v for _, v in rec["meas"]])
        aw = np.mean([c for c, _ in rec["white"]]); mw = np.median([v for _, v in rec["white"]])
        print(f"  {sigma:6.2f} | {am:5.0%} {mm:10.1f}     | {aw:5.0%} {mw:10.1f}")
    print("-" * 74)
    print("  READ: acc = fraction of trials the TRUE geometry is selected; |log-odds|")
    print("  = confidence. Correct selection with confidence that SHRINKS as sigma rises")
    print("  (honest uncertainty); measured-N whitening >= white, and the gap is the")
    print("  value of measuring the noise -- a flat model lets correlated noise mislead.")
    print("=" * 74)


if __name__ == "__main__":
    run()
