"""
brain_phantom_pipeline.py -- run the WHOLE pipeline end-to-end on a 3D phantom that
contains ALL features, with realistic noise, against known ground truth. This is the
capstone check: every stage at once, on truth.

Features (four independent 64^3 blocks, known label each):
  SINGLE   -- one fibre                       -> expect classify "single"
  CROSSING -- two fibres through each other   -> classify "crossing"; tracts TRAVERSE
  KISSING  -- two fibres touching, not crossing-> classify "single" per voxel; tracts
                                                  TERMINATE (only tractography tells it
                                                  from a crossing -- the stage division)
  COMPLEX  -- three random fibres             -> classify "complex"/"disorganised"

Noise is CORRELATED + ANISOTROPIC (the form the lung half-maps taught us), injected as
TWO independent realisations A=S+n_A, B=S+n_B, so the half-map stages run for real.

Stages scored:
  D  measure noise from n=(A-B)/2 -> recover correlated+anisotropic form?
  B  HOST classify each block on the averaged map -> per-block accuracy vs truth
  C  global tractography on CROSSING vs KISSING -> traverse vs terminate?
  F  noise-whitened generative likelihood -> pick crossing vs kissing correctly?
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter

from crossing_shape_3d import _tube_family, _coords
from host_geometry_test import _dir_inplane, _finalise, _random_dirs
from host_fod import analyse, classify
import noise_floor as nf
import phase3_i74 as Q
import phase3_global as P

SHAPE = (64, 64, 64)
NOISE_AMP = 0.07             # noise std ~ 0.07 vs signal std ~0.16 -> SNR ~2-3 (realistic;
                            # the lung's good blocks were ~3-10). 0.15 gave SNR~1 (too harsh).
SMOOTH = (2.5, 0.8, 0.8)     # anisotropic+correlated (streaks along axis0)
HALF = 8
SIGMAS = np.array([0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.7])


def build(kind, seed=0):
    c = _coords(SHAPE)
    if kind == "single":
        s = _tube_family(c, _dir_inplane(0.0), 11.0, 1.5)
    elif kind == "crossing":
        s = _tube_family(c, _dir_inplane(0.0), 11.0, 1.5) + _tube_family(c, _dir_inplane(70.0), 11.0, 1.5)
    elif kind == "kissing":
        s = np.where(c[2] < 0.0, _tube_family(c, _dir_inplane(0.0), 11.0, 1.5),
                     _tube_family(c, _dir_inplane(70.0), 11.0, 1.5))
    else:                                       # complex: 3 random fibres
        s = sum(_tube_family(c, d, 11.0, 1.5) for d in _random_dirs(3, 30.0, seed=seed))
    return _finalise(s)


def anoise(amp, seed):
    rng = np.random.default_rng(seed)
    n = gaussian_filter(rng.standard_normal(SHAPE).astype(np.float64), SMOOTH)
    return n * (amp / (n.std() + 1e-9))


def halfmaps(S, seed):
    return S + anoise(NOISE_AMP, 2 * seed), S + anoise(NOISE_AMP, 2 * seed + 1)


# --- stage D: recover the noise form ---
def stage_D(n):
    trN = np.array([np.trace(nf.N_matrix(n, sg)) for sg in SIGMAS])
    slope = nf.loglog_slope(SIGMAS[SIGMAS >= 2], trN[SIGMAS >= 2])
    aniso = nf.westin(nf.N_matrix(n, 4.0))["aniso_ratio"]
    return slope, aniso


# --- stage B: HOST classification on the averaged map ---
def stage_B(avg, npatch=12, seed=0):
    rng = np.random.default_rng(seed)
    lab = []
    for _ in range(npatch):
        ctr = rng.integers(HALF + 2, SHAPE[0] - HALF - 2, 3)
        pt = avg[ctr[0]-HALF:ctr[0]+HALF+1, ctr[1]-HALF:ctr[1]+HALF+1, ctr[2]-HALF:ctr[2]+HALF+1]
        m = analyse(pt)
        lab.append(classify(m))
    vals, cnt = np.unique(lab, return_counts=True)
    return dict(zip(vals, cnt)), lab


# --- stage C: tractography traverse vs terminate ---
def stage_C(vol, stride=4):
    pos, dirs, w, is_cross, seed_id = Q.seed_segments(vol, stride)
    cand = P.candidate_links(pos, dirs, r_max=2.6 * stride)
    active, adj = P.greedy_forest(cand, len(pos))
    tracts = P.tracts_from(adj, len(pos))
    tract_of = np.full(len(pos), -1, int); tract_len = np.zeros(len(tracts), int)
    for ti, comp in enumerate(tracts):
        tract_len[ti] = len(comp)
        for k in comp:
            tract_of[k] = ti
    conf, dem = Q.crossing_verdicts(seed_id, is_cross, tract_of, tract_len)
    tot = conf + dem
    return conf, dem, (conf / tot if tot else float("nan"))


# --- stage F: noise-whitened generative selection ---
def stage_F(obs, Tc, Tk, P_n, sigma):
    ws = lambda x: float(np.sum((np.abs(np.fft.fftn(x)) ** 2) / P_n))
    return -0.5 / sigma**2 * (ws(obs - Tc) - ws(obs - Tk))    # >0 favours crossing


def run(save="brain_phantom_pipeline.png"):
    print("=" * 74)
    print("WHOLE-PIPELINE CHECK on a 3D all-features phantom (correlated+aniso noise)")
    feats = {k: build(k, seed=1) for k in ("single", "crossing", "kissing", "complex")}
    halves = {k: halfmaps(S, seed=i) for i, (k, S) in enumerate(feats.items())}
    avg = {k: 0.5 * (A + B) for k, (A, B) in halves.items()}

    # ---- D ----
    A, B = halves["crossing"]
    n = (A - B) / np.sqrt(2.0)
    slope, aniso = stage_D(n)
    okD = (slope > -5 + 0.5) and (aniso > 1.2)
    print(f"\n[D noise ]  slope={slope:.2f} (white=-5), aniso={aniso:.2f} (iso=1)  "
          f"-> {'RECOVERED correlated+anisotropic' if okD else 'MISSED'}")

    # ---- B ----
    expect = {"single": "single", "crossing": "crossing", "kissing": "single",
              "complex": ("complex", "disorganised", "dispersed")}
    print("\n[B class ]  per-block HOST label on the averaged map (expected in parens):")
    okB = 0
    for k in feats:
        dist, _ = stage_B(avg[k])
        dom = max(dist, key=dist.get)
        exp = expect[k]
        hit = dom in (exp if isinstance(exp, tuple) else (exp,))
        okB += hit
        print(f"   {k:>8}: {dict(dist)}  dominant={dom:>12}  (expect {exp})  "
              f"{'OK' if hit else 'x'}")

    # ---- C ----
    # the seeder's foreground/anisotropy gates are tuned for dense i74 tissue; the
    # phantom's sparse Gaussian tubes have low patch-mean, so relax them here.
    Q.FG_FRAC, Q.ANISO_MIN = 0.05, 0.5
    print("\n[C tracts]  global tractography (crossing should TRAVERSE, kissing TERMINATE):")
    cc, cd, cr = stage_C(avg["crossing"])
    kc, kd, kr = stage_C(avg["kissing"])
    okC = (cr > 0.5) and (kr < 0.4)
    print(f"   crossing: {cc} confirm / {cd} demote -> confirm-rate {cr:.0%} (want high)")
    print(f"    kissing: {kc} confirm / {kd} demote -> confirm-rate {kr:.0%} (want low)")
    print(f"   -> {'SEPARATED' if okC else 'not separated'}")

    # ---- F ----
    Tc, Tk = feats["crossing"], feats["kissing"]
    P_n = gaussian_filter(np.abs(np.fft.fftn(anoise(NOISE_AMP, 99))) ** 2, 1.0) + 1e-9
    lo_c = stage_F(avg["crossing"], Tc, Tk, P_n, NOISE_AMP)
    lo_k = stage_F(avg["kissing"], Tc, Tk, P_n, NOISE_AMP)
    okF = (lo_c > 0) and (lo_k < 0)
    print(f"\n[F calib ]  whitened likelihood log-odds: crossing obs {lo_c:+.0f} (want>0), "
          f"kissing obs {lo_k:+.0f} (want<0)  -> {'CORRECT' if okF else 'wrong'}")

    print("-" * 74)
    score = sum([okD, okB == 4, okC, okF])
    print(f"  SCORECARD: D {'PASS' if okD else 'FAIL'} | B {okB}/4 | "
          f"C {'PASS' if okC else 'FAIL'} | F {'PASS' if okF else 'FAIL'}  "
          f"=> {score + (okB/4):.2f}/5 stages clean")
    print("=" * 74)

    # figure: top-down of each averaged block
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for a, k in zip(ax, feats):
        a.imshow(avg[k].mean(0), cmap="gray", origin="upper")
        a.set_title(k); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("All-features phantom (z-projection of the averaged half-maps)", fontsize=12)
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    run()
