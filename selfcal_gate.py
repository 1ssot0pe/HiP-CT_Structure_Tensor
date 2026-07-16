"""
selfcal_gate.py

Steps 1 (B1) + 2 (A1) of the gate fix plan.

The gate problem, established by gate_battery / gate_protocol:
  * absolute lambda3 does NOT transfer synthetic->real (phantom threshold landed
    at the 100th i74 percentile), and
  * no single cheap structure-tensor scalar we tried separates crossings cleanly
    (best test AUC ~0.78), because a single fibre's periodic texture looks
    spherical at sigma=3.

Two fixes, tested here together:

A1 -- PLANARITY as a discriminant. A single fibre is PLANAR (Westin c_p high);
     a crossing is less planar. We only ever gated on sphericity (c_s) / lambda3
     -- the crossing-HIGH measures. c_p is the single-fibre-HIGH measure aimed
     exactly at "is this one fibre?". We evaluate it as a crossing-positive gate
     signal cp_inv = -c_p, alone and combined with lambda3.

B1 -- SELF-CALIBRATING threshold (Otsu). Instead of transferring a phantom
     threshold (fails) or an arbitrary percentile, split each signal's OWN
     distribution at the Otsu valley (max between-class variance). It needs no
     labels and no cross-dataset transfer, so it adapts to whatever contrast the
     data has. We validate on phantoms (does the unsupervised Otsu split land
     near the supervised high-recall operating point?) then apply to real i74
     and SHOW the histograms + valleys, since real bimodality is not guaranteed.

Reuses make_plus / low_centre_gain / region_of + geometry (gate_battery),
patch_gradients (host_fod), load_interior (selfcalib_threshold),
threshold_at_catch / auc (gate_protocol).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from skimage.filters import threshold_otsu

from host_fod import patch_gradients
from gate_battery import (make_plus, low_centre_gain, region_of,
                          HALF, Z, CENTERS)
from gate_protocol import threshold_at_catch, auc, _phantoms

EPS = 1e-12


# ---------------------------------------------------------------------------
# per-patch cheap features at sigma=3 (one gradient pass): lambda3, c_s, c_p
# ---------------------------------------------------------------------------

def patch_feats(patch, sigma=3.0):
    G = patch_gradients(patch, sigma)
    if G.shape[0] < 40:
        return None
    lam = np.sort(np.linalg.eigvalsh(G.T @ G))       # ascending: l3,l2,l1
    l3, l2, l1 = lam
    s = lam.sum() + EPS
    c_s = 3.0 * l3 / s
    c_p = 2.0 * (l2 - l3) / s
    # crossing-positive signals (higher = more crossing-like): lam3, c_s, -c_p
    return dict(lam3=float(l3), cs=float(c_s), cp=float(c_p), cp_inv=float(-c_p))


# crossing-positive signals we score (all "higher => fire HOST")
SIGNALS = ("lam3", "cs", "cp_inv")


# ---------------------------------------------------------------------------
# phantom collection with ground-truth crossing labels
# ---------------------------------------------------------------------------

def collect_phantom(seeds=(0, 1, 2, 3)):
    pos = {k: [] for k in SIGNALS}
    neg = {k: [] for k in SIGNALS}
    for vol, masks in _phantoms(seeds):
        for cy in CENTERS:
            for cx in CENTERS:
                patch = vol[Z - HALF:Z + HALF + 1, cy - HALF:cy + HALF + 1,
                            cx - HALF:cx + HALF + 1]
                f = patch_feats(patch)
                if f is None:
                    continue
                is_cross = (region_of(masks, cy, cx) == "crossing")
                for k in SIGNALS:
                    (pos if is_cross else neg)[k].append(f[k])
    return {k: (np.array(pos[k]), np.array(neg[k])) for k in SIGNALS}


def _catch_fp(pos, neg, thr):
    return float(np.mean(pos > thr)), float(np.mean(neg > thr))


# ---------------------------------------------------------------------------
# A1 + B1 on phantoms: AUC per signal, and does Otsu ~ supervised operating pt?
# ---------------------------------------------------------------------------

def phantom_report():
    data = collect_phantom()
    print("=" * 84)
    print("A1/B1 on PHANTOMS (ground-truth crossings) -- AUC + Otsu vs supervised")
    print(f"  {'signal':>7} | {'AUC':>5} | {'Otsu thr':>9} {'catch':>6} {'FP':>5} | "
          f"{'sup@95 thr':>10} {'catch':>6} {'FP':>5}")
    out = {}
    for k in SIGNALS:
        pos, neg = data[k]
        a = auc(pos, neg)
        allv = np.concatenate([pos, neg])
        t_otsu = float(threshold_otsu(allv))
        c_o, f_o = _catch_fp(pos, neg, t_otsu)
        t_sup = threshold_at_catch(pos, neg, target=0.95)
        c_s2, f_s = _catch_fp(pos, neg, t_sup)
        out[k] = dict(auc=a, t_otsu=t_otsu, catch_o=c_o, fp_o=f_o,
                      t_sup=t_sup, catch_s=c_s2, fp_s=f_s)
        print(f"  {k:>7} | {a:5.2f} | {t_otsu:9.4g} {c_o:6.0%} {f_o:5.0%} | "
              f"{t_sup:10.4g} {c_s2:6.0%} {f_s:5.0%}")

    # A1 payoff: combine lambda3 (crossing-high) with cp_inv (single-fibre-low)
    # by a z-scored sum, and score its separability.
    pl, nl = data["lam3"]; pc, nc = data["cp_inv"]
    alll = np.concatenate([pl, nl]); allc = np.concatenate([pc, nc])
    zl = lambda v: (v - alll.mean()) / (alll.std() + EPS)
    zc = lambda v: (v - allc.mean()) / (allc.std() + EPS)
    combo_pos = zl(pl) + zc(pc); combo_neg = zl(nl) + zc(nc)
    a_combo = auc(combo_pos, combo_neg)
    t_combo = threshold_at_catch(combo_pos, combo_neg, target=0.95)
    c_cb, f_cb = _catch_fp(combo_pos, combo_neg, t_combo)
    print("-" * 84)
    print(f"  COMBINED lambda3 + planarity (z-sum): AUC={a_combo:.2f}, "
          f"at 95% catch -> FP={f_cb:.0%}  (best single: "
          f"AUC={max(out[k]['auc'] for k in SIGNALS):.2f})")
    print("=" * 84)
    return data, out, (combo_pos, combo_neg, a_combo)


# ---------------------------------------------------------------------------
# apply to real i74: cheap dense grid, Otsu self-calibrated valley + histograms
# ---------------------------------------------------------------------------

def i74_selfcal(z_slice=100, step=14, save="selfcal_gate.png"):
    import time
    from selfcalib_threshold import load_interior
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    vol = load_interior()
    sub = vol[z_slice - HALF:z_slice + HALF + 1].astype(np.float32)
    sub = (sub - sub.min()) / (np.ptp(sub) + EPS)          # match phantom scale
    centers = list(range(HALF + 8, vol.shape[1] - HALF - 5, step))
    feats = {k: [] for k in ("lam3", "cp")}
    for cy in centers:
        for cx in centers:
            patch = sub[:, cy - HALF:cy + HALF + 1, cx - HALF:cx + HALF + 1]
            f = patch_feats(patch)
            if f is None:
                continue
            feats["lam3"].append(f["lam3"]); feats["cp"].append(f["cp"])
    lam3 = np.array(feats["lam3"]); cp = np.array(feats["cp"])
    n = len(lam3)
    print("=" * 78)
    print(f"B1 on REAL i74 (slice z={z_slice}, {n} cheap patches, no HOST)")
    t_l = float(threshold_otsu(lam3)); t_c = float(threshold_otsu(cp))
    # gate fires on crossings: lambda3 HIGH  OR  planarity c_p LOW
    fire_l = lam3 > t_l
    fire_c = cp < t_c
    fire_or = fire_l | fire_c
    print(f"  lambda3 Otsu valley = {t_l:.4g}  -> fires {fire_l.mean():.0%} "
          f"(lambda3 above valley)")
    print(f"  c_p    Otsu valley = {t_c:.4g}  -> fires {fire_c.mean():.0%} "
          f"(planarity below valley = not-single)")
    print(f"  OR rule (lambda3-high OR planarity-low): fires {fire_or.mean():.0%}, "
          f"skips {(~fire_or).mean():.0%}")
    # sanity: is either distribution actually bimodal enough for Otsu to mean
    # something? report the valley depth (min between the two class means).
    for name, v, t in (("lambda3", lam3, t_l), ("c_p", cp, t_c)):
        lo = v[v <= t]; hi = v[v > t]
        sep = (hi.mean() - lo.mean()) / (v.std() + EPS)
        print(f"    {name}: class means {lo.mean():.3g} | {hi.mean():.3g}  "
              f"(separation {sep:.2f} sd -- >1 suggests a real valley)")
    print("=" * 78)

    # ---- figure: real histograms with Otsu valleys ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    C_FIRE, C_SKIP, INK = "#eb6834", "#2a78d6", "#0b0b0b"
    for a, v, t, name, fires_high in (
            (ax[0], lam3, t_l, "lambda3  (fire if HIGH = crossing-like)", True),
            (ax[1], cp, t_c, "planarity c_p  (fire if LOW = not single)", False)):
        a.hist(v, bins=24, color="#b9c0c9", edgecolor="white")
        a.axvline(t, color="#6a4bc4", ls="--", lw=2.2, label=f"Otsu valley={t:.3g}")
        a.set_xlabel(name); a.set_ylabel("patches"); a.legend(fontsize=9)
        a.set_title(f"real i74 {name.split()[0]} distribution", fontsize=10, color=INK)
    fig.suptitle("B1 self-calibrating gate: does the real i74 distribution have a "
                 "valley Otsu can split?", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save, dpi=130)
    print(f"({time.time()-t0:.0f}s)  Saved {save}")
    return dict(lam3=lam3, cp=cp, t_l=t_l, t_c=t_c, fire_or=fire_or)


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "phantom"):
        phantom_report()
    if stage in ("all", "i74"):
        i74_selfcal()
