"""
gate_protocol.py

Turn the gate_battery finding ("the lambda3 gate misses real crossings") into a
DEFENSIBLE choice of gate, with the train/test discipline the old c_s guard used
(crossing_shape_3d.layer3) but the new cheap gate skipped.

The question: among CHEAP per-patch signals (cheaper than running HOST-FOD),
which best separates a genuine crossing from a non-crossing region, ROBUSTLY,
including under a contrast gradient -- and with a threshold fit on TRAIN phantoms
that still works on DISJOINT TEST phantoms?

CANDIDATE CHEAP SIGNALS (all structure-tensor only -- no sphere sampling / dipy
peak_directions, so all strictly cheaper than HOST analyse()):
  lam3   -- absolute smallest eigenvalue at sigma=3   (the CURRENT gate; cheap_gate)
  cs3    -- normalised sphericity c_s at sigma=3       (contrast-invariant, 1 scale)
  cs_med -- MEDIAN c_s over sigma in {2,3,4}           (multi-scale, robust centre)
  cs_min -- MIN    c_s over sigma in {2,3,4}           (persistence: crossing
            sphericity must be present at EVERY scale, not just the largest --
            the cheap analogue of the HOST multi-scale persistence filter)

Why not "gate on HOST persistence" directly: that runs analyse() at every scale
= the full HOST cost, so it saves nothing. cs_min is the cheap stand-in --
structure-tensor sphericity that PERSISTS across scale.

METHOD (threshold-independent first, then an operating point):
  * AUC = P(crossing scores above non-crossing). Independent of any threshold,
    so it settles "which signal separates" cleanly. Reported on TEST phantoms.
  * Operating point: Youden-optimal threshold (max TPR-FPR) fit on TRAIN
    phantoms, then catch-rate (TPR on crossings) and false-fire (FPR on
    non-crossings) measured on TEST phantoms. A gate is only trustworthy if the
    train threshold transfers.

Ground truth per patch = crossing vs non-crossing, from the exact region masks
of gate_battery.make_plus (uniform, width, contrast-gradient, angled). TRAIN and
TEST use DISJOINT phantom seeds.

Reuses make_plus / low_centre_gain (gate_battery), patch_gradients (host_fod).
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root (host_fod)
from host_fod import patch_gradients
from gate_battery import (make_plus, low_centre_gain, region_of,
                          SHAPE, CROSS_HALF, HALF, Z, CENTERS)

EPS = 1e-12
SCALES = (2.0, 3.0, 4.0)        # cheap multi-scale set (structure tensor only)

# where the fitted gate operating point is cached for run_i74_gated to load.
# The threshold is an ABSOLUTE lambda3, valid only for patches whose intensities
# are normalised to [0,1] the SAME way make_plus normalises its phantoms -- the
# consumer (refined_field.run_i74_gated) must normalise the volume to match.
GATE_FILE = os.path.join(os.path.dirname(__file__), "gate_threshold.json")
TRAIN_SEEDS = (0, 1)
RECALL_TARGET = 0.95


# ---------------------------------------------------------------------------
# per-patch cheap signals: lam3 & cs at sigma=3, plus c_s across SCALES
# ---------------------------------------------------------------------------

def _cs_lam3(patch, sigma):
    G = patch_gradients(patch, sigma)
    if G.shape[0] < 40:
        return None, None
    lam = np.sort(np.linalg.eigvalsh(G.T @ G))       # ascending
    return float(3.0 * lam[0] / (lam.sum() + EPS)), float(lam[0])


def patch_signals(patch):
    cs_by_scale = {}
    lam3_3 = None
    for sg in SCALES:
        cs, lam3 = _cs_lam3(patch, sg)
        if cs is None:
            return None
        cs_by_scale[sg] = cs
        if sg == 3.0:
            lam3_3 = lam3
    css = np.array([cs_by_scale[s] for s in SCALES])
    return dict(lam3=lam3_3, cs3=cs_by_scale[3.0],
                cs_med=float(np.median(css)), cs_min=float(css.min()))


SIGNALS = ("lam3", "cs3", "cs_med", "cs_min")


# ---------------------------------------------------------------------------
# build labelled patches (crossing = 1, non-crossing = 0) over many phantoms
# ---------------------------------------------------------------------------

def _phantoms(seeds):
    """Family of phantoms for a set of seeds -- uniform, width, contrast, angle."""
    out = []
    for s in seeds:
        out.append(make_plus(noise_sigma=0.2, seed=s))                       # uniform
        out.append(make_plus(noise_sigma=0.4, seed=s))                       # noisier
        out.append(make_plus(spacing=7.0, spacing2=10.0, noise_sigma=0.2, seed=s))  # fine
        out.append(make_plus(noise_sigma=0.2, seed=s, gain=low_centre_gain(g_min=0.25)))  # contrast
        out.append(make_plus(angle_deg=60.0, noise_sigma=0.2, seed=s))       # angled
    return out


def collect_rows(seeds):
    """-> list of per-patch dicts (all signals + is_cross), joint pairing kept
    so a multi-signal gate (two-tier) can see both signals on the SAME patch."""
    rows = []
    for vol, masks in _phantoms(seeds):
        for cy in CENTERS:
            for cx in CENTERS:
                patch = vol[Z - HALF:Z + HALF + 1, cy - HALF:cy + HALF + 1,
                            cx - HALF:cx + HALF + 1]
                sig = patch_signals(patch)
                if sig is None:
                    continue
                sig["is_cross"] = (region_of(masks, cy, cx) == "crossing")
                rows.append(sig)
    return rows


def marginals(rows):
    """-> dict signal -> (pos_values, neg_values) from collected rows."""
    out = {}
    for k in SIGNALS:
        pos = np.array([r[k] for r in rows if r["is_cross"]])
        neg = np.array([r[k] for r in rows if not r["is_cross"]])
        out[k] = (pos, neg)
    return out


# ---------------------------------------------------------------------------
# threshold-free separability (AUC) + Youden operating point
# ---------------------------------------------------------------------------

def _rankdata(a):
    """Average ranks (ties shared), 1-based -- like scipy.stats.rankdata."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average tied groups
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auc(pos, neg):
    """P(random crossing scores > random non-crossing). Higher = better gate."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _rankdata(np.concatenate([pos, neg]))
    r_pos = r[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def youden_threshold(pos, neg):
    """Threshold (score > thr => fire) maximising TPR - FPR on these values."""
    cand = np.unique(np.concatenate([pos, neg]))
    mids = (cand[:-1] + cand[1:]) / 2.0 if len(cand) > 1 else cand
    best_t, best_j = cand[0], -np.inf
    for t in np.concatenate([[cand[0] - 1e-9], mids, [cand[-1] + 1e-9]]):
        tpr = np.mean(pos > t)
        fpr = np.mean(neg > t)
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, t
    return float(best_t)


def threshold_at_catch(pos, neg, target=0.95):
    """Lowest threshold (=> least waste) on TRAIN that still catches >= target of
    crossings. This is the gate operating rule: a missed crossing is costly, a
    wasted HOST call is cheap, so we FIX high recall and minimise false-fire."""
    cand = np.unique(np.concatenate([pos, neg]))
    grid = np.concatenate([[cand[0] - 1e-9], (cand[:-1] + cand[1:]) / 2.0,
                           [cand[-1] + 1e-9]])
    ok = [t for t in grid if np.mean(pos > t) >= target]
    return float(max(ok)) if ok else float(grid[0])      # highest thr meeting recall


def roc_curve(pos, neg):
    thr = np.unique(np.concatenate([pos, neg]))
    thr = np.concatenate([[thr[0] - 1e-9], (thr[:-1] + thr[1:]) / 2.0, [thr[-1] + 1e-9]])
    tpr = np.array([np.mean(pos > t) for t in thr])
    fpr = np.array([np.mean(neg > t) for t in thr])
    o = np.argsort(fpr)
    return fpr[o], tpr[o]


# ---------------------------------------------------------------------------
# TWO-TIER gate: a cheap "confidently single -> SKIP" pass (c_s-min low, where
# c_s-min separates best in the low-false-fire corner of the ROC) removes clear
# single-fibre patches, THEN lam3 gates the remainder for crossings. Goal: push
# the HOST-skip fraction back up WITHOUT dropping crossings that single-tier lam3
# would have to waste HOST on.
#   FIRE HOST  iff  (cs_min > t1)  AND  (lam3 > t2)
#   SKIP       otherwise   (t1 = confident-single cut, t2 = crossing cut)
# ---------------------------------------------------------------------------

def fit_two_tier(rows, recall_target=0.95, tier1_keep=0.99):
    """t1: highest c_s-min cut that still keeps `tier1_keep` of crossings above it
    (so the confident-skip pass barely touches real crossings). t2: highest lam3
    cut on the tier-1 survivors keeping COMBINED recall >= recall_target."""
    cross = [r for r in rows if r["is_cross"]]
    csm_cross = np.array([r["cs_min"] for r in cross])
    t1 = float(np.quantile(csm_cross, 1.0 - tier1_keep)) - 1e-9
    # combined recall as a function of t2 (with t1 fixed)
    lam_cand = np.unique([r["lam3"] for r in rows])
    grid = np.concatenate([[lam_cand[0] - 1e-9], (lam_cand[:-1] + lam_cand[1:]) / 2.0,
                           [lam_cand[-1] + 1e-9]])
    best_t2 = grid[0]
    for t2 in grid:
        rec = np.mean([(r["cs_min"] > t1) and (r["lam3"] > t2) for r in cross])
        if rec >= recall_target:
            best_t2 = t2                      # grid ascending -> keep highest feasible
    return t1, float(best_t2)


def eval_two_tier(rows, t1, t2):
    cross = [r for r in rows if r["is_cross"]]
    non = [r for r in rows if not r["is_cross"]]
    fire = lambda r: (r["cs_min"] > t1) and (r["lam3"] > t2)
    catch = float(np.mean([fire(r) for r in cross]))
    fpr = float(np.mean([fire(r) for r in non]))
    saved = float(np.mean([not fire(r) for r in rows]))
    return dict(catch=catch, fpr=fpr, saved=saved, t1=t1, t2=t2)


# ---------------------------------------------------------------------------
# fitted lambda3 gate: the ONE validated operating point run_i74_gated should
# use INSTEAD of the ad-hoc percentile-55 rule. Fit on train seeds for high
# recall, cache to GATE_FILE.
# ---------------------------------------------------------------------------

def fit_lam3_gate(train_seeds=TRAIN_SEEDS, recall_target=RECALL_TARGET):
    """Fit the absolute-lambda3 high-recall threshold on synthetic phantoms with
    KNOWN crossing regions. Returns (thr, diag)."""
    rows = collect_rows(train_seeds)
    pos = np.array([r["lam3"] for r in rows if r["is_cross"]])
    neg = np.array([r["lam3"] for r in rows if not r["is_cross"]])
    thr = threshold_at_catch(pos, neg, target=recall_target)
    allv = np.concatenate([pos, neg])
    diag = dict(lam3_threshold=thr, train_seeds=list(train_seeds),
                recall_target=recall_target,
                train_catch=float(np.mean(pos > thr)),
                train_fire_fraction=float(np.mean(allv > thr)),
                normalisation="patch volume scaled to [0,1] as in make_plus",
                note="ABSOLUTE lambda3; consumer MUST normalise its volume to "
                     "[0,1] before comparing. Fit on synthetic tube phantoms; "
                     "transfer to real tissue is unverified -- check the printed "
                     "percentile diagnostic on real data.")
    return float(thr), diag


def save_gate(path=GATE_FILE, **kw):
    thr, diag = fit_lam3_gate(**kw)
    with open(path, "w") as f:
        json.dump(diag, f, indent=2)
    return thr, diag


def load_gate(path=GATE_FILE):
    """Load the cached fitted gate; fit + save it if missing."""
    if not os.path.exists(path):
        print(f"[gate] {os.path.basename(path)} missing -- fitting once...")
        save_gate(path)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# run: fit on train seeds, report AUC + transferred operating point on test
# ---------------------------------------------------------------------------

def run(train_seeds=(0, 1), test_seeds=(2, 3), save="gate_protocol.png"):
    import time
    t0 = time.time()
    print("=" * 88)
    print(f"GATE PROTOCOL -- train seeds {train_seeds}, test seeds {test_seeds} (disjoint)")
    print("  candidate CHEAP signals; threshold fit on TRAIN, scored on TEST")
    print("=" * 88)

    train_rows = collect_rows(train_seeds)
    test_rows = collect_rows(test_seeds)
    train = marginals(train_rows)
    test = marginals(test_rows)
    n_pos_tr, n_neg_tr = len(train["lam3"][0]), len(train["lam3"][1])
    n_pos_te, n_neg_te = len(test["lam3"][0]), len(test["lam3"][1])
    print(f"  patches: train {n_pos_tr} crossing / {n_neg_tr} non ; "
          f"test {n_pos_te} crossing / {n_neg_te} non")
    print("-" * 88)
    print("  operating point = threshold on TRAIN that catches >=95% of crossings "
          "(gate = high-recall screen)")
    print(f"  {'signal':>7} | {'AUC train':>9} {'AUC test':>9} | "
          f"{'thr@95':>8} | {'catch test':>10} {'waste(FPR)':>10} {'HOST saved':>10}")

    results = {}
    for k in SIGNALS:
        p_tr, n_tr = train[k]
        p_te, n_te = test[k]
        a_tr, a_te = auc(p_tr, n_tr), auc(p_te, n_te)
        thr = threshold_at_catch(p_tr, n_tr, target=0.95)
        catch = float(np.mean(p_te > thr))       # TPR on test crossings
        fpr = float(np.mean(n_te > thr))          # false-fire on test non-crossings
        # fraction of ALL test patches the gate SKIPS (the actual HOST speedup)
        allv = np.concatenate([p_te, n_te])
        saved = float(np.mean(allv <= thr))
        results[k] = dict(a_tr=a_tr, a_te=a_te, thr=thr, catch=catch, fpr=fpr, saved=saved)
        print(f"  {k:>7} | {a_tr:9.3f} {a_te:9.3f} | {thr:8.4g} | "
              f"{catch:10.0%} {fpr:10.0%} {saved:10.0%}")

    print("-" * 88)
    # at matched high recall, the best gate is the one that WASTES least / saves most
    feasible = [k for k in SIGNALS if results[k]["catch"] >= 0.90]
    if feasible:
        winner = min(feasible, key=lambda k: results[k]["fpr"])
        w = results[winner]
        print(f"  BEST gate at >=90% test catch: '{winner}'  "
              f"(catch={w['catch']:.0%}, waste={w['fpr']:.0%}, HOST saved={w['saved']:.0%})")
    else:
        winner = max(SIGNALS, key=lambda k: results[k]["catch"])
        w = results[winner]
        print(f"  NO signal reaches 90% catch on test at the 95%-train threshold; "
              f"best catch = '{winner}' at {w['catch']:.0%}")
    cur = results["lam3"]
    print(f"  CURRENT gate 'lam3': catch={cur['catch']:.0%}, waste={cur['fpr']:.0%}, "
          f"HOST saved={cur['saved']:.0%}  (AUC {cur['a_te']:.3f})")
    if max(results[k]["a_te"] for k in SIGNALS) < 0.75:
        print(f"  WARNING: best test AUC < 0.75 -- no cheap gate separates cleanly; "
              f"any high-recall gate here wastes most of its HOST calls.")

    # ---- two-tier gate: fit on train, score on test ----
    t1, t2 = fit_two_tier(train_rows, recall_target=0.95)
    tt = eval_two_tier(test_rows, t1, t2)
    results["two_tier"] = dict(**tt, a_te=float("nan"))
    print("-" * 88)
    print(f"  TWO-TIER (c_s-min>{t1:.3g} AND lam3>{t2:.3g}):  "
          f"catch={tt['catch']:.0%}, waste={tt['fpr']:.0%}, HOST saved={tt['saved']:.0%}")
    dsave = tt["saved"] - cur["saved"]
    verdict = ("beats single-tier lam3" if (tt["catch"] >= 0.90 and dsave > 0.02)
               else "no better than single-tier lam3")
    print(f"  -> vs single-tier lam3 (saved {cur['saved']:.0%}): "
          f"{dsave:+.0%} HOST saved at matched recall -> {verdict}")
    print("=" * 88)

    # cache the fitted lambda3 operating point for run_i74_gated to load
    _, diag = save_gate()
    print(f"  fitted lambda3 gate -> {os.path.basename(GATE_FILE)}: "
          f"thr={diag['lam3_threshold']:.4g} (train catch {diag['train_catch']:.0%}, "
          f"fires {diag['train_fire_fraction']:.0%})")

    _figure(results, test, save)
    print(f"\n({time.time()-t0:.0f}s)  Saved {save}")
    return results


# ---------------------------------------------------------------------------
# figure: ROC per signal (test) + AUC-test bar. Fixed categorical hue order.
# ---------------------------------------------------------------------------

INK = "#0b0b0b"
# fixed hue order for the four signals (current gate first)
SIG_COL = {"lam3": "#6a4bc4", "cs3": "#1f9e89", "cs_med": "#d9a520", "cs_min": "#c0457a"}
SIG_LABEL = {"lam3": "lam3 (current gate)", "cs3": "c_s @sigma3",
             "cs_med": "c_s median (multi-scale)", "cs_min": "c_s min (persistence)"}


def _figure(results, test, save):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    # panel 1: ROC on test
    ax[0].plot([0, 1], [0, 1], ls=":", color="0.6", lw=1.5, label="chance (AUC 0.5)")
    for k in SIGNALS:
        fpr, tpr = roc_curve(*test[k])
        ax[0].plot(fpr, tpr, "-", color=SIG_COL[k], lw=2.3,
                   label=f"{SIG_LABEL[k]}  (AUC {results[k]['a_te']:.2f})")
    ax[0].set_xlabel("false-fire rate  (HOST wasted on non-crossing)")
    ax[0].set_ylabel("catch rate  (crossings that fire HOST)")
    ax[0].set_title("ROC on TEST phantoms: crossing vs non-crossing\n"
                    "(threshold fit on disjoint train seeds)", fontsize=10, color=INK)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_xlim(-0.02, 1.02); ax[0].set_ylim(-0.02, 1.02)

    # panel 2: AUC test bar, fixed order
    ks = list(SIGNALS)
    aucs = [results[k]["a_te"] for k in ks]
    xpos = np.arange(len(ks))
    ax[1].bar(xpos, aucs, 0.6, color=[SIG_COL[k] for k in ks])
    ax[1].axhline(0.5, color="0.6", ls=":", lw=1.5)
    ax[1].axhline(0.75, color="0.4", ls="--", lw=1.2)
    for x, a in zip(xpos, aucs):
        ax[1].text(x, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels([SIG_LABEL[k].split(" (")[0].replace(" ", "\n") for k in ks],
                          fontsize=8)
    ax[1].set_ylabel("AUC on TEST  (0.5 = chance, higher = better gate)")
    ax[1].set_ylim(0.4, 1.0)
    ax[1].set_title("Threshold-free separability of each cheap gate signal\n"
                    "dashed = 0.75 usable floor", fontsize=10, color=INK)

    # panel 3: the decision that matters -- HOST saved at matched high recall,
    # single-tier lam3 vs two-tier. Bar height = HOST saved; label = test catch.
    C_TT = "#2f7d3a"
    gates = [("lam3", "lam3\n(single tier)", SIG_COL["lam3"]),
             ("cs_min", "c_s min\n(single tier)", SIG_COL["cs_min"]),
             ("two_tier", "two-tier\n(c_s-min -> lam3)", C_TT)]
    xs = np.arange(len(gates))
    saved = [results[k]["saved"] for k, _, _ in gates]
    catch = [results[k]["catch"] for k, _, _ in gates]
    ax[2].bar(xs, saved, 0.6, color=[c for _, _, c in gates])
    for x, s, c in zip(xs, saved, catch):
        ax[2].text(x, s + 0.01, f"saved {s:.0%}\ncatch {c:.0%}", ha="center",
                   va="bottom", fontsize=9, color=INK)
    ax[2].set_xticks(xs); ax[2].set_xticklabels([lab for _, lab, _ in gates], fontsize=8)
    ax[2].set_ylabel("HOST calls SAVED on test  (higher = faster)")
    ax[2].set_ylim(0, max(saved) + 0.15)
    ax[2].set_title("Does a two-tier gate save more HOST at matched >=95% recall?\n"
                    "(bar = HOST saved; label = crossings caught)", fontsize=10, color=INK)

    fig.suptitle("Choosing the HOST gate by train/test protocol: which cheap "
                 "signal actually separates crossings?", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save, dpi=130)


if __name__ == "__main__":
    run()
