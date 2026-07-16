"""
coherence_robustness.py

Robustness / statistics pass for the coherence-onset sigma selector. The Layer
A-E gates in coherence_sigma.py each passed on ONE phantom / ONE seed; noise is
stochastic and crossing angle matters, so those are only suggestive. Here we
repeat the key tests over many seeds x many angles and report DISTRIBUTIONS.

Reuses everything (no rebuilding):
  phantom.make_phantom                          -- phantom generator (explicit seed)
  coherence_sigma.coherence_vs_sigma / knee_sigma  -- coherence-onset selector
  coherence_sigma.lindeberg_peak_sigma          -- bare Lindeberg selector
  coherence_sigma.angular_error_vs_sigma        -- supervised angular-error ruler
  coherence_sigma._selectors_on_image           -- all three picks on one image

Randomness note: the only RNG in the pipeline is make_phantom's noise seed;
every selector is a deterministic function of the image. So fixing the seed
fixes the whole run -- no hidden single-seed assumptions downstream.
"""

import numpy as np

from phantom import make_phantom
from coherence_sigma import (
    coherence_vs_sigma, knee_sigma, angular_error_vs_sigma, _selectors_on_image,
)

SIGMAS = np.arange(0.5, 8.01, 0.5)
N_SEEDS = 20


def _stats(x):
    """mean, sd, min, max, IQR (q25, q75) of a 1D array (NaNs ignored)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(mean=float(np.mean(x)), sd=float(np.std(x)),
                min=float(np.min(x)), max=float(np.max(x)),
                q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                n=int(x.size))


def _fmt(name, s):
    return (f"  {name:24s} mean={s['mean']:+.2f}  sd={s['sd']:.2f}  "
            f"IQR=[{s['q25']:+.2f},{s['q75']:+.2f}]  min/max=[{s['min']:+.2f},{s['max']:+.2f}]  n={s['n']}")


# ---------------------------------------------------------------------------
# RUN 1 -- Layer C robustness (noisy single fibre)
# ---------------------------------------------------------------------------

def run1(noise=0.5, angles=(0, 15, 30, 45, 60, 75), n_seeds=N_SEEDS,
         save="robust_run1.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 74)
    print(f"RUN 1 -- Layer C robustness | noise={noise}, "
          f"{len(angles)} angles x {n_seeds} seeds = {len(angles)*n_seeds} runs")
    lin, knee, truth = [], [], []
    for ang in angles:
        for seed in range(n_seeds):
            img, ta = make_phantom((200, 200), theta_deg=float(ang), spacing=12.0,
                                   width=1.2, noise_sigma=noise, seed=seed)
            d = _selectors_on_image(img, ta, SIGMAS)
            lin.append(d["sig_lin"]); knee.append(d["sig_knee"]); truth.append(d["sig_truth"])
    lin, knee, truth = map(np.array, (lin, knee, truth))
    knee_err, lin_err = knee - truth, lin - truth
    knee_abs, lin_abs = np.abs(knee_err), np.abs(lin_err)
    wins = knee_abs < lin_abs
    ties = knee_abs == lin_abs
    win_frac = float(np.mean(wins))

    print(_fmt("bare Lindeberg sigma", _stats(lin)))
    print(_fmt("coherence-onset sigma", _stats(knee)))
    print(_fmt("angular-error truth", _stats(truth)))
    print("  --- signed error vs truth ---")
    print(_fmt("coherence-onset - truth", _stats(knee_err)))
    print(_fmt("bare Lindeberg - truth", _stats(lin_err)))
    print("  --- |error| vs truth ---")
    print(_fmt("|coherence-onset - truth|", _stats(knee_abs)))
    print(_fmt("|bare Lindeberg - truth|", _stats(lin_abs)))
    print(f"  WIN FRACTION (coherence closer than Lindeberg) = {win_frac:.3f} "
          f"({int(wins.sum())}/{wins.size}), ties={int(ties.sum())}")
    passed = win_frac > 0.6 and _stats(knee_abs)["mean"] < _stats(lin_abs)["mean"]
    print(f"  GATE {'PASS' if passed else 'FAIL'}: coherence-onset "
          f"{'wins clear majority + centred nearer truth' if passed else 'does NOT clearly beat Lindeberg'}")
    print("=" * 74)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].boxplot([lin, knee, truth], tick_labels=["Lindeberg", "coherence-onset", "truth"])
    ax[0].set_ylabel("chosen sigma"); ax[0].set_title(f"Run1: sigma distributions (noise={noise})")
    ax[1].boxplot([lin_err, knee_err], tick_labels=["Lindeberg - truth", "coherence - truth"])
    ax[1].axhline(0, color="k", lw=1, ls=":")
    ax[1].set_ylabel("sigma error vs truth")
    ax[1].set_title(f"Run1: error vs truth (win frac={win_frac:.2f})")
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return dict(lin=lin, knee=knee, truth=truth, win_frac=win_frac, passed=passed)


# ---------------------------------------------------------------------------
# RUN 2 -- Layer D robustness (crossing guard)
# ---------------------------------------------------------------------------

# (theta1, theta2, separation-label). spacing2=17 keeps the two families off a
# shared period (as in the Layer D single run).
CROSSINGS = [
    (20.0, 110.0, "wide"), (25.0, 105.0, "wide"), (30.0, 115.0, "wide"),
    (40.0, 75.0, "narrow"), (50.0, 85.0, "narrow"), (55.0, 95.0, "narrow"),
]
RESOLVE_THRESH = 10.0   # crossing "resolved" if angular error < this (deg)


def run2(noise=0.35, n_seeds=N_SEEDS, save="robust_run2.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 74)
    print(f"RUN 2 -- Layer D robustness (crossing) | noise={noise}, "
          f"{len(CROSSINGS)} pairs x {n_seeds} seeds = {len(CROSSINGS)*n_seeds} runs")
    rows = []
    for t1, t2, sep in CROSSINGS:
        for seed in range(n_seeds):
            img, ta = make_phantom((200, 200), theta_deg=t1, spacing=12.0, width=1.2,
                                   theta2_deg=t2, spacing2=17.0, width2=1.2,
                                   noise_sigma=noise, seed=seed)
            sigs, C, _ = coherence_vs_sigma(img, SIGMAS)
            sig_knee, _ = knee_sigma(sigs, C)
            err = angular_error_vs_sigma(img, ta, SIGMAS)
            err_min = float(np.nanmin(err))
            sig_opt = float(SIGMAS[int(np.nanargmin(err))])
            err_at_onset = float(err[int(np.argmin(np.abs(sigs - sig_knee)))])
            win_mask = err < RESOLVE_THRESH
            win_w = float(SIGMAS[win_mask].max() - SIGMAS[win_mask].min()) if win_mask.any() else 0.0
            inside = bool(err_at_onset < RESOLVE_THRESH)
            rows.append(dict(sep=sep, sig_knee=sig_knee, sig_opt=sig_opt,
                             err_at_onset=err_at_onset, err_min=err_min,
                             win_w=win_w, inside=inside))

    def subset(sep):
        return [r for r in rows if sep is None or r["sep"] == sep]

    for sep in ("wide", "narrow", None):
        rs = subset(sep)
        eao = np.array([r["err_at_onset"] for r in rs])
        ww = np.array([r["win_w"] for r in rs])
        frac_in = float(np.mean([r["inside"] for r in rs]))
        label = sep if sep else "ALL"
        print(f"  --- {label} ({len(rs)} runs) ---")
        print(_fmt("coherence-onset sigma", _stats([r["sig_knee"] for r in rs])))
        print(_fmt("error-optimal sigma", _stats([r["sig_opt"] for r in rs])))
        print(_fmt("angular error at onset", _stats(eao)))
        print(_fmt("minimum angular error", _stats([r["err_min"] for r in rs])))
        print(_fmt("resolving-window width", _stats(ww)))
        print(f"  RESOLVED fraction (err@onset<{RESOLVE_THRESH:.0f} deg) = {frac_in:.3f}"
              f"  -> FUSED fraction = {1-frac_in:.3f}")

    frac_all = float(np.mean([r["inside"] for r in rows]))
    passed = frac_all > 0.8
    print(f"  GATE {'PASS' if passed else 'FAIL'}: onset resolves crossing in "
          f"{frac_all:.0%} of runs {'(large majority)' if passed else '(NOT a large majority)'}")
    print("=" * 74)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].boxplot([[r["err_at_onset"] for r in subset("wide")],
                   [r["err_at_onset"] for r in subset("narrow")]],
                  tick_labels=["wide", "narrow"])
    ax[0].axhline(RESOLVE_THRESH, color="r", ls="--", label=f"resolve thresh {RESOLVE_THRESH:.0f} deg")
    ax[0].set_ylabel("angular error at onset (deg)")
    ax[0].set_title("Run2: error at onset by separation"); ax[0].legend(fontsize=8)
    ax[1].boxplot([[r["win_w"] for r in subset("wide")],
                   [r["win_w"] for r in subset("narrow")]],
                  tick_labels=["wide", "narrow"])
    ax[1].set_ylabel("resolving-window width (sigma)")
    ax[1].set_title("Run2: resolving-window width by separation")
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return dict(rows=rows, frac_all=frac_all, passed=passed)


# ---------------------------------------------------------------------------
# RUN 3 -- low-noise undershoot (is the sigma bias harmless?)
# ---------------------------------------------------------------------------

def run3(noise=0.1, angles=(0, 15, 30, 45, 60, 75), n_seeds=N_SEEDS,
         save="robust_run3.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 74)
    print(f"RUN 3 -- low-noise undershoot | noise={noise}, "
          f"{len(angles)} angles x {n_seeds} seeds = {len(angles)*n_seeds} runs")
    knee_gap, err_gap, err_at_onset_all, err_min_all = [], [], [], []
    for ang in angles:
        for seed in range(n_seeds):
            img, ta = make_phantom((200, 200), theta_deg=float(ang), spacing=12.0,
                                   width=1.2, noise_sigma=noise, seed=seed)
            sigs, C, _ = coherence_vs_sigma(img, SIGMAS)
            sig_knee, _ = knee_sigma(sigs, C)
            err = angular_error_vs_sigma(img, ta, SIGMAS)
            err_min = float(np.nanmin(err))
            sig_truth = float(SIGMAS[int(np.nanargmin(err))])
            err_at_onset = float(err[int(np.argmin(np.abs(sigs - sig_knee)))])
            knee_gap.append(sig_knee - sig_truth)
            err_gap.append(err_at_onset - err_min)
            err_at_onset_all.append(err_at_onset); err_min_all.append(err_min)
    knee_gap = np.array(knee_gap); err_gap = np.array(err_gap)

    print(_fmt("coherence-onset - truth (sigma)", _stats(knee_gap)))
    print(_fmt("angular error at onset (deg)", _stats(np.array(err_at_onset_all))))
    print(_fmt("minimum angular error (deg)", _stats(np.array(err_min_all))))
    print(_fmt("EXTRA error at onset (deg)", _stats(err_gap)))
    # "Benign" = the sigma undershoot costs little angular error.
    benign = _stats(err_gap)["mean"] < 2.0 and _stats(err_gap)["q75"] < 3.0
    print(f"  VERDICT: low-noise undershoot is "
          f"{'BENIGN (extra angular error small)' if benign else 'NOT benign (extra error material)'}")
    print("=" * 74)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].boxplot([knee_gap], tick_labels=["onset - truth"]); ax[0].axhline(0, color="k", ls=":")
    ax[0].set_ylabel("sigma gap"); ax[0].set_title(f"Run3: sigma undershoot (noise={noise})")
    ax[1].boxplot([err_gap], tick_labels=["extra error at onset"]); ax[1].axhline(0, color="k", ls=":")
    ax[1].set_ylabel("deg above minimum"); ax[1].set_title("Run3: angular-error cost of the undershoot")
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return dict(knee_gap=knee_gap, err_gap=err_gap, benign=benign)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("1", "all"):
        run1()
    if which in ("2", "all"):
        run2()
    if which in ("3", "all"):
        run3()
