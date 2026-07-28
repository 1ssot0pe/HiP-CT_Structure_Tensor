"""
coherence_sigma.py -- a REPLACEMENT sigma selector based on orientation coherence.

Motivation
----------
The bare Lindeberg gamma-normalised selector (unsupervised.lindeberg_response)
picks sigma at the peak of sigma^2 * mean(largest structure-tensor eigenvalue).

On noisy data that peak collapses onto the smallest sigma: noise is the sharpest
fine-scale "structure", so gradient energy is maximal there. Demonstrated to
fail on the real HiP-CT chunk (see repo-root lindeberg_real_test.py) and flagged
in pick_sigma_rho.py.

Idea
----
Noise is INCOHERENT; a fibre is COHERENT. The structure-tensor eigenvalue spread
gives a scalar orientation-coherence per pixel,

        C = ((lambda_max - lambda_min) / (lambda_max + lambda_min))^2   in [0, 1]

(1 = perfectly oriented, 0 = isotropic). Averaged over a region, C(sigma) rises
from ~0 (small sigma, noise dominated) to a plateau (noise suppressed, structure
resolved). We pick sigma at the ONSET / KNEE of that rise -- NOT its maximum,
because over-smoothing ALSO raises coherence (it fuses crossing fibres into one
false direction), so max-coherence is a trap.

Reuses the Layer-1 forward pipeline (forward.compute_e3, crop_margin, _crop),
the Layer-0 phantom (phantom.make_phantom), the Layer-2 angular-error ruler
(supervised.mean_angular_error) and the Layer-3 bare Lindeberg selector
(unsupervised.lindeberg_response). Nothing is rebuilt.
"""

import numpy as np
from structure_tensor import structure_tensor_2d, eig_special_2d

from forward import crop_margin, _crop
from phantom import make_phantom


# ---------------------------------------------------------------------------
# LAYER A -- coherence measure
# ---------------------------------------------------------------------------

def coherence_map(image, sigma, rho, truncate=4.0, eps=1e-12):
    """Per-pixel structure-tensor coherence ((l1-l2)/(l1+l2))^2 in [0,1].

    Uses the SAME structure_tensor_2d / eig_special_2d as the forward pipeline;
    val[0] <= val[1] are the two eigenvalues.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        S = structure_tensor_2d(image.astype(np.float64), sigma, rho, truncate=truncate)
        val, _vec = eig_special_2d(S)
    lo, hi = val[0], val[1]
    return ((hi - lo) / (hi + lo + eps)) ** 2


def mean_coherence(image, sigma, rho, truncate=4.0):
    """Region-averaged coherence at one (sigma, rho), cropped to valid border."""
    C = coherence_map(image, sigma, rho, truncate)
    C = _crop(C, crop_margin(sigma, rho, truncate))
    return float(np.nanmean(C))


def _noise_image(shape=(200, 200), seed=0):
    """Pure incoherent noise, normalised to [0,1] (coherence is scale-free, but
    keeps it comparable to the phantom's range)."""
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(shape)
    return (n - n.min()) / (np.ptp(n) + 1e-12)


def layer_a(sigma=1.5, rho=3.0):
    """GATE: coherence must be HIGH on a clean fibre, LOW on pure noise."""
    print("=" * 66)
    print("LAYER A -- coherence separates fibre from noise?")
    print(f"  measured at sigma={sigma}, rho={rho}")
    fib, _ = make_phantom(shape=(200, 200), theta_deg=30.0, spacing=12.0,
                          width=1.2, noise_sigma=0.0)
    noise = _noise_image((200, 200), seed=0)
    c_fib = mean_coherence(fib, sigma, rho)
    c_noise = mean_coherence(noise, sigma, rho)
    print(f"  coherence(clean fibre) = {c_fib:.4f}")
    print(f"  coherence(pure noise)  = {c_noise:.4f}")
    passed = c_fib > 0.5 and c_noise < 0.2 and c_fib > 3 * c_noise
    print(f"  GATE {'PASS' if passed else 'FAIL'}: "
          f"fibre {'>>' if passed else 'NOT >>'} noise")
    print("=" * 66)
    return passed, c_fib, c_noise


# ---------------------------------------------------------------------------
# LAYER B -- coherence-ONSET sigma selector (knee, NOT maximum)
# ---------------------------------------------------------------------------

def coherence_vs_sigma(image, sigmas, rho_ratio=2.0, truncate=4.0):
    """C(sigma) with rho tied to sigma (rho = rho_ratio * sigma), so a single
    sigma sweep drives both scales. Returns (sigmas, C, rhos)."""
    sigmas = np.asarray(sigmas, float)
    rhos = rho_ratio * sigmas
    C = np.array([mean_coherence(image, s, r, truncate)
                  for s, r in zip(sigmas, rhos)])
    return sigmas, C, rhos


def knee_sigma(sigmas, C, tau=0.2):
    """Coherence-ONSET sigma via an explicit slope-threshold knee.

    Method (printed so it can be checked):
      1. slope[i] = (C[i+1]-C[i]) / (sigma[i+1]-sigma[i]).
      2. i_rise = argmax(slope) -- the steepest part of the coherence rise.
      3. onset = first i >= i_rise where slope[i] < tau * slope[i_rise]
         -- i.e. the sigma at which the rise first FLATTENS (the plateau starts).
    Taking the FIRST flattening (not the global max of C) is what avoids the
    over-smoothing trap: if C rises again later (crossing fusion), that second
    rise is ignored because we stop at the first knee.
    """
    sigmas = np.asarray(sigmas, float)
    C = np.asarray(C, float)
    slope = np.diff(C) / np.diff(sigmas)
    i_rise = int(np.argmax(slope))
    thresh = tau * slope[i_rise]
    onset = None
    for i in range(i_rise, slope.size):
        if slope[i] < thresh:
            onset = i
            break
    if onset is None:                       # never flattened in range
        onset = sigmas.size - 1
    info = dict(slope=slope, i_rise=i_rise, thresh=float(thresh),
                onset_idx=int(onset), max_slope=float(slope[i_rise]))
    return float(sigmas[onset]), info


def layer_b(noise_sigma=0.35, seed=1, save="coherence_layerB.png"):
    """Show the coherence curve and the knee the selector picks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 66)
    print("LAYER B -- coherence-onset selector (knee, not max)")
    image, true_angles = make_phantom(shape=(200, 200), theta_deg=30.0,
                                       spacing=12.0, width=1.2,
                                       noise_sigma=noise_sigma, seed=seed)
    sigmas = np.arange(0.5, 8.01, 0.5)
    sigmas, C, rhos = coherence_vs_sigma(image, sigmas)
    sigma_onset, info = knee_sigma(sigmas, C)
    sigma_maxC = float(sigmas[int(np.argmax(C))])
    print(f"  noise_sigma={noise_sigma}, true angle={true_angles[0]:.0f} deg")
    print(f"  steepest rise at sigma={sigmas[info['i_rise']]:.2f} "
          f"(slope={info['max_slope']:.3f}), knee threshold={info['thresh']:.3f}")
    print(f"  chosen ONSET sigma = {sigma_onset:.2f}   "
          f"(vs argmax-coherence sigma = {sigma_maxC:.2f}, the trap)")
    print("=" * 66)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(sigmas, C, "o-", label="mean coherence")
    ax.axvline(sigma_onset, color="g", ls="--", label=f"onset (knee) sigma={sigma_onset:.2f}")
    ax.axvline(sigma_maxC, color="orange", ls=":", label=f"max-coherence sigma={sigma_maxC:.2f} (trap)")
    ax.set_xlabel("sigma (rho = 2*sigma)")
    ax.set_ylabel("mean coherence")
    ax.set_title(f"Layer B: coherence rise + knee (noisy single-angle, noise={noise_sigma})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return sigma_onset, sigma_maxC


# ---------------------------------------------------------------------------
# LAYER C -- head-to-head on a NOISY single-angle phantom
# ---------------------------------------------------------------------------

def angular_error_vs_sigma(image, true_angles, sigmas, rho_ratio=2.0, truncate=4.0):
    """Supervised mean angular error(sigma) at rho = rho_ratio*sigma (the SAME
    rho policy the coherence selector uses, so the sigmas are comparable)."""
    from supervised import mean_angular_error
    sigmas = np.asarray(sigmas, float)
    return np.array([mean_angular_error(image, s, rho_ratio * s, true_angles, truncate)
                     for s in sigmas])


def lindeberg_peak_sigma(image, sigmas, rho=0.5):
    """Bare Lindeberg gamma-normalised selector: argmax of sigma^2*mean(l_max)."""
    from unsupervised import lindeberg_response
    resp = lindeberg_response(image, sigmas, rho=rho)
    return float(sigmas[int(np.nanargmax(resp))]), resp


def _selectors_on_image(image, true_angles, sigmas, rho_ratio=2.0):
    """All three sigma picks on one image. Returns dict of curves + picks."""
    sig_lin, resp = lindeberg_peak_sigma(image, sigmas)
    sigs, C, _rhos = coherence_vs_sigma(image, sigmas, rho_ratio)
    sig_knee, info = knee_sigma(sigs, C)
    err = angular_error_vs_sigma(image, true_angles, sigmas, rho_ratio)
    sig_truth = float(sigmas[int(np.nanargmin(err))])
    return dict(sigmas=np.asarray(sigmas, float), resp=resp, C=C, err=err,
                sig_lin=sig_lin, sig_knee=sig_knee, sig_truth=sig_truth, info=info)


def layer_c(noise_sigma=0.5, seed=2, save="coherence_layerC.png"):
    """GATE: coherence-onset ~ angular-error truth, while bare Lindeberg
    collapses toward the smallest sigma."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 66)
    print("LAYER C -- head-to-head on NOISY single-angle phantom")
    image, true_angles = make_phantom(shape=(200, 200), theta_deg=30.0,
                                       spacing=12.0, width=1.2,
                                       noise_sigma=noise_sigma, seed=seed)
    sigmas = np.arange(0.5, 8.01, 0.5)
    d = _selectors_on_image(image, true_angles, sigmas)
    print(f"  noise_sigma={noise_sigma}")
    print(f"  bare Lindeberg peak sigma  = {d['sig_lin']:.2f}")
    print(f"  coherence-onset sigma      = {d['sig_knee']:.2f}")
    print(f"  angular-error truth sigma  = {d['sig_truth']:.2f}")
    lin_collapses = d["sig_lin"] <= sigmas[1] + 1e-9
    coh_tracks = abs(d["sig_knee"] - d["sig_truth"]) <= 1.0
    passed = lin_collapses and coh_tracks
    print(f"  bare Lindeberg collapsed to smallest sigma? {lin_collapses}")
    print(f"  coherence-onset within 1.0 of truth?        {coh_tracks}")
    print(f"  GATE {'PASS' if passed else 'FAIL'}: "
          f"coherence tracks truth where bare Lindeberg fails")
    print("=" * 66)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    s = d["sigmas"]
    ax.plot(s, d["C"], "o-", color="tab:blue", label="coherence")
    ax.plot(s, d["resp"] / np.nanmax(d["resp"]), "s-", color="tab:red",
            alpha=0.7, label="Lindeberg resp (norm)")
    ax.plot(s, d["err"] / np.nanmax(d["err"]), "^-", color="0.5",
            alpha=0.7, label="angular error (norm)")
    ax.axvline(d["sig_knee"], color="tab:blue", ls="--", label=f"coherence-onset={d['sig_knee']:.2f}")
    ax.axvline(d["sig_lin"], color="tab:red", ls="--", label=f"Lindeberg peak={d['sig_lin']:.2f}")
    ax.axvline(d["sig_truth"], color="k", ls=":", lw=2, label=f"angular-error truth={d['sig_truth']:.2f}")
    ax.set_xlabel("sigma (rho = 2*sigma for coherence/error)")
    ax.set_ylabel("normalised value")
    ax.set_title(f"Layer C: noisy phantom (noise={noise_sigma}) -- coherence tracks truth, Lindeberg collapses")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return d, passed


# ---------------------------------------------------------------------------
# LAYER D -- crossing-phantom guard (the mirror failure: over-smoothing)
# ---------------------------------------------------------------------------

def layer_d(noise_sigma=0.35, seed=3, save="coherence_layerD.png"):
    """GATE: on a CROSSING phantom, coherence-onset must NOT over-smooth --
    the chosen sigma must still sit in the angular-error valley (resolving both
    directions), not out where smoothing has fused them into one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 66)
    print("LAYER D -- crossing guard (does coherence-onset over-smooth?)")
    image, true_angles = make_phantom(shape=(200, 200), theta_deg=20.0,
                                       spacing=12.0, width=1.2,
                                       theta2_deg=110.0, spacing2=17.0, width2=1.2,
                                       noise_sigma=noise_sigma, seed=seed)
    sigmas = np.arange(0.5, 8.01, 0.5)
    sigs, C, _ = coherence_vs_sigma(image, sigmas)
    sig_knee, info = knee_sigma(sigs, C)
    err = angular_error_vs_sigma(image, true_angles, sigmas)
    sig_truth = float(sigmas[int(np.nanargmin(err))])
    err_min = float(np.nanmin(err))
    err_at_knee = float(err[int(np.argmin(np.abs(sigs - sig_knee)))])
    print(f"  crossing angles = {true_angles}, noise_sigma={noise_sigma}")
    print(f"  coherence-onset sigma     = {sig_knee:.2f}")
    print(f"  angular-error truth sigma = {sig_truth:.2f} (min error {err_min:.2f} deg)")
    print(f"  angular error AT onset    = {err_at_knee:.2f} deg")
    # Over-smoothing shows up as a LARGE angular error at the chosen sigma
    # (fused direction far from both truths) and/or onset >> truth.
    not_oversmoothed = (sig_knee <= sig_truth + 1.0) and (err_at_knee <= err_min + 3.0)
    print(f"  onset resolves the crossing (no fusion)? {not_oversmoothed}")
    print(f"  GATE {'PASS' if not_oversmoothed else 'FAIL'}: "
          f"{'coherence-onset stays in the resolving regime' if not_oversmoothed else 'OVER-SMOOTHED -- fused the crossing'}")
    print("=" * 66)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(sigs, C, "o-", color="tab:blue", label="coherence")
    ax.plot(sigs, err / np.nanmax(err), "^-", color="0.5", label="angular error (norm)")
    ax.axvline(sig_knee, color="tab:blue", ls="--", label=f"coherence-onset={sig_knee:.2f}")
    ax.axvline(sig_truth, color="k", ls=":", lw=2, label=f"error truth={sig_truth:.2f}")
    ax.set_xlabel("sigma (rho = 2*sigma)")
    ax.set_ylabel("normalised value")
    ax.set_title(f"Layer D: crossing {true_angles[0]:.0f}/{true_angles[1]:.0f} deg -- onset must not over-smooth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return sig_knee, sig_truth, err_at_knee, err_min, not_oversmoothed


# ---------------------------------------------------------------------------
# LAYER E -- noise sweep: how each selector's sigma moves with noise
# ---------------------------------------------------------------------------

def layer_e(noises=(0.1, 0.2, 0.35, 0.5, 0.7, 1.0), seed=4,
            save="coherence_layerE.png"):
    """Track bare-Lindeberg / coherence-onset / angular-error-truth sigma as
    noise grows. Expected: coherence-onset stays near truth and rises gently;
    bare Lindeberg degrades toward the smallest sigma."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 66)
    print("LAYER E -- noise sweep")
    sigmas = np.arange(0.5, 8.01, 0.5)
    lin, knee, truth = [], [], []
    for ns in noises:
        image, ta = make_phantom(shape=(200, 200), theta_deg=30.0, spacing=12.0,
                                 width=1.2, noise_sigma=ns, seed=seed)
        d = _selectors_on_image(image, ta, sigmas)
        lin.append(d["sig_lin"]); knee.append(d["sig_knee"]); truth.append(d["sig_truth"])
        print(f"  noise={ns:4.2f} | Lindeberg={d['sig_lin']:.2f}  "
              f"coherence-onset={d['sig_knee']:.2f}  truth={d['sig_truth']:.2f}")
    lin, knee, truth = map(np.array, (lin, knee, truth))
    coh_mae = float(np.mean(np.abs(knee - truth)))
    lin_mae = float(np.mean(np.abs(lin - truth)))
    print(f"  mean|sigma - truth|:  coherence-onset={coh_mae:.2f}  bare Lindeberg={lin_mae:.2f}")
    passed = coh_mae < lin_mae
    print(f"  GATE {'PASS' if passed else 'FAIL'}: coherence-onset "
          f"{'closer to' if passed else 'NOT closer to'} truth than bare Lindeberg across noise")
    print("=" * 66)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(noises, truth, "ks:", lw=2, label="angular-error truth")
    ax.plot(noises, knee, "o-", color="tab:blue", label="coherence-onset")
    ax.plot(noises, lin, "s-", color="tab:red", label="bare Lindeberg")
    ax.set_xlabel("phantom noise_sigma")
    ax.set_ylabel("chosen sigma")
    ax.set_title("Layer E: selector sigma vs noise (coherence tracks truth; Lindeberg collapses)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    print(f"  saved {save}")
    return dict(noises=np.array(noises), lin=lin, knee=knee, truth=truth,
                coh_mae=coh_mae, lin_mae=lin_mae, passed=passed)


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "A"
    if stage == "A":
        layer_a()
    elif stage == "B":
        layer_a()
        layer_b()
    elif stage == "C":
        layer_c()
    elif stage == "D":
        layer_d()
    elif stage == "E":
        layer_e()
    elif stage == "all":
        layer_a(); layer_b(); layer_c(); layer_d(); layer_e()
