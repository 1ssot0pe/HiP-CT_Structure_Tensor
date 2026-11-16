"""
i74_noise_transfer_falsify.py -- FALSIFICATION check on the cross-organ noise
transfer that every "real noise" validation in this project relies on.

Context. i74 (brain) has NO half-maps, so its own noise cannot be measured by
the clean n=(A-B)/sqrt(2) route. Yet the pipeline's real-noise validations
(mass_conservation_validate.py, sigma_rho_robustness.py, phase3.py,
the 89%->57% tractography degradation) all inject LUNG half-map noise
(lung_A.npy/lung_B.npy) as a stand-in. That is a cross-ORGAN proxy. This
script cannot prove the transfer is valid -- with a single averaged volume
there is no signal-free region to isolate i74's noise spectrum (its air is
clipped to exactly 0, carrying no noise; tissue carries unresolved real
structure). It can only try to BREAK the assumption.

The one place i74 noise is visible in a single volume is the HIGH-FREQUENCY
TAIL of its radial power spectrum -- above the spatial scale where fibre
signal lives, what remains is noise + unresolved sub-resolution structure +
reconstruction roll-off. We compare that tail's log-log slope against the
SAME tail of pure lung noise n=(A-B)/sqrt(2), both through the identical
estimator (noise_floor.power_spectrum_3d + radial binning).

Logic (asymmetric on purpose):
  - if i74's high-freq tail slope is WILDLY different from lung noise's
    -> the two noise structures differ -> transfer FALSIFIED (a real result).
  - if they are COMPARABLE -> transfer SURVIVES this attempt -> weak
    corroboration only, NOT proof (the i74 tail is an upper bound: noise +
    unresolved structure + recon filter, not clean noise).

A white-noise reference is included as a scale anchor: white has a flat mean
radial power (slope ~ 0); anything markedly negative is correlated/low-pass.

Run:  .venv/Scripts/python -u i74_noise_transfer_falsify.py [--crop N]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))  # repo root (reconstruction, noise_floor)
from core.reconstruction import load_volume, DEFAULT_VOLUME_DIR
from core import noise_floor as nf

# high-frequency band (cycles/voxel) where i74 fibre SIGNAL should be gone and
# what remains is noise-dominated. Fibres at sigma~1-4 vox carry power below
# ~0.25 cyc/vox; we fit the top band well above that. Nyquist = 0.5.
FIT_BAND = (0.30, 0.48)
NBINS = 40


def radial_power_profile(block, nbins=NBINS):
    """Mean radial power spectrum P(k) vs frequency k in cycles/voxel, using the
    project's own windowed FFT (noise_floor.power_spectrum_3d). Returns
    (k_centres, mean_power) with empty shells dropped."""
    P, wmag, _wcomp = nf.power_spectrum_3d(block)
    k = (wmag / (2.0 * np.pi)).ravel()            # rad/vox -> cycles/vox
    Pf = P.ravel()
    edges = np.linspace(0.0, 0.5 * np.sqrt(3.0), nbins + 1)
    idx = np.digitize(k, edges) - 1
    cen = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([Pf[idx == i].mean() if np.any(idx == i) else np.nan
                     for i in range(nbins)])
    good = np.isfinite(prof) & (prof > 0)
    return cen[good], prof[good]


def band_slope(k, p, band=FIT_BAND):
    """Least-squares log-log slope of P(k) over the frequency band."""
    m = (k >= band[0]) & (k <= band[1])
    if m.sum() < 3:
        return np.nan, m
    s, _c = np.polyfit(np.log(k[m]), np.log(p[m]), 1)
    return float(s), m


def _cube(vol, side):
    c = [(s - side) // 2 for s in vol.shape]
    return vol[c[0]:c[0]+side, c[1]:c[1]+side, c[2]:c[2]+side].astype(np.float64)


def run(crop=128, save="i74_noise_transfer_falsify.png"):
    print("=" * 88)
    print(f"i74 NOISE-TRANSFER FALSIFICATION -- high-freq spectral tail vs lung noise "
          f"({DEFAULT_VOLUME_DIR.name})")
    print("=" * 88)

    # --- i74 tissue block (signal+noise; high-freq tail is noise-dominated) ---
    c0 = (768 - crop) // 2
    i74 = load_volume(crop=(c0, c0 + crop), dtype=np.float64)
    print(f"  i74 block {i74.shape}, mean={i74.mean():.1f}")

    # --- lung pure noise n=(A-B)/sqrt(2), matched cube ---
    A = np.load("lung_A.npy").astype(np.float64)
    B = np.load("lung_B.npy").astype(np.float64)
    n_lung = (A - B) / np.sqrt(2.0)
    side = min(crop, *n_lung.shape)
    lung = _cube(n_lung, side)
    print(f"  lung pure-noise cube {lung.shape}, std={lung.std():.3g}")

    # --- white-noise reference (scale anchor, slope ~ 0) ---
    rng = np.random.default_rng(0)
    white = rng.standard_normal((side, side, side))

    ki, pi = radial_power_profile(i74)
    kl, pl = radial_power_profile(lung)
    kw, pw = radial_power_profile(white)

    si, mi = band_slope(ki, pi)
    sl, ml = band_slope(kl, pl)
    sw, mw = band_slope(kw, pw)

    print(f"\n  high-frequency tail slope in band {FIT_BAND} cyc/vox (mean radial power P(k)):")
    print(f"    i74 tissue block : slope = {si:+.2f}")
    print(f"    lung pure noise  : slope = {sl:+.2f}   <-- the proxy actually used everywhere")
    print(f"    white reference  : slope = {sw:+.2f}   (anchor: ~0 = uncorrelated)")

    diff = abs(si - sl)
    print(f"\n  |i74 - lung| tail-slope difference = {diff:.2f}")
    # thresholds: the white<->lung gap is the natural scale of "a real difference
    # in noise correlation". Falsify if i74 differs from lung by more than that gap;
    # survive if it sits within ~half of it.
    lung_white_gap = abs(sl - sw)
    print(f"  (reference: lung-vs-white gap = {lung_white_gap:.2f} -- the scale of a "
          f"genuine correlation difference)")
    if diff > lung_white_gap:
        verdict = ("FALSIFIED: i74's high-freq noise tail differs from lung's by MORE than "
                   "the lung-vs-white gap -- the lung noise proxy does not describe i74's "
                   "noise structure. Real-noise validations built on lung noise do not "
                   "transfer to i74 and must be caveated as such.")
    elif diff < 0.5 * lung_white_gap:
        verdict = ("SURVIVES (weak): i74's high-freq tail is close to lung's, within half "
                   "the lung-vs-white gap. The transfer is NOT falsified -- but this is an "
                   "upper-bound tail (noise + unresolved structure + recon roll-off), so it "
                   "is weak corroboration, NOT proof the noise structures match.")
    else:
        verdict = ("INCONCLUSIVE: i74 sits between lung and white by an intermediate margin "
                   "-- neither cleanly falsified nor clearly consistent. Report as an open "
                   "limitation; half-maps would be needed to settle it.")
    print(f"\n  => {verdict}")
    print("=" * 88)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].loglog(ki, pi / pi.max(), "o-", color="tab:blue", label=f"i74 tissue (slope {si:+.2f})")
    ax[0].loglog(kl, pl / pl.max(), "s-", color="tab:red", label=f"lung noise (slope {sl:+.2f})")
    ax[0].loglog(kw, pw / pw.max(), "^-", color="0.5", label=f"white (slope {sw:+.2f})")
    ax[0].axvspan(FIT_BAND[0], FIT_BAND[1], color="0.85", alpha=0.5, label="fit band")
    ax[0].set_xlabel("frequency (cycles/voxel)"); ax[0].set_ylabel("mean radial power (peak-norm)")
    ax[0].set_title("Radial power spectra (log-log)")
    ax[0].legend(fontsize=8)

    labels = ["i74\ntissue", "lung\nnoise", "white\nref"]
    slopes = [si, sl, sw]
    colors = ["tab:blue", "tab:red", "0.5"]
    ax[1].bar(labels, slopes, color=colors)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel(f"high-freq tail slope, band {FIT_BAND} cyc/vox")
    ax[1].set_title(f"|i74 - lung| = {diff:.2f}  vs  lung-white gap = {lung_white_gap:.2f}")
    fig.suptitle("i74 noise-transfer falsification: does lung noise describe i74's high-freq tail?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save, dpi=130)
    print(f"Saved {save}")
    return dict(slope_i74=si, slope_lung=sl, slope_white=sw, diff=diff, gap=lung_white_gap)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=128)
    args = ap.parse_args()
    run(crop=args.crop)
