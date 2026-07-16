"""
width_sense_check.py

Quick sense check, not a new layer: does the Layer 1 pipeline (angle
recovery) stay accurate, and does the Layer 3(a) Lindeberg criterion still
find the phantom's true line width(s), when TWO very different widths (2
and 10 px) are combined into ONE phantom (same angle, different spacing per
population -- see phantom.py's spacing2/width2), across several angles?

This is a harder test than checking each width alone: it asks whether the
Lindeberg response is genuinely BIMODAL (a real peak near each width) when
both scales coexist in the same image, not just whether it can find a
single dominant scale.
"""

import numpy as np
from phantom import make_phantom
from forward import recover_angle
from supervised import angular_distance_deg
from unsupervised import lindeberg_response

WIDTH_A, WIDTH_B = 2.0, 10.0
SPACING_A, SPACING_B = 6.0 * WIDTH_A, 6.0 * WIDTH_B
ANGLES = [0.0, 36.0, 72.0, 108.0, 144.0]
SHAPE = (200, 200)
SIGMAS_1D = np.linspace(0.5, 20.0, 18)


def local_maxima(sigmas, response):
    """Indices of interior local maxima (response[i-1] < response[i] > response[i+1])."""
    return [i for i in range(1, len(response) - 1)
            if response[i - 1] < response[i] > response[i + 1]]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    print(f"combined phantom: width_A={WIDTH_A} (spacing={SPACING_A:.0f}), "
          f"width_B={WIDTH_B} (spacing={SPACING_B:.0f})\n")
    print(f"{'theta':>6} | {'peaks found (sigma)':>22} | {'recovered @ each peak':>28} | {'true':>6}")

    for theta in ANGLES:
        image, true_angles = make_phantom(
            shape=SHAPE, theta_deg=theta, spacing=SPACING_A, width=WIDTH_A,
            theta2_deg=theta, spacing2=SPACING_B, width2=WIDTH_B, noise_sigma=0.0)
        true_theta = true_angles[0]   # both populations share this angle

        response = lindeberg_response(image, SIGMAS_1D, rho=0.5)
        peak_idx = local_maxima(SIGMAS_1D, response)
        peak_sigmas = [float(SIGMAS_1D[i]) for i in peak_idx]

        recovered_at_peaks = []
        for s in peak_sigmas:
            rec = recover_angle(image, s, 2.0 * s)
            err = angular_distance_deg(rec, true_theta)
            recovered_at_peaks.append((s, rec, err))

        peaks_str = ", ".join(f"{s:.2g}" for s in peak_sigmas) or "none"
        rec_str = ", ".join(f"{rec:.2f}(err {err:.3f})" for s, rec, err in recovered_at_peaks) or "-"
        print(f"{theta:6.1f} | {peaks_str:>22} | {rec_str:>28} | {true_theta:6.1f}")

        ax.plot(SIGMAS_1D, response, "o-", alpha=0.7, label=f"theta={theta:.0f}")

    ax.axvline(WIDTH_A, color="k", ls="--", lw=1, label=f"true width A={WIDTH_A:.0f}")
    ax.axvline(WIDTH_B, color="gray", ls="--", lw=1, label=f"true width B={WIDTH_B:.0f}")
    ax.set_xlabel("sigma")
    ax.set_ylabel("Lindeberg response")
    ax.set_title("Combined two-width phantom: Lindeberg response, all angles merged")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("width_sense_check.png", dpi=120)
    print("\nSaved width_sense_check.png")
