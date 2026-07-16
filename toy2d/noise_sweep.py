"""
noise_sweep.py -- Layer 4

Re-run the Layer 3 pipeline on the crossing phantom at increasing noise
levels, using the criterion validated in Layer 3: Lindeberg gamma-normalised
response picks sigma, rho-stability plateau picks rho (given that sigma).
The naive/decoupled orthogonality residual was found NOT to track truth on
this phantom (see unsupervised.py docstring) and is not used here.

Question: as noise grows, does the (sigma, rho) that supervised error (and
the unsupervised pair) picks move toward MORE smoothing (larger sigma
and/or rho)? That would demonstrate there is no single fixed-forever
"correct" parameter setting -- the optimum is a function of the data.
"""

import numpy as np
from phantom import make_phantom
from supervised import sweep_angular_error, mean_angular_error, DEFAULT_SIGMAS, DEFAULT_RHOS
from unsupervised import lindeberg_response, sweep_rho_stability

THETA1, THETA2 = 20.0, 110.0
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.4]
SIGMAS_1D = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])


def run_one_noise_level(noise_sigma, seed=0):
    image, true_angles = make_phantom(shape=(200, 200), theta_deg=THETA1,
                                       spacing=12.0, width=1.2,
                                       theta2_deg=THETA2, spacing2=17.0, width2=1.2,
                                       noise_sigma=noise_sigma, seed=seed)

    A, bestA = sweep_angular_error(image, true_angles, DEFAULT_SIGMAS, DEFAULT_RHOS, verbose=False)

    response = lindeberg_response(image, SIGMAS_1D, rho=0.5)
    sigma_peak = float(SIGMAS_1D[int(np.argmax(response))])

    Sens, _bestSens, rho_mid = sweep_rho_stability(image, DEFAULT_SIGMAS, DEFAULT_RHOS, verbose=False)
    i_row = int(np.argmin(np.abs(DEFAULT_SIGMAS - sigma_peak)))
    rho_combined = float(rho_mid[int(np.nanargmin(Sens[i_row]))])

    err_combined = mean_angular_error(image, sigma_peak, rho_combined, true_angles)

    return dict(noise=noise_sigma,
                sigma_true=bestA["sigma"], rho_true=bestA["rho"], err_true=bestA["value"],
                sigma_comb=sigma_peak, rho_comb=rho_combined, err_comb=err_combined)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [run_one_noise_level(n) for n in NOISE_LEVELS]

    print(f"{'noise':>6} | {'sigma*':>7} {'rho*':>6} {'err*':>7} | "
          f"{'sigma_c':>7} {'rho_c':>6} {'err_c':>7}")
    print("-" * 66)
    for r in results:
        print(f"{r['noise']:6.2f} | {r['sigma_true']:7.3g} {r['rho_true']:6.3g} "
              f"{r['err_true']:7.3f} | {r['sigma_comb']:7.3g} {r['rho_comb']:6.3g} "
              f"{r['err_comb']:7.3f}")

    noises = [r["noise"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(noises, [r["sigma_true"] for r in results], "o-", label="supervised sigma*")
    axes[0].plot(noises, [r["sigma_comb"] for r in results], "s--", label="unsupervised (Lindeberg) sigma")
    axes[0].set_xlabel("noise_sigma")
    axes[0].set_ylabel("sigma")
    axes[0].set_title("Optimal sigma vs noise")
    axes[0].legend()

    axes[1].plot(noises, [r["rho_true"] for r in results], "o-", label="supervised rho*")
    axes[1].plot(noises, [r["rho_comb"] for r in results], "s--", label="unsupervised (stability) rho")
    axes[1].set_xlabel("noise_sigma")
    axes[1].set_ylabel("rho")
    axes[1].set_title("Optimal rho vs noise")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("layer4_noise_sweep.png", dpi=120)
    print("Saved layer4_noise_sweep.png")
