"""
half_map_analysis.py -- the analysis to run the MOMENT two half-reconstructions
of i74 arrive. Two independent reconstructions A, B of the SAME scan (e.g. odd
vs even projections) share the signal S but carry independent noise:

    A = S + n_A ,  B = S + n_B
    n = (A - B) / sqrt(2)   -> a PURE NOISE realisation (signal cancels)
    avg = (A + B) / 2       -> best signal estimate (noise reduced)

Noise bookkeeping. Each half uses half the projections, so A, B carry ~2x the
full-reconstruction noise variance. Hence:
    var(n)   = 2 * var_full            (n is at the half-map noise level)
    var(avg) =     var_full            (the averaged map ~ the full recon)
So N measured on `n` is the HALF-MAP floor; the full-recon-equivalent floor is
N/2, which is what we subtract from `avg`. Full-recon sigma_n = std(n)/sqrt(2).

Four tasks, each answering a question that a single volume could not:
  T1 NOISE      -- measure N(sigma) directly on n: trace, slope (-5=white vs
                   shallower=correlated), anisotropy + principal axis (ring
                   artifact points at the rotation axis?), power spectrum,
                   stationarity. [reuses noise_floor verbatim]
  T2 PREMISE    -- correct c_s on `avg` with the MEASURED N (not an upper bound):
                   does the pervasive high c_s collapse (=noise) or survive
                   (=real tissue)? [the definitive version of the noise thread]
  T3 CERTIFY    -- orientation FSC: structure-tensor field from A and from B,
                   correlated in Fourier shells -> finest scale at which the
                   ORIENTATION is reproducible. Sweep (sigma,rho); the argmax is a
                   VALID independent certifier (real signal split, not a guessed
                   noise level). Compare to the Lindeberg pick.
  T4 RESOLUTION -- plain FSC(A,B) -> the reconstruction's own resolution, to
                   confirm whether sigma~1 vox is genuinely at the resolution limit.

Run on real data:   python half_map_analysis.py --a PATH_A --b PATH_B
Validate the code:  python half_map_analysis.py --synthetic
(--synthetic builds a fanning phantom + two noise realisations, so every task has
a known answer -- the pipeline's own self-test / trust gate.)
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import noise_floor as nf
from resolution_sweep import fsc, resolution_frequency, _shell_index
from orthogonality_residual import fibre_direction, crop_margin

VOX_UM = 20.0
SIGMAS_N = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.7, 8.0])
SIG_GRID = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
RHO_GRID = np.array([1.0, 2.0, 3.0, 4.0])
EPS = 1e-12


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_any(path, dtype=np.float64):
    """Load a volume from a TIFF stack folder, a (multipage) .tif, .npy, or
    NIfTI. Returns a float64 3D array."""
    if os.path.isdir(path):
        import tifffile
        from pathlib import Path
        files = sorted(Path(path).glob("*.tif")) + sorted(Path(path).glob("*.tiff"))
        if not files:
            raise FileNotFoundError(f"no TIFF slices in {path}")
        return np.stack([tifffile.imread(f) for f in files]).astype(dtype)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path).astype(dtype)
    if ext == ".npy":
        return np.load(path).astype(dtype)
    if ext in (".nii", ".gz"):
        import nibabel as nib
        return np.asarray(nib.load(path).dataobj).astype(dtype)
    raise ValueError(f"unrecognised volume format: {path}")


def make_synthetic(shape=(96, 96, 96), noise_sigma=0.25, seed=0):
    """Fanning-phantom signal + two INDEPENDENT noise realisations -> (A, B,
    truth). Gives every task a known answer for the self-test."""
    from orthogonality_residual import make_fanning_phantom
    S, truth = make_fanning_phantom(shape=shape, noise_sigma=0.0)
    rng = np.random.default_rng(seed)
    A = S + noise_sigma * rng.standard_normal(shape)
    B = S + noise_sigma * rng.standard_normal(shape)
    return A.astype(np.float64), B.astype(np.float64), truth, noise_sigma


# ---------------------------------------------------------------------------
# T1 -- noise floor measured directly on n = (A-B)/sqrt(2)
# ---------------------------------------------------------------------------

def task1_noise(A, B, log, save="halfmap_T1_noise.png", truth_sigma_n=None):
    n = (A - B) / np.sqrt(2.0)
    sig_full = float(n.std() / np.sqrt(2.0))       # full-recon-equivalent sigma_n
    log(f"[T1] pure-noise n: std={n.std():.3g} (half-map level), "
        f"full-recon sigma_n~{sig_full:.3g}")
    trN, iso, csN, e1 = [], [], [], []
    for sg in SIGMAS_N:
        N = nf.N_matrix(n, sg); w = nf.westin(N)
        trN.append(np.trace(N)); iso.append(w["aniso_ratio"]); csN.append(w["cs"])
        e1.append(nf.principal_eigvec(N))
    trN = np.array(trN)
    slope_hi = nf.loglog_slope(SIGMAS_N[SIGMAS_N >= 2], trN[SIGMAS_N >= 2])
    fit = nf.fit_broken_power_law(SIGMAS_N, trN)
    kind = ("WHITE (slope ~ -5)" if abs(slope_hi + 5) < 1.0
            else f"CORRELATED (slope {slope_hi:.1f} != -5 -> low-frequency)")
    iso4 = iso[int(np.argmin(np.abs(SIGMAS_N - 4)))]
    aniso = "ANISOTROPIC" if iso4 > 1.2 else "~isotropic"
    log(f"[T1] high-sigma slope={slope_hi:.2f} -> {kind}; knee ell={fit['ell']:.2f} vox")
    log(f"[T1] N is {aniso} (lambda1/lambda3 @sigma=4 = {iso4:.2f}); "
        f"principal axis @sigma=4 = {np.round(e1[int(np.argmin(np.abs(SIGMAS_N-4)))],2)} "
        f"(check vs rotation axis for ring bias)")
    if truth_sigma_n is not None:
        ana4 = truth_sigma_n ** 2 * nf.C_WHITE * 4.0 ** -5
        log(f"[T1 self-test] injected white sigma_n={truth_sigma_n}: trN(4) "
            f"meas={trN[int(np.argmin(np.abs(SIGMAS_N-4)))]:.3e} vs analytic {ana4:.3e} "
            f"(ratio {trN[int(np.argmin(np.abs(SIGMAS_N-4)))]/ana4:.2f})")

    # power spectrum of n
    P, wmag, wcomp = nf.power_spectrum_3d(n)
    fr = np.fft.fftfreq(n.shape[0]); nb = 30
    edges = np.linspace(0, 0.5 * np.sqrt(3), nb + 1)
    idx = np.digitize(np.sqrt(sum(w ** 2 for w in np.meshgrid(*([2*np.pi*fr]*3), indexing="ij"))).ravel() / (2*np.pi), edges) - 1
    prof = np.array([P.ravel()[idx == i].mean() if np.any(idx == i) else np.nan for i in range(nb)])
    cen = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].loglog(SIGMAS_N, trN, "o-"); ax[0].loglog(SIGMAS_N, trN[2]*SIGMAS_N**-5, ":",
                 color="0.6", label="$\\sigma^{-5}$ (white)")
    ax[0].set_xlabel("$\\sigma$"); ax[0].set_ylabel("tr N (MEASURED)")
    ax[0].set_title(f"T1: real noise floor\nslope={slope_hi:.1f} -> {kind.split()[0]}")
    ax[0].legend(fontsize=8)
    ax[1].plot(SIGMAS_N, iso, "o-"); ax[1].axhline(1.05, color="0.6", ls=":")
    ax[1].set_xlabel("$\\sigma$"); ax[1].set_ylabel("$\\lambda_1/\\lambda_3$ of N")
    ax[1].set_title(f"T1: noise anisotropy ({aniso})")
    ax[2].loglog(cen, prof, "o-"); ax[2].set_xlabel("cycles/voxel")
    ax[2].set_ylabel("radial power of n"); ax[2].set_title("T1: real noise spectrum")
    fig.suptitle("T1: noise floor measured DIRECTLY from half-map difference")
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    log(f"[T1] saved {save}")
    # N(sigma) full 3x3 for reuse by T2 (full-recon-equivalent = /2)
    N_full = {float(sg): nf.N_matrix(n, sg) / 2.0 for sg in SIGMAS_N}
    return N_full


# ---------------------------------------------------------------------------
# T2 -- premise: correct c_s on avg with the MEASURED N
# ---------------------------------------------------------------------------

def task2_premise(A, B, N_full, log, sigma=4.0, rho=4.0, thr=0.15,
                  save="halfmap_T2_premise.png"):
    avg = (A + B) / 2.0
    N = N_full[min(N_full, key=lambda s: abs(s - sigma))]
    Su = nf.structure_tensor_field(avg, sigma, rho)
    csu, _ = nf.cs_field(Su)
    Sc = nf.structure_tensor_field(avg, sigma, rho, subtract_N=N)
    csc, clip = nf.cs_field(Sc, psd_clip=True)
    m = int(np.ceil(4 * (sigma + rho)))
    csu = nf.crop_border(csu, m).ravel(); csc = nf.crop_border(csc, m).ravel()
    fu, fc = float((csu > thr).mean()), float((csc > thr).mean())
    verdict = ("c_s COLLAPSES -> complexity was largely NOISE" if fc < 0.6 * fu
               else "c_s SURVIVES -> complexity is largely REAL tissue")
    log(f"[T2] c_s>thr on avg @sigma={sigma}: uncorrected {fu:.0%} -> "
        f"MEASURED-N corrected {fc:.0%} (clip {clip:.0%})  => {verdict}")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(csu, bins=60, range=(0, 1), alpha=0.6, color="tab:red", label=f"uncorrected ({fu:.0%}>thr)")
    ax.hist(csc, bins=60, range=(0, 1), alpha=0.6, color="tab:green", label=f"measured-N corrected ({fc:.0%}>thr)")
    ax.axvline(thr, color="k", ls=":"); ax.set_xlabel("c_s"); ax.set_ylabel("voxels")
    ax.legend(); ax.set_title(f"T2 PREMISE: {verdict}\n(clip {clip:.0%})")
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    log(f"[T2] saved {save}")
    return fu, fc


# ---------------------------------------------------------------------------
# T3 -- certify sigma,rho via orientation FSC between the two half-maps
# ---------------------------------------------------------------------------

def _tensor6(vol, sigma, rho, m):
    e3 = fibre_direction(vol, sigma, rho); e3 = np.where(np.isfinite(e3), e3, 0.0)
    comps = [e3[0]*e3[0], e3[1]*e3[1], e3[2]*e3[2],
             e3[0]*e3[1], e3[0]*e3[2], e3[1]*e3[2]]
    return [nf.crop_border(c, m) for c in comps]


def orientation_fsc(A, B, sigma, rho, nbins=30):
    """FSC cutoff of the sign-free structure-tensor field between A and B. Higher
    = orientation reproducible to finer scale = better (sig,rho)."""
    m = crop_margin(sigma, rho)
    Ta, Tb = _tensor6(A, sigma, rho, m), _tensor6(B, sigma, rho, m)
    num = da = db = None
    for ca, cb in zip(Ta, Tb):
        Fa = np.fft.fftn(ca).ravel(); Fb = np.fft.fftn(cb).ravel()
        idx, freqs = _shell_index(ca.shape, nbins)
        keep = (idx >= 0) & (idx < nbins); idx = idx[keep]
        n_ = np.bincount(idx, weights=np.real(Fa*np.conj(Fb))[keep], minlength=nbins)
        a_ = np.bincount(idx, weights=(np.abs(Fa)**2)[keep], minlength=nbins)
        b_ = np.bincount(idx, weights=(np.abs(Fb)**2)[keep], minlength=nbins)
        num = n_ if num is None else num + n_
        da = a_ if da is None else da + a_
        db = b_ if db is None else db + b_
    return resolution_frequency(freqs, num / (np.sqrt(da*db) + EPS), 0.143)


def task3_certify(A, B, log, truth=None, save="halfmap_T3_certify.png"):
    from reconstruction import iter_progress, grid_cells
    Q = np.full((SIG_GRID.size, RHO_GRID.size), np.nan)
    for i, j in iter_progress(grid_cells(SIG_GRID.size, RHO_GRID.size),
                              desc="orientation FSC", enable=True):
        Q[i, j] = orientation_fsc(A, B, SIG_GRID[i], RHO_GRID[j])
    i, j = np.unravel_index(np.nanargmax(Q), Q.shape)
    sig_c, rho_c = float(SIG_GRID[i]), float(RHO_GRID[j])
    log(f"[T3] orientation-FSC optimum: sigma={sig_c:.2g}, rho={rho_c:.2g} "
        f"(cutoff {Q[i,j]:.4f} cyc/vox)")
    # self-test: does it coincide with supervised angular error?
    passed = None
    if truth is not None:
        from resolution_sweep import sweep_angular_error
        Aerr, bA = sweep_angular_error((A + B) / 2.0, truth, SIG_GRID, RHO_GRID, verbose=False)
        d = int(max(abs(i - bA["i"]), abs(j - bA["j"])))
        ok = np.isfinite(Q) & np.isfinite(Aerr)
        corr = float(np.corrcoef(Q[ok], Aerr[ok])[0, 1])
        passed = (d <= 1) and (corr < -0.3)
        log(f"[T3 self-test] angular argmin sigma={bA['sigma']:.2g},rho={bA['rho']:.2g}; "
            f"sep={d} cell(s), corr={corr:+.2f} -> certifier {'VALID' if passed else 'INVALID'}")
    fig, ax = plt.subplots(figsize=(7, 5.4))
    extent = [RHO_GRID[0], RHO_GRID[-1], SIG_GRID[-1], SIG_GRID[0]]
    im = ax.imshow(Q, origin="upper", aspect="auto", extent=extent, cmap="viridis")
    ax.plot(rho_c, sig_c, "r*", ms=18, mec="white", label=f"opt ({sig_c:.1f},{rho_c:.1f})")
    ax.set_xlabel("$\\rho$"); ax.set_ylabel("$\\sigma$")
    ax.set_title("T3: orientation-FSC certifier (higher=reproducible finer)")
    ax.legend(fontsize=9); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    log(f"[T3] saved {save}")
    return sig_c, rho_c, passed


# ---------------------------------------------------------------------------
# T4 -- image resolution: plain FSC(A, B)
# ---------------------------------------------------------------------------

def task4_resolution(A, B, log, save="halfmap_T4_resolution.png"):
    freqs, curve = fsc(A, B, nbins=50)
    fcut = resolution_frequency(freqs, curve, 0.143)
    res_vox = 1.0 / fcut if fcut > 0 else np.inf
    log(f"[T4] image FSC cutoff = {fcut:.4f} cyc/vox -> resolution ~{res_vox:.2f} vox "
        f"(~{res_vox*VOX_UM:.0f} um)")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(freqs, curve, "o-"); ax.axhline(0.143, color="r", ls="--", label="0.143")
    ax.axvline(fcut, color="0.5", ls=":", label=f"cutoff {fcut:.3f}")
    ax.set_xlabel("cycles/voxel"); ax.set_ylabel("FSC(A,B)")
    ax.set_title(f"T4: reconstruction resolution ~{res_vox:.1f} vox (~{res_vox*VOX_UM:.0f} um)")
    ax.legend(); fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    log(f"[T4] saved {save}")
    return fcut, res_vox


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--crop", type=int, default=0, help="central NxNxN crop for speed")
    args = ap.parse_args()

    LOG = []
    def log(m): print(m); LOG.append(m)

    truth = truth_sigma = None
    if args.synthetic:
        log("=== SYNTHETIC SELF-TEST (phantom + 2 noise realisations) ===")
        A, B, truth, truth_sigma = make_synthetic()
    else:
        if not (args.a and args.b):
            ap.error("provide --a and --b (or --synthetic)")
        log(f"=== HALF-MAP ANALYSIS  A={args.a}  B={args.b} ===")
        A, B = load_any(args.a), load_any(args.b)
        if A.shape != B.shape:
            ap.error(f"shape mismatch {A.shape} vs {B.shape}")
    if args.crop and args.crop < min(A.shape):
        c = [(s - args.crop) // 2 for s in A.shape]; k = args.crop
        A = A[c[0]:c[0]+k, c[1]:c[1]+k, c[2]:c[2]+k]
        B = B[c[0]:c[0]+k, c[1]:c[1]+k, c[2]:c[2]+k]
    log(f"volume {A.shape}, A mean={A.mean():.3g} B mean={B.mean():.3g}")

    log("-" * 66)
    N_full = task1_noise(A, B, log, truth_sigma_n=truth_sigma)
    log("-" * 66); task2_premise(A, B, N_full, log)
    log("-" * 66); sig_c, rho_c, ok = task3_certify(A, B, log, truth=truth)
    log("-" * 66); task4_resolution(A, B, log)
    log("-" * 66)
    log(f"SUMMARY: certified sigma~{sig_c:.2g}, rho~{rho_c:.2g}"
        + (f" (certifier VALID on self-test)" if ok else
           " (certifier self-test: see T3)" if ok is not None else ""))
    open("HALFMAP_RESULTS.md", "w").write(
        "# Half-map analysis results\n\n```\n" + "\n".join(LOG) + "\n```\n")
    log("Wrote HALFMAP_RESULTS.md")


if __name__ == "__main__":
    main()
