"""
orthogonality_residual.py

Open-loop, ground-truth-free (sigma, rho) tuning metric for 3D structure-tensor
fibre extraction: the *gradient-orthogonality residual*.

Idea
----
A fibre lies along the direction of LEAST intensity change, so at every voxel
the image gradient should be perpendicular to the estimated fibre direction
e3 (the eigenvector of the smallest structure-tensor eigenvalue). Define, over
the volume,

        R(sigma, rho) = sum( (grad L . e3)^2 ) / sum( |grad L|^2 )

with grad L the gradient of the volume smoothed at the SAME sigma the structure
tensor uses (derivative-of-Gaussian at scale sigma). Numerator and denominator
are accumulated as separate scalar sums over the whole (cropped) volume and only
THEN divided -- so strong-gradient edge voxels dominate, which is intended: they
carry the orientation information. R lies in [0, 1]; 0 means grad L is exactly
orthogonal to e3 everywhere (perfect fibre alignment). Lower is better.

Axis convention (the one trap in this file)
-------------------------------------------
Skielex `structure_tensor`:
  * `eig_special_3d(S)` returns `vec` = the eigenvector of the SMALLEST
    eigenvalue = e3, shape (3, *spatial). (Confirmed in st3d.py.)
  * The three components of `vec` are ordered [x, y, z] = [axis2, axis1, axis0]
    of the volume (st3d.py comment: "vec is [x1 y1 z1]"), i.e. the REVERSE of
    the volume's (z, y, x) indexing.
  * `structure_tensor_3d` forms the gradient with scipy order tuples
        Vx = order(0,0,1), Vy = order(0,1,0), Vz = order(1,0,0), mode='nearest'.
We therefore build grad L with the EXACT same tuples and stack it as [Vx,Vy,Vz]
so that grad[i] and e3[i] live in the same basis and grad . e3 is trap-free --
no manual z/y/x juggling anywhere.

Crossings caveat
----------------
At a genuine fibre crossing no single e3 can be orthogonal to the gradients of
all the crossing populations at once, so R is penalised there *by construction*.
Consequently the SPATIAL MAP of the per-voxel residual (orthogonality_residual_field)
is a diagnostic of crossing / disorganised regions, not purely a parameter error.
Keep that in mind when reading R over anisotropic tissue.

VALIDATION FINDING (read before trusting R)
--------------------------------------------
Measured against ground truth on a discriminating phantom (single fibre
population, spatially varying orientation, moderate noise -- make_fanning_phantom
below), R as defined here does NOT track the supervised angular error: the two
surfaces are anti-correlated and R's minimum sits in a degenerate corner
(rho -> small, sigma -> large). Mechanism: e3 is (approximately) the null
direction of the SAME sigma-scale gradient the numerator dots against, so at
small rho `grad . e3 -> 0` trivially and large sigma smooths the test gradient
-- both shrink R regardless of whether the orientation is correct. So R rewards
under-integration / over-smoothing, not accuracy.
=> Do NOT use this R's raw argmin to pick (sigma, rho) on real data yet.
   The fix direction is to DECOUPLE the test gradient from e3's scale (e.g.
   evaluate grad L at a fixed fine analysis scale independent of rho, or test e3
   against a held-out gradient). The module is structured so you can swap the R
   definition in one place -- see below.

The existing Gray-style make_phantom (in reconstruction.py) CANNOT reveal this:
its truth-defined region has a single constant orientation (so supervised error
is monotone in rho -- no interior optimum), and R aggregates over the whole
heterogeneous volume while the supervised score only covers the straight third.
Use make_fanning_phantom for a fair test.

Everything is vectorised over voxels; the sweep reuses the same primitives.
Swap in a different R by replacing `_dot_and_magsq` (or the two public R
functions) -- the sweep / validation layers don't care how R is computed.
"""

import numpy as np
from scipy import ndimage

from structure_tensor import structure_tensor_3d, eig_special_3d

from core.reconstruction import iter_progress, grid_cells


# ---------------------------------------------------------------------------
# Core primitives (shared by the scalar metric and the per-voxel field)
# ---------------------------------------------------------------------------

def fibre_direction(volume, sigma, rho, truncate=4.0):
    """e3 field, shape (3, *spatial). Components ordered [x,y,z]=[axis2,axis1,axis0].

    Thin wrapper over the same Skielex calls as run_structure_tensor: `vec` is
    already the smallest-eigenvalue eigenvector, so it IS e3 -- no reselection.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # Skielex normalises the eigenvector by a norm that is 0 at isotropic
        # voxels -> a benign divide-by-zero we mask downstream. Silence it here.
        S = structure_tensor_3d(volume.astype(np.float32), sigma, rho, truncate=truncate)
        _val, vec = eig_special_3d(S)
    return vec


def smoothed_gradient(volume, sigma, truncate=4.0):
    """Derivative-of-Gaussian gradient at scale sigma, shape (3, *spatial).

    Uses the identical scipy order tuples / boundary mode as structure_tensor_3d
    so component i aligns with fibre_direction()[i]. Stacked as [Vx, Vy, Vz].
    """
    vol = volume.astype(np.float32)
    Vx = ndimage.gaussian_filter(vol, sigma, order=(0, 0, 1), mode="nearest", truncate=truncate)
    Vy = ndimage.gaussian_filter(vol, sigma, order=(0, 1, 0), mode="nearest", truncate=truncate)
    Vz = ndimage.gaussian_filter(vol, sigma, order=(1, 0, 0), mode="nearest", truncate=truncate)
    return np.stack([Vx, Vy, Vz])


def crop_margin(sigma, rho, truncate=4.0):
    """Gray two-thirds kernel-radius rule: ceil( (2/3) * truncate * max(sigma, rho) )."""
    return int(np.ceil((2.0 / 3.0) * truncate * max(sigma, rho)))


def _crop3(arr3d, m):
    """Crop `m` voxels from every face of a 3D scalar array."""
    if m <= 0:
        return arr3d
    s = tuple(slice(m, dim - m) for dim in arr3d.shape)
    return arr3d[s]


def _dot_and_magsq(volume, sigma, rho, truncate):
    """Return (grad.e3, |grad|^2, valid) as three 3D fields (uncropped).

    Skielex returns a NON-FINITE e3 at isotropic voxels (three ~equal
    eigenvalues -> the eigenvector solve divides by a zero norm), which happens
    in flat / uniform-noise regions and at small rho. Those voxels carry no
    orientation, so we mask them out of every sum rather than let inf/nan poison
    the accumulation. `valid` is True where e3 is finite.
    """
    e3 = fibre_direction(volume, sigma, rho, truncate)   # (3, *spatial), unit
    grad = smoothed_gradient(volume, sigma, truncate)    # (3, *spatial)
    valid = np.isfinite(e3).all(axis=0)
    e3 = np.where(valid, e3, 0.0)                        # kill inf/nan before mult
    dot = np.sum(grad * e3, axis=0)                       # grad . e3
    magsq = np.sum(grad * grad, axis=0)                  # |grad|^2
    return dot, magsq, valid


# ---------------------------------------------------------------------------
# The metric (scalar) and its diagnostic per-voxel field
# ---------------------------------------------------------------------------

def orthogonality_residual(volume, sigma, rho, truncate=4.0):
    """R(sigma, rho) as a single float. Separate sums, THEN divide."""
    dot, magsq, valid = _dot_and_magsq(volume, sigma, rho, truncate)
    m = crop_margin(sigma, rho, truncate)
    v = _crop3(valid, m)
    dot = _crop3(dot, m)[v]
    magsq = _crop3(magsq, m)[v]
    num = np.sum(np.square(dot, dtype=np.float64), dtype=np.float64)
    den = np.sum(magsq, dtype=np.float64)
    return float(num / den) if den > 0 else float("nan")


def orthogonality_residual_field(volume, sigma, rho, truncate=4.0, eps=1e-12):
    """Per-voxel residual (grad.e3)^2 / |grad|^2, cropped. Diagnostic of crossings.

    NOTE: this is the per-voxel RATIO (each voxel in [0,1]); it is NOT what the
    scalar metric averages (that one is a ratio of sums). Use it as a spatial map
    to see WHERE orthogonality breaks down, not to recompute R.
    """
    dot, magsq, valid = _dot_and_magsq(volume, sigma, rho, truncate)
    field = np.square(dot) / (magsq + eps)
    field = np.where(valid, field, np.nan)              # undefined where e3 invalid
    return _crop3(field, crop_margin(sigma, rho, truncate))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_residual(volume, sigmas, rhos, truncate=4.0, verbose=True):
    """Grid-evaluate R over sigmas x rhos.

    Returns
    -------
    R : (len(sigmas), len(rhos)) array   R[i, j] = R(sigmas[i], rhos[j])
    best : dict with keys sigma, rho, i, j, value  (the argmin)
    """
    sigmas = np.asarray(sigmas, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    R = np.full((sigmas.size, rhos.size), np.nan)
    cells = grid_cells(sigmas.size, rhos.size)
    for i, j in iter_progress(cells, desc="residual sweep", enable=verbose):
        R[i, j] = orthogonality_residual(volume, sigmas[i], rhos[j], truncate)
    i, j = np.unravel_index(np.nanargmin(R), R.shape)
    best = dict(sigma=float(sigmas[i]), rho=float(rhos[j]), i=int(i), j=int(j),
                value=float(R[i, j]))
    if verbose:
        print(f"[residual]  argmin R = {best['value']:.4f} "
              f"at sigma={best['sigma']:.3g}, rho={best['rho']:.3g} "
              f"(grid cell [{i},{j}])")
    return R, best


# ---------------------------------------------------------------------------
# Supervised cross-check (phantom only) -- angular error vs known ground truth
# ---------------------------------------------------------------------------

def _truth_to_e3_basis(truth):
    """Reorder a (*spatial, 3) truth field [x,y,z]=[axis0,axis1,axis2] into the
    e3 component basis [axis2,axis1,axis0], returning (3, *spatial)."""
    return np.moveaxis(truth[..., ::-1], -1, 0)


def angular_error_field(volume, truth, sigma, rho, truncate=4.0):
    """Per-voxel angular error (degrees) between e3 and the ground-truth
    direction, masked to voxels where truth is defined. Sign-agnostic (e3 ~ -e3).

    Returns (ang, mask) both cropped 3D arrays; ang is NaN outside mask.
    """
    e3 = fibre_direction(volume, sigma, rho, truncate)   # (3,*), unit (or non-finite)
    t = _truth_to_e3_basis(truth)                        # (3,*), unit or NaN
    valid = np.isfinite(e3).all(axis=0) & np.isfinite(t).all(axis=0)
    e3 = np.where(valid, e3, 0.0)
    t = np.where(valid, t, 0.0)
    cos = np.clip(np.abs(np.sum(e3 * t, axis=0)), 0.0, 1.0)
    ang = np.degrees(np.arccos(cos))                    # (*spatial,), meaningful only in mask
    m = crop_margin(sigma, rho, truncate)
    return _crop3(ang, m), _crop3(valid, m)


def mean_angular_error(volume, truth, sigma, rho, truncate=4.0):
    """Mean angular error (deg) over defined + cropped voxels; NaN if none."""
    ang, mask = angular_error_field(volume, truth, sigma, rho, truncate)
    return float(np.mean(ang[mask])) if mask.any() else float("nan")


def sweep_angular_error(volume, truth, sigmas, rhos, truncate=4.0, verbose=True):
    """Grid-evaluate supervised mean angular error. Same layout as sweep_residual."""
    sigmas = np.asarray(sigmas, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    A = np.full((sigmas.size, rhos.size), np.nan)
    cells = grid_cells(sigmas.size, rhos.size)
    for i, j in iter_progress(cells, desc="angular  sweep", enable=verbose):
        A[i, j] = mean_angular_error(volume, truth, sigmas[i], rhos[j], truncate)
    i, j = np.unravel_index(np.nanargmin(A), A.shape)
    best = dict(sigma=float(sigmas[i]), rho=float(rhos[j]), i=int(i), j=int(j),
                value=float(A[i, j]))
    if verbose:
        print(f"[angular ]  argmin err = {best['value']:.3f} deg "
              f"at sigma={best['sigma']:.3g}, rho={best['rho']:.3g} "
              f"(grid cell [{i},{j}])")
    return A, best


# ---------------------------------------------------------------------------
# Validation script: does the unsupervised R minimum land at the supervised one?
# ---------------------------------------------------------------------------

def make_fanning_phantom(shape=(96, 96, 96), width=1.5, period=6.0,
                         fan_deg=240.0, noise_sigma=0.25, seed=0):
    """Single-population phantom whose fibre orientation FANS across the volume.

    Straight tubes are laid on a periodic lattice in the plane perpendicular to a
    local fibre direction d(x) = (0, sin th, cos th) that rotates by `fan_deg`
    over the x extent. Spatially varying orientation is what gives the (sigma,
    rho) landscape a genuine INTERIOR optimum (too-small sigma = noise-dominated,
    too-large rho = integrates across the fan), so the metric can actually be
    tested -- unlike the constant-orientation region of the Gray phantom.

    Ground truth is EXACT despite the curvature: the intensity is I = f(a, b)
    with a = x-x0 and b = (y-y0)cos th - (z-z0)sin th, so grad I lies in
    span{grad a, grad b}; the fibre direction is the remaining null direction
        grad a x grad b = (1,0,0) x (b_x, cos th, -sin th) = (0, sin th, cos th),
    and the th' (curvature) terms sit only in b_x, which the cross product drops.
    Hence truth = (0, sin th, cos th) with no small-angle approximation.

    Returns (volume, truth) with truth (Nx,Ny,Nz,3), unit and defined everywhere.
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = shape
    x, y, z = np.indices(shape).astype(float)
    theta = np.deg2rad(fan_deg) * (x / (Nx - 1))
    xc, yc, zc = x - Nx / 2, y - Ny / 2, z - Nz / 2

    a = xc                                      # perpendicular axis 1 (x-hat)
    b = yc * np.cos(theta) - zc * np.sin(theta)  # perpendicular axis 2 (rotates)
    da = ((a + period / 2) % period) - period / 2
    db = ((b + period / 2) % period) - period / 2
    vol = np.exp(-(da ** 2 + db ** 2) / (2 * width ** 2))
    vol = vol + noise_sigma * rng.standard_normal(shape)
    vol = (vol - vol.min()) / (np.ptp(vol) + 1e-12)

    truth = np.zeros(shape + (3,))
    truth[..., 1] = np.sin(theta)               # [x, y, z] = [axis0, axis1, axis2]
    truth[..., 2] = np.cos(theta)
    return vol, truth


def _fit_dark_region(M, sigmas, rhos, dark_pct=25.0):
    """Weighted least-squares line  sigma = m*rho + b  through the DARKER (lower-M,
    i.e. better) cells of a heatmap. For overlaying the 'good region' trend.

    'Darker' = cells at or below the dark_pct percentile of M (viridis maps low
    values dark). Cells are weighted by how good they are (lower M -> heavier),
    so the line follows the trough of the valley rather than its outline.

    Returns (m, b, rho_endpoints, sigma_endpoints, mask) for plotting, or None if
    the dark cells are too few or share a single rho (a vertical valley, which a
    sigma=f(rho) line cannot represent).
    """
    M = np.asarray(M, float)
    sigmas = np.asarray(sigmas, float)
    rhos = np.asarray(rhos, float)
    finite = np.isfinite(M)
    if finite.sum() < 3:
        return None
    thr = np.percentile(M[finite], dark_pct)
    mask = finite & (M <= thr)
    ii, jj = np.where(mask)                        # i indexes sigma, j indexes rho
    if ii.size < 2 or np.unique(rhos[jj]).size < 2:
        return None
    x = rhos[jj]                                   # rho on the x-axis
    y = sigmas[ii]                                 # sigma on the y-axis
    w = 1.0 / (M[mask] - M[finite].min() + 1e-6)   # better (lower M) -> heavier
    W = w.sum()
    xbar = np.sum(w * x) / W
    ybar = np.sum(w * y) / W
    denom = np.sum(w * (x - xbar) ** 2)
    if denom <= 0:
        return None
    m = np.sum(w * (x - xbar) * (y - ybar)) / denom
    b = ybar - m * xbar
    rl = np.array([rhos.min(), rhos.max()])
    return m, b, rl, m * rl + b, mask


def _log_norm(M):
    """A LogNorm spanning the finite POSITIVE cells of M (for heatmaps whose
    interesting detail lives near the low end). Returns None if a log scale is
    ill-defined (no positive cells, or all equal) so the caller falls back to
    linear."""
    from matplotlib.colors import LogNorm
    M = np.asarray(M, float)
    good = np.isfinite(M) & (M > 0)
    if good.sum() == 0:
        return None
    vmin, vmax = float(M[good].min()), float(M[good].max())
    if vmin == vmax:
        return None
    return LogNorm(vmin=vmin, vmax=vmax)


def _overlay_dark_fit(ax, M, sigmas, rhos, dark_pct=25.0):
    """Draw _fit_dark_region's line + the dark cells it used onto a heatmap axis,
    keeping the axis limits fixed to the data extent. Returns the fit tuple/None."""
    fit = _fit_dark_region(M, sigmas, rhos, dark_pct)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if fit is not None:
        m, b, rl, sl, mask = fit
        ii, jj = np.where(mask)
        ax.scatter(np.asarray(rhos)[jj], np.asarray(sigmas)[ii], s=18,
                   facecolors="none", edgecolors="white", linewidths=0.8, alpha=0.7)
        ax.plot(rl, sl, "w--", lw=2,
                label=f"dark-region fit  σ={m:+.2g}·ρ{b:+.2g}")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    return fit


def validate(phantom="fanning", shape=(96, 96, 96),
             sigmas=(0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0),
             rhos=(0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0),
             truncate=4.0, save_path="orthogonality_validation.png",
             show=False, **phantom_kw):
    """Phantom validation. Sweeps R and angular error on the same grid, prints
    both argmins, their separation, the surface correlation, and a verdict, and
    writes a side-by-side heatmap.

    phantom : "fanning" (discriminating, default) or "gray" (existing
              make_phantom -- kept for reference; cannot test rho, see the
              module 'VALIDATION FINDING' note).
    **phantom_kw is forwarded to the chosen generator.

    Returns a dict with R, A, both `best` records, correlation, and verdict.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = np.asarray(sigmas, float)
    rhos = np.asarray(rhos, float)

    if phantom == "fanning":
        vol, truth = make_fanning_phantom(shape=shape, **phantom_kw)
    elif phantom == "gray":
        from core.reconstruction import make_phantom      # existing generator
        vol, truth = make_phantom(shape=shape, **phantom_kw)
    else:
        raise ValueError("phantom must be 'fanning' or 'gray'")
    print(f"phantom={phantom} {vol.shape}  "
          f"truth-defined voxels={int(np.isfinite(truth).all(-1).sum())}")

    R, bestR = sweep_residual(vol, sigmas, rhos, truncate)
    A, bestA = sweep_angular_error(vol, truth, sigmas, rhos, truncate)

    # Separation between the two argmins + how the whole surfaces relate.
    d_param = float(np.hypot(bestR["sigma"] - bestA["sigma"],
                             bestR["rho"] - bestA["rho"]))
    d_cells = int(max(abs(bestR["i"] - bestA["i"]), abs(bestR["j"] - bestA["j"])))
    ok = np.isfinite(R) & np.isfinite(A)
    corr = float(np.corrcoef(R[ok], A[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    # Trustworthy requires BOTH: argmins land together AND the surfaces agree in
    # trend (positive correlation). Anti-correlation is the failure signature.
    trustworthy = (d_cells <= 1) and (corr > 0.3)

    print("-" * 70)
    print(f"unsupervised R  argmin -> sigma={bestR['sigma']:.3g}, rho={bestR['rho']:.3g}")
    print(f"supervised ang  argmin -> sigma={bestA['sigma']:.3g}, rho={bestA['rho']:.3g}")
    print(f"separation: {d_param:.3g} in (sigma,rho) units, {d_cells} grid cell(s)")
    print(f"surface correlation corr(R, angular_error) = {corr:+.3f} "
          f"(want strongly positive)")
    if trustworthy:
        print("VERDICT: R TRACKS angular error (argmins coincide + positive "
              "correlation). Metric looks trustworthy on this phantom.")
    else:
        print("VERDICT: R does NOT track angular error -> NOT trustworthy as-is. "
              "Do NOT use R's argmin to pick (sigma, rho) on real data. "
              "See the 'VALIDATION FINDING' note at the top of this module for "
              "the mechanism and the fix direction (decouple the test-gradient "
              "scale from e3).")
    print("-" * 70)

    # --- side-by-side heatmaps -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    extent = [rhos[0], rhos[-1], sigmas[-1], sigmas[0]]  # x=rho, y=sigma
    for ax, M, best, title, cbl in (
        (axes[0], R, bestR, "Unsupervised residual  R(sigma, rho)", "R (lower = better)"),
        (axes[1], A, bestA, "Supervised mean angular error (deg)", "degrees (lower = better)"),
    ):
        im = ax.imshow(M, origin="upper", aspect="auto", extent=extent,
                       cmap="viridis", norm=_log_norm(M))
        ax.set_xlabel("rho"); ax.set_ylabel("sigma"); ax.set_title(title)
        ax.plot(best["rho"], best["sigma"], "r*", ms=16, mec="white",
                label=f"argmin ({best['sigma']:.2g}, {best['rho']:.2g})")
        if M is R:                       # dark-region fit only on the residual panel
            _overlay_dark_fit(ax, M, sigmas, rhos)
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, label=cbl, fraction=0.046, pad=0.04)
    fig.suptitle(f"Gradient-orthogonality residual vs angular error  "
                 f"[{phantom} phantom]  corr={corr:+.2f}  "
                 + ("R TRACKS error" if trustworthy else "R does NOT track error"))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"heatmaps written to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return dict(R=R, A=A, bestR=bestR, bestA=bestA, sigmas=sigmas, rhos=rhos,
                d_param=d_param, d_cells=d_cells, corr=corr,
                trustworthy=trustworthy)


def sweep_residual_on_volume(volume, sigmas, rhos, truncate=4.0,
                             save_path="orthogonality_residual_real.png",
                             show=False, verbose=True):
    """Compute the unsupervised residual R over a (sigma, rho) grid on a REAL
    volume (no ground truth, so no angular-error cross-check) and write a heatmap.

    WARNING: as documented in this module's 'VALIDATION FINDING' note, R's raw
    argmin is NOT trustworthy for picking (sigma, rho) yet -- e3 and the test
    gradient share the sigma scale. Read R here as a diagnostic surface, not an
    optimiser, until that coupling is fixed.
    """
    sigmas = np.asarray(sigmas, float)
    rhos = np.asarray(rhos, float)
    R, best = sweep_residual(volume, sigmas, rhos, truncate, verbose=verbose)

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [rhos[0], rhos[-1], sigmas[-1], sigmas[0]]   # x=rho, y=sigma
    im = ax.imshow(R, origin="upper", aspect="auto", extent=extent,
                   cmap="viridis", norm=_log_norm(R))
    ax.set_xlabel("rho"); ax.set_ylabel("sigma")
    ax.set_title("Unsupervised residual R(sigma, rho)  [real volume, log color]")
    ax.plot(best["rho"], best["sigma"], "r*", ms=16, mec="white",
            label=f"argmin ({best['sigma']:.2g}, {best['rho']:.2g})")
    fit = _overlay_dark_fit(ax, R, sigmas, rhos)
    if fit is not None:
        print(f"[dark fit] best-region trend sigma = {fit[0]:+.3g}*rho {fit[1]:+.3g}")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="R (lower = better; see caveat)",
                 fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"residual heatmap written to {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return R, best


if __name__ == "__main__":
    from pathlib import Path

    # Synthetic anti-parallel-cylinders phantom (single stacked TIFF). No ground
    # truth field, so this is the R surface alone (no angular-error panel).
    PHANTOM_TIF = Path(
        r"C:\Users\issyh\OneDrive\Desktop\Stack_Synthetic_Data_Anti-Parallel_Cylinders_Random_Dia_512X512.tif"
    )

    # sigma/rho grid: 0..30 inclusive, step 4  -> 0, 4, 8, 12, 16, 20, 24, 28.
    SIGMAS = tuple(float(v) for v in np.arange(0.0, 30.0 + 1e-9, 4.0))
    RHOS = tuple(float(v) for v in np.arange(0.0, 30.0 + 1e-9, 4.0))

    if PHANTOM_TIF.exists():
        import tifffile

        print(f"\n=== synthetic phantom ({PHANTOM_TIF.name}): residual sweep ===")
        volume = tifffile.imread(str(PHANTOM_TIF)).astype(float)
        print(f"loaded volume {volume.shape}")
        sweep_residual_on_volume(volume, sigmas=SIGMAS, rhos=RHOS)
    else:
        print(f"\n[phantom skipped: {PHANTOM_TIF} not found]")
