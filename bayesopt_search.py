"""
bayesopt_search.py

Sample-efficient, Gaussian-process-driven search for structure-tensor
parameters per region, replacing the exhaustive grids in
lindeberg_stability_3d.py (which needed ~50-90 expensive fibre_direction()/
structure_tensor_3d() calls per region).

The two sub-problems need DIFFERENT strategies, not the same "Bayesian
optimisation" recipe applied twice:

  - SIGMA (Lindeberg response) -- we want the actual peak. Classic Bayesian
    optimisation: a GP surrogate over sigma, refined by picking the next
    sigma that maximises Expected Improvement. This homes in on an
    extremum, which is exactly what we want here.

  - RHO (stability sensitivity) -- we do NOT just want "the lowest value":
    the scan in the previous step showed sensitivity is often trivially
    lowest at the largest rho tried (heavier smoothing is "more stable"
    almost by definition), so an argmin-seeking search just runs to
    whatever edge you give it -- no better than the exhaustive grid did.
    What we actually want is the CURVE'S SHAPE: is there a genuine interior
    bump (a real crossing/complexity signature), or is it monotonic? That's
    an active-learning problem, not an optimisation problem -- so rho uses
    GP uncertainty sampling (query where the surrogate is least certain) to
    map the curve's shape with fewer, adaptively-placed points instead of a
    uniform grid.

Minimal from-scratch GP (RBF kernel, numpy/scipy only -- no extra
dependency): inputs are 1D and budgets are ~10-15 points, so a full library
(GPy, scikit-optimize, ...) would be overkill.

Sensitivity here is redefined slightly from toy2d/unsupervised.py and
lindeberg_stability_3d.py's grid version: instead of finite-differencing
between two ADJACENT GRID rhos (which reuses e3 fields across transitions
but only exists on a pre-committed grid), it's a function of a SINGLE rho:
    sensitivity(rho) = angular change per unit log(rho) between e3(rho) and
                        e3(rho * step_factor)
computed fresh at query time (2 fibre_direction() calls per query) so it
can be queried at any continuous rho, which is what active learning needs.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

from structure_tensor import structure_tensor_3d, eig_special_3d
from orthogonality_residual import fibre_direction, crop_margin, _crop3


# ---------------------------------------------------------------------------
# Minimal 1D Gaussian process (RBF kernel)
# ---------------------------------------------------------------------------

class GP1D:
    def __init__(self, noise=1e-6):
        self.noise = noise
        self.lengthscale = 1.0
        self.variance = 1.0

    def _kernel(self, a, b):
        a = np.asarray(a, dtype=float).reshape(-1, 1)
        b = np.asarray(b, dtype=float).reshape(1, -1)
        sqdist = (a - b) ** 2
        return self.variance * np.exp(-0.5 * sqdist / self.lengthscale ** 2)

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self._fit_hyperparams()
        K = self._kernel(self.X, self.X) + self.noise * np.eye(len(self.X))
        self._chol = cho_factor(K)
        self._alpha = cho_solve(self._chol, self.y)

    def _fit_hyperparams(self):
        """Maximise the log marginal likelihood over (lengthscale, variance)
        -- crude but fine for the ~10-point regime used here."""
        y = self.y
        x_range = float(self.X.max() - self.X.min()) if len(self.X) > 1 else 1.0

        def neg_log_marg_lik(log_params):
            ls, var = np.exp(log_params)
            self.lengthscale, self.variance = ls, var
            K = self._kernel(self.X, self.X) + self.noise * np.eye(len(self.X))
            try:
                c = cho_factor(K)
            except np.linalg.LinAlgError:
                return 1e10
            alpha = cho_solve(c, y)
            log_det = 2 * np.sum(np.log(np.diag(c[0])))
            return float(0.5 * y @ alpha + 0.5 * log_det)

        x0 = np.log([max(x_range / 4, 1e-3), max(np.var(y), 1e-6)])
        res = minimize(neg_log_marg_lik, x0, method="Nelder-Mead")
        self.lengthscale, self.variance = np.exp(res.x)

    def predict(self, Xs):
        Xs = np.asarray(Xs, dtype=float)
        Ks = self._kernel(self.X, Xs)
        Kss_diag = self.variance * np.ones(Xs.shape[0])
        mu = Ks.T @ self._alpha
        v = cho_solve(self._chol, Ks)
        var = np.clip(Kss_diag - np.einsum("ij,ij->j", Ks, v), 1e-12, None)
        return mu, var


# ---------------------------------------------------------------------------
# Sigma: classic BO (Expected Improvement) on the Lindeberg response
# ---------------------------------------------------------------------------

def lindeberg_response_at(volume, sigma, rho_ref=1.0, truncate=4.0):
    """sigma^2 * mean(sum of two largest structure-tensor eigenvalues), a
    single scalar evaluation (one structure_tensor_3d + eig call)."""
    S = structure_tensor_3d(volume.astype(np.float32), sigma, rho_ref, truncate=truncate)
    val, _vec = eig_special_3d(S)          # default order='desc'
    m = crop_margin(sigma, rho_ref, truncate)
    plane_energy = _crop3(val[0] + val[1], m)
    return float(sigma ** 2 * np.mean(plane_energy))


def _first_local_max_bracket(X, y):
    """Index i of the first interior local max (y[i-1] < y[i] > y[i+1])
    among points sorted by X, plus its bracketing (X[i-1], X[i+1]), or None
    if the sequence has no interior local max (e.g. monotonic increasing)."""
    order = np.argsort(X)
    Xs, ys = np.asarray(X)[order], np.asarray(y)[order]
    for i in range(1, len(ys) - 1):
        if ys[i - 1] < ys[i] > ys[i + 1]:
            return float(Xs[i - 1]), float(Xs[i + 1])
    return None


def bayesopt_sigma_peak(volume, bounds, rho_ref=1.0, n_init=7, n_iter=6,
                         truncate=4.0, verbose=True):
    """GP + Expected-Improvement search for the sigma maximising the
    Lindeberg response, n_init + n_iter total expensive evaluations.

    Real (non-phantom) volumes can have non-fibre, large-scale content
    (shading, curvature) whose energy keeps growing with sigma with no
    interior peak -- plain global-EI BO then does exactly what it's
    designed to do and walks to the domain boundary, reproducing the same
    failure the naive global argmax had in lindeberg_stability_3d.py. So:
    log-spaced initial points (denser at small sigma, where a genuine
    fibre-scale peak is expected), and if THOSE already show an interior
    local max, all subsequent refinement is bracketed around it (zoom in)
    rather than left free to chase the global EI back to the boundary. Only
    falls back to unrestricted global search if no local max shows up in
    the initial batch, with an explicit warning (that result should be
    treated as "hit the non-fibre boundary", not a real peak).

    Returns (sigma_best, response_best, (X, y)).
    """
    lo, hi = bounds
    X = list(np.exp(np.linspace(np.log(lo), np.log(hi), n_init)))   # log-spaced
    y = [lindeberg_response_at(volume, x, rho_ref, truncate) for x in X]
    if verbose:
        for x, yy in zip(X, y):
            print(f"  [init] sigma={x:.3g} -> response={yy:.4g}")

    bracket = _first_local_max_bracket(X, y)
    if bracket is None:
        if verbose:
            print("  WARNING: no interior local max in initial samples -- "
                  "response looks monotonic over this range. Falling back "
                  "to unrestricted global search; treat the result as "
                  "'hit the boundary', not a genuine fibre-scale peak.")
        search_lo, search_hi = lo, hi
    else:
        search_lo, search_hi = bracket
        if verbose:
            print(f"  local max bracketed in [{search_lo:.3g}, {search_hi:.3g}] "
                  f"-- refining within this bracket only")

    gp = GP1D()
    candidates = np.linspace(search_lo, search_hi, 300)
    for it in range(n_iter):
        gp.fit(X, y)
        mu, var = gp.predict(candidates)
        sd = np.sqrt(var)
        best_y = max(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu - best_y) / sd
            ei = (mu - best_y) * norm.cdf(z) + sd * norm.pdf(z)
        ei[sd < 1e-9] = 0.0
        x_next = float(candidates[int(np.argmax(ei))])
        y_next = lindeberg_response_at(volume, x_next, rho_ref, truncate)
        X.append(x_next)
        y.append(y_next)
        if verbose:
            print(f"  [BO {it + 1}/{n_iter}] sigma={x_next:.3g} -> response={y_next:.4g}")

    X, y = np.array(X), np.array(y)
    # Select "best" ONLY among evaluations inside the search region actually
    # used (the bracket, if one was found) -- otherwise the very first
    # boundary-hijacked point from the initial scan can silently win the
    # argmax even after correctly restricting the refinement phase.
    in_region = (X >= search_lo) & (X <= search_hi)
    i_best = int(np.argmax(np.where(in_region, y, -np.inf)))
    if verbose:
        print(f"[sigma-BO] best sigma={X[i_best]:.3g}, response={y[i_best]:.4g} "
              f"({len(X)} evaluations total)")
    return X[i_best], y[i_best], (X, y)


# ---------------------------------------------------------------------------
# Rho: active learning (uncertainty sampling) to map the sensitivity curve
# ---------------------------------------------------------------------------

def rho_sensitivity_at(volume, rho, sigma, step_factor=1.15, truncate=4.0):
    """Antipodally-aware angular change in e3 per unit log(rho), estimated
    at a single rho via a small fixed multiplicative step (2 fibre_direction
    calls). Lower = e3 more stable w.r.t. rho at this point = "smoother"."""
    rho2 = rho * step_factor
    e3_a = fibre_direction(volume, sigma, rho, truncate)
    e3_b = fibre_direction(volume, sigma, rho2, truncate)
    valid = np.isfinite(e3_a).all(axis=0) & np.isfinite(e3_b).all(axis=0)

    m = crop_margin(sigma, max(rho, rho2), truncate)
    e3_a_c = np.stack([_crop3(e3_a[c], m) for c in range(3)])
    e3_b_c = np.stack([_crop3(e3_b[c], m) for c in range(3)])
    v = _crop3(valid, m)
    if not v.any():
        return float("nan")

    cos = np.clip(np.abs(np.sum(e3_a_c * e3_b_c, axis=0)), 0.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    dlogrho = np.log(step_factor)
    return float(np.mean(ang[v]) / dlogrho)


def active_learn_rho_curve(volume, sigma, bounds, n_init=4, n_iter=6,
                            step_factor=1.15, truncate=4.0, verbose=True):
    """GP uncertainty-sampling search that maps the shape of the rho
    sensitivity curve (at fixed sigma) with adaptively-placed points,
    instead of a uniform grid. Queries where the GP is LEAST certain, not
    where the predicted value is lowest -- the goal is characterising the
    curve's shape (monotonic vs. a genuine interior bump), not chasing a
    trivial argmin at the range's edge.

    Returns (X, y, gp) -- the query history and the fitted GP (so the
    caller can inspect the posterior mean/variance over a fine grid).
    """
    lo, hi = bounds
    log_lo, log_hi = np.log(lo), np.log(hi)
    X = list(np.exp(np.linspace(log_lo, log_hi, n_init)))   # log-spaced init
    y = [rho_sensitivity_at(volume, x, sigma, step_factor, truncate) for x in X]
    if verbose:
        for x, yy in zip(X, y):
            print(f"  [init] rho={x:.3g} -> sensitivity={yy:.4g}")

    gp = GP1D()
    candidates = np.exp(np.linspace(log_lo, log_hi, 300))
    for it in range(n_iter):
        gp.fit(np.log(X), y)          # model in log(rho) -- matches the
        _mu, var = gp.predict(np.log(candidates))   # "per e-fold" convention
        x_next = float(candidates[int(np.argmax(var))])
        y_next = rho_sensitivity_at(volume, x_next, sigma, step_factor, truncate)
        X.append(x_next)
        y.append(y_next)
        if verbose:
            print(f"  [AL {it + 1}/{n_iter}] rho={x_next:.3g} -> sensitivity={y_next:.4g}")

    order = np.argsort(X)
    X, y = np.array(X)[order], np.array(y)[order]
    gp.fit(np.log(X), y)
    if verbose:
        print(f"[rho-AL] {len(X)} evaluations total")
    return X, y, gp


def curve_shape(X, y, gp, n_check=200, rel_threshold=0.15, edge_frac=0.05):
    """Classify the GP posterior mean over [X.min(), X.max()] as 'monotonic'
    or 'non-monotonic' (has an interior local max, i.e. a genuine bump --
    the crossing/complexity signature from the earlier grid scan).

    A plain sign-of-derivative check is too sensitive: crop_margin(rho)
    changes in discrete integer steps as rho grows continuously, so the set
    of voxels averaged over jumps slightly at certain rho thresholds --
    real, deterministic, but has nothing to do with fibre crossings, and it
    produces small (~1-3% of the curve's range) wiggles even on genuinely
    monotonic regions (confirmed on C_deepstriated, whose exhaustive-grid
    result was cleanly monotonic). So a rise only counts if its size is at
    least `rel_threshold` of the curve's overall (max-min) range, and rises
    within the last `edge_frac` of the domain are ignored (GP extrapolation
    is least constrained right at the boundary).
    """
    grid = np.exp(np.linspace(np.log(X.min()), np.log(X.max()), n_check))
    mu, _ = gp.predict(np.log(grid))
    span = float(mu.max() - mu.min())
    edge_cut = int(n_check * (1 - edge_frac))

    interior_rise = False
    i = 0
    while i < edge_cut - 1:
        if mu[i + 1] > mu[i]:
            j = i
            while j + 1 < edge_cut and mu[j + 1] >= mu[j]:
                j += 1
            if span > 0 and (mu[j] - mu[i]) / span >= rel_threshold:
                interior_rise = True
                break
            i = j + 1
        else:
            i += 1
    return "non-monotonic (bump found)" if interior_rise else "monotonic", grid, mu


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile
    from pathlib import Path

    folder = Path(r"C:\Users\issyh\Downloads\sensitivity_chunk_IC_768\sensitivity_chunk_IC_768")
    files = sorted(folder.glob("*.tif"))

    def load_box(z0, z1, r0, r1, c0, c1, dtype=np.float32):
        return np.stack([tifffile.imread(f)[r0:r1, c0:c1] for f in files[z0:z1]]).astype(dtype)

    # Same three regions as the earlier blind scan: a coherent-looking
    # region, the curved-fold region, and a deeper striated region -- reuse
    # them so results are directly comparable to the exhaustive-grid scan.
    REGIONS = {
        "A_coherent": (60, 180, 40, 160, 40, 160),
        "B_fold": (60, 180, 350, 470, 150, 270),
        "C_deepstriated": (440, 560, 600, 720, 30, 150),   # known-monotonic negative control
    }

    SIGMA_BOUNDS = (0.5, 10.0)
    RHO_BOUNDS = (1.0, 18.0)

    fig, axes = plt.subplots(2, len(REGIONS), figsize=(6 * len(REGIONS), 9))
    for col, (name, box) in enumerate(REGIONS.items()):
        print(f"\n=== {name} ===")
        volume = load_box(*box)

        print("-- sigma (Bayesian optimisation) --")
        sigma_best, resp_best, (sX, sy) = bayesopt_sigma_peak(
            volume, SIGMA_BOUNDS, n_init=7, n_iter=5)

        print("-- rho (active learning) --")
        rX, ry, gp = active_learn_rho_curve(
            volume, sigma_best, RHO_BOUNDS, n_init=4, n_iter=6)
        shape, grid, mu = curve_shape(rX, ry, gp)
        print(f"[{name}] sigma={sigma_best:.3g}, rho curve shape: {shape}, "
              f"total evaluations: sigma={len(sX)}, rho={len(rX)}")

        ax0, ax1 = axes[0][col], axes[1][col]
        ax0.plot(sX, sy, "o", ms=8)
        order = np.argsort(sX)
        ax0.plot(sX[order], sy[order], "--", alpha=0.4)
        ax0.axvline(sigma_best, color="r", ls="--", label=f"best={sigma_best:.2g}")
        ax0.set_title(f"{name}: sigma BO ({len(sX)} evals)")
        ax0.set_xlabel("sigma"); ax0.set_ylabel("Lindeberg response")
        ax0.legend()

        ax1.plot(rX, ry, "o", ms=8, label="queries")
        ax1.plot(grid, mu, "-", alpha=0.7, label="GP mean")
        ax1.set_xscale("log")
        ax1.set_title(f"{name}: rho active learning ({len(rX)} evals) -- {shape}")
        ax1.set_xlabel("rho"); ax1.set_ylabel("sensitivity (deg/e-fold)")
        ax1.legend()

    fig.tight_layout()
    fig.savefig("bayesopt_search_results.png", dpi=120)
    print("\nSaved bayesopt_search_results.png")
