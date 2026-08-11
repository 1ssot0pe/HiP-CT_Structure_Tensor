"""
cone_objective.py -- candidate objectives for the cone search, and why each one
is a candidate.

`theta_validation` established that the objective Stage 3 descends on --
mse(TDI, D_obs) -- is ANTI-correlated with tract angular error across the one
transition that matters (0 rings -> 1 ring). The field is not at fault: e3 sits
0.88 deg from the true tangent on the curved phantom. So the fix belongs in the
objective, and this module is the set of things worth trying.

Every term here is a pure function of an already-tracked path set, so the sweep
tracks ONCE per (L, theta) and scores all of them on identical streamlines.
That is not only cheaper; it is the only way the comparison is fair, because
two objectives evaluated on separately-tracked path sets differ by the tracking
as well as by the objective.

The diagnosis
-------------
Why does the current data term prefer a tracker that cannot turn? D_obs is a
tensor invariant computed at the integration scale rho -- a SMOOTH field with
wings several voxels wide. A TDI is a line measure, near-singular on the fibre
core. Comparing them directly does not reward following the fibre; it rewards
reproducing a blurred profile, and a tracker whose tracts spray off the fibre
produces exactly that blur by accident. Turning concentrates density back onto
the core and makes the profile mismatch WORSE. The loss is not measuring what
it is believed to measure.

That diagnosis suggests two independent repairs, and they are separable, so
both are here and both get tested:

  SCALE MATCHING   smooth the TDI to the scale D_obs was measured at before
                   comparing. Then blur is not free -- both sides carry it --
                   and the residual is about WHERE the density is, not how
                   sharp it is.

  AN ALIGNMENT TERM  score the tracts against the field they are supposed to be
                   following. This is the term the loss never had. It is the
                   ground-truth-free analogue of the angular error: it uses e3,
                   which is independently measured, not the phantom's truth,
                   so it is available on real data where no truth exists.

Nothing here is asserted to work. `objective_search.py` ranks them by the only
thing that matters -- the TRUE angular error you end up with if you descend on
each one.
"""

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    sys.path.insert(0, _p)

import dense_tracker as DT                     # noqa: E402
from tract_density import rasterise            # noqa: E402
from weight_fit import design_matrix, fit_weights  # noqa: E402

EPS = 1e-12
L1_WEIGHT = 1e-3          # price per unit of tract weight; see weight_fit

# Alignment weights tried alongside a data term. NOT tuned against the phantom
# truth -- a weight chosen to make the answer come out right would make the
# validation circular. Three values an order of magnitude apart, reported as a
# sensitivity rather than a single fitted number.
ALIGN_WEIGHTS = (0.25, 1.0, 4.0)


class Context:
    """Everything the objectives need that does not depend on the tracks.

    Built once per volume. `rho` is the integration scale D_obs was measured at
    and is what SCALE MATCHING smooths the TDI by -- it is read from the
    certified scales, not chosen here, so the match is to the scale Stage 1
    actually used.
    """

    def __init__(self, vol, dense, D_obs, support, rho, step=DT.STEP,
                 l1=L1_WEIGHT, sigma=1.0):
        self.vol, self.dense, self.support = vol, dense, support
        self.shape, self.rho, self.step, self.l1 = vol.shape, float(rho), step, l1
        # Image gradient at the DERIVATIVE scale sigma, for `image_align`.
        # Deliberately NOT the integration scale rho: rho is the smoothing that
        # builds the structure tensor, and the whole point of that term is to
        # test the tracts against the greyscale WITHOUT the processing that
        # could have gone wrong. sigma is only "the scale at which a gradient
        # is meaningful", which any gradient needs.
        self.sigma = float(sigma)
        v = np.asarray(vol, float)
        self.grad = np.stack([
            gaussian_filter(v, self.sigma, order=(1, 0, 0)),   # d/dz
            gaussian_filter(v, self.sigma, order=(0, 1, 0)),   # d/dy
            gaussian_filter(v, self.sigma, order=(0, 0, 1)),   # d/dx
        ]).astype(np.float32)                                  # (3, Z, Y, X)
        self.d_obs = np.asarray(D_obs, float)[support]
        # D_obs smoothed the same way the TDI will be, so the scale-matched
        # comparison is symmetric. Smoothing only one side would trade one
        # mismatch for another.
        self.d_obs_smooth = gaussian_filter(np.asarray(D_obs, float),
                                            self.rho)[support]
        v1, v2 = DT._fields(dense)
        self.T1 = DT.dyadic(v1)
        self.T2 = None if v2 is None else DT.dyadic(v2)


# ---------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------

def _steps(paths):
    """(directions, midpoints) for every step of every tract.

    Midpoints rather than the leading endpoint: the step is a chord, and the
    field it should be compared against is the field it passes through, not the
    field at the vertex it happens to end on.
    """
    D, P = [], []
    for p in paths:
        p = np.asarray(p, float)
        if p.ndim != 2 or len(p) < 2:
            continue
        d = np.diff(p, axis=0)
        n = np.linalg.norm(d, axis=1)
        ok = n > EPS
        if not ok.any():
            continue
        D.append(d[ok] / n[ok, None])
        P.append((0.5 * (p[:-1] + p[1:]))[ok])
    if not D:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(D), np.concatenate(P)


def _align2_at(T, pts, dirs):
    """(d . v)^2 from the interpolated dyadic, one direction per point.

    Dyadic and not the director itself: e3's sign is arbitrary per voxel, so
    componentwise interpolation cancels v against -v wherever two neighbours
    disagree. See `dense_tracker.dyadic` -- 13.2% of adjacent voxel pairs are
    sign-opposed on the crossing phantom.
    """
    if not len(pts):
        return np.zeros(0)
    out = np.zeros(len(pts))
    for (i, j, w), comp in zip(DT._DYAD, T):
        out += w * DT._sample(comp, pts) * dirs[:, i] * dirs[:, j]
    return np.clip(out, 0.0, 1.0)


def alignment_error(paths, ctx):
    """1 - mean (d . v)^2 over every tract step. Zero when tracts follow e3.

    This is the term the objective never had, and it is the one quantity here
    that answers to ORIENTATION rather than to density. It needs no ground
    truth -- only the field the tracker was handed -- so unlike the angular
    error it is computable on real data.

    Where the field carries two directions the better-aligned one is taken, by
    the same `max` the tracker uses: a walker arriving along either fibre of a
    crossing is following the field, and penalising it for disagreeing with the
    other one would score a correct traversal as an error.
    """
    d, p = _steps(paths)
    if not len(d):
        return 1.0
    a2 = _align2_at(ctx.T1, p, d)
    if ctx.T2 is not None:
        a2 = np.maximum(a2, _align2_at(ctx.T2, p, d))
    return float(1.0 - a2.mean())


def image_align_error(paths, ctx):
    """(grad I . d)^2 / |grad I|^2 along the tracts. The GREYSCALE check.

    Along a fibre the intensity does not change, so a correct step satisfies
    grad I . d = 0. That is a statement about the raw image: no tensor, no rho,
    no eigendecomposition, no HOST.

    Why this is not circular where `alignment_error` is
    ---------------------------------------------------
    The tracker CHOOSES its direction from e3; this SCORES it against grad I.
    In `alignment_error` the field appears on both sides, so a tract that
    faithfully follows a WRONG e3 scores perfectly and the loss cannot object.
    Here it can: if e3 is wrong at a voxel, a tract obeying it has
    grad I . d != 0 and the term registers it.

    The honest limit: grad I and the structure tensor come from the same data,
    so this is not an independent MEASUREMENT. It is independent of the
    PROCESSING that could have gone wrong -- the rho smoothing, the
    eigendecomposition and its sign ambiguity, the eigen-degeneracy that makes
    e3 report the bisector at a crossing, and HOST's patch grid. Those are
    exactly the failure modes `alignment_error` is blind to.

    Weighted by |grad I|^2 through the shared denominator, which is what makes
    a magnitude threshold unnecessary: a flat voxel contributes nearly nothing
    to the numerator AND nearly nothing to the denominator, instead of
    contributing a ratio of noise to noise.

    At a genuine crossing grad I carries both fibres, so this is non-zero even
    for a correct tract. It therefore mildly penalises correct traversal and
    cannot be the only term -- the same second-order limit as everything else
    built from grad I grad I^T.
    """
    d, p = _steps(paths)
    if not len(d):
        return 1.0
    G = np.stack([DT._sample(ctx.grad[c], p) for c in range(3)], axis=-1)
    num = float((np.einsum("mc,mc->m", G, d) ** 2).sum())
    den = float((G ** 2).sum())
    return num / den if den > EPS else 1.0


def image_mse(paths, ctx):
    """MSE against the PREPROCESSED IMAGE, with a fitted intercept AND scale.

        min over a, b of   mean( (a + b * TDI - I)^2 )   ==   var(I) * (1 - r^2)

    The intercept is the whole point. Every other mse in this module fits only a
    scale, which makes it an UNCENTRED cosine -- and an uncentred cosine between
    two non-negative fields is dominated by their means, not their patterns. On
    i74 that inflated a real agreement of r^2 = 0.10% into a reported "explains
    18.9%". With `a` free, the image's mean is absorbed exactly and what is left
    is pure pattern disagreement, which is what the number should have been
    measuring all along.

    The identity above is exact and is asserted in the sweep's output as a check.

    NOTE this does not fix the DIRECTION of the loss, only its honesty: measured
    across the cone ladder, the minimum still sits at 0 rings, because the HipCT
    image is bright over ~98% of a tissue chunk and straight tracts cover more of
    it than turning ones. See the module docstring's standing result.
    """
    D = paths if isinstance(paths, np.ndarray) else rasterise(
        paths, ctx.shape, step=ctx.step)
    d = np.asarray(D, float)[ctx.support]
    I = np.asarray(ctx.vol, float)[ctx.support]
    if d.size == 0 or float(d @ d) <= EPS:
        return float(I.var())
    A = np.stack([np.ones_like(d), d], axis=1)
    coef, *_ = np.linalg.lstsq(A, I, rcond=None)
    return float(np.mean((A @ coef - I) ** 2))


def _mse_scalar(D, d_obs, support):
    """mse after the single best global scale. Removes the seed budget, and
    nothing else -- a TDI and a unit-mean target share no unit."""
    d = np.asarray(D, float)[support]
    dd = float(d @ d)
    s = float(d @ d_obs) / dd if dd > EPS else 0.0
    return float(np.mean((s * d - d_obs) ** 2))


def data_mse(paths, ctx):
    """The CURRENT objective: raw TDI against D_obs, one fitted scalar."""
    return _mse_scalar(rasterise(paths, ctx.shape, step=ctx.step),
                       ctx.d_obs, ctx.support)


def data_mse_scalematched(paths, ctx):
    """The same comparison with both sides at the scale D_obs was measured at.

    The repair for the diagnosis in the module docstring: a TDI is a line
    measure and D_obs is a rho-smoothed tensor invariant, so the raw residual is
    dominated by a sharpness mismatch that following the fibre makes worse.
    """
    D = gaussian_filter(rasterise(paths, ctx.shape, step=ctx.step), ctx.rho)
    return _mse_scalar(D, ctx.d_obs_smooth, ctx.support)


def data_mse_nnls(paths, ctx):
    """The repo's real data term: per-streamline non-negative weights, L1-priced.

    Included because `cone_descent` deliberately simplified it away, and a
    finding about the objective must not turn out to be a finding about that
    simplification. If the degenerate preference survives the full weight fit,
    the simplification was not the cause.
    """
    Phi = design_matrix(paths, ctx.shape, ctx.support, step=ctx.step)
    if Phi.shape[1] == 0:
        return float(np.mean(ctx.d_obs ** 2))
    w, _fit = fit_weights(Phi, ctx.d_obs, l1=ctx.l1)
    return float(np.mean((Phi @ w - ctx.d_obs) ** 2))


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def score_all(paths, ctx, align_weights=ALIGN_WEIGHTS):
    """Every candidate objective, on ONE set of tracks. Lower is better for all.

    The combinations are formed here from the already-computed pieces rather
    than recomputed, so `mse_scalar + w*align` is exactly its two parts and a
    difference between objectives cannot come from anything else.
    """
    out = {
        "mse_raw": data_mse(paths, ctx),
        "mse_scalematched": data_mse_scalematched(paths, ctx),
        "mse_nnls": data_mse_nnls(paths, ctx),
        "align": alignment_error(paths, ctx),
        "image_align": image_align_error(paths, ctx),
    }
    for w in align_weights:
        out[f"mse_raw+{w:g}align"] = out["mse_raw"] + w * out["align"]
        out[f"mse_scalematched+{w:g}align"] = (out["mse_scalematched"]
                                               + w * out["align"])
    # The two orientation terms together: consistency with the field, plus
    # agreement with the greyscale the field was derived from. Equal weight,
    # not tuned -- both are dimensionless and both live in [0, 1].
    out["align+image_align"] = out["align"] + out["image_align"]
    return out


def names(align_weights=ALIGN_WEIGHTS):
    base = ["mse_raw", "mse_scalematched", "mse_nnls", "align", "image_align"]
    for w in align_weights:
        base += [f"mse_raw+{w:g}align", f"mse_scalematched+{w:g}align"]
    return base + ["align+image_align"]


# What each objective is, in one line, for the report and the figure.
BLURB = {
    "mse_raw": "the current objective",
    "mse_scalematched": "TDI smoothed to rho before comparing",
    "mse_nnls": "current objective, full per-tract NNLS weights",
    "align": "tract vs e3 only -- no density term at all",
    "image_align": "tract vs the raw image gradient -- bypasses Stages 1-2",
    "align+image_align": "field consistency + greyscale validity",
}
