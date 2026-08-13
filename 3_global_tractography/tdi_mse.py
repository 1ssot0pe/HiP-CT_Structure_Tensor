r"""
tdi_mse.py -- MSE between the PREPROCESSED IMAGE and a NORMALISED TDI.

The objective this loop descends on. Both sides are put on [0, 1] before they
are compared, so the residual is a plain mean square with no fitted scale and no
fitted intercept:

    image   clip to its own 1-99 percentiles, rescale onto [0, 1]   (preprocess.py)
    TDI     log10(1 + D), rescale onto [0, 1]
    loss    mean over the support of (TDI_norm - image)^2

Why each side is treated differently
------------------------------------
They are different KINDS of quantity and the same transform on both would be the
mistake. The image is a bounded greyscale whose extremes are outliers -- dust,
hot voxels, reconstruction spikes -- which is exactly what a percentile clip is
for. A TDI is a line measure: near-singular on a fibre core, exactly zero over
most of the volume, with a heavy right tail whose largest values are the SIGNAL
rather than outliers. Clipping that tail would flatten the tract cores, which are
the only voxels carrying information about where the tracts went; compressing it
with a log does not.

Why the normalisation is the point
----------------------------------
The previous run measured a raw MSE of 0.327 against a `mean(I^2)` of 0.314 --
the number was almost entirely the SCALE MISMATCH between a TDI whose mean is
~0.04 and an image whose mean is ~0.55, not a statement about where the tracts
were. `cone_objective.image_mse` handles that by fitting an intercept and a
scale, which is honest but hands the loss two free parameters that absorb any
change in the tracts' overall mass. Fixing both sides onto [0, 1] up front
removes the mismatch WITHOUT the free parameters, so a change in the loss is a
change in the tracts.

What this does NOT fix, and the number that says so
---------------------------------------------------
Normalisation cannot make a sparse field dense. The image is bright over most of
a tissue chunk; a TDI is zero wherever no tract went, and log10(1 + 0) = 0, so
the zeros stay zero however the non-zeros are rescaled. The residual is
therefore still liable to be dominated by that sparsity offset rather than by
pattern agreement, which is the standing result recorded in `cone_objective` --
the minimum sits at the degenerate no-turn cone because straight tracts cover
more bright voxels than turning ones do.

`decompose` exists to make that visible instead of arguable. It splits

    MSE = (mean_TDI - mean_I)^2  +  var(TDI) + var(I) - 2 cov(TDI, I)
          \_______ offset ______/    \________ pattern _____________/

so the report always says how much of the loss is the sparsity offset and how
much is genuine disagreement about WHERE the density is. If `offset_frac` comes
back near 1, the descent is minimising coverage, not tracking quality, and the
loss should not be believed no matter which direction it moved.

Frame: `support` is a boolean (Z, Y, X) mask -- the tissue mask, so the air
around the sample cannot dilute the residual with voxels where both sides are
trivially zero and agree for no reason.
"""

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    sys.path.insert(0, _p)

import smooth_tracker as ST                      # noqa: E402
from preprocess import preprocess_volume         # noqa: E402

EPS = 1e-12
CLIP_PCT = (1.0, 99.0)
N_KEEP = 0.5          # survival below this is a decayed tail, not a tract


# ---------------------------------------------------------------------------
# the two sides
# ---------------------------------------------------------------------------

def prepare_image(vol, clip_pct=CLIP_PCT, mask_air=True):
    """The image side: clipped to [p_lo, p_hi] and rescaled onto [0, 1].

    `reconstruction.load_volume` already does this (preprocess defaults ON), so
    the usual case is that the volume arrives preprocessed and this is a no-op.
    It is checked rather than assumed, because the transform is NOT idempotent:
    applying it twice re-clips an already-[0, 1] volume at its OWN 1-99
    percentiles and stretches the survivors back out, which silently changes the
    data and leaves the range looking exactly as correct as before.

    Returns (image_01, was_already_preprocessed).
    """
    v = np.asarray(vol, float)
    lo, hi = float(v.min()), float(v.max())
    if lo >= -EPS and hi <= 1.0 + EPS:
        return v, True
    return preprocess_volume(v, clip_pct=clip_pct, mask_air=mask_air,
                             normalise=True), False


def normalise_tdi(tdi, support=None, smooth=0.0):
    """The TDI side: log10(1 + D), then rescaled onto [0, 1].

    log10(1 + D) rather than log10(D): a TDI is exactly zero over most of the
    volume and log10(0) is -inf. The 1 + also fixes the origin, so an empty voxel
    maps to 0 and stays there -- "no tract here" must not become a finite value
    that the residual can be paid for.

    The rescale is a min-max over the SUPPORT, not over the whole array. A
    tractogram that escapes into the air would otherwise set the maximum from a
    voxel the loss never looks at, and every value inside the tissue would be
    divided by it.

    `smooth` is a Gaussian applied BEFORE the log, in voxels, and defaults to
    off. It is the scale-matching repair `cone_objective` argues for -- a line
    measure and a greyscale disagree about sharpness before they disagree about
    anything else -- but it is a second transform on top of the one asked for,
    so it is a switch that gets reported, not a default that gets hidden.
    """
    d = np.asarray(tdi, float)

    if smooth and smooth > 0:
        d = gaussian_filter(d, float(smooth))
    
    d = np.log10(1.0 + np.maximum(d, 0.0))

    inside = d if support is None else d[support]
    if not inside.size:
        return np.zeros_like(d)
    lo, hi = float(inside.min()), float(inside.max())
    if hi - lo <= EPS:
        return np.zeros_like(d)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------

def decompose_abs(a, b):
    """The MAE analogue of `decompose` -- and NOT an identity, unlike that one.

        MAE      = mean |a - b|
        level    = |mean a - mean b|          how far apart the two LEVELS are
        shape    = mean |(a - a_bar) - (b - b_bar)|   disagreement after centring

    `level + shape >= MAE` by the triangle inequality, with equality only when
    the centred residual never changes sign. So the two parts BOUND the error
    rather than partitioning it, and `level/MAE` is not a fraction-of-loss the
    way `offset_frac` is. Reported anyway because the question it answers -- is
    this a level gap or a shape gap -- is the same question, and refusing to
    answer it because the algebra is less tidy would be worse than answering it
    with the caveat attached.
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if not a.size:
        return dict(mae=float("nan"), level=float("nan"), shape=float("nan"),
                    level_frac=float("nan"))
    ma, mb = a.mean(), b.mean()
    mae = float(np.mean(np.abs(a - b)))
    level = float(abs(ma - mb))
    shape = float(np.mean(np.abs((a - ma) - (b - mb))))
    return dict(mae=mae, level=level, shape=shape,
                level_frac=float(level / mae) if mae > EPS else float("nan"))


def decompose(a, b):
    """Split mean((a - b)^2) into its offset and pattern halves.

    Exact, not an approximation:

        mean((a-b)^2) = (mean a - mean b)^2 + var a + var b - 2 cov(a, b)

    The first term is what the two fields' overall LEVELS disagree about; the
    rest is what their SHAPES disagree about. A loss whose offset term is nearly
    all of it is being minimised by coverage, and moving it says nothing about
    whether the tracts followed the fibres.
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if not a.size:
        return dict(mse=float("nan"), offset=float("nan"),
                    pattern=float("nan"), offset_frac=float("nan"),
                    pearson_r=float("nan"))
    ma, mb = a.mean(), b.mean()
    offset = float((ma - mb) ** 2)
    da, db = a - ma, b - mb
    pattern = float(np.mean(da ** 2) + np.mean(db ** 2) - 2 * np.mean(da * db))
    mse = offset + pattern
    sa, sb = float(np.std(a)), float(np.std(b))
    r = float(np.mean(da * db) / (sa * sb)) if sa > EPS and sb > EPS else float("nan")
    return dict(mse=mse, offset=offset, pattern=pattern,
                offset_frac=float(offset / mse) if mse > EPS else float("nan"),
                pearson_r=r)


class TdiMse:
    """The objective, as a callable `gradient_optimise.Problem` can descend on.

    Built once per (volume, field variant); every evaluation reuses the prepared
    image and the support, so a change in the loss between two parameter
    settings can only have come from the tracts.
    """

    def __init__(self, vol, support, smooth=0.0, n_keep=N_KEEP,
                 clip_pct=CLIP_PCT, metric="mse"):
        self.image, self.was_preprocessed = prepare_image(vol, clip_pct=clip_pct)
        self.shape = self.image.shape
        self.support = (np.ones(self.shape, bool) if support is None
                        else np.asarray(support, bool))
        self.smooth, self.n_keep = float(smooth), n_keep
        self.img_s = self.image[self.support]
        self.n_calls = 0
        # WHICH residual is descended on. Both are reported either way; this
        # only picks the one the optimiser sees.
        #
        # They differ in how the residual weights the gradient. d/dp of the
        # squared error carries a factor of the residual itself, so voxels that
        # are already badly wrong dominate the step; d/dp of the absolute error
        # carries only sign(residual), so every voxel contributes equally
        # regardless of how wrong it is. On this data the residual distribution
        # is near-Gaussian (RMSE/MAE = 1.24 against 1.25 for a normal), so there
        # is no heavy tail for MAE to be robust AGAINST -- the expected effect is
        # a change in which voxels steer, not a change in outlier handling.
        if metric not in ("mse", "mae"):
            raise ValueError(f"metric must be 'mse' or 'mae', got {metric!r}")
        self.metric = metric

    # -- the two steps, separately, so either can be inspected ---------------

    def tdi(self, pos, dirs, w):
        """The raw (un-normalised) TDI of a smooth tractogram."""
        return ST.as_tdi(pos, dirs, w, self.shape, n_keep=self.n_keep)
    
    def normalised(self, pos, dirs, w):
        return normalise_tdi(self.tdi(pos, dirs, w), support=self.support,
                             smooth=self.smooth)

    # -- the loss ------------------------------------------------------------

    def __call__(self, pos, dirs, w):
        self.n_calls += 1
        d = self.normalised(pos, dirs, w)[self.support]
        e = d - self.img_s
        return float(np.mean(e ** 2) if self.metric == "mse"
                     else np.mean(np.abs(e)))

    def report(self, pos, dirs, w):
        """The loss plus everything needed to judge whether to believe it.

        BOTH residuals are always reported, whichever one is being descended
        on, so an mae run and an mse run can be compared without re-tracking
        either. `loss` is the one the optimiser saw.
        """
        raw = self.tdi(pos, dirs, w)
        norm = normalise_tdi(raw, support=self.support, smooth=self.smooth)
        d, r = norm[self.support], raw[self.support]
        out = decompose(d, self.img_s)
        out.update(decompose_abs(d, self.img_s))
        out.update(
            metric=self.metric,
            loss=float(out["mse"] if self.metric == "mse" else out["mae"]),
            rmse=float(np.sqrt(out["mse"])),
            coverage=float((r > 0).mean()),          # support voxels a tract touched
            tdi_mean_raw=float(r.mean()), tdi_max_raw=float(r.max()),
            tdi_mean_norm=float(d.mean()), image_mean=float(self.img_s.mean()),
            image_var=float(self.img_s.var()),
            n_support=int(self.support.sum()), smooth=self.smooth)
        return out


# ---------------------------------------------------------------------------

def _selftest():
    """The three properties the loss has to have, checked on synthetic input."""
    print("tdi_mse self-test")
    rng = np.random.default_rng(0)
    shape = (24, 24, 24)
    sup = np.ones(shape, bool)

    # 1. the decomposition is exact
    a, b = rng.random(400), rng.random(400)
    dc = decompose(a, b)
    direct = float(np.mean((a - b) ** 2))
    print(f"  decompose exact:   {dc['mse']:.9f} vs {direct:.9f}   "
          f"(offset {dc['offset']:.6f} + pattern {dc['pattern']:.6f})")

    # 2. normalise_tdi lands on [0,1], keeps zeros at zero, is monotone
    tdi = np.zeros(shape)
    tdi[12, 12, :] = np.linspace(0.5, 40.0, shape[2])
    n = normalise_tdi(tdi, sup)
    line = n[12, 12, :]
    print(f"  normalise range:   [{n.min():.3f}, {n.max():.3f}]  "
          f"zeros stay zero: {bool((n[tdi == 0] == 0).all())}  "
          f"monotone: {bool((np.diff(line) > 0).all())}")

    # 3. prepare_image is a no-op on an already-[0,1] volume
    v01 = rng.random(shape)
    out, flag = prepare_image(v01)
    print(f"  image no-op on [0,1]: unchanged={np.allclose(out, v01)} "
          f"already_preprocessed={flag}")

    ok = (abs(dc["mse"] - direct) < 1e-12
          and abs(n.max() - 1.0) < 1e-9 and n.min() == 0.0
          and bool((n[tdi == 0] == 0).all())
          and bool((np.diff(line) > 0).all())
          and np.allclose(out, v01) and flag)
    print("  " + ("ALL PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    _selftest()
