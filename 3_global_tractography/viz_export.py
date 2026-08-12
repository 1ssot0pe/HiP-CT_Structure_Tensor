r"""
viz_export.py -- everything this loop writes for a human to look at.

Five outputs, one module, because they all depend on the same two conventions
and splitting them is how the two get out of step:

    NIfTI     preprocessed image, normalised TDI, and their difference, all on
              ONE affine so they overlay in FreeView without being moved
    .trk      the tractogram, in the frame that same affine defines
    figure    TDI against image, as slices and as a joint histogram
    volume    the loss over (L, theta, alpha), as a NIfTI whose axes are
              PARAMETERS -- see `param_volume` for why that is legitimate
    HTML      the descent path and the search cone it moved, interactive

The two conventions
-------------------
FRAME. Arrays are (z, y, x); NIfTI and TrackVis both want (x, y, z). `nifti_io`
documents the trap and does the transpose for volumes; streamlines have to be
reversed here, because a tractogram written with the axes in array order lands
in the volume rotated and still looks like a plausible tractogram. The rotation
is only obvious when it is overlaid on the image, which is the one check nobody
does before believing the picture.

SPACE. `nifti_affine(crop_origin, voxel_um)` carries the crop offset, so a
64-voxel chunk out of the 768^3 volume sits where it belongs rather than at the
world origin. Streamlines are handed to nibabel in VOXEL coordinates together
with that affine as `affine_to_rasmm`, so the file records the same mapping the
images use and the two are registered by construction rather than by luck.

Why .trk and not .tck
---------------------
FreeView reads both, but a .tck stores only the points and leans on a separate
image for its frame; a .trk carries the dimensions, the voxel size and the
voxel-to-RAS matrix in its own header. For a crop out of a larger volume that
difference is the whole game -- a .tck opened against the wrong reference lands
somewhere else with no error.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    sys.path.insert(0, _p)

from nifti_io import nifti_affine, save_nifti     # noqa: E402

EPS = 1e-12

IMG_C = "#2a78d6"
TDI_C = "#eb6834"
PATH_C = "#7d3ac1"
START_C = "#1baf7a"


# ---------------------------------------------------------------------------
# volumes
# ---------------------------------------------------------------------------

def export_volumes(image, tdi_norm, outdir, tag, crop_origin=(0, 0, 0),
                   voxel_um=20.0, support=None, verbose=True):
    """Image, normalised TDI, and (TDI - image), on one shared affine.

    The difference map is written because it is the loss, per voxel, and it is
    the only one of the three that answers the question the descent is actually
    asking. Two volumes side by side in a viewer are compared by eye at whatever
    window level each happens to get; the residual is not.

    Both inputs are already on [0, 1], so the difference is on [-1, 1] and a
    symmetric colour map reads directly: positive is density the image does not
    support, negative is image the tracts did not cover.
    """
    os.makedirs(outdir, exist_ok=True)
    A = nifti_affine(crop_origin, voxel_um)
    diff = np.asarray(tdi_norm, float) - np.asarray(image, float)
    if support is not None:
        diff = np.where(np.asarray(support, bool), diff, 0.0)

    paths = {
        "image": save_nifti(np.asarray(image, float),
                            os.path.join(outdir, f"image_{tag}.nii.gz"),
                            A, dtype=np.float32),
        "tdi": save_nifti(np.asarray(tdi_norm, float),
                          os.path.join(outdir, f"tdi_{tag}.nii.gz"),
                          A, dtype=np.float32),
        "residual": save_nifti(diff,
                               os.path.join(outdir, f"residual_{tag}.nii.gz"),
                               A, dtype=np.float32),
    }
    if verbose:
        for k, v in paths.items():
            print(f"      {k:9} {v}")
    return paths


# ---------------------------------------------------------------------------
# tractogram
# ---------------------------------------------------------------------------

def save_trk(streamlines, shape, path, crop_origin=(0, 0, 0), voxel_um=20.0,
             weights=None, verbose=True):
    """Write (z, y, x)-voxel streamlines as a TrackVis .trk for FreeView.

    `streamlines` : list of (M, 3) arrays of (z, y, x) positions in voxels.
    `shape`       : the array shape (Nz, Ny, Nx) they were tracked in.

    Positions are reversed to (x, y, z) and handed over as voxel coordinates
    with `affine_to_rasmm` set to the same affine the NIfTI exports use, which
    is what registers the tracts to the images. Doing the mm conversion by hand
    and passing an identity affine also "works" and produces a file that opens
    -- in the wrong place, silently -- so the conversion is left to nibabel.

    The per-tract survival weight is written as a TRK per-point SCALAR named
    "survival" (the same value repeated along the tract), because that is the
    field FreeView can colour and threshold by; TRK's per-streamline
    "properties" are not exposed the same way. Carrying it at all is what keeps
    a tract that decayed after three steps distinguishable from one that held.
    """
    import nibabel as nib
    from nibabel.streamlines import Tractogram
    from nibabel.streamlines.trk import TrkFile

    A = nifti_affine(crop_origin, voxel_um)
    lines = [np.ascontiguousarray(np.asarray(s, float)[:, ::-1])
             for s in streamlines if len(np.asarray(s)) >= 2]

    dps = {}
    if weights is not None and len(lines):
        w = np.asarray(weights, float)[:len(lines)]
        dps["survival"] = [np.full((len(s), 1), float(wi), np.float32)
                           for s, wi in zip(lines, w)]

    tg = Tractogram(lines, data_per_point=dps, affine_to_rasmm=A)
    mm = float(voxel_um) / 1000.0
    hdr = {
        # Dimensions are (X, Y, Z), matching the reversed points and the
        # transposed volumes -- not the array's (Nz, Ny, Nx).
        "dimensions": np.array(tuple(shape)[::-1], np.int16),
        "voxel_sizes": np.array([mm, mm, mm], np.float32),
        "voxel_to_rasmm": A.astype(np.float32),
        "voxel_order": "RAS",
    }
    TrkFile(tg, header=hdr).save(path)
    if verbose:
        n_pts = sum(len(s) for s in lines)
        print(f"      trk       {path}  ({len(lines)} tracts, {n_pts} points)")
    return path


# ---------------------------------------------------------------------------
# the comparison figure
# ---------------------------------------------------------------------------

def comparison_figure(image, tdi_norm, support, path, title="", info=None,
                      tdi_cmap="magma"):
    """TDI against image: three slices, the residual, and the joint histogram.

    `tdi_cmap` is "magma" by default and "gray" for the like-for-like version.
    Two different maps make the TDI easy to SEE -- a sparse line measure on a
    perceptually uniform map reads clearly against a black background -- but
    they make the two panels impossible to compare by eye, because any
    difference in apparent brightness could be the colour map rather than the
    data. Drawing both in grey on the identical [0, 1] scale means a voxel that
    looks equally bright in the two panels IS equally valued. The cost is that a
    TDI covering ~12-20% of the support becomes a faint line drawing, which is
    an honest picture of how sparse it is rather than a flattering one.

    The residual keeps its diverging map in both versions: it is a SIGNED
    quantity, and grey would render +0.5 and -0.5 as the same tone.

    The joint histogram is the panel that decides the question. Slices show that
    the tracts are roughly where the tissue is, which they always are -- both
    fields are confined to the same mask, so any pair would look agreeable.
    Whether density tracks INTENSITY is a two-dimensional question and only the
    2D histogram answers it: a real relationship is a ridge, and the standing
    result is a vertical smear at TDI = 0 with everything else scattered.

    Plotted in log counts because most support voxels have no tract at all, so
    on a linear count scale a single bin at TDI = 0 is the entire picture.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    I = np.asarray(image, float)
    D = np.asarray(tdi_norm, float)
    S = np.asarray(support, bool)
    k = I.shape[0] // 2

    fig = plt.figure(figsize=(14.0, 8.4))
    fig.patch.set_facecolor("white")
    fig.suptitle(title or "preprocessed image vs normalised TDI", fontsize=12)
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.20)

    for j, (arr, nm, cm, vr) in enumerate((
            (I, "preprocessed image  (clip 1-99%, 0-1)", "gray", (0, 1)),
            (D, "normalised TDI  (log10, 0-1)", tdi_cmap, (0, 1)),
            (np.where(S, D - I, np.nan), "residual  TDI - image", "coolwarm",
             (-1, 1)))):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(arr[k], cmap=cm, vmin=vr[0], vmax=vr[1],
                       interpolation="nearest")
        ax.set_title(f"{nm}\nz = {k}", fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    d, i = D[S], I[S]

    # --- joint histogram ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    H, xe, ye = np.histogram2d(d, i, bins=(64, 64), range=[[0, 1], [0, 1]])
    ax.imshow(np.log10(H.T + 1.0), origin="lower", aspect="auto",
              extent=(0, 1, 0, 1), cmap="viridis")
    ax.set_xlabel("normalised TDI"); ax.set_ylabel("preprocessed image")
    r = (np.corrcoef(d, i)[0, 1] if d.std() > EPS and i.std() > EPS
         else float("nan"))
    ax.set_title(f"joint histogram (log counts)   Pearson r = {r:+.4f}",
                 fontsize=9, loc="left")

    # --- marginals ---------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(i, bins=60, range=(0, 1), color=IMG_C, alpha=0.65, label="image",
            log=True)
    ax.hist(d, bins=60, range=(0, 1), color=TDI_C, alpha=0.65, label="TDI",
            log=True)
    ax.set_xlabel("value on [0, 1]"); ax.set_ylabel("support voxels (log)")
    ax.set_title(f"marginals   image mean {i.mean():.3f}, "
                 f"TDI mean {d.mean():.3f}", fontsize=9, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # --- what the loss is made of -------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    if info:
        lines = [
            f"loss  mean (TDI - image)^2   = {info.get('mse', float('nan')):.6f}",
            "",
            f"  offset  (mean diff)^2      = {info.get('offset', float('nan')):.6f}"
            f"   [{info.get('offset_frac', float('nan')):.1%} of the loss]",
            f"  pattern (shape disagreement) = {info.get('pattern', float('nan')):.6f}",
            "",
            f"  coverage: {info.get('coverage', float('nan')):.2%} of support "
            f"voxels touched",
            f"  raw TDI  mean {info.get('tdi_mean_raw', float('nan')):.4f}, "
            f"max {info.get('tdi_max_raw', float('nan')):.2f}",
            f"  image    mean {info.get('image_mean', float('nan')):.4f}, "
            f"var {info.get('image_var', float('nan')):.5f}",
            f"  support  {info.get('n_support', 0)} voxels",
            f"  pre-log smoothing: {info.get('smooth', 0.0):g} vox",
        ]
        ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8.6,
                family="monospace", transform=ax.transAxes)
        of = info.get("offset_frac", float("nan"))
        if np.isfinite(of) and of > 0.5:
            ax.text(0.0, 0.10,
                    "offset dominates: this loss is mostly measuring how much\n"
                    "of the tissue the tracts COVER, not where they went.",
                    va="top", ha="left", fontsize=8.6, color="#b3261e",
                    transform=ax.transAxes)

    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# the loss volume over (L, theta, alpha)
# ---------------------------------------------------------------------------

def save_param_volume(Ls, thetas, alphas, Z, path, descrip=b"tdi_mse",
                      alpha_in_degrees=True):
    """The sweep as a NIfTI whose three axes are the three parameters.

    Identical convention to `param_volume.save_volume`, repeated here rather
    than imported so this module has no dependency on that script's `main`:
    index -> log10(parameter), angles in DEGREES, units marked unknown because
    they are not spatial and a viewer printing millimetres for a cone width
    would be worse than printing nothing.

    `alpha_in_degrees=False` for a `turn="relax"` sweep, where alpha is a
    dimensionless turn FRACTION and not an angle. Converting a fraction to
    degrees would put a plausible-looking angle on the axis of a quantity that
    is not one -- see `descent_loop.fmt_alpha`.
    """
    import nibabel as nib

    thd = np.degrees(np.asarray(thetas))
    ald = (np.degrees(np.asarray(alphas)) if alpha_in_degrees
           else np.asarray(alphas, float))

    def _step(v):
        lg = np.log10(np.asarray(v, float))
        return float(lg[1] - lg[0]) if len(lg) > 1 else 1.0

    A = np.diag([_step(Ls), _step(thd), _step(ald), 1.0])
    A[:3, 3] = [np.log10(Ls[0]), np.log10(thd[0]), np.log10(ald[0])]
    img = nib.Nifti1Image(np.ascontiguousarray(Z, np.float32), A)
    img.header.set_xyzt_units("unknown")
    img.header["descrip"] = descrip
    nib.save(img, path)
    return path


# ---------------------------------------------------------------------------
# the loss volume with the descent drawn on it
# ---------------------------------------------------------------------------

def _upsample_axis(v, n_new):
    """Log-spaced axis resampled to n_new points. The grids are log-spaced by
    construction, so interpolating in log and exponentiating is exact at the
    original nodes rather than assuming the parameter is linear in index."""
    lg = np.log(np.asarray(v, float))
    return np.exp(np.interp(np.linspace(0, len(lg) - 1, n_new),
                            np.arange(len(lg)), lg))


def _read_sweep(outdir, variant, json_name="descent_loop.json", up=1):
    """(volume, log10 axes, descent path in log10, grid-min, run dict).

    Coordinates are log10(parameter), not the parameter. The three grids are
    log-spaced by construction, so in log10 the sample points are EVENLY spaced
    -- which is what a volume renderer requires and what `save_param_volume`'s
    affine already encodes. Handing raw parameter values to a renderer that
    assumes a regular grid would draw the cube stretched, and the stretch would
    look like structure in the loss.

    The two ANGLES are in degrees, as everywhere else a human reads them; L
    stays in voxels because it is a length.
    """
    import json
    import nibabel as nib
    from scipy.ndimage import zoom

    with open(os.path.join(outdir, json_name)) as f:
        run = json.load(f)
    d = run["runs"][variant]
    if "sweep" not in d:
        raise ValueError(f"{variant} in {outdir} has no sweep; use --sweep-n")
    sw = d["sweep"]
    Z = np.asarray(nib.load(os.path.join(
        outdir, f"lossvol_{variant.replace('+', '_')}.nii.gz")).get_fdata(),
        float)
    if up > 1:
        Z = zoom(Z, up, order=3)
    ax = [_upsample_axis(np.array(sw[k], float), Z.shape[i])
          for i, k in enumerate(("L", "theta_deg", "alpha_deg"))]
    lg = [np.log10(a) for a in ax]

    tr = d["trace"]
    P = np.array([[np.log10(t["L"]), np.log10(np.degrees(t["theta"])),
                   np.log10(np.degrees(t["alpha"]))] for t in tr])
    loss = np.array([t["loss"] for t in tr], float)
    gm = sw["grid_min"]
    g = np.log10([gm["L"], gm["theta_deg"], gm["alpha_deg"]])
    return Z, lg, P, loss, g, d, run


def _log_ticks(lo, hi):
    """Tick positions in log10 with labels in the real parameter, so the axis
    reads in voxels/degrees even though the coordinate is a logarithm."""
    cand = np.array([1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40], float)
    v = cand[(cand >= 10 ** lo * 0.999) & (cand <= 10 ** hi * 1.001)]
    if len(v) < 3:
        v = 10 ** np.linspace(lo, hi, 5)
    return np.log10(v), [f"{x:g}" for x in v]


def descent_volume_3d(outdir, path, variants=None, up=5, opacity=0.10,
                      surface_count=18, json_name="descent_loop.json",
                      title=""):
    """The (L, theta, alpha) loss as a VOLUME, with the descent flown through it.

    One rotatable scene per field variant, side by side, each containing:

      * the loss as a semi-transparent volume rendering -- every sampled
        (L, theta, alpha) contributes, so a basin is a solid dark region rather
        than something inferred from three flat projections
      * the descent path as a line through that volume, coloured by its own loss
      * the start, the descent's end, and the GRID minimum as distinct markers

    The grid minimum is drawn because it is usually not where the descent
    stopped, and the gap between them is the finding. A picture with only the
    path in it cannot show that the optimiser settled somewhere the sweep had
    already beaten.

    HONESTY ABOUT THE PICTURE: the sweep is 5x5x5 = 125 MEASURED points. `up`
    spline-upsamples so the render is not 125 cubes; everything between the
    nodes is the interpolant, not measurement. The measured nodes are drawn as
    small dots inside the volume so the two can never be confused. Same caveat
    and same reason as `param_volume_figure.iso_shell`.

    Self-contained HTML with plotly inlined -- opens offline in a browser, or in
    VS Code with the Live Preview extension.
    """
    import json
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    with open(os.path.join(outdir, json_name)) as f:
        variants = variants or [v for v in json.load(f)["runs"]]
    keep = []
    for v in variants:
        try:
            _read_sweep(outdir, v, json_name)
            keep.append(v)
        except (ValueError, FileNotFoundError):
            pass
    if not keep:
        raise ValueError(f"no variant in {outdir} has a sweep volume")

    fig = make_subplots(rows=1, cols=len(keep),
                        specs=[[{"type": "scene"}] * len(keep)],
                        subplot_titles=keep, horizontal_spacing=0.03)

    run = None
    for c, v in enumerate(keep, start=1):
        Zf, lg, P, loss, g, d, run = _read_sweep(outdir, v, json_name, up=up)
        Z0, _lg0, *_ = _read_sweep(outdir, v, json_name, up=1)
        X, Y, W = np.meshgrid(lg[0], lg[1], lg[2], indexing="ij")
        show = c == 1

        fig.add_trace(go.Volume(
            x=X.ravel(), y=Y.ravel(), z=W.ravel(), value=Zf.ravel(),
            isomin=float(Zf.min()), isomax=float(Zf.max()),
            opacity=opacity, surface_count=surface_count,
            colorscale="Viridis", reversescale=True, showscale=show,
            colorbar=dict(title="loss", len=0.65, thickness=12, x=-0.06),
            name=f"{v} loss", hoverinfo="skip"), row=1, col=c)

        # the 125 points that were actually measured
        m0 = [np.log10(np.array(d["sweep"][k], float))
              for k in ("L", "theta_deg", "alpha_deg")]
        MX, MY, MW = np.meshgrid(*m0, indexing="ij")
        fig.add_trace(go.Scatter3d(
            x=MX.ravel(), y=MY.ravel(), z=MW.ravel(), mode="markers",
            marker=dict(size=1.6, color="#888888"), showlegend=show,
            name="measured node (125)",
            text=[f"loss {q:.6f}" for q in Z0.ravel()], hoverinfo="text"),
            row=1, col=c)

        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines+markers",
            line=dict(color=PATH_C, width=7),
            marker=dict(size=5, color=loss, colorscale="Plasma"),
            name="descent path", showlegend=show,
            text=[f"it {t['iter']}<br>loss {t['loss']:.6f}<br>"
                  f"L {t['L']:.3f} vox<br>theta {np.degrees(t['theta']):.2f} deg"
                  f"<br>alpha {np.degrees(t['alpha']):.2f} deg"
                  for t in d["trace"]], hoverinfo="text"), row=1, col=c)

        gm = d["sweep"]["grid_min"]
        for xyz, nm, col, sym, txt in (
                (P[0], "start", START_C, "circle",
                 f"start, loss {loss[0]:.6f}"),
                (P[-1], "descent end", "#d62728", "diamond",
                 f"descent end, loss {loss[-1]:.6f}"),
                (g, "grid minimum", "#ffd400", "x",
                 f"grid min, loss {gm['loss']:.6f}"
                 + (" (ON A BOUNDARY)" if gm.get("on_boundary") else ""))):
            fig.add_trace(go.Scatter3d(
                x=[xyz[0]], y=[xyz[1]], z=[xyz[2]], mode="markers",
                marker=dict(size=9, color=col, symbol=sym,
                            line=dict(color="white", width=1)),
                name=nm, showlegend=show, hovertext=txt, hoverinfo="text"),
                row=1, col=c)

        tl = _log_ticks(lg[0][0], lg[0][-1])
        tt = _log_ticks(lg[1][0], lg[1][-1])
        ta = _log_ticks(lg[2][0], lg[2][-1])
        scene = dict(
            xaxis=dict(title="L  lookahead (vox)", tickvals=tl[0],
                       ticktext=tl[1]),
            yaxis=dict(title="theta  half-width (deg)", tickvals=tt[0],
                       ticktext=tt[1]),
            zaxis=dict(title="alpha  resolution (deg)", tickvals=ta[0],
                       ticktext=ta[1]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.5, z=1.1)))
        fig.layout[f"scene{c if c > 1 else ''}"].update(scene)

    gaps = []
    for v in keep:
        d = run["runs"][v]
        gaps.append(f"{v}: end {d['end']['mse']:.6f} vs grid "
                    f"{d['sweep']['grid_min']['loss']:.6f} "
                    f"({d['end']['mse'] - d['sweep']['grid_min']['loss']:+.6f})")
    fig.update_layout(
        title=dict(text=(title or
                   f"{run.get('label', 'i74')} {tuple(run['shape'])} -- "
                   f"MSE(preprocessed image, normalised TDI) over "
                   f"(L, theta, alpha), with the descent path<br>"
                   f"<sub>5x5x5 measured (grey dots), spline-upsampled "
                   f"{up}x for rendering &nbsp;|&nbsp; "
                   + " &nbsp;|&nbsp; ".join(gaps) + "</sub>"), x=0.01,
                   font=dict(size=13)),
        template="plotly_white", height=760,
        legend=dict(orientation="h", y=-0.02))
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def descent_volume_3d_png(outdir, path, variants=None, up=7,
                          json_name="descent_loop.json", title="",
                          quantiles=(0.02, 0.12, 0.35), band=0.018,
                          max_pts=2600, rng_seed=0):
    """A still of the same scene, for contexts that cannot run the HTML.

    Drawn as NESTED ISO-SHELLS, not as a fog. matplotlib has no volume renderer,
    and the obvious substitute -- every sample as a low-alpha point -- fails for
    a reason worth recording: thousands of points at the alpha needed to see
    through them composite to nothing, so the box renders empty and the picture
    silently shows no data at all.

    A shell at loss quantile q is the set of parameter settings the loss is
    (roughly) equal at, so nesting three of them shows the basin the way contour
    lines show a hill: the innermost shell is where the loss is lowest, and its
    SHAPE says which parameters the loss actually cares about. A shell that
    spans the full range of an axis means that axis does not matter.

    HONESTY: the sweep is 5x5x5 = 125 measured points; `up` spline-upsamples for
    display, so the shells are surfaces of the interpolant. Same caveat as
    `param_volume_figure.iso_shell`, which uses the identical construction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import json

    with open(os.path.join(outdir, json_name)) as f:
        variants = variants or [v for v in json.load(f)["runs"]]
    keep = []
    for v in variants:
        try:
            _read_sweep(outdir, v, json_name)
            keep.append(v)
        except (ValueError, FileNotFoundError):
            pass
    if not keep:
        raise ValueError(f"no variant in {outdir} has a sweep volume")

    rng = np.random.default_rng(rng_seed)
    fig = plt.figure(figsize=(7.6 * len(keep), 7.0))
    fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("viridis")
    run = None
    for c, v in enumerate(keep, start=1):
        Zf, lg, P, loss, g, d, run = _read_sweep(outdir, v, json_name, up=up)
        ax = fig.add_subplot(1, len(keep), c, projection="3d")
        X, Y, W = np.meshgrid(lg[0], lg[1], lg[2], indexing="ij")
        val = Zf.ravel()
        lo, hi = float(val.min()), float(val.max())

        for si, q in enumerate(quantiles):
            iso = float(np.quantile(val, q))
            m = np.abs(val - iso) <= band * (hi - lo)
            idx = np.flatnonzero(m)
            if len(idx) > max_pts:
                idx = rng.choice(idx, max_pts, replace=False)
            col = cmap((iso - lo) / max(hi - lo, 1e-12))
            # Innermost (lowest-loss) shell is the most opaque: it is the one
            # carrying the answer, and the outer ones are context.
            ax.scatter(X.ravel()[idx], Y.ravel()[idx], W.ravel()[idx],
                       color=col, s=13.0 - 3.0 * si,
                       alpha=0.55 - 0.16 * si, linewidths=0, depthshade=False,
                       zorder=1 + si)

        # Dropped to the theta-alpha floor as well as drawn in place: five
        # iterates inside a translucent cube are almost impossible to localise
        # in depth from a still, and the shadow pins where the path sits.
        ax.plot(P[:, 0], P[:, 1], np.full(len(P), lg[2][0]), "-",
                color="#999999", lw=1.4, zorder=2, alpha=0.9)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], "-", color=PATH_C, lw=4.0, zorder=8,
                solid_capstyle="round")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color=PATH_C, s=46,
                   edgecolors="white", linewidths=1.0, zorder=9,
                   depthshade=False)
        ax.scatter(*P[0], s=190, color=START_C, edgecolors="white",
                   linewidths=1.3, zorder=10, depthshade=False)
        ax.scatter(*P[-1], s=300, marker="*", color="#d62728",
                   edgecolors="white", linewidths=1.1, zorder=10,
                   depthshade=False)
        ax.scatter(*g, s=230, marker="X", color="#ffd400",
                   edgecolors="#222222", linewidths=1.0, zorder=10,
                   depthshade=False)

        for setter, lab_setter, lim, lab in (
                (ax.set_xticks, ax.set_xticklabels, lg[0], "L  lookahead (vox)"),
                (ax.set_yticks, ax.set_yticklabels, lg[1],
                 "theta  half-width (deg)"),
                (ax.set_zticks, ax.set_zticklabels, lg[2],
                 "alpha  resolution (deg)")):
            tv, tt = _log_ticks(lim[0], lim[-1])
            setter(tv)
            lab_setter(tt, fontsize=8)
        ax.set_xlabel("L  lookahead (vox)", fontsize=9, labelpad=2)
        ax.set_ylabel("theta  half-width (deg)", fontsize=9, labelpad=2)
        ax.set_zlabel("alpha  resolution (deg)", fontsize=9, labelpad=2)

        gm = d["sweep"]["grid_min"]
        gap = d["end"]["mse"] - gm["loss"]
        ax.set_title(
            f"{v}\ndescent end {d['end']['mse']:.6f}    grid min "
            f"{gm['loss']:.6f}    ({gap:+.6f})\n"
            + (f"grid min ON A BOUNDARY -- theta at the range floor "
               f"({gm['theta_deg']:.2f} deg)" if gm.get("on_boundary")
               else "grid min is interior"),
            fontsize=9.5, pad=2)
        ax.view_init(elev=22, azim=-60)

        # How much of the box the path actually crosses. Without this the
        # reader cannot tell a descent that barely moved from a path the
        # renderer failed to draw -- and here it really did barely move.
        span = [(P[:, i].max() - P[:, i].min()) / (lg[i][-1] - lg[i][0])
                for i in range(3)]
        ax.text2D(0.02, 0.02,
                  f"path spans {span[0]:.1%} of L, {span[1]:.1%} of theta, "
                  f"{span[2]:.1%} of alpha\n"
                  f"({len(P)} iterates; the low-loss shell fills most of the "
                  f"box, so there is no basin to fall into)",
                  transform=ax.transAxes, fontsize=8, color="#444444")

        if c == 1:
            handles = [Line2D([], [], marker="o", ls="", color=cmap(
                (float(np.quantile(val, q)) - lo) / max(hi - lo, 1e-12)),
                label=f"loss shell, {q:.0%} quantile "
                      f"({np.quantile(val, q):.4f})", ms=7)
                for q in quantiles]
            handles += [
                Line2D([], [], color=PATH_C, lw=3, label="descent path"),
                Line2D([], [], marker="o", ls="", color=START_C, ms=9,
                       label="start"),
                Line2D([], [], marker="*", ls="", color="#d62728", ms=13,
                       label="descent end"),
                Line2D([], [], marker="X", ls="", color="#ffd400", ms=11,
                       label="grid minimum")]
            ax.legend(handles=handles, fontsize=8, loc="upper left",
                      bbox_to_anchor=(-0.14, 1.02), frameon=False)

    fig.suptitle(title or
                 f"{run.get('label', 'i74')} {tuple(run['shape'])} -- "
                 f"loss VOLUME over (L, theta, alpha) with the descent path   "
                 f"[5x5x5 measured, {up}x spline-upsampled for display]",
                 fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.03,
                        wspace=0.06)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# the interactive 3D plot
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# redraw a figure from what a run already wrote
# ---------------------------------------------------------------------------

def regenerate_figure(outdir, variant="vec1", tdi_cmap="gray", suffix="_gray",
                      verbose=True):
    """Redraw one run's comparison figure from its saved outputs.

    Exists so a change of colour map costs a file read instead of the ~20 minute
    descent + sweep that produced the numbers. Everything the figure needs was
    already written: the two [0, 1] volumes as NIfTI, and the diagnostics in
    `descent_loop.json`.

    The support mask is the one thing NOT written out, and it is recomputed here
    rather than guessed: `reconstruction.tissue_mask` is `vol > AIR_FRAC *
    vol.max()`, a pure function of the volume, and the volume it was originally
    computed on IS the saved preprocessed image (`load_chunk` preprocesses, and
    `tdi_mse.prepare_image` then detects the [0, 1] range and does nothing). So
    re-running it on the saved image reproduces the original mask exactly, up to
    the float32 the NIfTI was written in.

    Arrays come back off disk in (x, y, z) -- `nifti_io.save_nifti` transposes on
    the way out -- so they are transposed back to (z, y, x). Without that the
    figure would slice a DIFFERENT plane of the same volume and quietly not be
    comparable to the figure it is supposed to be a recolour of.
    """
    import json
    import nibabel as nib
    from reconstruction import tissue_mask

    tag = variant.replace("+", "_")
    with open(os.path.join(outdir, "descent_loop.json")) as f:
        run = json.load(f)
    r = run["runs"][variant]

    def _load(nm):
        a = np.asarray(nib.load(os.path.join(outdir, f"{nm}_{tag}.nii.gz")
                                ).get_fdata(), float)
        return np.ascontiguousarray(a.transpose(2, 1, 0))      # -> (z, y, x)

    image, tdi = _load("image"), _load("tdi")
    support = tissue_mask(image)
    path = os.path.join(outdir, f"compare_{tag}{suffix}.png")
    # alpha in ITS OWN units: degrees for turn="sample", a bare fraction for
    # turn="relax". `alpha_deg` is deliberately None on a relax run so it cannot
    # be picked up and printed as an angle -- see `descent_loop.fmt_alpha`.
    turn = r.get("turn", run.get("turn", "sample"))
    a_str = (f"{r['alpha_deg']:.1f} deg" if turn == "sample"
             else f"{r['alpha']:.4f} (turn fraction)")
    p = comparison_figure(
        image, tdi, support, path, info=r["end"], tdi_cmap=tdi_cmap,
        title=f"{run.get('label', 'i74')} -- {variant} field, converged cone "
              f"(L={r['L']:.2f} vox, theta={r['theta_deg']:.1f} deg, "
              f"alpha={a_str}, turn={turn})   [TDI in {tdi_cmap}]")
    if verbose:
        n_orig, n_new = run["tissue_voxels"], int(support.sum())
        print(f"  {variant:>10}: support {n_new} voxels "
              f"({'matches' if n_new == n_orig else f'DIFFERS from {n_orig}'}"
              f" the run)  ->  {p}")
    return p


def _cone_mesh(L, theta, n_phi=48):
    """A search cone of length L and half-angle theta, apex at the origin,
    opening along +x. Returns (x, y, z) vertex arrays for a triangle fan."""
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=True)
    r = float(L) * np.tan(float(theta))
    x = np.concatenate([[0.0], np.full(n_phi, float(L))])
    y = np.concatenate([[0.0], r * np.cos(phi)])
    z = np.concatenate([[0.0], r * np.sin(phi)])
    return x, y, z


def descent_3d_html(trace, path, title="", alphas_shown=3):
    """The descent path in (L, theta, alpha), plus the cone geometry it moved.

    Two linked things in one rotatable scene, which is the point -- the numbers
    (L = 4.07, theta = 8.0 deg) do not tell you what changed about the SHAPE of
    the search, and the shape is what the tracker uses.

      * the trajectory, as a line through the three-parameter space, log-scaled
        on all three axes because they are positive scale parameters spanning an
        order of magnitude and a linear box would put every iterate in one corner
      * the START and END cones, drawn to scale in real voxel units, with the
        alpha rings that set how finely the tracker can tell two directions apart

    Written as a self-contained HTML with plotly inlined, so it opens offline in
    a browser, or inside VS Code with the Live Preview extension.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tr = [t for t in trace if np.isfinite(t.get("loss", np.nan))]
    L = np.array([t["L"] for t in tr], float)
    TH = np.degrees([t["theta"] for t in tr])
    AL = np.degrees([t["alpha"] for t in tr])
    loss = np.array([t["loss"] for t in tr], float)

    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("descent path in (L, theta, alpha)",
                        "the search cone, start vs end (voxels)"),
        horizontal_spacing=0.04)

    # ---- left: the trajectory ---------------------------------------------
    fig.add_trace(go.Scatter3d(
        x=L, y=TH, z=AL, mode="lines+markers", name="descent path",
        line=dict(color=PATH_C, width=5),
        marker=dict(size=5, color=loss, colorscale="Viridis",
                    colorbar=dict(title="loss", x=0.44, len=0.7, thickness=12),
                    showscale=True),
        text=[f"it {t['iter']}<br>loss {t['loss']:.6f}<br>"
              f"L {t['L']:.3f} vox<br>theta {np.degrees(t['theta']):.2f} deg<br>"
              f"alpha {np.degrees(t['alpha']):.2f} deg" for t in tr],
        hoverinfo="text"), row=1, col=1)
    for idx, nm, col, sym in ((0, "start", START_C, "circle"),
                              (len(tr) - 1, "end", "#d62728", "diamond")):
        fig.add_trace(go.Scatter3d(
            x=[L[idx]], y=[TH[idx]], z=[AL[idx]], mode="markers", name=nm,
            marker=dict(size=11, color=col, symbol=sym,
                        line=dict(color="white", width=1)),
            hovertext=f"{nm}: L {L[idx]:.3f}, theta {TH[idx]:.2f} deg, "
                      f"alpha {AL[idx]:.2f} deg, loss {loss[idx]:.6f}",
            hoverinfo="text"), row=1, col=1)

    # ---- right: the geometry ----------------------------------------------
    for idx, nm, col in ((0, "start cone", START_C),
                         (len(tr) - 1, "end cone", "#d62728")):
        Lv, th, al = tr[idx]["L"], tr[idx]["theta"], tr[idx]["alpha"]
        cx, cy, cz = _cone_mesh(Lv, th)
        n = len(cx) - 1
        fig.add_trace(go.Mesh3d(
            x=cx, y=cy, z=cz,
            i=np.zeros(n - 1, int), j=np.arange(1, n), k=np.arange(2, n + 1),
            color=col, opacity=0.28, name=nm, showlegend=True,
            hovertext=f"{nm}: L {Lv:.2f} vox, half-angle "
                      f"{np.degrees(th):.2f} deg", hoverinfo="text"),
            row=1, col=2)
        # the axis, and the alpha rings on the cap: alpha is the angular
        # RESOLUTION, so what it sets is how many distinguishable tilts fit
        # inside theta -- drawn, not asserted.
        fig.add_trace(go.Scatter3d(
            x=[0, Lv], y=[0, 0], z=[0, 0], mode="lines",
            line=dict(color=col, width=4), showlegend=False,
            hoverinfo="skip"), row=1, col=2)
        n_ring = max(int(round(th / max(al, 1e-9))), 1)
        for m in range(1, min(n_ring, alphas_shown * 4) + 1):
            t_m = min(m * al, th)
            phi = np.linspace(0, 2 * np.pi, 60)
            rr = Lv * np.tan(t_m)
            fig.add_trace(go.Scatter3d(
                x=np.full(60, Lv), y=rr * np.cos(phi), z=rr * np.sin(phi),
                mode="lines", line=dict(color=col, width=2),
                opacity=0.75, showlegend=False,
                hovertext=f"{nm}: tilt ring {m} at "
                          f"{np.degrees(t_m):.2f} deg", hoverinfo="text"),
                row=1, col=2)

    fig.update_layout(
        title=dict(text=title or "gradient descent on the cone", x=0.02),
        template="plotly_white", height=720,
        scene=dict(xaxis=dict(title="L  lookahead (vox)", type="log"),
                   yaxis=dict(title="theta  half-width (deg)", type="log"),
                   zaxis=dict(title="alpha  resolution (deg)", type="log"),
                   aspectmode="cube"),
        scene2=dict(xaxis_title="along the current heading (vox)",
                    yaxis_title="voxels", zaxis_title="voxels",
                    aspectmode="data"),
        legend=dict(orientation="h", y=-0.04))

    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="redraw a run's comparison figure with a different colour "
                    "map, from what it already wrote")
    ap.add_argument("outdir", help="a directory containing descent_loop.json")
    ap.add_argument("--cmap", default="gray",
                    help="colour map for the TDI panel (default gray, to match "
                         "the image panel)")
    ap.add_argument("--suffix", default="_gray",
                    help="appended to the output filename, so the original is "
                         "not overwritten")
    ap.add_argument("--variant", default="both")
    a = ap.parse_args()

    import json as _json
    with open(os.path.join(a.outdir, "descent_loop.json")) as _f:
        _variants = list(_json.load(_f)["runs"])
    for _v in (_variants if a.variant == "both" else [a.variant]):
        regenerate_figure(a.outdir, _v, tdi_cmap=a.cmap, suffix=a.suffix)
