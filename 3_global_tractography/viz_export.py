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

def comparison_figure(image, tdi_norm, support, path, title="", info=None):
    """TDI against image: three slices, the residual, and the joint histogram.

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
            (D, "normalised TDI  (log10, 0-1)", "magma", (0, 1)),
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

def save_param_volume(Ls, thetas, alphas, Z, path, descrip=b"tdi_mse"):
    """The sweep as a NIfTI whose three axes are the three parameters.

    Identical convention to `param_volume.save_volume`, repeated here rather
    than imported so this module has no dependency on that script's `main`:
    index -> log10(parameter), angles in DEGREES, units marked unknown because
    they are not spatial and a viewer printing millimetres for a cone width
    would be worse than printing nothing.
    """
    import nibabel as nib

    thd, ald = np.degrees(np.asarray(thetas)), np.degrees(np.asarray(alphas))

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
# the interactive 3D plot
# ---------------------------------------------------------------------------

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
