"""
param_volume_figure.py -- label the (L, theta, alpha) volume and draw the descent.

Reads what `param_volume.py` already computed -- the loss volume and the descent
trace -- and produces two things:

  1. A PATH OVERLAY NIfTI on the identical grid and affine, so NiiVue can show
     the descent inside the volume it walked on. Voxel value is the iteration
     number, so the overlay's colour bar reads as time.
  2. A FIGURE with the axes named, because a NIfTI cannot name its own axes.
     NIfTI has no per-axis label field: a viewer can only be told the affine and
     the units, and the units here are "log10 of a parameter", which no viewer
     has a word for. So the naming lives in the sidecar JSON, in the header's
     `descrip` string, and in this figure -- which is the only one of the three
     a reader actually sees.

Nothing is recomputed. Re-running the sweep would produce a second sample of the
same surface, and the path drawn on it would then be a path across a volume it
never visited.

Run
---
    python param_volume_figure.py
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    sys.path.insert(0, _p)

PATH_C = "#eb6834"       # categorical slot 2 -- the descent
START_C = "#1baf7a"      # slot 3 -- where it began
MIN_C = "#1A1A1A"


def iso_shell(Z, Ls, thd, ald, iso, up=9, tol_frac=0.035):
    """Points lying on the loss iso-surface, as a shell in parameter space.

    marching_cubes would be the usual tool; scikit-image is not a dependency of
    this repo and one figure is not a reason to make it one. A spline upsample
    plus a narrow band around the isovalue gives the same read -- a translucent
    shell you can see the basin's shape through -- with scipy alone.

    HONESTY ABOUT WHAT THIS IS: the sweep is a 5x5x5 grid. Everything between
    those 125 nodes is `zoom`'s cubic spline, not measurement. The shell shows
    the shape the interpolant implies; it is an illustration of the basin, not
    evidence about its fine structure, and the figure says so.

    Returns (L, theta_deg, alpha_deg) of the shell points.
    """
    from scipy.ndimage import zoom

    Zf = zoom(np.asarray(Z, float), up, order=3)
    # Index -> parameter is exact on the ORIGINAL log-spaced axes, so the
    # upsampled index is mapped back through the same log interpolation rather
    # than assuming the parameters are linear in index.
    def _axis(v, n_new):
        lg = np.log(v)
        return np.exp(np.interp(np.linspace(0, len(v) - 1, n_new),
                                np.arange(len(v)), lg))

    gl = _axis(Ls, Zf.shape[0])
    gt = _axis(thd, Zf.shape[1])
    ga = _axis(ald, Zf.shape[2])
    band = np.abs(Zf - iso) < tol_frac * iso
    i, j, k = np.nonzero(band)
    return gl[i], gt[j], ga[k]


def _frac_index(values, v):
    """Fractional index of `v` on a log-spaced axis. Extrapolates: the descent
    left the swept box on L, and silently clamping that would draw the path
    ending inside the volume when it did not."""
    lg = np.log(values)
    return (np.log(max(float(v), 1e-9)) - lg[0]) / (lg[1] - lg[0])


def path_overlay(side, run, outdir, tag="i74"):
    """The descent path as a NIfTI on the loss volume's own grid."""
    import nibabel as nib

    ref = nib.load(os.path.join(outdir, f"param_volume_{tag}.nii.gz"))
    Ls = np.array(side["L"], float)
    ths = np.array(side["theta_rad"], float)
    als = np.array(side["alpha_rad"], float)
    n = ref.shape[0]

    P = np.zeros(ref.shape, np.float32)
    n_out = 0
    for step in run["descent"]["trace"]:
        idx = [_frac_index(Ls, step["L"]), _frac_index(ths, step["theta"]),
               _frac_index(als, step["alpha"])]
        ijk = [int(round(v)) for v in idx]
        if any(v < 0 or v >= n for v in ijk):
            n_out += 1
            continue
        # Iteration number, so the overlay reads as time rather than presence.
        P[tuple(ijk)] = max(P[tuple(ijk)], float(step["iter"]) + 2.0)

    img = nib.Nifti1Image(P, ref.affine)
    img.header.set_xyzt_units("unknown")
    img.header["descrip"] = b"descent path; value = iteration+2; 0 = not visited"
    p = os.path.join(outdir, f"param_path_{tag}.nii.gz")
    nib.save(img, p)
    return p, n_out


def figure(side, run, Z, outdir, tag="i74"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D           # noqa: F401

    Ls = np.array(side["L"], float)
    thd = np.array(side["theta_deg"], float)
    ald = np.array(side["alpha_deg"], float)

    tr = run["descent"]["trace"]
    pL = np.array([t["L"] for t in tr])
    pT = np.degrees([t["theta"] for t in tr])
    pA = np.degrees([t["alpha"] for t in tr])

    g = run["grid_min"]
    gL, gT, gA = g["L"], np.degrees(g["theta"]), np.degrees(g["alpha"])
    dL, dT, dA = (run["descent"]["L"], np.degrees(run["descent"]["theta"]),
                  np.degrees(run["descent"]["alpha"]))

    LL, TT, AA = np.meshgrid(Ls, thd, ald, indexing="ij")
    lo, hi = float(Z.min()), float(Z.max())
    # LOG colour scale. The loss spans 0.068 to 0.45, but nearly all of the
    # volume sits between 0.07 and 0.11 -- on a linear ramp the handful of
    # degenerate high-loss cells at small theta consume the whole range and
    # every point that matters renders the same near-white. A log norm spends
    # the ramp where the samples are.
    norm = matplotlib.colors.LogNorm(vmin=lo, vmax=hi)

    fig = plt.figure(figsize=(15.6, 8.4))
    fig.patch.set_facecolor("white")
    fig.suptitle("(L, theta, alpha) loss volume on i74  --  descent path and "
                 "minima     [L in voxels, angles in DEGREES]", fontsize=13)

    # ---- the volume ------------------------------------------------------
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    f = Z.ravel()

    # ---- the volume SHAPE: two nested iso-shells around the minimum --------
    # Multiples of the minimum, not absolute levels, so the same figure works
    # on a chunk with a different overall loss scale.
    shell_proxies = []
    for iso_mult, col, a_, sz in ((1.20, "#9dc3ec", 0.10, 7),
                                  (1.05, "#2a78d6", 0.26, 9)):
        sl, st, sa = iso_shell(Z, Ls, thd, ald, lo * iso_mult)
        if not len(sl):
            continue
        ax.scatter(sl, st, sa, s=sz, color=col, alpha=a_, depthshade=False,
                   linewidths=0)
        # A translucent point cloud makes an unreadable legend swatch, so the
        # legend gets an opaque proxy of the same colour instead.
        shell_proxies.append(
            (plt.Line2D([], [], marker="s", ms=8, ls="none", color=col),
             f"basin shell: loss = {iso_mult:.2f} x min "
             f"({lo * iso_mult:.4f}),  {len(sl)} pts"))

    # The measured grid nodes stay visible on top of the interpolated shape, so
    # what was actually sampled is never confused with what was interpolated.
    keep = f <= np.percentile(f, 55)
    ax.scatter(LL.ravel()[~keep], TT.ravel()[~keep], AA.ravel()[~keep],
               s=5, color="#C9D3DE", alpha=0.30, depthshade=False)
    lg = np.log(f[keep] / lo) / np.log(hi / lo)
    sc = ax.scatter(LL.ravel()[keep], TT.ravel()[keep], AA.ravel()[keep],
                    c=f[keep], cmap="Blues", norm=norm,
                    s=14 + 90 * (1.0 - lg),
                    edgecolors="#00000040", linewidths=0.4, depthshade=False)
    ax.plot(pL, pT, pA, color=PATH_C, lw=2.4, marker="o", ms=4,
            label="descent path", zorder=8)
    ax.scatter([pL[0]], [pT[0]], [pA[0]], s=120, color=START_C,
               depthshade=False, label=f"start  L={pL[0]:.1f}, "
                                       f"{pT[0]:.1f} deg, {pA[0]:.1f} deg")
    ax.scatter([dL], [dT], [dA], s=200, marker="*", color=PATH_C,
               depthshade=False,
               label=f"descent end  L={dL:.2f}, {dT:.1f} deg, {dA:.1f} deg")
    ax.scatter([gL], [gT], [gA], s=170, marker="D", color=MIN_C,
               depthshade=False, edgecolors="white", linewidths=1.4,
               label=f"grid minimum  L={gL:.1f}, {gT:.1f} deg, {gA:.1f} deg")
    ax.set_xlabel("L   lookahead (voxels)", labelpad=8)
    ax.set_ylabel("theta   cone half-width (deg)", labelpad=8)
    ax.set_zlabel("alpha   angular resolution (deg)", labelpad=8)
    ax.set_title("the basin as a solid   (shells = cubic interpolation of the "
                 f"{Z.shape[0]}^3 grid,\nnot measurement; the dots ARE the "
                 "measured nodes)", fontsize=9, loc="left")
    h, lb = ax.get_legend_handles_labels()
    ax.legend(h + [p for p, _t in shell_proxies],
              lb + [t for _p, t in shell_proxies],
              fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.view_init(elev=24, azim=-62)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.11)
    cb.set_label("align loss  (DARKER = worse)", fontsize=8)

    # ---- three faces -----------------------------------------------------
    # Each is the MINIMUM over the axis not shown, not a slice: a slice would
    # depend on where the third parameter happened to be pinned, and the point
    # is where the loss can get to, not where one arbitrary plane sits.
    faces = [(0, 1, 2, Ls, thd, pL, pT, gL, gT, dL, dT,
              "L (vox)", "theta (deg)"),
             (0, 2, 1, Ls, ald, pL, pA, gL, gA, dL, dA,
              "L (vox)", "alpha (deg)"),
             (1, 2, 0, thd, ald, pT, pA, gT, gA, dT, dA,
              "theta (deg)", "alpha (deg)")]
    for i, (a0, a1, drop, X, Y, px, py, gx, gy, dx, dy, xl, yl) in \
            enumerate(faces):
        a = fig.add_subplot(3, 2, 2 * (i + 1))
        M = Z.min(axis=drop)
        if a0 > a1:
            M = M.T
        im = a.pcolormesh(X, Y, M.T, cmap="Blues", norm=norm,
                          shading="nearest")
        a.plot(px, py, color=PATH_C, lw=2, marker="o", ms=3.5, zorder=5)
        a.scatter([px[0]], [py[0]], s=70, color=START_C, zorder=6,
                  edgecolors="white", linewidths=1.0)
        a.scatter([dx], [dy], s=150, marker="*", color=PATH_C, zorder=6,
                  edgecolors="white", linewidths=0.8)
        a.scatter([gx], [gy], s=80, marker="D", color=MIN_C, zorder=6,
                  edgecolors="white", linewidths=0.8)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(xl, fontsize=9)
        a.set_ylabel(yl, fontsize=9)
        a.tick_params(labelsize=8)
        a.set_title(f"min over the third axis: {xl.split()[0]} vs "
                    f"{yl.split()[0]}", fontsize=9, loc="left")
        # Only claimed when it is true. Whether the descent leaves the swept
        # box depends on the run -- it did on the radius-24 chunk and did not
        # here -- so the note is derived from the path, never assumed.
        outside = ((px < X.min()) | (px > X.max())
                   | (py < Y.min()) | (py > Y.max()))
        if i == 0 and outside.any():
            a.annotate(f"path leaves the swept box ({int(outside.sum())} of "
                       f"{len(px)} points)\n(blank = never sampled)",
                       (px[-1], py[-1]), xytext=(10, 12),
                       textcoords="offset points", fontsize=7.5, color=PATH_C,
                       bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                 ec="none", alpha=0.85))
        fig.colorbar(im, ax=a, shrink=0.9).ax.tick_params(labelsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(outdir, f"param_volume_{tag}.png")
    fig.savefig(p, dpi=150, facecolor="white")
    plt.close(fig)
    return p


def main(outdir=None, tag="i74"):
    import nibabel as nib
    outdir = outdir or os.path.join(_ROOT, "results")
    with open(os.path.join(outdir, f"param_volume_{tag}.json")) as f:
        side = json.load(f)
    with open(os.path.join(outdir, "param_volume_run.json")) as f:
        run = json.load(f)
    Z = nib.load(os.path.join(outdir,
                              f"param_volume_{tag}.nii.gz")).get_fdata()

    print("=" * 78)
    print("AXES:  x = L (voxels)   y = theta (degrees)   z = alpha (degrees)")
    print(f"  L     {np.array(side['L']).round(2)}")
    print(f"  theta {np.array(side['theta_deg']).round(2)} deg")
    print(f"  alpha {np.array(side['alpha_deg']).round(2)} deg")
    print("=" * 78)

    p, n_out = path_overlay(side, run, outdir, tag)
    print(f"  wrote {p}")
    print(f"    {n_out}/{len(run['descent']['trace'])} path points fall "
          f"OUTSIDE the swept grid (the descent left the box on L) and are "
          f"not in the overlay")
    q = figure(side, run, Z, outdir, tag)
    print(f"  wrote {q}")
    print("=" * 78)
    return p, q


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="label the volume, draw the path")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tag", default="i74")
    a = ap.parse_args()
    main(outdir=a.outdir, tag=a.tag)
