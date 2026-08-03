"""
render_feature_size_map.py

Draws the per-voxel Lindeberg feature-size heat map produced by
lindeberg_feature_size.py.

Colour choices (and why)
------------------------
* Feature size is a MAGNITUDE, so it gets a sequential, perceptually-uniform
  ramp with a scale legend (the colour bar) -- never a categorical set.
  viridis rather than a single-hue light->dark ramp for one specific reason:
  these panels sit beside the raw GREYSCALE tomogram, and a single-hue ramp
  that runs light->dark is exactly what the greyscale panel already looks
  like, so a reader glancing across the row can mistake anatomy for scale.
  viridis is unambiguous against grey, perceptually uniform, and CVD-safe.
* The scale axis is LOG, because sigma is a multiplicative quantity: the
  distance from 40 to 60 um is the same amount of scale-space as 60 to 90.
* Voxels with no certified scale (background air, or a railed response) are
  painted NEUTRAL GREY, not a ramp colour. Painting them at the ramp's end
  would invent a plateau of fake feature size along both rails -- the single
  easiest way to make this figure lie.
* The classification panel is the only categorical encoding here; it uses the
  reference palette's slots 1/2/3 verbatim in fixed order, and every class is
  named in the legend with its share, so identity is never colour-alone.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from reconstruction import VOXEL_SIZE_UM

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID_INK = "#dcdbd6"
UNCERTIFIED = "#c9c8c3"      # neutral grey: "no scale certified here"
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # palette slots 1, 2, 3
CMAP = plt.get_cmap("viridis")


def _load(prefix):
    sig = np.load(f"{prefix}_sigma.npy", mmap_mode="r")
    flg = np.load(f"{prefix}_flags.npy", mmap_mode="r")
    with open(f"{prefix}_meta.json") as f:
        meta = json.load(f)
    return sig, flg, meta


def _raw_slices(meta, iz, iy):
    """Axial slice iz and coronal slice iy of the source greyscale, read
    straight from the TIFF stack (never the whole volume)."""
    import tifffile
    files = sorted(Path(meta["folder"]).glob("*.tif"))
    crop = meta["crop"]
    if crop is None:
        axial = tifffile.imread(files[iz])
        coronal = np.stack([tifffile.imread(f)[iy] for f in files])
    else:
        a, b = crop
        axial = tifffile.imread(files[a + iz])[a:b, a:b]
        coronal = np.stack([tifffile.imread(f)[a + iy, a:b] for f in files[a:b]])
    return axial, coronal


def _heat_rgba(um_slice, certified, norm):
    """Feature size -> RGBA, with uncertified voxels forced to neutral grey."""
    rgba = CMAP(norm(np.where(certified, um_slice, norm.vmin)))
    rgba[~certified] = matplotlib.colors.to_rgba(UNCERTIFIED)
    return rgba


def _class_rgba(tissue, rail_lo, rail_hi):
    certified = tissue & ~rail_lo & ~rail_hi
    rgba = np.empty(tissue.shape + (4,))
    rgba[...] = matplotlib.colors.to_rgba(UNCERTIFIED)
    rgba[tissue & rail_hi] = matplotlib.colors.to_rgba(CLASS_COLORS[2])
    rgba[tissue & rail_lo] = matplotlib.colors.to_rgba(CLASS_COLORS[1])
    rgba[certified] = matplotlib.colors.to_rgba(CLASS_COLORS[0])
    return rgba


def _log_ticks(lo, hi, n=6):
    """`n` round-ish ticks spanning [lo, hi] geometrically.

    Matplotlib's log locators label almost nothing when the range is under one
    decade -- which it is here, because real tissue feature sizes are tightly
    clustered -- so the ticks are chosen explicitly rather than left to the
    default locator (an unlabelled colour bar is a chart with no units).
    """
    raw = np.geomspace(lo, hi, n)
    step = 10.0 ** np.floor(np.log10(raw))
    nice = np.unique(np.round(raw / (step / 2)) * (step / 2))
    return nice[(nice >= lo) & (nice <= hi)]


def _bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def render(prefix, out=None, stride=3):
    sig, flg, meta = _load(prefix)
    calib = meta["calib"]
    um_per_sigma = calib["fwhm_factor"] * calib["voxel_um"] / calib["k"]
    shape = tuple(meta["shape"])
    iz, iy = shape[0] // 2, shape[1] // 2

    # --- whole-volume statistics, subsampled so nothing large is materialised
    sub_f = np.asarray(flg[::stride, ::stride, ::stride])
    sub_s = np.asarray(sig[::stride, ::stride, ::stride])
    tis = (sub_f & 1) > 0
    lo = ((sub_f & 2) > 0) & tis
    hi = ((sub_f & 4) > 0) & tis
    cert = tis & ~lo & ~hi
    d_all = sub_s[cert] * um_per_sigma
    frac = dict(certified=cert.sum() / max(tis.sum(), 1),
                fine=lo.sum() / max(tis.sum(), 1),
                coarse=hi.sum() / max(tis.sum(), 1))
    vmin, vmax = np.percentile(d_all, [2, 98])
    norm = LogNorm(vmin=vmin, vmax=vmax)
    med = float(np.median(d_all))

    # --- slices
    def plane(arr, axis, idx):
        return np.asarray(arr[idx] if axis == 0 else arr[:, idx])
    panels = []
    for axis, idx in ((0, iz), (1, iy)):
        s = plane(sig, axis, idx) * um_per_sigma
        f = plane(flg, axis, idx)
        t = (f & 1) > 0
        panels.append((s, t & ~((f & 2) > 0) & ~((f & 4) > 0),
                       t, ((f & 2) > 0) & t, ((f & 4) > 0) & t))
    raw_ax, raw_cor = _raw_slices(meta, iz, iy)

    # --- figure
    fig = plt.figure(figsize=(15.5, 10.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05],
                          hspace=0.22, wspace=0.10,
                          left=0.02, right=0.975, top=0.885, bottom=0.155)

    for row, (raw, name) in enumerate(((raw_ax, f"axial  z = {iz}"),
                                       (raw_cor, f"coronal  y = {iy}"))):
        ax = fig.add_subplot(gs[row, 0], facecolor=SURFACE)
        ax.imshow(raw, cmap="gray", interpolation="nearest")
        ax.set_title(f"raw i74, {name}", color=INK, fontsize=11, pad=6)
        _bare(ax)

        s, certified, _t, _l, _h = panels[row]
        ax = fig.add_subplot(gs[row, 1], facecolor=SURFACE)
        ax.imshow(_heat_rgba(s, certified, norm), interpolation="nearest")
        ax.set_title("Lindeberg feature size, same slice",
                     color=INK, fontsize=11, pad=6)
        _bare(ax)

    # --- colour bar under the two heat panels. Each tick carries BOTH readings
    # of the same number: the micron size the structure has, and the sigma* the
    # pipeline should differentiate at there. One bar, two units, no second axis
    # to collide with anything.
    cax = fig.add_axes([0.075, 0.088, 0.40, 0.018])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = _log_ticks(vmin, vmax)
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels([f"{t:.0f}\n{t / um_per_sigma:.2f}" for t in ticks])
    cb.ax.tick_params(colors=INK_2, labelsize=8.5, length=3)
    cb.outline.set_visible(False)
    cb.ax.set_title("feature size, um  (top)   /   adaptive sigma* to use, "
                    f"rho = {meta['rho_ratio']:g} x sigma*  (bottom)",
                    color=INK_2, fontsize=9.5, pad=5)
    fig.text(0.075, 0.028, "grey = no scale certified", color=INK_2, fontsize=9)

    # --- histogram
    ax = fig.add_subplot(gs[0, 2], facecolor=SURFACE)
    # Clip the right edge to p99.9: the coarse tail runs to the sigma=6 rail,
    # and letting it set the axis would spend most of a log panel on a handful
    # of voxels. The overflow is stated rather than hidden.
    hi_edge = float(np.percentile(d_all, 99.9))
    n_over = int((d_all > hi_edge).sum())
    edges = np.geomspace(d_all.min(), hi_edge, 70)
    counts, _ = np.histogram(np.clip(d_all, None, hi_edge), bins=edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color=CMAP(norm(np.clip(centres, vmin, vmax))), linewidth=0)
    ax.set_xscale("log")
    ax.axvline(med, color=INK, lw=2, ls=(0, (4, 3)))
    ax.annotate(f"median {med:.0f} um\n({med / VOXEL_SIZE_UM:.1f} voxels)",
                xy=(med, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", color=INK, fontsize=9.5, va="top")
    # The two block-level picks this repo already certified, for reference.
    for sg, lbl in ((0.89, "dense-tissue block pick\n(sigma_L 0.88-0.90)"),
                    (6.00, "sparse block B pick\n(sigma_L 6.0)")):
        x = sg * um_per_sigma
        if edges[0] < x < edges[-1]:
            ax.axvline(x, color=INK_2, lw=1.4, ls=":")
            ax.annotate(lbl, xy=(x, ax.get_ylim()[1] * 0.55), xytext=(5, 0),
                        textcoords="offset points", color=INK_2, fontsize=8, va="top")
    ax.set_xlabel("feature size, um  (log)", color=INK_2, fontsize=9.5, labelpad=3)
    ax.set_ylabel("voxels", color=INK_2, fontsize=9.5)
    ax.annotate(f"{n_over / d_all.size:.1%} coarser than {hi_edge:.0f} um,\n"
                f"folded into the last bin",
                xy=(0.985, 0.62), xycoords="axes fraction", ha="right",
                color=INK_2, fontsize=8.2)
    ax.set_title("distribution over certified tissue voxels",
                 color=INK, fontsize=11, pad=6)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.grid(axis="y", color=GRID_INK, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_INK)
    ax.set_xticks(_log_ticks(edges[0], edges[-1], 7))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _p: f"{v:.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # --- classification panel
    _s, _c, t0, l0, h0 = panels[0]
    ax = fig.add_subplot(gs[1, 2], facecolor=SURFACE)
    ax.imshow(_class_rgba(t0, l0, h0), interpolation="nearest")
    ax.set_title("where a scale could be certified (axial)",
                 color=INK, fontsize=11, pad=6)
    _bare(ax)
    smin, smax = meta["sigmas"][0], meta["sigmas"][-1]
    fig.legend(handles=[
        Patch(facecolor=CLASS_COLORS[0], label=(
            f"scale certified -- interior peak  ({frac['certified']:.0%} of tissue)")),
        Patch(facecolor=CLASS_COLORS[1], label=(
            f"too fine / noise-dominated  ({frac['fine']:.0%})  "
            f"- response only falls from sigma={smin:g}; use the sigma floor")),
        Patch(facecolor=CLASS_COLORS[2], label=(
            f"too coarse  ({frac['coarse']:.0%})  - response still rising at "
            f"sigma={smax:g}; not fibre-scale")),
        Patch(facecolor=UNCERTIFIED, label="background / air"),
    ], loc="lower right", bbox_to_anchor=(0.982, 0.015), frameon=False,
        fontsize=9, labelcolor=INK_2, handlelength=1.3, borderpad=0,
        labelspacing=0.5)

    fig.suptitle(
        "Per-voxel Lindeberg feature size on i74  "
        f"({shape[0]}x{shape[1]}x{shape[2]} @ {VOXEL_SIZE_UM:g} um/voxel)",
        color=INK, fontsize=15, y=0.975)
    fig.text(0.5, 0.938,
             f"sigma*(x) = first interior local max over sigma of  "
             f"sigma^2 (lambda_1+lambda_2)(x; sigma, rho={meta['rho_ratio']:g}sigma),  "
             f"{len(meta['sigmas'])} geometric scales {smin:g}-{smax:g}, response pooled at "
             f"{meta.get('resp_smooth', 0):g}sigma;   "
             f"size = {calib['fwhm_factor']:.3f} sigma*/{calib['k']:.3f} x "
             f"{VOXEL_SIZE_UM:g} um  (tube-phantom fit, R^2={calib['r2']:.3f})",
             ha="center", color=INK_2, fontsize=9.5)
    fig.text(0.5, 0.915,
             "Calibration caveat: the phantom fit is linear to within 3% for tube widths >= 1.5 voxels, but its "
             "w = 1.0 voxel point sits 24% high -- sizes below ~50 um are upper bounds near the sampling limit.",
             ha="center", va="center", color=INK_2, fontsize=8.6)

    out = out or f"{prefix}_map.png"
    fig.savefig(out, dpi=145, facecolor=SURFACE)
    print(f"Saved {out}")
    print(f"  certified {frac['certified']:.1%} of tissue | too fine "
          f"{frac['fine']:.1%} | too coarse {frac['coarse']:.1%}")
    print(f"  feature size um: p5={np.percentile(d_all, 5):.0f} "
          f"median={med:.0f} p95={np.percentile(d_all, 95):.0f}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("prefix", nargs="?", default="fs_i74_full")
    p.add_argument("--out", default=None)
    p.add_argument("--stride", type=int, default=3,
                   help="subsampling stride for the whole-volume statistics")
    args = p.parse_args()
    render(args.prefix, args.out, args.stride)
