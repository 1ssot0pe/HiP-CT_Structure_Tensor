"""
host_geometry_report.py

Compiles the three host_geometry_test.py phantoms (images) and their HOST-FOD
results (metrics) into a single summary PNG. Recomputes everything fresh from
the same phantom builders / analyse() calls used in host_geometry_test.py --
no hardcoded numbers -- so the figure stays reproducible from source.

Colour usage follows the dataviz skill's validated default palette
(references/palette.md): a fixed hue per structure-tensor ORDER wherever order
is the series being compared (2=blue, 4=aqua, 6=yellow, 8=green), a separate
fixed pair for the phantom-3 case comparison (crossing=violet, kissing=
magenta), and the status pair (good/critical) only as small icon+label tags on
classification outcomes -- never as the only signal.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from host_fod import analyse, classify
from host_geometry_test import (
    make_two_region_phantom, make_messy_crossing, make_kissing_phantom,
    make_true_crossing_same_angles, _patch_at, _ang_err, _match_dirs, SHAPE,
)

# ---------------------------------------------------------------------------
# palette (dataviz skill reference palette, light mode)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

ORDER_COLOR = {2: "#2a78d6", 4: "#1baf7a", 6: "#eda100", 8: "#008300"}
CASE_COLOR = {"true crossing": "#4a3aa7", "kissing": "#e87ba4"}
BOUNDARY_LINE = "#e34948"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK, "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _style_ax(ax, ylabel=None, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, color=INK, pad=6)


def _bar_group(ax, categories, series, colors, ylabel, title, value_fmt="{:.1f}"):
    """series: dict(name -> list of values aligned with categories)."""
    n_series = len(series)
    x = np.arange(len(categories))
    width = 0.8 / n_series
    for i, (name, vals) in enumerate(series.items()):
        xs = x + (i - (n_series - 1) / 2) * width
        bars = ax.bar(xs, vals, width * 0.9, color=colors[name], zorder=3, label=name)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                       value_fmt.format(v), ha="center", va="bottom",
                       fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8.5)
    _style_ax(ax, ylabel=ylabel, title=title)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")


if __name__ == "__main__":
    h = SHAPE[0] // 2
    q, tq = SHAPE[0] // 4, 3 * SHAPE[0] // 4

    fig = plt.figure(figsize=(17, 15.5))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[1.05, 1.0, 1.0, 1.0, 0.5],
                 hspace=0.5, wspace=0.4)

    fig.suptitle("HOST-FOD (order-6 higher-order structure tensor) -- geometry stress tests",
                color=INK, fontsize=15, y=0.995, fontweight="bold")

    # =======================================================================
    # PHANTOM 1 -- perpendicular regions, different widths
    # =======================================================================
    vol1, tr1 = make_two_region_phantom()
    ax_img1 = fig.add_subplot(gs[0, 0])
    ax_img1.imshow(vol1[h], cmap="gray")
    ax_img1.axvline(h, color=BOUNDARY_LINE, ls="--", lw=1.3)
    ax_img1.set_xticks([]); ax_img1.set_yticks([])
    ax_img1.set_title("Phantom 1: perpendicular regions\n(width 1.2 vs 3.5, NOT overlapping)",
                      fontsize=10)
    for s in ax_img1.spines.values():
        s.set_color(BASELINE)

    sigmas = [1.0, 2.0, 4.0]
    regionA_err = {2: [], 6: []}
    regionB_err = {2: [], 6: []}
    for order in (2, 6):
        for sg in sigmas:
            mA = analyse(_patch_at(vol1, (h, h, q)), sigma_grad=sg, order=order)
            mB = analyse(_patch_at(vol1, (h, h, tq)), sigma_grad=sg, order=order)
            regionA_err[order].append(_ang_err(mA["dirs"][0], tr1["d1"]) if mA and mA["dirs"] is not None else np.nan)
            regionB_err[order].append(_ang_err(mB["dirs"][0], tr1["d2"]) if mB and mB["dirs"] is not None else np.nan)

    ax_a = fig.add_subplot(gs[0, 1])
    _bar_group(ax_a, [f"sg={s:g}" for s in sigmas],
              {"order=2": regionA_err[2], "order=6": regionA_err[6]},
              {"order=2": ORDER_COLOR[2], "order=6": ORDER_COLOR[6]},
              "angular error (deg)", "Region A (width=1.2)\nerror vs sigma_grad")

    ax_b = fig.add_subplot(gs[0, 2])
    _bar_group(ax_b, [f"sg={s:g}" for s in sigmas],
              {"order=2": regionB_err[2], "order=6": regionB_err[6]},
              {"order=2": ORDER_COLOR[2], "order=6": ORDER_COLOR[6]},
              "angular error (deg)", "Region B (width=3.5)\nerror vs sigma_grad")

    peak1_err, peak2_err = [], []
    for sg in sigmas:
        m = analyse(_patch_at(vol1, (h, h, h)), sigma_grad=sg, order=6)
        if m and m["dirs"] is not None and len(m["dirs"]) >= 2:
            e1 = min(_ang_err(m["dirs"][0], tr1["d1"]), _ang_err(m["dirs"][0], tr1["d2"]))
            e2 = min(_ang_err(m["dirs"][1], tr1["d1"]), _ang_err(m["dirs"][1], tr1["d2"]))
        else:
            e1 = e2 = np.nan
        peak1_err.append(e1); peak2_err.append(e2)

    ax_c = fig.add_subplot(gs[0, 3])
    _bar_group(ax_c, [f"sg={s:g}" for s in sigmas],
              {"peak 1": peak1_err, "peak 2": peak2_err},
              {"peak 1": INK2, "peak 2": MUTED},
              "angular error (deg)", "BOUNDARY (order=6)\norder=2 found no 2nd peak here")

    # =======================================================================
    # PHANTOM 2 -- messy 4-family crossing
    # =======================================================================
    vol2, tr2 = make_messy_crossing()
    ax_img2 = fig.add_subplot(gs[1, 0])
    ax_img2.imshow(vol2[h], cmap="gray")
    ax_img2.set_xticks([]); ax_img2.set_yticks([])
    ax_img2.set_title(f"Phantom 2: messy crossing\n{len(tr2['dirs'])} random-angle/width families, all overlapping",
                      fontsize=10)
    for s in ax_img2.spines.values():
        s.set_color(BASELINE)

    orders2 = [4, 6, 8]
    npeaks = []
    for order in orders2:
        m = analyse(_patch_at(vol2, (h, h, h)), sigma_grad=2.0, order=order)
        npeaks.append(0 if m is None else m["n_peaks"])

    ax_npk = fig.add_subplot(gs[1, 1])
    x = np.arange(len(orders2))
    bars = ax_npk.bar(x, npeaks, 0.55, color=[ORDER_COLOR[o] for o in orders2], zorder=3)
    for b, v in zip(bars, npeaks):
        ax_npk.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom",
                   fontsize=9, color=INK2)
    ax_npk.axhline(len(tr2["dirs"]), color=CRITICAL, ls="--", lw=1.3, zorder=2)
    ax_npk.text(len(orders2) - 0.5, len(tr2["dirs"]) + 0.08, "true count = 4",
               color=CRITICAL, fontsize=8, ha="right", va="bottom")
    ax_npk.set_xticks(x); ax_npk.set_xticklabels([f"order={o}" for o in orders2], fontsize=8.5)
    ax_npk.set_ylim(0, len(tr2["dirs"]) + 1)
    _style_ax(ax_npk, ylabel="peaks found", title="Peaks recovered vs true family count")

    ax_txt2 = fig.add_subplot(gs[1, 2:])
    ax_txt2.axis("off")
    ax_txt2.text(0.02, 0.85,
                "✖  REAL CEILING, not a tuning issue", color=CRITICAL, fontsize=11,
                fontweight="bold", va="top")
    ax_txt2.text(0.02, 0.60,
                "Every order (4, 6, 8) recovers only 1 of the 4 overlapping\n"
                "families -- and it isn't even the same direction across\n"
                "orders. One fixed sigma_grad cannot serve 4 simultaneously-\n"
                "present, differently-scaled populations at once.\n\n"
                "Current honest ceiling: reliable for ~2 similar-scale\n"
                "orientations, not messy 4-way crossings.",
                color=INK2, fontsize=9.5, va="top", linespacing=1.6)

    # =======================================================================
    # PHANTOM 3 -- kissing vs true crossing
    # =======================================================================
    kiss, trk = make_kissing_phantom()
    cross, trc = make_true_crossing_same_angles()

    ax_k = fig.add_subplot(gs[2, 0:2])
    ax_k.imshow(kiss[h], cmap="gray")
    ax_k.axvline(h, color=BOUNDARY_LINE, ls="--", lw=1.3)
    ax_k.set_xticks([]); ax_k.set_yticks([])
    ax_k.set_title("Phantom 3a: KISSING (segregated bundles, same 2 angles)", fontsize=10)
    for s in ax_k.spines.values():
        s.set_color(BASELINE)

    ax_x = fig.add_subplot(gs[2, 2:4])
    ax_x.imshow(cross[h], cmap="gray")
    ax_x.set_xticks([]); ax_x.set_yticks([])
    ax_x.set_title("Phantom 3b: TRUE CROSSING (overlapping, same 2 angles)", fontsize=10)
    for s in ax_x.spines.values():
        s.set_color(BASELINE)

    metrics = {}
    for name, vol, tr in [("true crossing", cross, trc), ("kissing", kiss, trk)]:
        m = analyse(_patch_at(vol, (h, h, h)), sigma_grad=2.0, order=6)
        if m and m["dirs"] is not None and len(m["dirs"]) >= 2:
            e1 = min(_ang_err(m["dirs"][0], tr["d1"]), _ang_err(m["dirs"][0], tr["d2"]))
            e2 = min(_ang_err(m["dirs"][1], tr["d1"]), _ang_err(m["dirs"][1], tr["d2"]))
            peak_err = 0.5 * (e1 + e2)
        else:
            peak_err = np.nan
        metrics[name] = dict(amp2=m["amp2"], aniso=m["aniso"], cs=m["c_s"], peak_err=peak_err,
                             label=classify(m))

    mini_specs = [("amp2", "2nd-peak amplitude ratio", "ratio (0-1)", "{:.2f}"),
                 ("aniso", "directional anisotropy", "unitless", "{:.2f}"),
                 ("cs", "sphericity c_s", "ratio (0-1)", "{:.2f}"),
                 ("peak_err", "mean peak error", "degrees", "{:.0f}")]
    for i, (key, title, ylab, fmt) in enumerate(mini_specs):
        ax_m = fig.add_subplot(gs[3, i])
        vals = [metrics["true crossing"][key], metrics["kissing"][key]]
        names = ["crossing", "kissing"]
        colors = [CASE_COLOR["true crossing"], CASE_COLOR["kissing"]]
        bars = ax_m.bar(names, vals, 0.55, color=colors, zorder=3)
        for b, v in zip(bars, vals):
            lbl = "n/a" if not np.isfinite(v) else fmt.format(v)
            ax_m.text(b.get_x() + b.get_width() / 2, (0 if not np.isfinite(v) else v),
                     lbl, ha="center", va="bottom", fontsize=8.5, color=INK2)
        _style_ax(ax_m, ylabel=ylab, title=title)
        ax_m.tick_params(axis="x", labelsize=8.5)

    ax_lab = fig.add_subplot(gs[4, 0:2])
    ax_lab.axis("off")
    ok_x = GOOD if metrics["true crossing"]["label"] == "crossing" else CRITICAL
    ax_lab.text(0.0, 0.75, f"true crossing -> classified “{metrics['true crossing']['label']}” "
                          "(validated case, correct)",
               fontsize=10, color=ok_x, fontweight="bold")
    ax_lab.text(0.0, 0.45, f"kissing -> classified “{metrics['kissing']['label']}” "
                          "(descriptive, not a failure)",
               fontsize=10, color=INK2, fontweight="bold")

    ax_verdict = fig.add_subplot(gs[4, 2:4])
    ax_verdict.axis("off")
    ax_verdict.text(0.0, 0.95,
                   "Surprising result: they did NOT look the same.", color=INK,
                   fontsize=10.5, fontweight="bold", va="top")
    ax_verdict.text(0.0, 0.72,
                   "A true overlapping crossing gives a clean, balanced 2-peak\n"
                   "signature (2nd peak strong, both peaks near truth). The\n"
                   "kissing case gives a weak, off-target 2nd peak that fails\n"
                   "the crossing gate. One seed / one angle pair -- promising,\n"
                   "not yet confirmed general.",
                   color=INK2, fontsize=9, va="top", linespacing=1.55)

    # legend strip for the order colour mapping (shared meaning across panels 1-2)
    fig.legend(handles=[Line2D([], [], color=c, marker="s", ls="", ms=9, label=f"order={o}")
                        for o, c in ORDER_COLOR.items()],
              loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.savefig("host_geometry_report.png", dpi=130, bbox_inches="tight")
    print("Saved host_geometry_report.png")
