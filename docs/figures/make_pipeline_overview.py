"""Figure: pipeline overview (README header).

A schematic of the four stages. Panels Input-2 are computed, not drawn by
hand: a 2D phantom of two crossing fibre bundles and the orientations a plain
2D structure tensor finds in it. Panel 3 is a line drawing of the two
junction types. Panel 4 plots the recorded Stage 4 sweep
(results/seed_rho_curves.json, 5,000-seed run).

Colour has one meaning throughout: blue = one orientation, orange = two.

Writes, next to this script:
    pipeline_overview.png       light theme (README, GitHub light mode)
    pipeline_overview_dark.png  dark theme  (README, GitHub dark mode)
    pipeline_overview.pdf       light theme, vector

    python docs/figures/make_pipeline_overview.py
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy.ndimage import gaussian_filter                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "results", "seed_rho_curves.json")
OUT = os.path.join(HERE, "pipeline_overview")

# Blue/orange pair validated for contrast and colour-vision deficiency
# (dataviz palette validator, both themes). Surfaces match GitHub's pages.
THEMES = {
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  box="#c3c2b7", one="#2a78d6", two="#eb6834",
                  mark="#ffffff"),
    "dark": dict(bg="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#8b8a84",
                 box="#3d444d", one="#3987e5", two="#d95926",
                 mark="#f0efec"),
}

FW, FH = 13.0, 3.5           # figure size (inches)
N = 128                      # phantom size (pixels)
ANG_A, ANG_B = 20.0, -45.0   # bundle directions (degrees from +x)


# --------------------------------------------------------------------------
# Phantom and structure tensor (2D, for illustration only)
# --------------------------------------------------------------------------
def bundle(yy, xx, ang_deg, offset, width=34.0, spacing=5.0, radius=1.2):
    a = np.deg2rad(ang_deg)
    n = np.array([-np.sin(a), np.cos(a)])
    u = (xx - N / 2) * n[0] + (yy - N / 2) * n[1] - offset
    d = (u + spacing / 2) % spacing - spacing / 2
    fibres = np.exp(-d ** 2 / (2 * radius ** 2))
    band = 1 / (1 + np.exp((np.abs(u) - width / 2) / 1.5))
    return fibres * band, band


def phantom():
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    a, band_a = bundle(yy, xx, ANG_A, offset=-6)
    b, band_b = bundle(yy, xx, ANG_B, offset=8)
    rng = np.random.default_rng(0)
    img = a + b + 0.12 + 0.18 * rng.standard_normal((N, N))
    return gaussian_filter(img, 0.6), band_a, band_b


def fibre_angle(img, sigma=1.0, rho=4.0):
    """Angle of the 2D structure tensor's smaller-eigenvalue eigenvector."""
    gy = gaussian_filter(img, sigma, order=(1, 0))
    gx = gaussian_filter(img, sigma, order=(0, 1))
    jxx = gaussian_filter(gx * gx, rho)
    jxy = gaussian_filter(gx * gy, rho)
    jyy = gaussian_filter(gy * gy, rho)
    return 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def panel_box(fig, t, x0, x1, y0, y1, header, title):
    fig.add_artist(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.012", fc="none",
        ec=t["box"], lw=1.1))
    fig.text(x0 + 0.010, y1, f" {header} ", fontsize=10, color=t["ink"],
             va="center", ha="left", backgroundcolor=t["bg"])
    fig.text((x0 + x1) / 2, y0 - 0.085, title, fontsize=12.5,
             fontweight="bold", color=t["ink"], ha="center", va="center")


def note(fig, t, x0, x1, y, s):
    fig.text((x0 + x1) / 2, y, s, fontsize=8.5, color=t["ink2"],
             ha="center", va="center")


def arrow(fig, t, x0, x1, y):
    fig.add_artist(FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=14, lw=1.4, color=t["ink2"]))


def image_axes(fig, rect, img, t, alpha=1.0):
    ax = fig.add_axes(rect)
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.6, origin="lower",
              interpolation="bilinear", alpha=alpha)
    ax.set_facecolor("#000000")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(t["box"])
        s.set_linewidth(0.8)
    return ax


def glyph(ax, x, y, ang, length, color, lw=2.2):
    # orientations are directors (v == -v): segments, never arrows
    dx, dy = 0.5 * length * np.cos(ang), 0.5 * length * np.sin(ang)
    ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=lw,
            solid_capstyle="round")


def side_key(fig, t, tx, y, swatch, label):
    """A label to the right of an image: swatch (callable) then text."""
    swatch(tx, y)
    fig.text(tx + 0.016, y, label, fontsize=8, color=t["ink2"], va="center",
             linespacing=1.1)


# --------------------------------------------------------------------------
def draw(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "figure.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "axes.facecolor": t["bg"], "pdf.fonttype": 42,
    })
    img, band_a, band_b = phantom()
    theta = fibre_angle(img)

    fig = plt.figure(figsize=(FW, FH))
    Y0, Y1 = 0.20, 0.93
    xs = [(0.008, 0.160), (0.186, 0.394), (0.420, 0.632), (0.658, 0.832),
          (0.858, 0.992)]
    heads = ["Input", "How wide to measure?", "Is one direction enough?",
             "Crossing or touching?", "Best single scale?"]
    titles = ["HiP-CT volume", "1. Scale selection", "2. Crossing gate",
              "3. Tractography", "4. Optimisation"]
    for (x0, x1), h, ti in zip(xs, heads, titles):
        panel_box(fig, t, x0, x1, Y0, Y1, h, ti)
    for (_, a), (b, _) in zip(xs[:-1], xs[1:]):
        arrow(fig, t, a + 0.003, b - 0.003, (Y0 + Y1) / 2)

    side = 0.50                       # image height, figure units
    w = side / (FW / FH)              # matching width -> square image
    yi = Y0 + 0.095                   # image bottom
    ynote = Y0 + 0.048
    y_hi, y_lo = yi + 0.40, yi + 0.13  # rows for the side keys

    # ---- Input -----------------------------------------------------------
    x0, x1 = xs[0]
    image_axes(fig, [(x0 + x1) / 2 - w / 2, yi, w, side], img, t)
    note(fig, t, x0, x1, ynote, "3D greyscale, µm voxels")

    # ---- 1. Scale selection ---------------------------------------------
    x0, x1 = xs[1]
    ax = image_axes(fig, [x0 + 0.010, yi, w, side], img, t)
    ax.add_patch(Circle((64, 64), 16, fc="none", ec=t["mark"], lw=1.6,
                        ls=(0, (3, 2))))
    ax.add_patch(Circle((64, 64), 3, fc=t["mark"], ec="none"))
    tx = x0 + 0.010 + w + 0.010
    side_key(fig, t, tx, y_hi,
             lambda x, y: fig.text(x, y, r"$\sigma$", fontsize=14,
                                   color=t["ink"], va="center"),
             "derivative\nscale")
    side_key(fig, t, tx, y_lo,
             lambda x, y: fig.text(x, y, r"$\rho$", fontsize=14,
                                   color=t["ink"], va="center"),
             "averaging\nwindow")
    note(fig, t, x0, x1, ynote, "set from the data, per 32³ block")

    # ---- 2. Crossing gate ----------------------------------------------
    x0, x1 = xs[2]
    ax = image_axes(fig, [x0 + 0.010, yi, w, side], img, t, alpha=0.35)
    step = 11
    for y in range(step // 2 + 2, N - 2, step):
        for x in range(step // 2 + 2, N - 2, step):
            ina, inb = band_a[y, x] > 0.5, band_b[y, x] > 0.5
            if ina and inb:      # routed to the higher-order tensor
                glyph(ax, x, y, np.deg2rad(ANG_A), 8.0, t["two"])
                glyph(ax, x, y, np.deg2rad(ANG_B), 8.0, t["two"])
            elif ina or inb:     # plain structure tensor
                glyph(ax, x, y, theta[y, x], 8.5, t["one"])
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    tx = x0 + 0.010 + w + 0.010

    def line_swatch(col):
        return lambda x, y: fig.add_artist(Line2D(
            [x, x + 0.010], [y, y], transform=fig.transFigure, color=col,
            lw=2.4, solid_capstyle="round"))
    side_key(fig, t, tx, y_hi, line_swatch(t["one"]), "one\ndirection")
    side_key(fig, t, tx, y_lo, line_swatch(t["two"]), "two\ndirections")
    note(fig, t, x0, x1, ynote, r"HOST only where $c_s$ > FA")

    # ---- 3. Tractography ----------------------------------------------
    x0, x1 = xs[3]
    pw = (x1 - x0 - 0.030) / 2
    s = np.linspace(-1.1, 1.1, 60)
    for k, kind in enumerate(("cross", "touch")):
        ax = fig.add_axes([x0 + 0.010 + k * (pw + 0.010), yi + 0.03, pw,
                           side - 0.03])
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.45, 1.35)
        ax.axis("off")
        if kind == "cross":
            ax.plot(s, 0.55 * s, color=t["ink2"], lw=2.2)
            ax.plot(s, -0.55 * s, color=t["ink2"], lw=2.2)
            lab, sval, ring = "crossing", r"$S \approx 1$", t["two"]
        else:
            ax.plot(s, 0.13 + 0.45 * s ** 2, color=t["ink2"], lw=2.2)
            ax.plot(s, -0.13 - 0.45 * s ** 2, color=t["ink2"], lw=2.2)
            lab, sval, ring = "touching", r"$S \approx 0.5$", t["one"]
        ax.plot(0, 0, "o", ms=12, mfc="none", mec=ring, mew=2.4)
        ax.text(0, -1.0, lab, ha="center", va="top", fontsize=9,
                color=t["ink"])
        ax.text(0, 1.05, sval, ha="center", va="bottom", fontsize=11,
                color=t["ink"])
    note(fig, t, x0, x1, ynote, "brightness test + global tracking")

    # ---- 4. Optimisation -----------------------------------------------
    x0, x1 = xs[4]
    with open(DATA) as f:
        d = json.load(f)
    rhos = np.array(d["rhos"])
    err = np.array(d["res"]["5000"]["median"])
    j = int(np.argmin(err))
    ax = fig.add_axes([x0 + 0.022, yi + 0.06, x1 - x0 - 0.034, side - 0.08])
    ax.plot(rhos, err, color=t["one"], lw=2)
    ax.plot(rhos[j], err[j], "o", ms=8, mfc=t["one"], mec=t["bg"], mew=1.2)
    ax.annotate(rf"$\rho^* = {rhos[j]:.2f}$", (rhos[j], err[j]),
                xytext=(0, -9), textcoords="offset points", ha="center",
                va="top", fontsize=11, color=t["ink"])
    ax.set_xscale("log")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(t["muted"])
    ax.set_xlabel(r"$\rho$", fontsize=12, color=t["ink2"], labelpad=1)
    ax.set_ylabel("angular error", fontsize=8, color=t["ink2"], labelpad=3)
    ax.set_ylim(err.min() - 11, err.max() + 2)
    note(fig, t, x0, x1, ynote, "one (σ, ρ) per chunk")
    return fig


def main():
    for theme in ("light", "dark"):
        fig = draw(theme)
        suffix = "" if theme == "light" else "_dark"
        fig.savefig(f"{OUT}{suffix}.png", dpi=250)
        if theme == "light":
            fig.savefig(f"{OUT}.pdf")
        plt.close(fig)
    print(f"wrote {OUT}.png, {OUT}_dark.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
