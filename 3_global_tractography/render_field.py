"""
render_field.py -- greyscale slice with the recovered orientations drawn on top.

One figure per case, so a run can be checked by eye rather than only by reading
numbers: does the field follow the fibres, and are the junctions where the
structure actually crosses?

Two decisions worth stating, because both could reasonably have gone the other
way:

* **Line segments, not arrows.** A fibre orientation is a *director*: e3 and -e3
  describe the same fibre. An arrowhead would assert a direction the measurement
  does not contain, and would invite reading flow into a static field. Each glyph
  is drawn double-ended through its patch centre.

* **In-plane projection, honestly short.** A fibre running through the slice has
  little in-plane component, so its glyph is drawn short rather than rescaled to
  full length. A short glyph means "this fibre is leaving the plane", not "this
  measurement is weak" -- rescaling every glyph to the same length would hide
  exactly that distinction.

Colour carries the routing decision, which is the thing under test. The three
hues are the first three categorical slots of the project's reference palette,
used unmodified (they are documented as passing all-pairs colour-vision
validation in both light and dark). Colour is never the only encoding: junctions
also carry a ring marker, and the legend is always drawn.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# Categorical slots 1-3, used as-is.
C_SINGLE = "#2a78d6"      # gate trusted one direction
C_CROSSING = "#eb6834"    # junction scored CROSSING
C_TANGENCY = "#1baf7a"    # junction scored TANGENCY
C_OTHER = "#8a8a86"       # routed but not scored -- absence of a verdict
INK = "#0b0b0b"

VERDICT_COLOUR = {"CROSSING": C_CROSSING, "TANGENCY": C_TANGENCY}


def _background(vol, slab=3, lo=1.0, hi=99.5):
    """Max projection over a thin slab about the mid-plane, contrast-stretched.

    A thin slab rather than one slice: a single slice through a sparse phantom
    can miss the fibres entirely. A full projection through a real volume turns
    to mush. Percentile stretch because real tissue sits in a narrow band well
    away from 0-255.
    """
    z = vol.shape[0] // 2
    k = max(1, int(slab))
    img = vol[max(z - k, 0):min(z + k + 1, vol.shape[0])].max(axis=0)
    a, b = np.percentile(img, [lo, hi])
    if b <= a:
        b = a + 1e-6
    return np.clip((img - a) / (b - a), 0, 1)


def _glyph_colour(patch, junction_centres):
    """Colour + z-order for one patch, by what the pipeline decided about it."""
    c = tuple(patch["centre"])
    if c in junction_centres:
        return VERDICT_COLOUR.get(junction_centres[c], C_OTHER), 4
    if not patch["gated"]:
        return C_SINGLE, 2
    return C_OTHER, 3


def render_field(vol, field, junctions, path, title="", subtitle="",
                 slab=3, glyph=7.0, max_glyphs=1500):
    """Write a PNG: greyscale slab + one orientation glyph per patch.

    vol       (z, y, x) volume
    field     list of patches from S3.field_from_gated -- dirs in (z, y, x)
    junctions list of scored junctions (centre + verdict)
    path      output PNG path
    """
    img = _background(vol, slab=slab)
    jc = {tuple(j["centre"]): j["verdict"] for j in junctions}

    h, w = img.shape
    scale = max(w, h) / 48.0                      # keep glyphs readable at any size
    L = glyph * scale

    fig, ax = plt.subplots(figsize=(7.2, 7.2 * h / w + 0.9))
    ax.imshow(img, cmap="gray", origin="lower", interpolation="nearest",
              extent=[0, w, 0, h], vmin=0, vmax=1)

    patches = field
    if len(patches) > max_glyphs:                 # thin out, keep it legible
        step = int(np.ceil(len(patches) / max_glyphs))
        patches = patches[::step]

    for p in patches:
        colour, zo = _glyph_colour(p, jc)
        cz, cy, cx = p["centre"]
        for d in p["dirs"]:
            d = np.asarray(d, float)
            inplane = np.hypot(d[1], d[2])        # (dy, dx)
            if inplane < 1e-6:
                continue
            half = 0.5 * L * inplane              # short => leaving the plane
            ux, uy = d[2] / inplane, d[1] / inplane
            ax.plot([cx - half * ux, cx + half * ux],
                    [cy - half * uy, cy + half * uy],
                    color=colour, lw=1.8, solid_capstyle="round", zorder=zo,
                    path_effects=[pe.Stroke(linewidth=3.0, foreground="black",
                                            alpha=0.55), pe.Normal()])
        if tuple(p["centre"]) in jc:              # secondary encoding
            ax.plot(cx, cy, "o", mfc="none", mec=VERDICT_COLOUR.get(
                        jc[tuple(p["centre"])], C_OTHER),
                    mew=1.8, ms=11, zorder=5,
                    path_effects=[pe.Stroke(linewidth=3.2, foreground="black",
                                            alpha=0.55), pe.Normal()])

    present = {c for p in patches for c in [_glyph_colour(p, jc)[0]]}
    order = [(C_SINGLE, "single (one direction trusted)"),
             (C_CROSSING, "junction: CROSSING"),
             (C_TANGENCY, "junction: TANGENCY"),
             (C_OTHER, "routed, no verdict")]
    handles = [Line2D([0], [0], color=col, lw=2.2, label=lab)
               for col, lab in order if col in present]
    if len(handles) >= 2:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=min(len(handles), 2), frameon=False, fontsize=9,
                  labelcolor=INK)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # Title sits above the subtitle, which sits just above the image. `pad` has
    # to clear the subtitle line or the two overlap.
    if title:
        ax.set_title(title, fontsize=12, color=INK,
                     pad=26 if subtitle else 10, loc="left")
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=9,
                color="#52514e", va="bottom", ha="left")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
