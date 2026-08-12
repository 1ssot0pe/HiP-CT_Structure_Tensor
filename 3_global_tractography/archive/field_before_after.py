"""
field_before_after.py -- render the Stage 3 vector field before and after.

`run_case` already writes the dense fields for both states (`vec1`, `vec2`,
`labels` for `<tag>_initial` and `<tag>_optimised`). This turns that pair into
a picture: the primary orientation as a quiver over the image, the SECOND
orientation drawn only where one survives, and the demotion diff underneath.

Reading the exported NIfTI rather than re-running the pipeline keeps the figure
honest -- it shows exactly the artifact that was written to disk, not a fresh
computation that might differ.

What the phantoms are expected to show, and why "no change" is the pass
----------------------------------------------------------------------
Neither pipeline phantom produces a demotion, and that IS the correct result:

  * TRUE CROSSING -- the gate routes 5 of 27 patches, HOST finds one
    two-orientation junction, S = 1.03, verdict CROSSING. Both orientations
    must SURVIVE. A demotion here would be a false negative.
  * TANGENCY (kissing) -- the gate routes 0 patches, so there is no junction to
    demote. The pipeline declines to invent a crossing, which is the whole
    behaviour under test.

So on phantoms the before/after pair is a REGRESSION check: any difference is a
bug. The pair only carries information where HOST over-calls, which on i74 is
59 of 67 junctions -- run with --tag i74 to see a non-empty diff.

Run
---
    python field_before_after.py --outdir <dir>            # both phantoms
    python field_before_after.py --tag i74 --outdir <dir>
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    sys.path.insert(0, _p)

from field_export import CODE_LABELS, LABEL_CODES              # noqa: E402

LABEL_COLOURS = {
    "single":       "#9aa0a6",
    "crossing":     "#ff4d4d",
    "kissing":      "#4dd2ff",
    "dispersed":    "#ffd24d",
    "disorganised": "#c77dff",
}


def _load(outdir, tag, state):
    import nibabel as nib
    out = {}
    for key in ("vec1", "vec2", "labels", "ndirs"):
        p = os.path.join(outdir, f"{key}_{tag}_{state}.nii.gz")
        if not os.path.exists(p):
            return None
        a = np.asanyarray(nib.load(p).dataobj)
        # written (x, y, z[, 3]) with (x,y,z) components; bring back to the
        # (z, y, x) indexing everything in this folder uses
        if a.ndim == 4:
            a = a.transpose(2, 1, 0, 3)[..., ::-1]
        else:
            a = a.transpose(2, 1, 0)
        out[key] = a
    p = os.path.join(outdir, f"raw_{tag}.nii.gz")
    if os.path.exists(p):
        out["raw"] = np.asanyarray(nib.load(p).dataobj).transpose(2, 1, 0)
    return out


def _best_slice(labels):
    """The z with the most non-background labelled voxels -- the slice where
    there is actually something to look at, rather than the middle by default."""
    counts = [(labels[z] > 0).sum() for z in range(labels.shape[0])]
    return int(np.argmax(counts)) if max(counts) > 0 else labels.shape[0] // 2


def _quiver(ax, F, z, stride, scale, second=True):
    v1, v2 = F["vec1"][z], F["vec2"][z]
    lab = F["labels"][z]
    ys, xs = np.mgrid[0:v1.shape[0]:stride, 0:v1.shape[1]:stride]
    for vec, col, w in ((v1, "#ff4d4d", 0.0035),
                        (v2, "#4dd2ff", 0.0035) if second else (None, None, 0)):
        if vec is None:
            continue
        dy = vec[ys, xs, 1] * scale
        dx = vec[ys, xs, 2] * scale
        keep = (dy ** 2 + dx ** 2) > 0
        if not keep.any():
            continue
        # nematic: draw both ways so no spurious head/tail is implied
        for s in (1, -1):
            ax.quiver(xs[keep], ys[keep], s * dx[keep], s * dy[keep],
                      color=col, angles="xy", scale_units="xy", scale=1,
                      width=w, headwidth=0, headlength=0, headaxislength=0,
                      zorder=3)
    return lab


def plot(outdir, tag, path=None, stride=3, scale=2.6, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    A = _load(outdir, tag, "initial")
    B = _load(outdir, tag, "optimised")
    if A is None or B is None:
        print(f"  [{tag}] exports not found in {outdir} -- run the pipeline first")
        return None

    z = _best_slice(A["labels"])
    bg = A.get("raw")
    n_dem = int(((A["ndirs"] >= 2) & (B["ndirs"] < A["ndirs"])).sum())

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), sharex=True, sharey=True)
    for ax, (F, name) in zip(axes[:2], ((A, "BEFORE  (Stage 2 hand-over)"),
                                        (B, "AFTER  (Stage 3 verdicts applied)"))):
        if bg is not None:
            ax.imshow(bg[z], cmap="gray", origin="lower", aspect="equal")
        _quiver(ax, F, z, stride, scale)
        present = sorted(set(np.unique(F["labels"][z])) - {0})
        note = ", ".join(f"{CODE_LABELS[c]} {int((F['labels'][z] == c).sum())}"
                         for c in present) or "nothing labelled"
        ax.set_title(f"{name}\n{note}", fontsize=9, loc="left")
        ax.set_xlabel("x (voxels)")
    axes[0].set_ylabel("y (voxels)")

    # --- what changed -------------------------------------------------------
    ax = axes[2]
    if bg is not None:
        ax.imshow(bg[z], cmap="gray", origin="lower", aspect="equal", alpha=0.55)
    dem = ((A["ndirs"] >= 2) & (B["ndirs"] < A["ndirs"]))[z]
    if dem.any():
        m = np.ma.masked_where(~dem, dem.astype(float))
        ax.imshow(m, cmap="cool", origin="lower", aspect="equal", alpha=0.85,
                  vmin=0, vmax=1)
        ax.set_title(f"DEMOTED  (2 orientations -> 1)\n{int(dem.sum())} voxels "
                     f"in this slice, {n_dem} in the volume", fontsize=9,
                     loc="left")
    else:
        ax.set_title("DEMOTED\nnothing -- on a phantom this is the PASS "
                     "condition", fontsize=9, loc="left")
    ax.set_xlabel("x (voxels)")

    handles = [Patch(color="#ff4d4d", label="primary orientation (vec1)"),
               Patch(color="#4dd2ff", label="second orientation (vec2)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9)
    fig.suptitle(title or f"Stage 3 vector field, before and after -- {tag} "
                          f"(slice z={z})", fontsize=11, y=0.99)
    # rect leaves room top AND bottom: the per-axis titles are two lines each,
    # so a full-height layout puts the suptitle straight through the middle one.
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    path = path or os.path.join(outdir, f"before_after_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [{tag}] slice z={z}, {n_dem} demoted voxels -> {path}")
    return path


def main(outdir, tags=("true", "tangency"), stride=3):
    print("=" * 78)
    print("STAGE 3 VECTOR FIELD -- before vs after")
    print("=" * 78)
    made = []
    for t in tags:
        p = plot(outdir, t, stride=stride)
        if p:
            made.append(p)
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="before/after Stage 3 vector field")
    ap.add_argument("--outdir", default=os.path.join(_ROOT, "results"))
    ap.add_argument("--tag", default=None, help="one tag; default both phantoms")
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    main(a.outdir, tags=(a.tag,) if a.tag else ("true", "tangency"),
         stride=a.stride)
