"""
refined_field.py -- improvements #1 (spatial regularisation) and #2 (denoising
+ multi-scale persistence) on top of the HOST-FOD triage pipeline.

The i74 orientation overlay showed the CROSSING tier is over-called: spurious
second peaks (noise-driven, appearing only at large sigma, and spatially
isolated) get reshaped into plausible-looking crossings. Two consistency filters
attack that, each making a peak EARN its place:

  #2 MULTI-SCALE PERSISTENCE. A real fibre peak recurs across a range of
     gradient scales; a noise-driven split appears at only one (usually the
     largest) sigma. We run HOST analyse() at every sigma, cluster the peaks by
     direction, and keep a peak only if its cluster spans >= min_scales distinct
     scales. Plus TV denoising of the volume first (edge-preserving) to cut the
     noise that drives the splitting in the first place.

  #1 SPATIAL REGULARISATION. A real orientation is spatially coherent; an
     isolated peak with no neighbour agreement is untrustworthy. Over a grid of
     patches we keep each surviving (persistent) direction only if enough
     neighbouring patches carry a similar direction, then RE-DERIVE the tier
     from the survivors (2 -> crossing, 1 -> single, 0 -> unresolvable). This is
     what demotes the spurious second arms the overlay exposed.

All per-scale analyse() results are computed ONCE per patch, so the same pass
yields both the BEFORE field (single-sigma v2 pick) and the AFTER field
(persistent + spatially-regularised) for an honest side-by-side.

Reuses analyse (host_fod), load_interior (selfcalib_threshold). Validated on
synthetic (genuine crossings must SURVIVE both filters) before real data.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root (host_fod, selfcalib_threshold)
warnings.filterwarnings("ignore")

import numpy as np

from host_fod import analyse

EPS = 1e-12
SIGMAS = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])   # dense, for persistence


def _ang(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0, 1))))


def _mean_axis(dirs):
    M = np.zeros((3, 3))
    for d in dirs:
        M += np.outer(d, d)
    _l, v = np.linalg.eigh(M)
    return v[:, -1]


# ---------------------------------------------------------------------------
# per-patch: all scales once, then multi-scale-persistent peaks (#2)
# ---------------------------------------------------------------------------

def analyse_all_scales(patch, sigmas=SIGMAS, order=6):
    out = {}
    for sg in sigmas:
        out[float(sg)] = analyse(patch, sigma_grad=float(sg), order=order)
    return out


def persistent_peaks(scale_ms, ang_tol=22.0, min_scales=2):
    """Cluster peaks across scales; keep clusters spanning >= min_scales sigmas."""
    valid = {sg: m for sg, m in scale_ms.items() if m is not None}
    if not valid:
        return None
    clusters = []          # each: dict(axis, members[list of dir], sigmas set)
    for sg, m in sorted(valid.items()):
        for d in m["dirs"]:
            best, best_a = None, ang_tol
            for c in clusters:
                a = _ang(d, c["axis"])
                if a < best_a:
                    best, best_a = c, a
            if best is None:
                clusters.append(dict(axis=np.asarray(d, float), members=[np.asarray(d, float)],
                                     sigmas={sg}))
            else:
                best["members"].append(np.asarray(d, float))
                best["sigmas"].add(sg)
                best["axis"] = _mean_axis(best["members"])
    persist = [c for c in clusters if len(c["sigmas"]) >= min_scales]
    persist.sort(key=lambda c: -len(c["members"]))
    cs = float(np.median([m["c_s"] for m in valid.values()]))
    aniso = float(np.median([m["aniso"] for m in valid.values()]))   # median: robust to one noisy scale
    return dict(dirs=[c["axis"] for c in persist],
                support=[len(c["members"]) for c in persist],
                scales=[len(c["sigmas"]) for c in persist], c_s=cs, aniso=aniso)


def best_single_scale(scale_ms, amp2_min=0.4, sep_min=30.0, cs_single=0.15):
    """BEFORE field: the v2 single-sigma pick (max amp2 among 2-peak scales)."""
    best, best_q = None, -np.inf
    for sg, m in scale_ms.items():
        q = m["amp2"] if (m is not None and m["n_peaks"] == 2 and np.isfinite(m["amp2"])) else -np.inf
        if q > best_q:
            best, best_q = m, q
    if best is not None and best_q >= amp2_min and best["sep"] >= sep_min:
        return "crossing", list(best["dirs"][:2])
    # fall back to a mid-scale single/dispersed characterisation
    mid = scale_ms.get(3.0) or next((m for m in scale_ms.values() if m is not None), None)
    if mid is None:
        return "n/a", []
    if mid["aniso"] < 0.9:
        return "disorganised", []
    lab = "single" if mid["c_s"] <= cs_single else "dispersed"
    return lab, ([mid["dirs"][0]] if len(mid["dirs"]) else [])


def classify_persistent(res, cs_single=0.15, aniso_iso=0.9, sep_min=30.0):
    """AFTER field, pre-spatial: tier from the persistent peaks."""
    if res is None:
        return "n/a", []
    if res["aniso"] < aniso_iso:
        return "disorganised", []
    dirs = res["dirs"]
    if len(dirs) >= 2 and _ang(dirs[0], dirs[1]) >= sep_min:
        return "crossing", [dirs[0], dirs[1]]
    lab = "single" if res["c_s"] <= cs_single else "dispersed"
    return lab, ([dirs[0]] if len(dirs) else [])


# ---------------------------------------------------------------------------
# #1 spatial regularisation across a grid of patches
# ---------------------------------------------------------------------------

def spatial_regularize(field, ang_tol=25.0, min_support=2):
    """field: dict[(gi,gj)] -> dict(label, dirs, cy, cx), keyed by GRID INDEX
    (not voxel coords) so (gi+-1, gj+-1) are real neighbours. Keep each direction
    only if >= min_support of the 8 neighbours carry a similar direction, then
    re-derive the tier from survivors."""
    keys = set(field.keys())
    out = {}
    for (gi, gj), cell in field.items():
        neigh = [field[(gi + di, gj + dj)] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                 if (di or dj) and (gi + di, gj + dj) in keys]
        survivors = []
        for d in cell["dirs"]:
            support = sum(1 for nb in neigh
                          if any(_ang(d, nd) <= ang_tol for nd in nb["dirs"]))
            if support >= min_support:
                survivors.append(d)
        if len(survivors) >= 2:
            lab = "crossing"
        elif len(survivors) == 1:
            lab = "single" if cell["label"] in ("single", "crossing") else cell["label"]
        else:
            lab = "unresolvable"
        out[(gi, gj)] = dict(label=lab, dirs=survivors, cy=cell["cy"], cx=cell["cx"],
                             c_s=cell.get("c_s", np.nan))
    return out


# ---------------------------------------------------------------------------
# synthetic validation: genuine crossings must SURVIVE persistence
# ---------------------------------------------------------------------------

def validate(noise_sigma=0.3, seeds=(0, 1, 2)):
    from crossing_shape_3d import (make_single_fibre_3d, make_crossing_phantom_3d,
                                   _noise_volume)
    from two_orientation_test import _central_patch

    print("=" * 74)
    print("MULTI-SCALE PERSISTENCE validation (do genuine crossings survive?)")
    cases = [("single", "single"), ("cross_50", "crossing"), ("cross_70", "crossing"),
             ("cross_90", "crossing"), ("noise", "disorganised")]
    ok = 0
    for kind, truth in cases:
        labs = []
        for s in seeds:
            if kind == "single":
                v, _ = make_single_fibre_3d(theta_deg=20.0, noise_sigma=noise_sigma, seed=s)
            elif kind == "noise":
                v = _noise_volume((96, 96, 96), seed=s)
            else:
                v, _ = make_crossing_phantom_3d(sep_deg=float(kind.split("_")[1]),
                                                noise_sigma=noise_sigma, seed=s, spacing2=12.0)
            res = persistent_peaks(analyse_all_scales(_central_patch(v, 18)))
            labs.append(classify_persistent(res)[0])
        lab = max(set(labs), key=labs.count)
        ok += (lab == truth)
        dbg = "-" if res is None else f"npk={len(res['dirs'])} c_s={res['c_s']:.2f} aniso={res['aniso']:.2f}"
        print(f"  {kind:>10} -> {lab:12s} (truth {truth}) {'OK' if lab == truth else 'XX'}   [{dbg}]")
    print(f"  {ok}/{len(cases)} correct")
    print("=" * 74)


# ---------------------------------------------------------------------------
# i74 before/after: single-sigma field vs persistent + spatially-regularised
# ---------------------------------------------------------------------------

C_SINGLE, C_CROSS, C_DISP, INK = "#2a78d6", "#eb6834", "#898781", "#0b0b0b"
HALF, Z_SLICE, STEP, SEG = 20, 100, 28, 12.0


def _draw(ax, cx, cy, d, color, lw=2.2):
    dx, dy = float(d[0]), float(d[1])
    ip = np.hypot(dx, dy)
    if ip < 1e-3:
        ax.plot(cx, cy, ".", color=color, ms=4); return
    ux, uy, L = dx / ip, dy / ip, SEG * ip
    ax.plot([cx - L * ux, cx + L * ux], [cy - L * uy, cy + L * uy], "-",
            color=color, lw=lw, solid_capstyle="round")


def _panel(ax, sl, field, title):
    ax.imshow(sl, cmap="gray", origin="upper"); ax.set_xticks([]); ax.set_yticks([])
    c = {"single": 0, "crossing": 0, "other": 0}
    for _key, cell in field.items():
        cy, cx = cell["cy"], cell["cx"]
        lab = cell["label"]
        if lab == "single" and cell["dirs"]:
            _draw(ax, cx, cy, cell["dirs"][0], C_SINGLE); c["single"] += 1
        elif lab == "crossing" and len(cell["dirs"]) >= 2:
            _draw(ax, cx, cy, cell["dirs"][0], C_CROSS); _draw(ax, cx, cy, cell["dirs"][1], C_CROSS)
            c["crossing"] += 1
        else:
            ax.plot(cx, cy, "o", color=C_DISP, ms=4, alpha=0.7); c["other"] += 1
    ax.set_title(f"{title}\nsingle {c['single']}, crossing {c['crossing']}, "
                 f"unresolvable {c['other']}", fontsize=11, color=INK)
    return c


def run_i74(save=None, denoise="none"):
    """denoise: 'none' (TV was found to over-smooth/erase striations) or 'tv'."""
    import time
    from selfcalib_threshold import load_interior
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if save is None:
        save = f"i74_refined_orientations_{denoise}.png"
    vol = load_interior()
    sub = vol[Z_SLICE - HALF:Z_SLICE + HALF + 1].astype(np.float32)
    if denoise == "tv":
        from skimage.restoration import denoise_tv_chambolle
        sub = denoise_tv_chambolle(sub / 255.0, weight=0.08) * 255.0
    sl = sub[HALF]
    centers = list(range(HALF + 10, vol.shape[1] - HALF - 5, STEP))
    grid = [(cy, cx) for cy in centers for cx in centers]
    print(f"i74 slice z={Z_SLICE}, {len(grid)} patches, denoise={denoise}, sigma grid={SIGMAS.tolist()}")

    from reconstruction import iter_progress
    cells = [(gi, cy, gj, cx) for gi, cy in enumerate(centers)
             for gj, cx in enumerate(centers)]
    before, after0 = {}, {}
    t0 = time.time()
    for gi, cy, gj, cx in iter_progress(cells, desc="i74 patches"):
        patch = sub[:, cy - HALF:cy + HALF + 1, cx - HALF:cx + HALF + 1]
        scale_ms = analyse_all_scales(patch)                   # ONE pass, both fields
        lb, db = best_single_scale(scale_ms)                   # BEFORE (single-sigma v2)
        res = persistent_peaks(scale_ms)                       # #2 persistence
        la, da = classify_persistent(res)
        before[(gi, gj)] = dict(label=lb, dirs=db, cy=cy, cx=cx)
        after0[(gi, gj)] = dict(label=la, dirs=da, cy=cy, cx=cx,
                                c_s=(np.nan if res is None else res["c_s"]))
    after = spatial_regularize(after0)                             # #1 spatial

    fig, axes = plt.subplots(1, 2, figsize=(21, 11))
    cb = _panel(axes[0], sl, before, "BEFORE: single-sigma HOST-FOD")
    after_desc = "multi-scale + spatial" + (" + TV" if denoise == "tv" else " (no denoise)")
    ca = _panel(axes[1], sl, after, f"AFTER: {after_desc}")
    fig.legend(handles=[
        Line2D([], [], color=C_SINGLE, lw=3, label="single (1 direction)"),
        Line2D([], [], color=C_CROSS, lw=3, label="crossing (2 directions)"),
        Line2D([], [], color=C_DISP, marker="o", ls="", label="unresolvable")],
        loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"i74 orientation field -- multi-scale persistence + spatial "
                 f"regularisation (denoise={denoise})", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(save, dpi=130)
    print(f"\n({time.time()-t0:.0f}s)  BEFORE {cb}  ->  AFTER {ca}")
    print(f"Saved {save}")


# ---------------------------------------------------------------------------
# lambda3 HOST-gate: run expensive HOST only where lambda3 (= sphericity x
# energy = smallest structure-tensor eigenvalue) is high; single-e3 elsewhere.
# ---------------------------------------------------------------------------

def cheap_gate(patch, sigma_ref=3.0):
    """Cheap 2nd-order read: (lambda3, e3) from one gradient pass. lambda3 =
    smallest eigenvalue = energy in the least-gradient direction (the HOST
    trigger); e3 = its eigenvector = single-fibre direction, in the (axis2,
    axis1, axis0) basis the glyph drawing uses."""
    from host_fod import patch_gradients
    G = patch_gradients(patch, sigma_ref)
    if G.shape[0] < 40:
        return 0.0, None
    S2 = G.T @ G
    lam, vec = np.linalg.eigh(S2)                 # ascending
    e3 = vec[:, 0] / (np.linalg.norm(vec[:, 0]) + EPS)
    return float(lam[0]), e3


def run_i74_gated(threshold_pct=55, save="i74_lambda3_gated.png"):
    import time
    from selfcalib_threshold import load_interior
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    vol = load_interior()
    sub = vol[Z_SLICE - HALF:Z_SLICE + HALF + 1].astype(np.float32)   # no denoise
    # normalise the sub-volume to [0,1] the SAME way make_plus normalises its
    # phantoms, so the ABSOLUTE lambda3 gate threshold fitted in gate_protocol is
    # on a comparable scale (raw i74 is 0-255 -> lambda3 ~1e4x larger otherwise).
    sub = (sub - sub.min()) / (np.ptp(sub) + EPS)
    sl = sub[HALF]
    centers = list(range(HALF + 10, vol.shape[1] - HALF - 5, STEP))
    grid = [(gi, gj, cy, cx) for gi, cy in enumerate(centers)
            for gj, cx in enumerate(centers)]
    print(f"i74 slice z={Z_SLICE}, {len(grid)} patches, lambda3 HOST-gate")

    from reconstruction import iter_progress
    rows = []          # per patch: lam3, e3, HOST label/dirs, cy, cx, keys
    t0 = time.time()
    t_host = 0.0
    for (gi, gj, cy, cx) in iter_progress(grid, desc="lambda3-gate patches"):
        patch = sub[:, cy - HALF:cy + HALF + 1, cx - HALF:cx + HALF + 1]
        lam3, e3 = cheap_gate(patch)
        th = time.time()
        res = persistent_peaks(analyse_all_scales(patch))            # HOST (everywhere, for comparison)
        t_host += time.time() - th
        hlabel, hdirs = classify_persistent(res)
        rows.append(dict(gi=gi, gj=gj, cy=cy, cx=cx, lam3=lam3, e3=e3,
                         hlabel=hlabel, hdirs=hdirs))

    lam3s = np.array([r["lam3"] for r in rows])
    # OPERATING POINT from the validated high-recall fit (gate_protocol.fit_lam3_gate),
    # replacing the old ad-hoc percentile-55 rule that skipped real crossings.
    #
    # The fit's ABSOLUTE lambda3 does NOT transfer across datasets: real i74
    # tissue has lower normalised gradient energy than the crisp synthetic tubes,
    # so the absolute value lands off the end of the i74 distribution. What DOES
    # transfer is the operating point as a FIRE-FRACTION (rank-based, hence
    # contrast-invariant): the fit fires ~65% of patches (= percentile-35), and
    # we apply that same fire-fraction here. Erring toward firing is the safe
    # direction (a missed crossing costs more than a wasted HOST call).
    from gate_protocol import load_gate
    gate = load_gate()
    fire_frac_target = float(gate["train_fire_fraction"])
    gate_pct = 100.0 * (1.0 - fire_frac_target)
    thr = float(np.percentile(lam3s, gate_pct))
    fire_frac = float((lam3s > thr).mean())
    abs_thr = float(gate["lam3_threshold"])
    abs_pct = float((lam3s <= abs_thr).mean() * 100.0)
    print("=" * 68)
    print(f"FITTED lambda3 gate (high-recall, from gate_threshold.json)")
    print(f"  fit on phantoms: train catch {gate['train_catch']:.0%}, "
          f"fires {fire_frac_target:.0%} of patches -> percentile-{gate_pct:.0f} rule")
    print(f"  applied to i74: thr={thr:.4g}, fires HOST on {fire_frac:.0%} of patches")
    print(f"  (diagnostic: the phantom ABSOLUTE thr={abs_thr:.4g} lands at the "
          f"{abs_pct:.0f}th i74 percentile -> absolute lambda3 does NOT transfer, "
          f"so the fire-fraction is used instead)")
    print("=" * 68)

    # build both fields (keyed by grid index for spatial_regularize)
    everywhere, gated = {}, {}
    for r in rows:
        key = (r["gi"], r["gj"])
        everywhere[key] = dict(label=r["hlabel"], dirs=r["hdirs"], cy=r["cy"], cx=r["cx"])
        if r["lam3"] > thr:
            gated[key] = dict(label=r["hlabel"], dirs=r["hdirs"], cy=r["cy"], cx=r["cx"])
        else:                                    # skip HOST -> cheap single-e3
            gated[key] = dict(label="single", dirs=([r["e3"]] if r["e3"] is not None else []),
                              cy=r["cy"], cx=r["cx"])
    everywhere = spatial_regularize(everywhere)
    gated = spatial_regularize(gated)

    # ---- threshold sweep (free: reuse the stored lam3 + HOST labels) ----
    print("=" * 68)
    print("lambda3 HOST-gate -- threshold sweep (HOST run everywhere for truth)")
    print(f"  {'pct':>4} {'thr':>10} {'HOST-run':>9} {'skipped':>8} {'crossings MISSED':>16}")
    n_cross_total = sum(1 for r in rows if r["hlabel"] == "crossing")
    for pct in (40, 50, 55, 60, 70):
        t = np.percentile(lam3s, pct)
        run = [r for r in rows if r["lam3"] > t]
        skip = [r for r in rows if r["lam3"] <= t]
        missed = sum(1 for r in skip if r["hlabel"] == "crossing")
        print(f"  {pct:>4} {t:10.4g} {len(run):>9} {len(skip):>8} "
              f"{missed:>10}/{n_cross_total} crossings")
    # the fitted operating point in the same terms (its missed count on i74's
    # HOST-everywhere "truth" -- the number that matters for the gate)
    run_fit = [r for r in rows if r["lam3"] > thr]
    skip_fit = [r for r in rows if r["lam3"] <= thr]
    missed_fit = sum(1 for r in skip_fit if r["hlabel"] == "crossing")
    print(f"  {'FIT':>4} {thr:10.4g} {len(run_fit):>9} {len(skip_fit):>8} "
          f"{missed_fit:>10}/{n_cross_total} crossings   <-- fitted high-recall gate")
    print(f"  (per-patch HOST cost ~{t_host/len(rows):.1f}s; gate saves that on every skipped patch)")
    print("=" * 68)

    fig, axes = plt.subplots(1, 2, figsize=(21, 11))
    cb = _panel(axes[0], sl, everywhere, "HOST everywhere")
    frac_skip = np.mean(lam3s <= thr)
    ca = _panel(axes[1], sl, gated,
                f"lambda3-gated (fitted high-recall thr={thr:.3g}): HOST on "
                f"{int((1-frac_skip)*len(rows))}/{len(rows)}, single-e3 on "
                f"{int(frac_skip*len(rows))}")
    fig.legend(handles=[
        Line2D([], [], color=C_SINGLE, lw=3, label="single (1 direction)"),
        Line2D([], [], color=C_CROSS, lw=3, label="crossing (2 directions)"),
        Line2D([], [], color=C_DISP, marker="o", ls="", label="unresolvable")],
        loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"i74: HOST-everywhere vs lambda3-gated HOST (saves HOST on "
                 f"{frac_skip:.0%} of patches)", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(save, dpi=130)
    print(f"\n({time.time()-t0:.0f}s)  HOST-everywhere {cb}  ->  gated {ca}")
    print(f"Saved {save}")


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if stage == "validate":
        validate()
    elif stage == "i74":
        run_i74(denoise=(sys.argv[2] if len(sys.argv) > 2 else "none"))
    elif stage == "gated":
        run_i74_gated()
