"""
run_dense_stage12.py -- Stages 1 and 2, per voxel, end to end.

Stage 1   select (sigma, rho)  ->  one dense structure tensor
Stage 2   gate on lambda3/lambda2  ->  HOST's ODF at the firing voxels only

What is different from the patch pipeline
-----------------------------------------
DETECTION AND SEPARATION ARE NOW SEPARATE QUESTIONS, because measurement says
they have different limits. At the selected scale, on the validated phantoms:

    case        lambda3/lambda2   c_s      FA      dense peaks (order 8)
    single           0.0000     0.0000   0.7071          1
    cross 40         0.0241     0.0309   0.7057          1
    cross 50         0.0618     0.0716   0.7039          2  (45 deg)
    cross 70         0.2774     0.2356   0.6982          2  (69 deg)
    cross 90         0.9991     0.5127   0.6948          2  (88 deg)

lambda3/lambda2 rises monotonically from exactly zero, so a 40 degree crossing
IS detected. Resolving it into two directions is a harder problem and the ODF
cannot do it below ~50 degrees at order 8 (~70 at order 6). A voxel that the
gate fires on but whose ODF yields one peak is therefore reported as
UNRESOLVED -- a crossing seen but not decomposed -- rather than silently
relabelled single.

That table also explains the rule this replaces. FA is essentially FLAT at ~0.70
across the whole range while c_s reaches only 0.51 even at 90 degrees, so
`c_s > FA` is unsatisfiable for a genuine crossing. It can fire only where FA
collapses -- degenerate voxels, or the boundary-replicated gradients of an
isolated patch. The old gate was firing on the artifact, not the structure.

Run
---
    python run_dense_stage12.py --radius 24            # 48^3 i74 cube
    python run_dense_stage12.py --radius 24 --fire 0.10
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection")):
    sys.path.insert(0, _p)

from core.reconstruction import load_volume, tissue_mask, VOXEL_SIZE_UM   # noqa: E402
from core.nifti_io import nifti_affine, save_nifti, save_vector_field     # noqa: E402
from core.tensor import field_for                                        # noqa: E402
from select_scales import select_scales, save_scales, paint_maps, summary  # noqa: E402
from dense_host import (moment_field, sphere_basis, odf_at, peaks_from_odf,
                        lambda_ratio, gate_mask, DEFAULT_FIRE_FRACTION)  # noqa: E402

# Certified on the validated phantoms: order 8 resolves 50/70/90 degrees with
# angular errors of 5/1/2 degrees, order 6 only reaches ~70. sigma_grad 2.5 is
# the peak-resolving scale; rho 5.0 is the neighbourhood.
ODF_ORDER = 8
ODF_SIGMA_GRAD = 2.5
ODF_RHO = 5.0
# Below this the ODF cannot separate two lobes, so a firing voxel returning one
# peak means "detected, not decomposed" rather than "single fibre".
RESOLUTION_FLOOR_DEG = 50.0

LABEL_CODES = {"background": 0, "single": 1, "candidate": 2, "unresolved": 3,
               "unresolvable": 4}


def triage_voxel(p):
    """candidate / unresolved / unresolvable, from the peak count at a voxel."""
    n = p["n_peaks"]
    if n >= 3:
        return "unresolvable"          # past the validated two-fibre ceiling
    if n == 2:
        return "candidate"             # two directions, ready for Stage 3
    return "unresolved"                # gate fired, ODF could not decompose


def main(radius=24, centre=(384, 384, 384), fire=DEFAULT_FIRE_FRACTION,
         mode="adaptive", block=32, outdir=None, verbose=True):
    t0 = time.time()
    outdir = outdir or os.path.join(_ROOT, "results", "dense")
    os.makedirs(outdir, exist_ok=True)
    lo = centre[0] - radius
    vol = load_volume(crop=(lo, centre[0] + radius),
                      dtype=np.uint8).astype(np.float32)
    A = nifti_affine((lo, lo, lo), VOXEL_SIZE_UM)
    tis = tissue_mask(vol)
    print("=" * 78)
    print(f"DENSE STAGES 1-2 -- i74 {vol.shape} at {tuple(centre)}, "
          f"crop origin ({lo}, {lo}, {lo})")
    print("=" * 78)
    print(f"  tissue {int(tis.sum())}/{tis.size} ({tis.mean():.1%})")
    save_nifti(vol, os.path.join(outdir, "raw.nii.gz"), A, dtype=np.float32)

    # ---- Stage 1 -----------------------------------------------------------
    print(f"\n[stage 1] selecting (sigma, rho), mode={mode} ...")
    sc = select_scales(vol, mode=mode, block=block, verbose=verbose)
    save_scales(sc, os.path.join(outdir, "scales.json"))
    print(f"  {summary(sc)}")
    t = field_for(vol, sc, invariants=False)
    e3, lam = t["e3"], t["lam"]
    save_vector_field(e3, os.path.join(outdir, "e3.nii.gz"), A)
    sig, rho = paint_maps(sc, vol.shape)
    save_nifti(sig, os.path.join(outdir, "sigma_map.nii.gz"), A, dtype=np.float32)
    save_nifti(rho, os.path.join(outdir, "rho_map.nii.gz"), A, dtype=np.float32)

    # ---- Stage 2a: the gate ------------------------------------------------
    print(f"\n[stage 2] gate on lambda3/lambda2, fire fraction {fire:.0%} ...")
    lr = lambda_ratio(lam)
    mask, thr = gate_mask(lam, tis, fire_fraction=fire)
    n_fire = int(mask.sum())
    print(f"  threshold lambda3/lambda2 > {thr:.4f}")
    print(f"  firing {n_fire} voxels = {n_fire / max(int(tis.sum()), 1):.1%} of tissue")
    save_nifti(np.where(np.isfinite(lr), lr, 0).astype(np.float32),
               os.path.join(outdir, "lambda_ratio.nii.gz"), A, dtype=np.float32)
    save_nifti(mask.astype(np.uint8), os.path.join(outdir, "gate_mask.nii.gz"),
               A, dtype=np.uint8)

    # ---- Stage 2b: the ODF, at the firing voxels only ----------------------
    print(f"\n[stage 2] dense ODF, order {ODF_ORDER}, sigma_grad "
          f"{ODF_SIGMA_GRAD:g}, rho {ODF_RHO:g} ...")
    M = moment_field(vol, ODF_SIGMA_GRAD, ODF_RHO, order=ODF_ORDER)
    print(f"  moment field {M.shape} ({M.nbytes / 1e6:.0f} MB)")
    vox = np.argwhere(mask)
    B = sphere_basis(ODF_ORDER)
    print(f"  extracting peaks at {len(vox)} gated voxels ...")
    E = odf_at(M, vox, basis=B, order=ODF_ORDER)
    del M

    vec1 = np.moveaxis(e3, 0, -1).astype(np.float32).copy()
    n = np.linalg.norm(vec1, axis=-1, keepdims=True)
    vec1 = np.where(n > 1e-12, vec1 / (n + 1e-12), 0.0).astype(np.float32)
    vec2 = np.zeros_like(vec1)
    labels = np.zeros(vol.shape, np.uint8)
    npk = np.zeros(vol.shape, np.uint8)
    seps = []
    counts = {"candidate": 0, "unresolved": 0, "unresolvable": 0}
    for i, (z, y, x) in enumerate(vox):
        if verbose and i and i % 5000 == 0:
            print(f"    {i}/{len(vox)}  ({time.time() - t0:.0f}s)")
        p = peaks_from_odf(E[i])
        b = triage_voxel(p)
        counts[b] += 1
        labels[z, y, x] = LABEL_CODES[b]
        npk[z, y, x] = min(int(p["n_peaks"]), 255)
        if len(p["dirs"]):
            # host_fod returns (x,y,z); the field is indexed (z,y,x)
            vec1[z, y, x] = np.asarray(p["dirs"][0], np.float32)[::-1]
        if b == "candidate":
            vec2[z, y, x] = np.asarray(p["dirs"][1], np.float32)[::-1]
            seps.append(p["sep"])
    labels[(labels == 0) & tis] = LABEL_CODES["single"]
    labels[~tis] = LABEL_CODES["background"]

    save_vector_field(np.moveaxis(vec1, -1, 0),
                      os.path.join(outdir, "vec1.nii.gz"), A)
    save_vector_field(np.moveaxis(vec2, -1, 0),
                      os.path.join(outdir, "vec2.nii.gz"), A)
    save_nifti(labels, os.path.join(outdir, "labels.nii.gz"), A, dtype=np.uint8)
    save_nifti(npk, os.path.join(outdir, "n_peaks.nii.gz"), A, dtype=np.uint8)

    seps = np.asarray(seps, float)
    res = dict(shape=[int(s) for s in vol.shape], crop_origin=[lo] * 3,
               sigma=sc["sigma_global"], rho=sc["rho_global"], mode=mode,
               fire_fraction=float(fire), gate_threshold=float(thr),
               n_tissue=int(tis.sum()), n_fired=n_fire, counts=counts,
               odf=dict(order=ODF_ORDER, sigma_grad=ODF_SIGMA_GRAD,
                        rho=ODF_RHO, resolution_floor_deg=RESOLUTION_FLOOR_DEG),
               sep_median=None if not seps.size else float(np.median(seps)),
               seconds=time.time() - t0)
    with open(os.path.join(outdir, "stage12.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "-" * 78)
    print(f"  gated voxels        {n_fire:8d}")
    for k in ("candidate", "unresolved", "unresolvable"):
        print(f"    {k:16} {counts[k]:8d}  ({counts[k] / max(n_fire, 1):5.1%} of fired)")
    if seps.size:
        print(f"  candidate separations: median {np.median(seps):.1f} deg, "
              f"IQR [{np.percentile(seps, 25):.1f}, {np.percentile(seps, 75):.1f}]")
    print(f"  wrote {outdir}   ({time.time() - t0:.0f}s)")
    print("=" * 78)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="dense Stages 1-2 on i74")
    ap.add_argument("--radius", type=int, default=24, help="half-width (24 -> 48^3)")
    ap.add_argument("--centre", nargs=3, type=int, default=[384, 384, 384])
    ap.add_argument("--fire", type=float, default=DEFAULT_FIRE_FRACTION)
    ap.add_argument("--mode", choices=("adaptive", "global"), default="adaptive")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    main(radius=a.radius, centre=tuple(a.centre), fire=a.fire, mode=a.mode,
         block=a.block, outdir=a.outdir)
