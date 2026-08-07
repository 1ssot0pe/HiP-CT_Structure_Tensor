"""
certify_dense_scales.py -- certify the ODF scales for the DENSE formulation.

`certify_gate_scales.py` certified (half, sigma_grad) for the per-patch gate.
Those numbers cannot be carried over, because the measurement they came from
includes the patch-boundary artifact: at sigma_grad = 5 the derivative kernel is
wider than the half = 14 patch, so `mode="nearest"` replication contributed
substantially to every gradient. The dense formulation has no such boundary, so
its scales have to be measured in their own terms.

What is certified here
----------------------
    dir_sigma_grad   the differentiation scale
    dir_rho          the NEIGHBOURHOOD width -- the sliding counterpart of the
                     patch's box half-width

against the same known truth as before, and with the same criterion: among
scales that resolve the validated 40 deg crossing into exactly two peaks, take
the one whose recovered directions are closest to the phantom's true ones. Peak
COUNT alone is not enough -- a scale can split one fibre into two lobes and
score 2 -- so angular error decides.

Two negative controls, both of which must NOT produce two peaks: the kissing
pair at its point of closest approach, and single-family tissue taken from the
crossing volume itself, 45 voxels off centre where the second family has tapered
to exactly zero.

Run
---
    python certify_dense_scales.py
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

from dense_host import (moment_field, sphere_basis, peaks_from_odf,  # noqa: E402
                        odf_at, DEFAULT_ORDER)
from localized_phantoms import (raw_localized_crossing, raw_kissing,  # noqa: E402
                                CROSS_SHAPE, CROSS_R1, CROSS_TAPER,
                                CROSS_CAL_OFFSETS)

CACHE = os.path.join(_HERE, "dense_scales.json")
# The validation separation. 40 degrees -- the repo's historical case -- is NOT
# resolvable by the artifact-free ODF at any order swept: measured n_peaks is 1
# at order 6 AND order 8, while the same phantom read through an isolated patch
# returns 2. The patch's boundary replication was manufacturing the second peak.
# Order 8 resolves 50 degrees (and 70, 90); order 6 only reaches ~70. So the
# certified floor is stated here rather than assumed, and a run that needs 40
# degrees has to widen the ORDER, not the sweep.
CROSS_SEP_DEG = 50.0
SIGMAS = (2.0, 2.5, 3.0)
RHOS = (3.0, 5.0, 8.0)
ORDERS = (6, 8)
# `sharpen` has to be certified here too, and it is the parameter that actually
# decides this. Measured at the crossing centre, EVERY dense variant already
# carries both true fibres at essentially equal fibre-ness (0.999 / 0.987 for a
# box window, 0.995 / 0.991 for a Gaussian) -- the second fibre is not missing
# from the ODF. What is missing is a SADDLE between the two lobes: 40 degrees
# apart, they merge into one broad maximum, and `peak_directions` correctly
# reports one peak because on the discrete sphere there is only one.
#
# fn = (f / f.max()) ** sharpen deepens the contrast between the lobe maxima and
# the ridge joining them, which is exactly the knob for resolving close lobes.
# The patch formulation's 3.0 worked because its boundary contamination carved
# that saddle for it; without the artifact the value has to be re-measured.
# Raising it too far splits a SINGLE fibre into spurious lobes, which is what
# the two negative controls are here to catch.
SHARPENS = (3.0, 6.0, 10.0, 15.0, 25.0)
SINGLE_OFFSET = int(CROSS_CAL_OFFSETS[0])


def _true_dirs(sep_deg=CROSS_SEP_DEG):
    h = np.deg2rad(sep_deg / 2.0)
    return [np.array([np.cos(h), np.sin(h), 0.0]),
            np.array([np.cos(h), -np.sin(h), 0.0])]


def _angle_err(dirs, truth):
    if not len(dirs):
        return np.nan
    return float(np.mean([min(np.degrees(np.arccos(np.clip(abs(d @ t), 0, 1)))
                              for d in dirs) for t in truth]))


def certify(verbose=True, sep_deg=CROSS_SEP_DEG):
    X = raw_localized_crossing(CROSS_SHAPE, sep_deg, 12.0, 1.5,
                               CROSS_R1, CROSS_TAPER)
    K = raw_kissing(4.0, 0.028, 1.6, 1.6)
    cx = [s // 2 for s in X.shape]
    ck = [s // 2 for s in K.shape]
    cs_ = list(cx); cs_[1] += SINGLE_OFFSET
    truth = _true_dirs(sep_deg)
    B = sphere_basis(DEFAULT_ORDER)

    rows = []
    if verbose:
        print(f"  {'sig':>4} {'rho':>4} {'ord':>4} {'shrp':>5} | "
              f"{'cross npk':>9} {'sep':>6} {'amp2':>5} {'err':>5} | "
              f"{'kiss':>5} {'sngl':>5}")
    for sg in SIGMAS:
        for rho in RHOS:
            for order in ORDERS:
                B = sphere_basis(order)
                Mx = moment_field(X, sg, rho, order=order)
                Mk = moment_field(K, sg, rho, order=order)
                # The ODF is computed ONCE per (sigma, rho, order); `sharpen`
                # only reshapes it for peak extraction, so it is swept inside.
                Ex = odf_at(Mx, [cx], basis=B, order=order)[0]
                Es = odf_at(Mx, [cs_], basis=B, order=order)[0]
                Ek = odf_at(Mk, [ck], basis=B, order=order)[0]
                del Mx, Mk
                for sh in SHARPENS:
                    px = peaks_from_odf(Ex, sharpen=sh)
                    ps = peaks_from_odf(Es, sharpen=sh)
                    pk = peaks_from_odf(Ek, sharpen=sh)
                    err = (_angle_err(px["dirs"][:2], truth)
                           if px["n_peaks"] >= 2 else np.nan)
                    ok = (px["n_peaks"] == 2 and pk["n_peaks"] < 2
                          and ps["n_peaks"] < 2)
                    rows.append(dict(
                        sigma_grad=float(sg), rho=float(rho), order=int(order),
                        sharpen=float(sh), cross_npeaks=int(px["n_peaks"]),
                        cross_sep=None if not np.isfinite(px["sep"]) else float(px["sep"]),
                        cross_amp2=None if not np.isfinite(px["amp2"]) else float(px["amp2"]),
                        angle_err_deg=None if not np.isfinite(err) else float(err),
                        kiss_npeaks=int(pk["n_peaks"]),
                        single_npeaks=int(ps["n_peaks"]), usable=bool(ok)))
                    if verbose and (ok or px["n_peaks"] >= 2):
                        sep = px["sep"] if np.isfinite(px["sep"]) else float("nan")
                        am = px["amp2"] if np.isfinite(px["amp2"]) else float("nan")
                        print(f"  {sg:4.1f} {rho:4.1f} {order:4d} {sh:5.1f} | "
                              f"{px['n_peaks']:9d} {sep:6.1f} {am:5.2f} {err:5.1f} | "
                              f"{pk['n_peaks']:5d} {ps['n_peaks']:5d}"
                              + ("   <- usable" if ok else ""))

    usable = [r for r in rows if r["usable"] and r["angle_err_deg"] is not None]
    if not usable:
        raise RuntimeError(
            "no (sigma_grad, rho) resolves the 40 deg crossing into exactly two "
            "peaks while leaving the kissing pair and single-fibre tissue at one. "
            "The dense ODF cannot be certified on these phantoms -- widen the "
            "sweep before trusting any value.")
    best = min(usable, key=lambda r: r["angle_err_deg"])
    edge = []
    if best["sigma_grad"] in (SIGMAS[0], SIGMAS[-1]):
        edge.append(f"sigma_grad={best['sigma_grad']:g}")
    if best["rho"] in (RHOS[0], RHOS[-1]):
        edge.append(f"rho={best['rho']:g}")
    if best["sharpen"] in (SHARPENS[0], SHARPENS[-1]):
        edge.append(f"sharpen={best['sharpen']:g}")
    if edge and verbose:
        print(f"\n  WARNING: pick on the sweep boundary ({', '.join(edge)}) -- "
              f"the optimum is probably outside the box.")

    return dict(dir_sigma_grad=best["sigma_grad"], dir_rho=best["rho"],
                order=int(best["order"]), sharpen=float(best["sharpen"]),
                evidence=dict(
                    validated_separation_deg=float(sep_deg),
                    criterion="min angular error against known directions, among "
                              "scales giving exactly 2 peaks on the crossing and "
                              "fewer than 2 on both the kissing pair and "
                              "single-family tissue",
                    angle_err_deg=best["angle_err_deg"],
                    sep_deg=best["cross_sep"], amp2=best["cross_amp2"],
                    on_sweep_boundary=bool(edge),
                    n_usable=len(usable), n_swept=len(rows), sweep=rows))


def load_dense_scales(path=CACHE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run:\n"
            f"    python 1_sigma_rho_selection/certify_dense_scales.py")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="certify the dense ODF scales")
    ap.add_argument("--out", default=CACHE)
    ap.add_argument("--sep", type=float, default=CROSS_SEP_DEG,
                    help="validation crossing separation in degrees")
    a = ap.parse_args()
    print("=" * 78)
    print("DENSE ODF SCALE CERTIFICATION -- no patch boundary")
    print("=" * 78)
    g = certify(sep_deg=a.sep)
    print("\n" + "-" * 78)
    print(f"  dir_sigma_grad={g['dir_sigma_grad']:g}, dir_rho={g['dir_rho']:g}, "
          f"order={g['order']}, sharpen={g['sharpen']:g}"
          f"   (angular error {g['evidence']['angle_err_deg']:.1f} deg, "
          f"sep {g['evidence']['sep_deg']:.1f} deg)")
    with open(a.out, "w") as f:
        json.dump(g, f, indent=2)
    print(f"  wrote {a.out}")
    print("=" * 78)
