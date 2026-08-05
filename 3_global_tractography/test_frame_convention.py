"""
test_frame_convention.py -- guards the Stage 2 -> Stage 3 frame conversion.

`self_optimising_tractography.field_from_gated` reverses every direction on the
way in, because `host_fod.analyse` and Skielex's e3 return (x, y, z) while
Stage 3 indexes positions as (z, y, x):

    dirs = [np.asarray(d, float)[::-1] for d in p["dirs"]]

That single `[::-1]` is the most dangerous line in the pipeline, and its own
docstring says why: dropping it "does not crash -- it just makes every
streamline walk a mirrored path, which reads as 'the structure curves away'
rather than as a bug". Nothing else in the repo would fail if it were removed.
The phantom validation would not catch it either: a symmetric crossing scores
much the same S either way.

So: build three volumes each containing one straight fibre along a KNOWN axis,
push them through the real Stage 1 + 2 + join, and assert the recovered
direction's dominant component is the axis the fibre was drawn along. An
inverted or dropped reversal maps z <-> x and the z and x cases fail.

`export_nifti` reverses in a SECOND place -- it writes a dense (z, y, x) peaks
image straight from the tensor, never passing through `field_from_gated` -- so
the same three fibres are run through that path too. The two can drift apart
independently; each needs its own known answer.

Run
---
    python test_frame_convention.py        # exits non-zero on failure
"""

import os
import sys

import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import self_optimising_tractography as S3          # noqa: E402
from sigma_rho_gated_field import build_gated_field  # noqa: E402
from export_nifti import dense_e3                    # noqa: E402

N = 48
AXES = ((0, "z"), (1, "y"), (2, "x"))
DOMINANCE_MIN = 0.90     # |component| along the true axis, after normalising
TISSUE_FRAC = 0.10       # dense case: score on voxels above this fraction of max


def single_fibre(axis, n=N, width=1.2, amplitude=50.0):
    """A volume holding one straight, smooth fibre along `axis` (0=z, 1=y, 2=x)."""
    vol = np.zeros((n, n, n), np.float32)
    sl = [slice(None)] * 3
    for a in range(3):
        if a != axis:
            sl[a] = n // 2
    vol[tuple(sl)] = 1.0
    return ndimage.gaussian_filter(vol, width) * amplitude


def recovered_direction(vol, block=32, stride=8):
    """Mean |direction| over the field, in Stage 3's (z, y, x) frame."""
    gated = build_gated_field(vol, block=block, stride=stride, verbose=False)
    field, _junctions, _I0 = S3.field_from_gated(gated, vol)
    if not field:
        return None
    d = np.abs(np.mean([p["dirs"][0] for p in field], axis=0))
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def dense_direction(vol, block=32):
    """Mean |direction| over the DENSE export path, in (z, y, x).

    Scored on tissue voxels only. The surrounding halo also points along the
    fibre, so including it would not change the answer, but it would let a
    vacuum voxel's zero vector dilute the mean.
    """
    e3 = dense_e3(vol, block=block, verbose=False)          # (3, ...) in (x,y,z)
    tissue = vol > TISSUE_FRAC * float(vol.max())
    v = e3[:, tissue]
    v = v[:, np.linalg.norm(v, axis=0) > 0]
    if not v.size:
        return None
    d = np.abs(np.mean(v, axis=1))[::-1]                    # -> (z, y, x)
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def _check(label, d, axis, name, failures, verbose):
    if d is None:
        failures.append(f"{label} / fibre along {name}: no directions")
        if verbose:
            print(f"  {label:6s} fibre along {name} (axis {axis}): NONE  -> FAIL")
        return
    got = int(np.argmax(d))
    ok = got == axis and d[axis] >= DOMINANCE_MIN
    if verbose:
        print(f"  {label:6s} fibre along {name} (axis {axis}) -> (z,y,x) = "
              f"[{d[0]:.3f} {d[1]:.3f} {d[2]:.3f}]  dominant={'zyx'[got]}  "
              f"{'OK' if ok else '*** FAIL ***'}")
    if not ok:
        failures.append(f"{label} / fibre along {name}: dominant axis "
                        f"{'zyx'[got]}, component {d[axis]:.3f} < {DOMINANCE_MIN}")


def run(verbose=True):
    if verbose:
        print("=" * 78)
        print("FRAME CONVENTION -- Stage 2 (x,y,z) -> Stage 3 / NIfTI (z,y,x)")
        print("=" * 78)
    failures = []
    for axis, name in AXES:
        vol = single_fibre(axis)
        _check("patch", recovered_direction(vol), axis, name, failures, verbose)
        _check("dense", dense_direction(vol), axis, name, failures, verbose)
    if verbose:
        print("-" * 78)
        if failures:
            print(f"FAIL -- {len(failures)}/{2 * len(AXES)} cases wrong:")
            for f in failures:
                print(f"    {f}")
            print("The (x,y,z) -> (z,y,x) reversal in field_from_gated or "
                  "export_nifti is inverted, dropped, or applied twice.")
        else:
            print(f"PASS -- all {2 * len(AXES)} cases recovered in the correct "
                  f"frame (patch join + dense export)")
        print("=" * 78)
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
