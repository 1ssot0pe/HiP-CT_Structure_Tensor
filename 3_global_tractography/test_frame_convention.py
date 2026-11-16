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
from select_scales import select_scales             # noqa: E402
from core.tensor import field_for                        # noqa: E402
from export_nifti import dense_e3                    # noqa: E402

N = 48
AXES = ((0, "z"), (1, "y"), (2, "x"))
DOMINANCE_MIN = 0.90     # |component| along the true axis, after normalising
TISSUE_FRAC = 0.10       # dense case: score on voxels above this fraction of max


def single_fibre(axis, n=N, width=1.2, amplitude=50.0, noise=0.0, seed=0):
    """A volume holding one straight, smooth fibre along `axis` (0=z, 1=y, 2=x).

    `noise` is a fraction of the peak amplitude, added AFTER smoothing. It is 0
    by default -- the end-to-end check wants the cleanest possible input -- and
    non-zero only where the perfect axial symmetry of a noiseless line would
    make the transverse eigenvalues exactly equal and e3 undefined.
    """
    vol = np.zeros((n, n, n), np.float32)
    sl = [slice(None)] * 3
    for a in range(3):
        if a != axis:
            sl[a] = n // 2
    vol[tuple(sl)] = 1.0
    out = ndimage.gaussian_filter(vol, width) * amplitude
    if noise > 0:
        rng = np.random.default_rng(seed)
        out = out + rng.normal(scale=noise * float(out.max()),
                               size=out.shape).astype(np.float32)
    return out


def recovered_direction(vol, block=32, stride=8):
    """Mean |direction| over the field, in Stage 3's (z, y, x) frame.

    The scale is selected, not assumed. `field_from_gated` no longer has a
    default (sigma, rho) to fall back on, which is the point of the change this
    test now also guards: a hidden default here would mean the frame is checked
    at a scale the pipeline never uses.
    """
    scales = select_scales(vol, mode="global", verbose=False)
    e3 = field_for(vol, scales, invariants=False)["e3"]
    gated = build_gated_field(vol, scales=scales, e3_xyz=e3[::-1], block=block,
                              stride=stride, verbose=False)
    # host_only=False, and it MUST be. A single straight fibre is exactly the
    # case the routing gate is supposed to refuse -- measured here, 0 of 27
    # patches route, at either gate scale -- so with host_only=True the field
    # comes back empty and all three axes report "no patches" rather than a
    # frame error. That is what happens on the current working tree, where
    # `host_only` was added to field_from_gated with a default of True: this
    # test began failing for a reason that has nothing to do with the frame it
    # exists to check. The directions under test live on the gated=False
    # patches, so those are the ones to read.
    field, _junctions, _I0 = S3.field_from_gated(gated, vol, e3=e3,
                                                 host_only=False)
    if not field:
        return None
    d = np.abs(np.mean([p["dirs"][0] for p in field], axis=0))
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def tensor_frame_selftest(verbose=True):
    """The same claim one level down: `tensor_field` returns (z, y, x) e3.

    Every reversal in the pipeline is now relative to that one contract, so if
    it silently flipped, each caller's compensating `[::-1]` would flip with it
    and the end-to-end test above could still pass while every exported map was
    mirrored. Checking the source directly costs one small volume.
    """
    fails = []
    for axis, name in AXES:
        # NOISE IS REQUIRED HERE, and the reason is worth stating. An ideal
        # Gaussian line is perfectly symmetric about its own axis, so the two
        # transverse eigenvalues are exactly equal (measured: lambda = [0.390,
        # 0.390, 0.000] at the centre). The eigenvector of the smallest
        # eigenvalue is then undefined, and Skielex returns NaN for it at EVERY
        # voxel -- the field is 0% finite at every (sigma, rho). That is correct
        # behaviour on a degenerate input, not a failure, and the pipeline
        # survives it because `build_gated_field` falls back to HOST's own
        # direction when e3 is not finite. But it means the tensor's frame
        # cannot be read off this phantom unless the symmetry is broken.
        vol = single_fibre(axis, noise=0.05)
        sc = select_scales(vol, mode="global", verbose=False)
        e3 = field_for(vol, sc, invariants=False)["e3"]
        c = [s // 2 for s in vol.shape]
        v = np.abs(e3[:, c[0], c[1], c[2]])
        if not np.all(np.isfinite(v)):
            fails.append(f"tensor_field e3 for {name} is not finite at the "
                         f"fibre centre even with the symmetry broken")
            if verbose:
                print(f"  tensor_field e3 at centre, fibre along {name}: "
                      f"NOT FINITE  *** FAIL ***")
            continue
        got = int(np.argmax(v))
        ok = got == axis
        if verbose:
            print(f"  tensor_field e3 at centre, fibre along {name}: "
                  f"[{v[0]:.3f} {v[1]:.3f} {v[2]:.3f}] -> {'zyx'[got]}  "
                  f"{'OK' if ok else '*** FAIL ***'}")
        if not ok:
            fails.append(f"tensor_field e3 for {name} is dominant in "
                         f"{'zyx'[got]} -- the (z,y,x) contract is broken")
    return fails


def dense_frame_selftest(verbose=True):
    """The same claim for the DENSE carrier: `DenseField.build` reverses patch
    dirs and passes e3 through.

    Stage 3 now consumes `DenseField`, not the patch list, so the reversal that
    matters has moved. It is checked here directly rather than through a
    phantom: hand-build one patch with directions along known axes, hand in an
    e3 pointing along a third, and read the grid back. A synthetic field makes
    the expected answer exact, so a wrong reversal cannot hide inside a
    plausible-looking angle.

    `max_radius` is deliberately small (2 in an 8^3 grid), because the test has
    to distinguish two things that a fully-painted volume conflates: the patch's
    own voxels, which must carry HOST's directions, and everything beyond its
    reach, which must still carry Stage 1's e3 untouched.
    """
    from sigma_rho_gated_field import GatedField
    from dense_field import DenseField, LABEL_CODES

    n, c, r = 8, 4, 2.0
    g = GatedField((n, n, n))
    # (x, y, z) components, host_fod's convention: first peak along x, second
    # along y. In Stage 3's (z, y, x) frame those are index 2 and index 1.
    g.add(centre=(c, c, c), n_dirs=2,
          dirs=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
          gated=True, label="crossing", c_s=0.5, FA=0.4, sigma=1.0, rho=2.0)
    e3 = np.zeros((3, n, n, n))
    e3[0] = 1.0                       # (z, y, x) components: e3 along z
    F = DenseField.build(g, e3, host_only=True, max_radius=r, fill=False)

    fails = []
    checks = (("vec1 at the patch centre", F.vec1[:, c, c, c], 2, "x"),
              ("vec2 at the patch centre", F.vec2[:, c, c, c], 1, "y"),
              ("vec1 beyond max_radius",   F.vec1[:, 0, 0, 0], 0, "z"))
    for what, v, want, axname in checks:
        v = np.abs(np.asarray(v, float))
        got = int(np.argmax(v))
        ok = got == want and v[want] > 0.99
        if verbose:
            print(f"  {what:28} -> (z,y,x) = [{v[0]:.3f} {v[1]:.3f} {v[2]:.3f}]"
                  f"  dominant={'zyx'[got]}, want {axname}  "
                  f"{'OK' if ok else '*** FAIL ***'}")
        if not ok:
            fails.append(f"DenseField {what}: dominant {'zyx'[got]}, "
                         f"expected {axname}")

    # The demotion must retract the SECOND direction, not the first.
    keep = F.vec1[:, c, c, c].copy()
    F.demote([0])
    ok = (np.linalg.norm(F.vec2[:, c, c, c]) < 1e-9
          and np.allclose(F.vec1[:, c, c, c], keep)
          and int(F.n_dirs[c, c, c]) == 1
          and int(F.labels[c, c, c]) == LABEL_CODES["kissing"])
    if verbose:
        print(f"  {'demote keeps vec1, drops vec2':28} -> "
              f"n_dirs={int(F.n_dirs[c, c, c])} "
              f"|vec2|={np.linalg.norm(F.vec2[:, c, c, c]):.3f}  "
              f"{'OK' if ok else '*** FAIL ***'}")
    if not ok:
        fails.append("DenseField.demote did not retract exactly vec2")
    return fails


def dense_direction(vol, block=32):
    """Mean |direction| over the DENSE export path, in (z, y, x).

    Scored on tissue voxels only. The surrounding halo also points along the
    fibre, so including it would not change the answer, but it would let a
    vacuum voxel's zero vector dilute the mean.
    """
    e3, _measured, _trace = dense_e3(vol, block=block, verbose=False)   # (x,y,z)
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
    failures = list(tensor_frame_selftest(verbose=verbose))
    if verbose:
        print("-" * 78)
    failures += dense_frame_selftest(verbose=verbose)
    if verbose:
        print("-" * 78)
    for axis, name in AXES:
        vol = single_fibre(axis)
        _check("patch", recovered_direction(vol), axis, name, failures, verbose)
        _check("export", dense_direction(vol), axis, name, failures, verbose)
    if verbose:
        print("-" * 78)
        if failures:
            print(f"FAIL -- {len(failures)} frame check(s) wrong:")
            for f in failures:
                print(f"    {f}")
            print("The (x,y,z) -> (z,y,x) reversal in field_from_gated or "
                  "export_nifti is inverted, dropped, or applied twice.")
        else:
            print(f"PASS -- tensor source, dense carrier, and all "
                  f"{len(AXES)} axes through both the patch join and the "
                  f"per-voxel NIfTI export are in the correct frame")
        print("=" * 78)
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
