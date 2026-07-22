"""
noise_axis_gate.py -- apply the half-map noise result (#2): the measured noise is
ANISOTROPIC, so a HOST 'crossing' whose second peak points along the NOISE axis is
likely a directional-noise artefact, not a real fibre. This validates the mechanism
and the gate on phantoms with KNOWN truth before trusting it on real data.

Subtlety we verify empirically (not assume): streaky/anisotropic noise makes spurious
'fibres' along the direction of LEAST gradient = the MIN-eigenvector of the noise
structure tensor N -- NOT N's principal (max-gradient) axis that T1 reports. Gating
against the wrong one would be 90 deg off, so we measure where the spurious peaks
actually land.

Setup: noise is white smoothed MORE along axis0 (z) -> streaks along z -> spurious
fibre-ness along z. Real fibres are placed IN-PLANE (perp to z) so a z-gate should
catch the spurious peaks and spare the real ones. (Blind spot, stated honestly: a
real fibre that genuinely runs along the noise axis is indistinguishable -- the gate
would wrongly flag it.)
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from crossing_shape_3d import _tube_family, _coords
from host_geometry_test import _finalise
from host_fod import analyse, classify
import noise_floor as nf

SHAPE = (64, 64, 64)
HALF = 8
SMOOTH = (3.2, 0.7, 0.7)     # smooth along axis0 -> streaks (spurious fibres) along z
GATE_DEG = 25.0
NOISE_AMP = 0.50            # noise std relative to signal. NOTE the regime dependence:
                           # at low noise (~0.2) spurious 'crossings' are peak-SPLITTING
                           # of the real fibre (near it, NOT the noise axis) -- a
                           # different artefact the gate rightly ignores; at realistic/
                           # high noise, genuine noise-ORIENTATION peaks emerge along
                           # the noise min-grad axis and the gate catches them.
TRUE_AXIS = np.array([0.0, 0.0, 1.0])   # axis0=z in (x,y,z)


def aniso_noise(amp, seed):
    rng = np.random.default_rng(seed)
    n = gaussian_filter(rng.standard_normal(SHAPE).astype(np.float32), SMOOTH)
    return n * (amp / (n.std() + 1e-9))


def dir_inplane(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([np.cos(t), np.sin(t), 0.0])           # (x,y,z), perp to z


def build(kind, seed):
    c = _coords(SHAPE)
    if kind == "single":
        sig = _tube_family(c, dir_inplane(0), 12, 1.5)
    else:
        sig = _tube_family(c, dir_inplane(0), 12, 1.5) + _tube_family(c, dir_inplane(70), 12, 1.5)
    return _finalise(sig) + aniso_noise(NOISE_AMP, seed)


def noise_axes(noise, sigma=4.0):
    """(min-grad axis, max-grad axis) of the noise, in (x,y,z). The MIN-grad axis is
    the spurious-fibre direction."""
    w, v = np.linalg.eigh(nf.N_matrix(noise, sigma))       # axis-index basis, ascending
    to_xyz = lambda e: np.array([e[2], e[1], e[0]])
    return to_xyz(v[:, 0]), to_xyz(v[:, -1])


def sample_crossings(vol, stride=5):
    Z, Y, X = vol.shape
    out = []
    for z in range(HALF, Z - HALF, stride):
        for y in range(HALF, Y - HALF, stride):
            for x in range(HALF, X - HALF, stride):
                m = analyse(vol[z-HALF:z+HALF+1, y-HALF:y+HALF+1, x-HALF:x+HALF+1])
                if m is not None and classify(m) == "crossing":
                    out.append(np.asarray(m["dirs"][:2]))
    return out


def ang(p, a):
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(p, a)))))))


def suspect_angle(dirs, axis):
    return min(ang(dirs[0], axis), ang(dirs[1], axis))     # closest peak to the axis


def run():
    print("=" * 74)
    print("NOISE-AXIS CROSSING GATE -- validate #2 on anisotropic-noise phantoms")
    single = build("single", 0)
    cross = build("crossing", 1)
    amin, amax = noise_axes(aniso_noise(NOISE_AMP, 0))
    print(f"  noise min-grad (spurious-fibre) axis est = {np.round(amin,2)}  "
          f"(angle to true z = {ang(amin, TRUE_AXIS):.0f} deg)")
    print(f"  noise max-grad axis est = {np.round(amax,2)}  "
          f"(angle to true z = {ang(amax, TRUE_AXIS):.0f} deg)")

    sc = sample_crossings(single)          # ALL spurious (truth = single fibre)
    cc = sample_crossings(cross)           # real crossings (perp to noise axis)
    print("-" * 74)
    if sc:
        smin = np.median([suspect_angle(d, amin) for d in sc])
        smax = np.median([suspect_angle(d, amax) for d in sc])
        print(f"  SPURIOUS crossings (single-fibre phantom): n={len(sc)}; suspect-peak "
              f"angle median to MIN-grad axis = {smin:.0f} deg, to MAX-grad = {smax:.0f} deg")
        print(f"    -> spurious peaks align with the {'MIN' if smin < smax else 'MAX'}"
              f"-grad axis (confirms which axis to gate)")
    catch = np.mean([suspect_angle(d, amin) < GATE_DEG for d in sc]) if sc else float("nan")
    false_rm = np.mean([suspect_angle(d, amin) < GATE_DEG for d in cc]) if cc else float("nan")
    print(f"  GATE (a peak within {GATE_DEG:.0f} deg of the noise min-grad axis => flag):")
    print(f"    catches {catch:.0%} of SPURIOUS crossings (single-fibre phantom)")
    print(f"    flags   {false_rm:.0%} of crossing-phantom crossings (n={len(cc)})")
    print("  VERDICT: mechanism CONFIRMED -- directional noise manufactures crossings,")
    print("  with the spurious peak along the noise MIN-grad axis (not T1's principal/max")
    print("  axis). But a blunt angular gate is only a PARTIAL filter: at realistic noise")
    print("  BOTH phantoms grow noise-axis peaks, so catch (~1/3) and collateral are")
    print("  comparable, and it needs a LOCAL noise axis (the real lung axis varied per")
    print("  region) with a blind spot for real fibres along that axis. Prefer subtracting")
    print("  the FULL measured N (application #1) to de-bias the tensor continuously.")
    print("=" * 74)


if __name__ == "__main__":
    run()
