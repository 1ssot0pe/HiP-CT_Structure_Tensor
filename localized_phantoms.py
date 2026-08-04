"""
localized_phantoms.py -- the two phantoms the live pipeline validates against.

A *localized* crossing and a curved *kissing* pair, both built from the tube and
bundle primitives already at root (`crossing_shape_3d`, `kissing_phantom`).
"Localized" is the point: the second fibre family exists only inside a bounded
zone, so genuine single-family tissue exists elsewhere in the SAME volume.

Why this file exists
--------------------
These two builders lived in `mass_conservation/mass_conservation_validate.py`,
which meant the live pipeline reached *down* into an evidence folder for its
phantoms: `3_global_tractography/run_pipeline.py`,
`self_optimising_tractography.py` and `2_host_classification/
sigma_rho_gated_field.py` all imported from a mass-conservation experiment,
while the other phantom builders (`crossing_shape_3d.py`, `kissing_phantom.py`)
sat at root. That is backwards -- a stage cannot be run without a downstream
folder present, and the builders read as mass-conservation-specific when they
are the pipeline's own validation fixtures.

They are shared code, so they live at root with the rest of the shared code.
`mass_conservation_validate` imports them back from here, so its own public
surface is unchanged for the scripts that depend on it.
"""

import numpy as np

from crossing_shape_3d import _tube_family, _coords as _coords_cross
from kissing_phantom import _curve_pts, curved_bundle, SHAPE as KSHAPE

# Patch half-widths match the ESTABLISHED, already-validated setups elsewhere
# in the repo: crossing uses half=14 (host_fod.validate's _central_patch, with
# matching tube spacing -- the exact configuration proven to resolve a 40deg+
# crossing); kissing uses HALF=8 (test_kissing_phantom.py's window, the one
# that actually reproduces the rare spurious "crossing" near the pinch point).
CROSS_HALF = 14
KISS_HALF = 8

# z kept small (real lung half-map noise only has 81 z-slices available); y is
# generous so the far calibration offsets (+/-45, +/-50) have room either side
# of the crossing zone without needing a huge, expensive volume.
CROSS_SHAPE = (56, 200, 60)
CROSS_R1, CROSS_TAPER = 20.0, 8.0
CROSS_CAL_OFFSETS = (45, 50, -45, -50)


def raw_kissing(gap, curv, radius_a, radius_b):
    """Two curved bundles that approach to `gap` and turn away -- never crossing."""
    C = _coords_cross(KSHAPE)
    ptsA, ptsB = _curve_pts(-1, gap, curv), _curve_pts(+1, gap, curv)
    A = curved_bundle(C, ptsA, radius_a)
    B = curved_bundle(C, ptsB, radius_b)
    return (A + B).astype(np.float32)


def raw_localized_crossing(shape, sep_deg, spacing, width, R1=CROSS_R1,
                           taper=CROSS_TAPER):
    """A crossing that only exists within a BOUNDED zone around the volume
    centre -- family 1 fills the whole volume, family 2 is smoothly windowed
    to r<R1 (flat, no gradient in the window itself) then tapers to exactly 0
    by r=R1+taper. This models real tissue: one dominant background
    population, locally crossed by a second bundle over a finite overlap zone
    -- so genuine single-family-only tissue exists elsewhere in the SAME
    volume, and the calibration can come from there instead of a synthetic
    companion. (A raw radial mask with no flat core was tried first and broke
    HOST's 2-peak detection at the intersection -- the mask's own smooth
    gradient, even a small one, was enough to corrupt the higher-order
    energy balance; a flat core avoids that entirely.)"""
    half = np.deg2rad(sep_deg / 2.0)
    d1 = np.array([np.cos(half), np.sin(half), 0.0])
    d2 = np.array([np.cos(half), -np.sin(half), 0.0])
    C = _coords_cross(shape)
    t1 = _tube_family(C, d1, spacing, width)
    t2 = _tube_family(C, d2, spacing, width)
    X, Y, Z = C
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    mask = np.ones_like(r)
    band = (r > R1) & (r < R1 + taper)
    mask[band] = 0.5 * (1 + np.cos(np.pi * (r[band] - R1) / taper))
    mask[r >= R1 + taper] = 0.0
    return (t1 + t2 * mask).astype(np.float32)
