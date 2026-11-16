"""
self_optimising_tractography.py -- Stage 3.

Takes the unified orientation field from Stages 1-2 (HOST-gated voxels carrying
several orientations, everything else carrying one e3) and optimises the
tractography against the image itself.

The loop
--------
    integrate streamlines through the field with parameters theta
      -> accumulate D_rec (tract_density.rasterise)
      -> L = MSE(D_rec, D_obs) + superposition penalty
      -> update theta
      -> re-integrate, until L converges

theta = (step, angle_threshold, seed_density, link_weight).

D_obs -- the target density
---------------------------
Derived from the field's OWN anisotropy magnitude, `lambda1 - lambda3`,
normalised to unit mean over tissue. The reasoning: where the structure tensor
is strongly anisotropic there is fibre there, and a tractography that explains
the image should put tracts in proportion. Using the anisotropy rather than raw
intensity keeps the target contrast-invariant, which is the same rank/ratio
lesson the gate work recorded (an absolute level never transfers).

This target is a proxy, not ground truth, and the honest limitation is that
`lambda1 - lambda3` cannot distinguish one bright fibre from two dim crossing
ones -- which is precisely what the superposition term below is for.

Superposition metric S
----------------------
At each junction where two orientations meet, with background I_0:

    S = (I_c - I_0) / ((I_a - I_0) + (I_b - I_0))

where I_c is the measured centre value and I_a, I_b are the two contributing
structures sampled away from the junction along their own directions.

    S ~ 1   both structures superpose  -> genuine CROSSING
    S << 1  only one structure present -> TANGENCY (kissing)

This is the background-corrected form of `mass_conservation`'s ratio, which was
validated on these phantoms at catch 100% / false-positive 0%. Subtracting I_0
matters on real data, where air is not at zero and an uncorrected ratio inherits
the offset.

The verdict enters the loop twice, as specified: it gates whether a junction is
allowed to link straight through (a tangency must not be traversed), and it adds
a penalty to L for junctions whose linking decision disagrees with their own S.
Per-junction S, verdict and loss contribution are logged every iteration.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "mass_conservation"),
           os.path.join(_ROOT, "2_host_classification"),
           os.path.join(_ROOT, "1_sigma_rho_selection")):
    sys.path.insert(0, _p)

from core.host_fod import analyse
from sigma_rho_gated_field import (BUCKET_CANDIDATE, BUCKET_SINGLE,
                                  BUCKET_UNRESOLVABLE)
from core.tensor import tensor_field, anisotropy as _anisotropy
from core.reconstruction import AIR_FRAC, tissue_mask
from tract_density import rasterise, density_at, mse
from weight_fit import design_matrix, fit_weights

EPS = 1e-12
# Physical values of S, not tuned ones. A genuine crossing superposes two
# fibres over the background: I_c - I_0 = 2f while (I_a - I_0) + (I_b - I_0) =
# f + f = 2f, so S = 1. A tangency has one fibre's worth at the centre while
# both contributors are still one fibre each: S = f / 2f = 0.5. The decision
# boundary is therefore the midpoint 0.75 -- which is exactly
# mass_conservation_validate.THRESH = 1.5 rescaled, since S = ratio / 2.
S_CROSSING = 1.0
S_TANGENCY = 0.5
S_DECISION = 0.75
S_BAND = 0.10            # +-band around the midpoint reported as AMBIGUOUS
# I_a / I_b are searched over a RANGE of distances, not read at one point: the
# contributing structure may curve away from a straight ray (the kissing
# phantom bends by ~1 voxel over 8), and a point read then lands in background
# and sends S to infinity. See _contributor_intensity.
# ARM_MIN must exceed the junction's OVERLAP EXTENT, or the contributor is
# measured inside the region where both fibres are still present and reads the
# sum instead of one arm. For fibres of radius r meeting at angle theta the
# overlap reaches ~r/sin(theta/2) along each arm: 7.1 voxels for r=3 at 50 deg,
# against the old ARM_MIN of 5.0. Noise-free the low quantile still recovered
# the clean samples further out, so this never showed; under noise it collapses
# -- a known crossing scored 0.51 at SNR 62 with arm_min=5 and 0.93 at
# arm_min=8. Measured on the toy: 5.0 fails at every noise level, 8.0 and above
# hold to sigma=4.
ARM_MIN, ARM_MAX, ARM_STEP = 8.0, 16.0, 1.0
ARM_HALF = 2             # perpendicular tolerance when tracking the structure
CONTRIB_QUANTILE = 0.25  # see _walk_contributor: median inflates I_a at wide angles
BASELINE_HALF = 14       # = GATE_HALF: the neighbourhood HOST itself saw
BASELINE_Q = 0.50        # see local_baseline + s_calibration.py


# ---------------------------------------------------------------------------
# D_obs -- target density from the field's own anisotropy
# ---------------------------------------------------------------------------

def host_support(field, shape, radius):
    """Voxels a streamline can actually reach: within `radius` of a HOST patch.

    The loss has to compare like with like. `integrate` terminates as soon as
    no patch centre lies within `assoc_radius`, so with Stage 3 confined to
    HOST-routed patches the tracts can only ever occupy a neighbourhood of
    those patches -- 344 of them on i74. Scoring D_rec against D_obs over ALL
    884,736 tissue voxels therefore measured, almost entirely, the tracts'
    absence from voxels they are forbidden to enter. That term is constant in
    theta, and it swamped the part that is not.
    """
    from scipy.spatial import cKDTree
    pts = np.array([p["centre"] for p in field], float)
    if not len(pts):
        return np.zeros(shape, bool)
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"),
                    axis=-1).reshape(-1, 3)
    d, _ = cKDTree(pts).query(grid, k=1)
    return (d <= radius).reshape(shape)


def renormalise(D, mask):
    """Unit mean over `mask` -- the target must be normalised on the same
    support it is scored on, or the comparison carries an arbitrary scale."""
    m = float(D[mask].mean()) if mask.any() else 0.0
    return D / m if m > 0 else D


def anisotropy_target(vol, sigma=None, rho=None, lam=None, tissue=None):
    """Normalised (lambda1 - lambda3) as the density a tractography should match.

    The scale is REQUIRED -- either as (sigma, rho), or by handing in the `lam`
    from a decomposition already computed. The old signature defaulted to
    sigma=1.0, rho=3.0, which meant the target density was measured at a
    different scale from the e3 field the streamlines followed (1.0, 2.0) and
    from the scale Stage 1 had certified. Three measurements of one volume, two
    of them at scales nobody had chosen.

    Passing `lam` is the preferred path: it makes D_obs a read of the SAME
    eigen-decomposition that produced e3, rather than a second computation that
    can silently drift from it.

    Normalised to unit MEAN over tissue, not unit max: a max-normalisation is set
    by the single brightest voxel and would make the target hostage to one
    outlier, the same failure mode the absolute-threshold work kept hitting.
    """
    if lam is None:
        if sigma is None or rho is None:
            raise TypeError(
                "anisotropy_target needs a scale: pass (sigma, rho) from "
                "select_scales, or the `lam` of an existing tensor_field. There "
                "is no default -- see the docstring.")
        lam = tensor_field(vol, sigma, rho, invariants=False)["lam"]
    aniso = np.asarray(_anisotropy(lam), dtype=np.float64)
    if tissue is None:
        tissue = tissue_mask(vol)
    m = float(aniso[tissue].mean()) if tissue.any() else 0.0
    if m <= 0:
        return np.zeros_like(aniso), tissue
    return aniso / m, tissue


# ---------------------------------------------------------------------------
# superposition metric
# ---------------------------------------------------------------------------

def background_level(vol, air_frac=AIR_FRAC):
    """I_0: the background the superposition is measured over.

    Median of the sub-air-threshold voxels, falling back to the volume's own
    low percentile when the phantom has no true background (synthetic volumes
    are often exactly 0 outside the tubes, which is a valid I_0 of 0).
    """
    thr = float(air_frac) * float(vol.max())
    bg = vol[vol <= thr]
    if bg.size >= 32:
        return float(np.median(bg))
    return float(np.percentile(vol, 1))


def local_baseline(vol, centre, half=14, q=BASELINE_Q):
    """The local NON-FIBRE level at a junction -- the right I_0 in dense tissue.

    `background_level` returns the AIR level, which is the correct reference
    only when a fibre sits in air. In i74 it does not: the air level is 115,
    the tissue a fibre is embedded in reads ~183, and the fibre itself ~214.

    That distinction is not cosmetic, because superposition applies to the
    FIBRE part of the signal and not to the tissue pedestal underneath it:

        I = I_tissue + n_fibres * f          (the pedestal does NOT double)

    Referencing to air folds the pedestal into f, so f is measured as 99
    instead of 31 and the model predicts a crossing at 115 + 2*99 = 313 in a
    uint8 volume that stops at 255. The prediction is outside the container, S
    saturates at ~0.69, and the CROSSING verdict becomes unreachable -- which
    is exactly what a full i74 run produced: 67 junctions, 67 demotions, S
    spanning only 0.42-0.61. Referencing to the local tissue level instead
    puts the same crossing at 183 + 2*31 = 245, comfortably inside uint8.

    A quantile of the local neighbourhood rather than its mean or median: the
    fibres in the patch are the bright minority, so a low quantile reads the
    tissue between them. See `s_calibration.py` for the sweep that picks q
    against known truth.
    """
    c = [int(round(v)) for v in centre]
    sl = tuple(slice(max(0, v - half), v + half + 1) for v in c)
    patch = np.asarray(vol)[sl]
    if patch.size == 0:
        return background_level(vol)
    return float(np.quantile(patch, q))


def _local_max(vol, p, half):
    idx = tuple(int(np.clip(round(v), 0, s - 1)) for v, s in zip(p, vol.shape))
    if half <= 0:
        return float(vol[idx])
    sl = tuple(slice(max(v - half, 0), min(v + half + 1, s))
               for v, s in zip(idx, vol.shape))
    return float(vol[sl].max())


def orientation_field(vol, sigma, rho):
    """e3 as a (3, *shape) field in (z, y, x) component order.

    `sigma` and `rho` are required. They used to default to (1.0, 2.0), which is
    how the field every streamline followed came to be measured at a scale
    nothing had selected -- see `tensor.py` for the three-scales problem this
    closes. Get them from `select_scales`.

    The (z, y, x) component order is `tensor_field`'s contract, decided once at
    the source; getting it wrong makes the walker follow a mirrored direction,
    which reads as the structure curving away rather than as a bug.
    """
    return tensor_field(vol, sigma, rho, invariants=False)["e3"]


def fields_for(vol, mode="global", block=32, verbose=False):
    """(scales, e3, lam) for a volume -- select the scale, then decompose once.

    The convenience the diagnostic scripts need. They each used to open with
    `orientation_field(vol)` and pick up its (1.0, 2.0) default, so a sweep
    measuring the loss, or S, or the cone tracker's behaviour was measuring it
    at a scale the pipeline does not use. Global mode by default: these scripts
    run on small crops and single phantoms, where a per-block map has nothing to
    adapt to.
    """
    from select_scales import select_scales
    from core.tensor import field_for
    sc = select_scales(vol, mode=mode, block=block, verbose=verbose)
    t = field_for(vol, sc, invariants=False)
    return sc, t["e3"], t["lam"]


def _perp_basis(u):
    """Two unit vectors spanning the plane perpendicular to u."""
    u = np.asarray(u, float)
    u = u / (np.linalg.norm(u) + EPS)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, u)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a); e1 /= (np.linalg.norm(e1) + EPS)
    e2 = np.cross(u, e1); e2 /= (np.linalg.norm(e2) + EPS)
    return e1, e2


def _snap_to_ridge(vol, p, u, radius=1.0, n=2):
    """Re-centre p on the ridge, moving ONLY perpendicular to the travel
    direction u.

    A full-neighbourhood argmax snap does not work here: at a crossing the
    brightest voxel nearby IS the junction, so the walker is dragged straight
    back to it and every contributor reads the summed value (measured: I_a =
    I_c = 2.00 at every crossing angle, pinning S at exactly 0.5). Restricting
    the correction to the perpendicular plane keeps the walker on the fibre
    while guaranteeing it keeps making forward progress away from the junction.
    """
    p = np.asarray(p, float)
    e1, e2 = _perp_basis(u)
    best_p, best_v = p, _local_max(vol, p, 0)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if i == 0 and j == 0:
                continue
            q = p + (radius * i / n) * e1 + (radius * j / n) * e2
            if not all(0 <= q[k] < vol.shape[k] for k in range(3)):
                continue
            v = _local_max(vol, q, 0)
            if v > best_v:
                best_p, best_v = q, v
    return best_p


def _walk_contributor(vol, e3, centre, d, arm_min=ARM_MIN, arm_max=ARM_MAX,
                      step=1.0, half=1, snap=True):
    """Brightness of one contributing structure, FOLLOWING it rather than
    assuming it is straight.

    From the junction the walker steps along `d`, and at every step (i) re-reads
    the local fibre direction from the e3 field and (ii) re-centres on the local
    ridge maximum. Samples taken beyond `arm_min` are collected from both
    directions, and their median over the on-structure subset is returned.

    A straight ray fails on both phantom families for opposite reasons: on a
    curved tangency it leaves the bundle and reads background (S diverged to
    136), and on a wide crossing it stays inside the overlap zone and reads the
    SUM of both fibres (I_a = 1.48 instead of 1.0, dragging S to 0.68). Tracking
    the structure fixes both, because it measures one fibre where that fibre
    actually is.
    """
    vol = np.asarray(vol)
    c = np.asarray(centre, float)
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) + EPS)
    vals = []
    for sgn in (1.0, -1.0):
        p = c.copy()
        u = sgn * d
        travelled = 0.0
        # RADIAL distance from the junction, not path length. The walk re-reads
        # its direction from a noisy e3 and re-centres on a noisy ridge, so it
        # wanders; gating on path length alone let it curl back INTO the
        # junction while `travelled` kept counting up, and there it reads the
        # SUMMED intensity as though it were one arm. I_a then doubles, the
        # denominator doubles, and S halves to ~0.5 -- a known crossing scored
        # 0.507 at SNR 62, i.e. the crossing verdict was lost to noise that is
        # barely visible. Requiring genuine displacement, and stopping as soon
        # as the walk turns back, is what keeps the contributor on its own arm.
        radius = 0.0
        while travelled < arm_max:
            p = p + step * u
            travelled += step
            if not all(0 <= p[k] < vol.shape[k] for k in range(3)):
                break
            # snap=False exists to test the ESTIMATOR ASYMMETRY: the arms get
            # re-centred on the brightest perpendicular neighbour and the
            # junction centre does not, so under noise only the denominator is
            # optimised upward and S is pushed down. Noise-free phantoms cannot
            # show this.
            if snap:
                p = _snap_to_ridge(vol, p, u)
            idx = tuple(int(np.clip(round(v), 0, s - 1))
                        for v, s in zip(p, vol.shape))
            # Hold the GIVEN direction until clear of the junction, and only
            # then start following e3. Inside the overlap the structure tensor
            # is degenerate -- two comparable eigenvalues in the plane of the
            # two fibres -- so e3 there points along their BISECTOR, along
            # neither fibre (measured at a crossing centre: e3 = [0.03, -0.01,
            # -1.00], the bisector). Re-reading it immediately steered the walk
            # off its own arm and back into the junction: at sigma=0.5 the walk
            # took 16 unit steps and reached a radius of 7.05, circling inside
            # the overlap and never producing a single valid sample, so the
            # contributor came back NaN. Larger noise perturbs the degeneracy
            # enough to escape, which is why this failed at LOW noise and
            # passed at high -- the diagnostic that gives it away.
            #
            # `d` is HOST's peak direction, which is exactly the estimate that
            # is trustworthy at the junction; e3 becomes the better estimate
            # only once the arm is on its own again.
            if float(np.linalg.norm(p - c)) >= arm_min:
                v = e3[:, idx[0], idx[1], idx[2]]
                if np.all(np.isfinite(v)) and np.linalg.norm(v) > EPS:
                    v = v / np.linalg.norm(v)
                    u = np.sign(np.dot(v, u) or 1.0) * v  # keep going forward
            # Gate on RADIAL displacement, not path length. A wandering walk
            # accumulates `travelled` while staying near the junction, so the
            # path-length gate admitted samples that were still inside it.
            # (An early break when the walk turned back was tried too and is
            # NOT kept: it fired on ordinary noise wobble, ended the walk
            # before arm_min, and returned NaN for the whole contributor.)
            r_now = float(np.linalg.norm(p - c))
            if r_now >= arm_min:
                vals.append(_local_max(vol, p, half))
    if not vals:
        return np.nan
    vals = np.asarray(vals, float)
    on = vals[vals > 0.2 * (vals.max() + EPS)]
    if not on.size:
        return float(np.median(vals))
    # LOW quantile, not the median. Walking along one fibre of a lattice
    # periodically passes through OTHER junctions, where the local reading is
    # already a sum; the median absorbs those and inflates the contributor
    # (measured I_a = 1.20 at 60 deg and 1.38 at 90 deg instead of 1.0, pushing
    # S down to 0.83 / 0.73 and losing both wide crossings). A low quantile
    # picks the value where this fibre is alone, which is what I_a means.
    # Swept: q = 0.25 -> 7/8 phantoms, q = 0.35 -> 7/8, q = 0.50 -> 5/8.
    return float(np.quantile(on, CONTRIB_QUANTILE))


def _contributor_intensity(vol, centre, d, arm_min=ARM_MIN, arm_max=ARM_MAX,
                           arm_step=ARM_STEP, half=ARM_HALF):
    """Brightness of ONE contributing structure, away from the junction.

    Two things have to be true at once and they pull in opposite directions:
    the sample must be far enough out that the other structure no longer
    overlaps (inside the junction both fibres contribute and the reading is
    already the sum), but close enough that the structure has not curved out of
    reach. So the ray is walked over a RANGE and the maximum local value taken,
    with a perpendicular tolerance of `half` voxels.

    A single point read at a fixed distance fails exactly where it matters: on
    a curved tangency it lands in background, the denominator collapses and S
    diverges (measured: S = 2.1, 21.9, 136 at arm 6, 8, 10).
    """
    c = np.asarray(centre, float)
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) + EPS)
    vals = []
    for s in np.arange(arm_min, arm_max + 1e-9, arm_step):
        for sgn in (1.0, -1.0):
            vals.append(_local_max(vol, c + sgn * s * d, half))
    vals = np.asarray(vals, float)
    # Keep only samples that are ON the structure, then take their MEDIAN.
    # max() is wrong: near the junction both families still overlap, so the
    # brightest sample on the ray reads the SUM and the crossing's own
    # contributor comes out at 1.58 instead of 1.0, dragging S down to 0.63.
    # min() is equally wrong: it reads the faded far end of a curving tangency.
    # The median of the on-structure samples is the one that is neither.
    on = vals[vals > 0.2 * (vals.max() + EPS)]
    return float(np.median(on)) if on.size else float(vals.max())


def superposition_S(vol, centre, d1, d2, I0=None, half=1, e3=None, snap=True,
                    **kw):
    """S = (I_c - I0) / ((I_a - I0) + (I_b - I0)) at one junction.

    S ~ 1.0 -> the two structures superpose  -> CROSSING
    S ~ 0.5 -> only one structure is present -> TANGENCY

    e3 : the orientation field from `orientation_field(vol)`. When supplied the
         contributors are measured by FOLLOWING each structure; without it the
         old straight-ray estimate is used, which is kept only so the failure
         mode stays reproducible.
    """
    vol = np.asarray(vol)
    if I0 is None:
        I0 = background_level(vol)
    c = np.asarray(centre, float)
    I_c = _local_max(vol, c, half)
    if e3 is not None:
        I_a = _walk_contributor(vol, e3, c, d1, snap=snap)
        I_b = _walk_contributor(vol, e3, c, d2, snap=snap)
    else:
        I_a = _contributor_intensity(vol, c, d1, **kw)
        I_b = _contributor_intensity(vol, c, d2, **kw)
    denom = (I_a - I0) + (I_b - I0)
    if abs(denom) < EPS:
        return np.nan, I_c, I_a, I_b, I0
    return float((I_c - I0) / denom), I_c, I_a, I_b, I0


def verdict_from_S(S):
    """CROSSING / TANGENCY against the physical midpoint, AMBIGUOUS in a band."""
    if not np.isfinite(S):
        return "unscored"
    if S >= S_DECISION + S_BAND:
        return "CROSSING"
    if S <= S_DECISION - S_BAND:
        return "TANGENCY"
    return "AMBIGUOUS"


# ---------------------------------------------------------------------------
# streamline integration through the unified orientation field
# ---------------------------------------------------------------------------

def _nearest_patch(field_pts, p, max_d, tree=None):
    """Index of the field patch nearest p, or None if none within max_d.

    A KD-tree when scipy is available: the brute-force scan is O(N) per
    integration step, and with a few hundred patches and thousands of steps per
    loss evaluation that dominates everything else in the loop. The linear path
    is kept as a fallback so the module does not hard-depend on scipy.
    """
    if tree is not None:
        d, i = tree.query(np.asarray(p, float), k=1,
                          distance_upper_bound=float(max_d))
        if not np.isfinite(d):
            return None, d
        return int(i), float(d)
    d = np.linalg.norm(field_pts - np.asarray(p, float)[None, :], axis=1)
    i = int(np.argmin(d))
    return (i, d[i]) if d[i] <= max_d else (None, d[i])


def integrate(field, shape, theta, junction_verdicts=None, rng=None):
    """Walk streamlines through the orientation field under parameters theta.

    theta keys: step, angle_deg, seed_density, link_weight.

    At a junction the walker picks the continuing orientation closest to its
    current heading, subject to the angular threshold. If that junction's
    superposition verdict is TANGENCY the walker is NOT allowed to switch onto
    the other structure -- which is exactly how "S folds into the linking
    decision": a tangency may be followed along, never crossed through.
    """
    rng = rng or np.random.default_rng(0)
    pts = np.array([p["centre"] for p in field], float)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
    except Exception:
        tree = None
    dirs = [np.atleast_2d(np.asarray(p["dirs"], float)) for p in field]
    step = float(theta["step"])
    cosmax = np.cos(np.deg2rad(float(theta["angle_deg"])))
    max_d = max(float(theta.get("assoc_radius", 4.0)), step * 2.0)
    verdicts = junction_verdicts or {}

    # Seeds. `seed_density` is seeds PER PATCH and may exceed 1: sampling
    # without replacement capped the stream count at the patch count, which is
    # why a 96^3 i74 crop produced 396 streamlines covering 1.27% of tissue and
    # a D_rec 175x below the target. The loss was then numerically identical to
    # one computed with no streamlines at all -- deleting every tract moved the
    # MSE by 0.0001 -- so nothing downstream could respond to theta.
    #
    # Repeated seeds are JITTERED within the patch. Without that, drawing the
    # same patch twice re-runs an identical deterministic walk and deposits the
    # same line twice, which raises the density along existing tracts instead
    # of covering the volume.
    n_seed = max(int(round(len(field) * float(theta["seed_density"]))), 1)
    idx = rng.choice(len(field), size=n_seed, replace=n_seed > len(field))
    jitter = float(theta.get("seed_jitter", 0.0))

    streams, weights = [], []
    for i in idx:
        for d0 in dirs[i]:
            for sgn in (1.0, -1.0):
                p = pts[i].copy()
                if jitter > 0:
                    p = p + rng.normal(scale=jitter, size=3)
                    if not all(0 <= p[k] < shape[k] for k in range(3)):
                        continue
                u = sgn * d0 / (np.linalg.norm(d0) + EPS)
                path = [p.copy()]
                for _ in range(int(theta.get("max_steps", 200))):
                    p = p + step * u
                    if not all(0 <= p[k] < shape[k] for k in range(3)):
                        break
                    j, dist = _nearest_patch(pts, p, max_d, tree)
                    if j is None:
                        break
                    cand = dirs[j]
                    dots = cand @ u
                    k = int(np.argmax(np.abs(dots)))
                    if abs(dots[k]) < cosmax:
                        break                       # turn too sharp: terminate
                    # tangency: forbid switching onto the other structure
                    if verdicts.get(j) == "TANGENCY" and cand.shape[0] > 1:
                        same = int(np.argmax(np.abs(cand @ u)))
                        if same != k:
                            break
                    u = np.sign(dots[k]) * cand[k] / (np.linalg.norm(cand[k]) + EPS)
                    path.append(p.copy())
                if len(path) >= 2:
                    streams.append(np.asarray(path))
                    weights.append(float(theta["link_weight"]))
    return streams, weights


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def loss(vol, field, D_obs, tissue, theta, junctions, I0, rng=None,
         verdict_penalty=1.0):
    """L = MSE(D_rec, D_obs) over tissue + a penalty for links that contradict S.

    The penalty is what makes the streamlines "learn" the topology: a junction
    whose S says TANGENCY but which the current tracts drive straight through
    contributes to the loss, so parameter settings that traverse tangencies are
    pushed away from.
    """
    verdicts = {j["field_index"]: j["verdict"] for j in junctions}
    streams, w = integrate(field, vol.shape, theta, verdicts, rng=rng)
    D_rec = rasterise(streams, vol.shape, step=float(theta["step"]), weights=w)

    # Fit the per-streamline weights before scoring: convex, w >= 0, unique
    # minimum. Seeding more streamlines improves COVERAGE and inflates
    # AMPLITUDE together, and an unfitted MSE cannot separate them -- on i74,
    # 1532 -> 24538 streams took coverage 42% -> 90% but drove the raw loss
    # 1.21 -> 16.95, so the optimiser's rational move was to seed FEWER tracts.
    # Fitting the weights removes that confound entirely and, unlike a single
    # global scale, lets an unsupported tract go to exactly zero.
    d_obs = D_obs[tissue]
    Phi = design_matrix(streams, vol.shape, tissue, step=float(theta["step"]))
    w_fit, fit_info = fit_weights(Phi, d_obs,
                                  l1=float(theta.get("l1", L1_WEIGHT)))
    data_term = float(np.mean((Phi @ w_fit - d_obs) ** 2))
    D_rec = rasterise(streams, vol.shape, step=float(theta["step"]),
                      weights=w_fit)

    # Junction targets as a RATIO to the single-fibre density, not the absolute
    # 1 and 2. Once the weights are fitted, D_rec is in the units of D_obs
    # (unit-mean anisotropy over the support), in which "one fibre" is not 1.0.
    # Testing a fitted density against an absolute threshold was the same
    # mistake as an absolute intensity gate: the level never transfers, only
    # the ratio does. `unit` is measured from the patches HOST calls
    # single-orientation, which is where exactly one fibre is believed to be.
    singles = [p["centre"] for p in field if p["n_dirs"] == 1]
    dens_single = [density_at(D_rec, c, half=1) for c in singles]
    unit = float(np.median(dens_single)) if dens_single else 0.0
    if not np.isfinite(unit) or unit <= EPS:
        unit = 1.0                       # no single-fibre reference: ratio off

    contribs = []
    pen = 0.0
    for j in junctions:
        d_here = density_at(D_rec, j["centre"], half=1)
        n_fibres = d_here / unit
        if j["verdict"] == "TANGENCY":
            # a tangency should carry ~1 fibre's worth, not 2
            c = max(n_fibres - 1.0, 0.0) ** 2
        elif j["verdict"] == "CROSSING":
            # a crossing should carry ~2
            c = (n_fibres - 2.0) ** 2
        else:
            c = 0.0
        pen += c
        contribs.append(dict(centre=[int(v) for v in j["centre"]],
                             S=float(j["S"]), verdict=j["verdict"],
                             density=float(d_here), n_fibres=float(n_fibres),
                             loss_contribution=float(c)))
    pen = verdict_penalty * pen / max(len(junctions), 1)
    total = (0.0 if not np.isfinite(data_term) else data_term) + pen
    return total, dict(mse=float(data_term), penalty=float(pen),
                       total=float(total), n_streams=len(streams),
                       single_fibre_unit=float(unit),
                       n_active=int(fit_info["n_active"]),
                       nnls_iterations=int(fit_info["iterations"]),
                       junctions=contribs)


# ---------------------------------------------------------------------------
# theta optimisation -- coordinate search
# ---------------------------------------------------------------------------

# seed_density is seeds PER PATCH and is no longer capped at 1; assoc_radius
# was 4.0, below the 5.2 needed to cover a stride-6 patch grid, so the tracts
# could not even reach between neighbouring patches. See loss_sensitivity.py.
THETA0 = dict(step=0.75, angle_deg=45.0, seed_density=32.0, link_weight=1.0,
              assoc_radius=8.0, seed_jitter=3.0, max_steps=200,
              l1=1e-3)
# Only TWO parameters are searched now, which is what makes an exhaustive
# surface sweep possible instead of a descent (see loss_surface.py).
#
#   link_weight   left the search because it multiplies every streamline
#                 equally, which the fitted per-streamline weights subsume
#                 exactly -- searching both makes the pair unidentifiable.
#   seed_density  left the search because it is MONOTONE: more streamlines is
#                 more columns in Phi, a larger feasible set for the fit, so
#                 the residual can essentially only fall. It pinned to the grid
#                 ceiling at 32 and again at 64. It is a compute BUDGET, not
#                 something to optimise -- set it as high as runtime allows.
L1_WEIGHT = 1e-3         # price per unit of tract weight; see fit_weights
GRID = dict(step=[0.5, 0.75, 1.0, 1.5],
            angle_deg=[25.0, 35.0, 45.0, 60.0])


def optimise(vol, field, junctions, D_obs, tissue, I0, n_iter=6, seed=0,
             verbose=True):
    """Coordinate descent on theta. Returns (theta, trace).

    Coordinate descent rather than a gradient: L is a non-differentiable
    function of theta (streamlines terminate discretely), so a derivative does
    not exist. Each iteration sweeps one parameter at a time over its grid and
    keeps the best -- cheap, and it cannot diverge.
    """
    rng = np.random.default_rng(seed)
    theta = dict(THETA0)
    best, info = loss(vol, field, D_obs, tissue, theta, junctions, I0, rng=rng)
    trace = [dict(iteration=-1, theta=dict(theta), **{k: info[k] for k in
                                                      ("mse", "penalty", "total",
                                                       "n_streams")})]
    if verbose:
        print(f"    init  L={best:.4f} (mse {info['mse']:.4f} + pen "
              f"{info['penalty']:.4f}), {info['n_streams']} streams")

    for it in range(n_iter):
        improved = False
        for key, values in GRID.items():
            cur = theta[key]
            for v in values:
                if v == cur:
                    continue
                trial = dict(theta); trial[key] = v
                L, inf2 = loss(vol, field, D_obs, tissue, trial, junctions, I0,
                               rng=np.random.default_rng(seed))
                if np.isfinite(L) and L < best - 1e-9:
                    best, theta, info, improved = L, trial, inf2, True
        trace.append(dict(iteration=it, theta=dict(theta),
                          **{k: info[k] for k in ("mse", "penalty", "total",
                                                  "n_streams")}))
        if verbose:
            print(f"    iter {it}: L={best:.4f} (mse {info['mse']:.4f} + pen "
                  f"{info['penalty']:.4f})  theta="
                  f"step {theta['step']}, angle {theta['angle_deg']}, "
                  f"seed {theta['seed_density']}, w {theta['link_weight']}")
        if not improved:
            if verbose:
                print("    converged (no coordinate move reduces L)")
            break
    return theta, trace, info


# ---------------------------------------------------------------------------
# phantom validation
# ---------------------------------------------------------------------------

def field_from_gated(gated, vol, I0=None, e3=None, host_only=True, scales=None):
    """Consume Stage 2's GatedField -- the actual Stage 2 -> Stage 3 join.

    Two frame conversions have to happen here and both are easy to get silently
    wrong:

    * `host_fod.analyse` returns directions as (x, y, z) = (axis2, axis1,
      axis0), and Skielex's e3 uses the same order. Stage 3 indexes positions as
      (z, y, x). So every direction is reversed on the way in. Skipping this
      does not crash -- it just makes every streamline walk a mirrored path,
      which reads as "the structure curves away" rather than as a bug.
    * Stage 2 stores patch centres already in (z, y, x), so those pass through.

    Junctions are the multi-orientation patches, each scored by superposition S
    at the point Stage 2 routed to HOST.
    """
    if e3 is None:
        # No hidden scale. Either the caller hands in the field it already
        # built (the pipeline path -- one decomposition, reused), or it hands in
        # the `scales` that certify what to build. Falling back to a default
        # (sigma, rho) here is what let Stage 3 measure the volume at a scale
        # Stage 1 had never chosen.
        if scales is None:
            raise TypeError(
                "field_from_gated needs an orientation field: pass e3=, or "
                "scales= from select_scales. There is no default scale.")
        from core.tensor import field_for
        e3 = field_for(vol, scales, invariants=False)["e3"]
    # I0=None now means PER-JUNCTION local baseline, not one global air level.
    # A single air-referenced I_0 made the CROSSING verdict arithmetically
    # unreachable on i74 -- see `local_baseline` and `s_calibration.py`. An
    # explicit I0 still overrides, so the old behaviour stays reproducible.
    global_I0 = background_level(vol) if I0 is None else float(I0)
    field, junctions = [], []
    for patch_index, p in enumerate(gated.patches):
        # host_only: Stage 3 sees ONLY what the routing gate passed to HOST.
        # Previously every patch entered the field, so streamlines were seeded
        # from -- and the TDI accumulated over -- single-orientation patches the
        # gate had already declined to route. Those can never be demoted, so
        # they contributed nothing but cost and diluted the density target.
        if host_only and not p["gated"]:
            continue
        dirs = [np.asarray(d, float)[::-1] for d in p["dirs"]]     # (x,y,z)->(z,y,x)
        idx = len(field)
        # patch_index is the position in gated.patches, which is NOT idx once
        # anything is filtered out. The demotion writes back to the GatedField,
        # so it needs the original index, not the field's.
        field.append(dict(centre=tuple(p["centre"]), dirs=dirs,
                          n_dirs=len(dirs), c_s=p["c_s"], FA=p["FA"],
                          gated=p["gated"], label=p["label"],
                          bucket=p.get("bucket", BUCKET_CANDIDATE),
                          patch_index=patch_index))
        # A junction requires the CANDIDATE bucket, not merely two directions.
        # Stage 2 used to hand over `dirs[:2]` for every routed patch, so a
        # >=3-peak "complex" patch or a flat "disorganised" one arrived looking
        # exactly like a clean crossing and was scored, and demoted, as one. On
        # the i74 run that was 33 of 67 junctions. Patches outside the candidate
        # bucket now carry a single direction and make no two-fibre claim.
        if p.get("bucket", BUCKET_CANDIDATE) == BUCKET_CANDIDATE and len(dirs) >= 2:
            I0_here = (local_baseline(vol, p["centre"], half=BASELINE_HALF)
                       if I0 is None else global_I0)
            S, Ic, Ia, Ib, _ = superposition_S(vol, p["centre"], dirs[0], dirs[1],
                                               I0=I0_here, e3=e3)
            junctions.append(dict(field_index=idx, patch_index=patch_index,
                                  centre=tuple(p["centre"]),
                                  dirs=dirs, S=S, verdict=verdict_from_S(S),
                                  I_c=Ic, I_a=Ia, I_b=Ib, I_0=I0_here))
    return field, junctions, global_I0


def build_field_and_junctions(vol, half, centre, sigma_grad, search=6,
                              stride=6, air_frac=AIR_FRAC):
    """Minimal Stage-1/2 field for the phantom: patches on a grid, HOST peaks
    where it finds them, plus the junction at/near `centre`."""
    field, junctions = [], []
    air = air_frac * float(vol.max())
    I0 = background_level(vol, air_frac)
    lo = half + 1
    for z in range(lo, vol.shape[0] - lo, stride):
        for y in range(lo, vol.shape[1] - lo, stride):
            for x in range(lo, vol.shape[2] - lo, stride):
                p = vol[z - half:z + half + 1, y - half:y + half + 1,
                        x - half:x + half + 1]
                if p.mean() < air:
                    continue
                m = analyse(p, sigma_grad=sigma_grad, order=4)
                if m is None or not len(m["dirs"]):
                    continue
                ds = [np.asarray(d, float)[::-1] for d in m["dirs"][:2]]
                field.append(dict(centre=(z, y, x), dirs=ds,
                                  n_dirs=len(ds), c_s=m["c_s"], FA=m["FA"]))
    # the labelled junction: nearest field patch with >=2 dirs to `centre`
    multi = [(i, p) for i, p in enumerate(field) if p["n_dirs"] >= 2]
    if multi:
        d = [np.linalg.norm(np.array(p["centre"]) - np.array(centre))
             for _i, p in multi]
        i, p = multi[int(np.argmin(d))]
        S, Ic, Ia, Ib, _ = superposition_S(vol, p["centre"], p["dirs"][0],
                                           p["dirs"][1], I0=I0)
        junctions.append(dict(field_index=i, centre=p["centre"],
                              dirs=p["dirs"], S=S, verdict=verdict_from_S(S),
                              I_c=Ic, I_a=Ia, I_b=Ib, I_0=I0))
    return field, junctions, I0


def validate(n_iter=4, verbose=True):
    # The phantoms are at the repo root, which the preamble above guarantees is
    # on sys.path. This used to be a two-deep try/except that reached into
    # `mass_conservation/` and silently returned {} -- printing "validation
    # skipped" and exiting 0 -- whenever neither import shape worked. A
    # validation that quietly does nothing is worse than one that crashes, and
    # the fallback only existed because the builders lived in a sibling folder.
    from core.localized_phantoms import (raw_localized_crossing, raw_kissing,
                                    CROSS_SHAPE, CROSS_R1, CROSS_TAPER,
                                    CROSS_HALF, KISS_HALF)
    print("=" * 84)
    print("STAGE 3 -- self-optimising tractography, phantom validation")
    print("=" * 84)

    X = raw_localized_crossing(CROSS_SHAPE, 40.0, 12.0, 1.5, CROSS_R1, CROSS_TAPER)
    K = raw_kissing(4.0, 0.028, 1.6, 1.6)
    cases = [("TRUE CROSSING", X, tuple(s // 2 for s in CROSS_SHAPE), CROSS_HALF,
              "CROSSING"),
             ("TANGENCY (kissing)", K, tuple(s // 2 for s in K.shape), KISS_HALF,
              "TANGENCY")]

    from select_scales import select_scales
    from certify_gate_scales import load_gate_scales
    gs = load_gate_scales()

    out = {}
    for name, vol, centre, half, truth in cases:
        print(f"\n--- {name}  (truth: {truth}) ---")
        # Each phantom certifies its own scale rather than inheriting a literal.
        sc = select_scales(vol, mode="global", verbose=False)
        print(f"    scales: sigma={sc['sigma_global']:.3g}, "
              f"rho={sc['rho_global']:.3g}; dir sigma_grad="
              f"{gs['dir_sigma_grad']:g}")
        lam = tensor_field(vol, sc["sigma_global"], sc["rho_global"],
                           invariants=False)["lam"]
        D_obs, tissue = anisotropy_target(vol, lam=lam)
        field, junctions, I0 = build_field_and_junctions(
            vol, half, centre, sigma_grad=gs["dir_sigma_grad"])
        print(f"    field: {len(field)} patches, "
              f"{sum(p['n_dirs'] >= 2 for p in field)} multi-orientation; I_0={I0:.3f}")
        if not junctions:
            print("    no multi-orientation junction found -- nothing to score")
            out[name] = dict(truth=truth, skipped=True)
            continue
        j = junctions[0]
        print(f"    junction at {j['centre']}: I_c={j['I_c']:.3f} "
              f"I_a={j['I_a']:.3f} I_b={j['I_b']:.3f}")
        print(f"    S = {j['S']:.3f}  ->  verdict {j['verdict']}   "
              f"(truth {truth}) {'CORRECT' if j['verdict'] == truth else 'WRONG'}")
        theta, trace, info = optimise(vol, field, junctions, D_obs, tissue, I0,
                                      n_iter=n_iter, verbose=verbose)
        out[name] = dict(truth=truth, skipped=False, S=float(j["S"]),
                         verdict=j["verdict"], correct=j["verdict"] == truth,
                         theta=theta, trace=trace,
                         junction_log=info["junctions"],
                         L_initial=float(trace[0]["total"]),
                         L_final=float(trace[-1]["total"]))
        print(f"    L: {trace[0]['total']:.4f} -> {trace[-1]['total']:.4f}")

    scored = [v for v in out.values() if not v.get("skipped")]
    ok = sum(v["correct"] for v in scored)
    print("\n" + "=" * 84)
    print(f"SUPERPOSITION VERDICTS: {ok}/{len(scored)} correct")
    for k, v in out.items():
        if not v.get("skipped"):
            print(f"  {k:22} S={v['S']:.3f} -> {v['verdict']:10} "
                  f"(truth {v['truth']})")
    print("=" * 84)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 3 self-optimising tractography")
    ap.add_argument("--iters", type=int, default=4)
    a = ap.parse_args()
    res = validate(n_iter=a.iters)
    with open("self_optimising_results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("wrote self_optimising_results.json")
