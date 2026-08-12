"""
dense_tracker.py -- cone-search tractography over the dense per-voxel field.

This is `cone_search`'s tracker, generalised to the two-direction field and
rewritten to walk every streamline at once. Both changes are load-bearing:

TWO DIRECTIONS
--------------
Support was `intensity * |d . e3|`: one direction per voxel, so at a crossing the
tracker could only ever agree with whichever fibre e3 happened to report -- and
at the junction centre e3 reports neither, it reports their bisector. Now

    support(d) = intensity * max(|d . vec1|, |d . vec2|)

so a walker arriving along either fibre keeps its own evidence. 

-> `vec2` is zero
aka wherever Stage 2 found one direction, and |d . 0| = 0, so the max degrades
EXACTLY to the old expression in single-fibre tissue -- there is no branch, and
no behaviour change where there is nothing to choose between. That is also why a
demotion needs no special case in the tracker: `DenseField.demote` zeroes vec2,
and the second option simply stops existing.

SPEED
-----
The recorded sweep in `results/cone_search.json` cost 76 minutes for 24 settings
on a 40^3 crop, worst cell 53 minutes. An optimiser that re-tracks every
iteration cannot pay that, and the cost was not in the physics: it was one
Python loop per seed, per step, rebuilding the cone from scratch each time.

Three fixes, no approximation in any of them:
  * all live walkers advance together, so each step is a handful of array ops
    rather than N interpreter round-trips;
  * the cone is precomputed ONCE per (theta, alpha) as weights in the walker's
    own frame, then rotated per walker -- `cone_directions` used to rebuild the
    identical trig at every step of every walker;
  * dead walkers are dropped from the working set instead of being carried.

Results are identical to the scalar tracker up to floating point; `selftest`
below checks that against `cone_search.track` rather than asserting it.

Frame: positions and directions are both (z, y, x). See `dense_field`.
"""

import numpy as np
from scipy.ndimage import map_coordinates

EPS = 1e-12

STEP = 1.0          # voxels advanced per iteration; lookahead is a separate knob
MAX_STEPS = 90
N_LOOKAHEAD = 6     # samples along each candidate ray

# How much brightness is allowed to influence WHICH WAY the walker turns. 0 =
# not at all (the direction comes from the field, brightness only decides when
# the walk has ended); 1 = `cone_search`'s original. See `support` for the
# measurement behind the default, and `single_fibre_check` for the gate it has
# to pass.
INTENSITY_EXP = 0.0


def cone_weights(theta, alpha):
    """Cone directions as (w_v, w_a, w_b) weights in the walker's own frame.

    d = w_v * v + w_a * a + w_b * b, for an orthonormal (v, a, b). Precomputing
    these is what makes the cone independent of the walker: the trig depends
    only on (theta, alpha), and the per-walker part is one rotation.

    Work scales as (theta/alpha)^2, which is why alpha is a discretisation knob
    and not a model parameter -- halving it quadruples the cost for a strictly
    finer sampling of the same cone.
    """
    wv, wa, wb = [1.0], [0.0], [0.0]                    # the straight-ahead ray
    for i in range(1, max(int(round(theta / alpha)), 0) + 1):
        tilt = i * alpha
        n_phi = max(int(round(2 * np.pi * np.sin(tilt) / alpha)), 1)
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            wv.append(np.cos(tilt))
            wa.append(np.sin(tilt) * np.cos(phi))
            wb.append(np.sin(tilt) * np.sin(phi))
    return (np.asarray(wv, float), np.asarray(wa, float), np.asarray(wb, float))


RELAX_RINGS = 3          # tilt rings used to SAMPLE the cone in "relax" mode


def cone_sample_weights(theta, rings=RELAX_RINGS):
    """Cone sampling directions for `turn="relax"`, at a FIXED resolution.

    In "sample" mode alpha is the angular step, so `cone_weights` derives its
    ring count from it. In "relax" mode alpha means something else entirely --
    how far along the turn the walker stops -- so the cone still has to be
    sampled, but by something that is NOT alpha. That is this: a fixed number of
    rings spanning theta, so the sampling density is a cost constant rather than
    a model parameter, and two different alphas sample the identical cone.

    Keeping it fixed is the point. If the sampling tracked alpha, then changing
    alpha would change both the turn fraction AND which vectors were averaged,
    and no measurement could separate the two.
    """
    return cone_weights(theta, theta / max(int(rings), 1))


def _mean_director(vec1, vec2, x, dirs, lookahead, n, v_now, dyadic_fields):
    """The mean orientation of the field inside the cone, as a unit director.

    THE AVERAGE IS OVER OUTER PRODUCTS, NOT VECTORS. e3 is a director -- v and
    -v are the same fibre and the eigensolver's sign is arbitrary per voxel --
    so a componentwise mean cancels v against -v wherever two samples disagree.
    On the crossing phantom 13.2% of adjacent voxel pairs are sign-opposed, so a
    naive mean would not be a small error, it would be near zero on a third of
    the field. `v v^T` has no sign to cancel, so summing it and taking the
    principal eigenvector is the correct mean of axial data. See `dyadic`.

    Where the field carries TWO directions the better-aligned one is taken per
    sample, by the same `max` the tracker's support uses: averaging both would
    return their bisector at every crossing, which is the one direction no fibre
    actually goes.

    Returns (N, 3) unit vectors, signed to agree with the current heading
    `v_now` -- an axis has no sign, and reversing the walker is not a turn.
    """
    N, K = dirs.shape[0], dirs.shape[1]
    s = np.linspace(lookahead / n, lookahead, n)
    pts = (x[:, None, None, :] + s[None, None, :, None]
           * dirs[:, :, None, :]).reshape(-1, 3)

    def _accum(T):
        return np.stack([_sample(c, pts).reshape(N, K * n) for c in T], axis=-1)

    if not dyadic_fields:
        vec1 = dyadic(vec1)
        vec2 = None if vec2 is None else dyadic(vec2)
    D1 = _accum(vec1)                                   # (N, K*n, 6)
    if vec2 is not None:
        D2 = _accum(vec2)
        # (d . v)^2 against the CURRENT heading picks which fibre each sample
        # belongs to; ties and empty vec2 fall to family 1.
        def _a2(D):
            out = np.zeros(D.shape[:2])
            for k, (i, j, w) in enumerate(_DYAD):
                out += w * D[:, :, k] * (v_now[:, i] * v_now[:, j])[:, None]
            return out
        take2 = (_a2(D2) > _a2(D1))[:, :, None]
        D1 = np.where(take2, D2, D1)

    M6 = D1.mean(axis=1)                                # (N, 6) mean v v^T
    T = np.empty((len(M6), 3, 3))
    for k, (i, j, _w) in enumerate(_DYAD):
        T[:, i, j] = M6[:, k]
        T[:, j, i] = M6[:, k]
    w_eig, V = np.linalg.eigh(T)                        # ascending
    m = V[:, :, -1]                                     # principal axis
    flip = np.einsum("ij,ij->i", m, v_now) < 0.0
    m[flip] *= -1.0
    return _unit(m, axis=1)


def _slerp(v, m, alpha, max_angle=None):
    """Rotate from v toward m along the great circle, stopping at fraction alpha.

    alpha = 0 keeps the current heading, alpha = 1 goes the whole way, and the
    path between is the geodesic on the sphere -- a linear blend would cut the
    chord and arrive off the unit sphere, which after renormalising is a
    DIFFERENT angle from the fraction requested.

    WHY max_angle EXISTS. It is tempting to argue the clamp is unnecessary --
    that a mean of vectors sampled inside a theta cone must itself lie within
    theta. That argument is false, and measurably so: the cone bounds WHERE the
    field is read, not WHAT is found there. The sampled directors are whatever
    the field holds at those positions and are under no angular constraint at
    all. Measured on the curved phantom at theta = 17.19 deg, the unclamped turn
    reached 89.87 deg -- an outright reversal, five times the supposed bound.

    Since theta is defined as "the widest turn allowed in one step", the target
    is first pulled back onto the theta cone, then alpha interpolates toward it.
    So alpha = 1 turns by min(angle-to-mean, theta), never more.
    """
    c = np.clip(np.einsum("ij,ij->i", v, m), -1.0, 1.0)
    ang = np.arccos(c)
    tgt = m.copy()
    if max_angle is not None:
        over = ang > float(max_angle)
        if over.any():
            # Pull the target back to exactly max_angle along the same geodesic.
            f = (float(max_angle) / ang[over])[:, None]
            s = np.sin(ang[over])[:, None]
            tgt[over] = (np.sin((1.0 - f) * ang[over][:, None]) * v[over]
                         + np.sin(f * ang[over][:, None]) * m[over]) / s
            tgt[over] = _unit(tgt[over], axis=1)
            ang[over] = float(max_angle)
    out = v.copy()
    move = ang > 1e-9
    if move.any():
        a = float(np.clip(alpha, 0.0, 1.0))
        s = np.sin(ang[move])
        out[move] = (np.sin((1.0 - a) * ang[move])[:, None] * v[move]
                     + np.sin(a * ang[move])[:, None] * tgt[move]) / s[:, None]
    return _unit(out, axis=1)


def _unit(v, axis=-1):
    """Normalise WITHOUT the `norm + EPS` idiom used elsewhere in the repo.

    That idiom divides an already-unit vector by 1 + 1e-12, which is invisible
    once but not when it compounds: the walker's heading is renormalised at
    every step, so each step comes out 1e-12 short and the error accumulates
    linearly along the tract. Measured against `cone_search.track` on a straight
    fibre, that alone moved the end point by 2e-11 and cost one extra step
    before the volume boundary -- a path one sample longer than the scalar
    tracker produces from identical inputs. Dividing by max(norm, EPS) is the
    same guard against a zero vector with no bias on a good one.
    """
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


def _basis(v):
    """Two unit vectors completing an orthonormal frame with each row of v."""
    a = np.zeros_like(v)
    a[:, 0] = 1.0
    flip = np.abs(v[:, 0]) > 0.9
    a[flip] = 0.0
    a[flip, 1] = 1.0
    a = a - (np.einsum("ij,ij->i", a, v))[:, None] * v
    a = _unit(a, axis=1)
    return a, np.cross(v, a)


def _sample(vol, pts):
    """Trilinear read of `vol` at (M, 3) continuous (z, y, x) positions."""
    return map_coordinates(vol, pts.T, order=1, mode="nearest")


# Upper triangle of a symmetric 3x3, and the multiplicity each entry carries in
# d^T T d (the off-diagonals appear twice).
_DYAD = ((0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0),
         (0, 1, 2.0), (0, 2, 2.0), (1, 2, 2.0))


def dyadic(v):
    """The 6 unique components of the outer product v v^T, per voxel.

    Why the tracker cannot interpolate v directly
    ---------------------------------------------
    e3 is a DIRECTOR: v and -v are the same fibre, and the eigensolver picks the
    sign arbitrarily at each voxel independently. Trilinear interpolation
    averages components, so wherever two neighbours were handed opposite signs
    it averages v against -v and returns something near ZERO -- not a direction
    at all. Measured on the crossing phantom: 13.2% of adjacent voxel pairs are
    sign-opposed, and sampling vec1 at half-voxel offsets gives |v| with a
    median of 0.78 and 31.8% of reads below 0.5.

    That is not a small numerical blemish. The alignment term |d . v| is
    attenuated by whatever fraction the cancellation destroyed, so on a third of
    the field the tracker was choosing its direction almost entirely on
    brightness -- which is exactly the symptom `single_fibre_check` measures, and
    why sharpening the alignment exponent barely helped: there was no signal
    left to sharpen.

    v v^T is invariant under v -> -v, so interpolating it has no sign to cancel,
    and d^T T d = (d . v)^2 recovers the alignment exactly at voxel centres and
    averages orientations correctly between them. This is the standard
    construction for averaging axial data.
    """
    v = np.asarray(v, float)
    return np.stack([v[i] * v[j] for i, j, _w in _DYAD]).astype(np.float32)


def _align2(T, pts, dirs, shape):
    """(d . v)^2 from an interpolated dyadic. T is (6, Z, Y, X)."""
    N, K, n = shape
    out = np.zeros((N, K, n))
    for (i, j, w), comp in zip(_DYAD, T):
        out += w * _sample(comp, pts).reshape(N, K, n) * (
            dirs[:, :, i] * dirs[:, :, j])[:, :, None]
    return np.maximum(out, 0.0)


def support(vol, vec1, vec2, x, dirs, lookahead, n=N_LOOKAHEAD,
            align_exp=1.0, dyadic_fields=True, intensity_exp=1.0):
    """Evidence for each candidate direction. Returns (score, gate).

    score : what the walker maximises,  mean(I^intensity_exp * align^align_exp)
    gate  : what decides it has run off the structure, always mean(I * align)

    x     (N, 3)      current positions
    dirs  (N, K, 3)   unit candidate directions
    -> two (N, K) arrays

    Why these are two quantities and not one
    ----------------------------------------
    They answer different questions. "Is there still a fibre here" is a
    statement about brightness and has to stay coupled to it. "Which way does
    the fibre go" is a statement about orientation, and weighting it by
    brightness makes the walker steer toward whatever is bright rather than
    along what the structure tensor measured.

    Measured on the crossing phantom, in voxels carrying exactly ONE direction,
    where there is nothing to decide and the tract should simply follow e3:

        intensity * align   (intensity_exp 1)   median 12.74 deg off e3
        align alone         (intensity_exp 0)   median  5.00 deg
        align alone, lookahead 1                median  2.71 deg
        e3's own voxel-to-voxel coherence       median  1.39 deg

    and the deviation is FLAT in the cone's angular sampling -- 11.98 deg at
    alpha 0.4 through 12.93 deg at alpha 0.025, a 16x refinement -- so it is the
    support function, not the discretisation. The lookahead accounts for the
    rest: averaging alignment over a ray several voxels long asks where the
    fibre goes next, not where it points here.

    `align_exp` sharpens the angular term instead of removing the intensity
    one. It was the first fix tried and it is NOT the one adopted, because it
    does not work well enough: with intensity still weighting the turn, going
    from exponent 1 to 16 only moved the deviation 12.46 -> 6.91 degrees and
    cost crossing traversal (0.195 -> 0.170). Dropping intensity from the turn
    outright reaches 4.82 degrees at no cost to traversal at all. Kept as a
    knob because the measurement is worth being able to reproduce.
    """
    N, K = dirs.shape[0], dirs.shape[1]
    t = np.linspace(STEP, lookahead, n)
    pts = x[:, None, None, :] + t[None, None, :, None] * dirs[:, :, None, :]
    flat = pts.reshape(-1, 3)
    inten = _sample(vol, flat).reshape(N, K, n)

    if dyadic_fields:
        a1 = np.sqrt(_align2(vec1, flat, dirs, (N, K, n)))
        align = a1 if vec2 is None else np.maximum(
            a1, np.sqrt(_align2(vec2, flat, dirs, (N, K, n))))
    else:
        # Componentwise interpolation of the director -- kept only so the
        # equivalence test against `cone_search.track` still has something to
        # compare to. See `dyadic` for why it is wrong.
        a1 = np.abs(np.einsum("nksc,nkc->nks",
                              np.stack([_sample(vec1[c], flat) for c in range(3)], -1)
                              .reshape(N, K, n, 3), dirs))
        align = a1 if vec2 is None else np.maximum(
            a1, np.abs(np.einsum("nksc,nkc->nks",
                                 np.stack([_sample(vec2[c], flat)
                                           for c in range(3)], -1)
                                 .reshape(N, K, n, 3), dirs)))
    gate = (inten * align).mean(axis=2)
    if align_exp != 1.0:
        align = align ** float(align_exp)
    if intensity_exp == 1.0:
        w = inten
    elif intensity_exp == 0.0:
        w = 1.0
    else:
        w = np.power(np.maximum(inten, 0.0), float(intensity_exp))
    score = (w * align).mean(axis=2)
    return score, gate


def track(vol, field, seeds, seed_dirs, alpha, lookahead, theta, floor,
          step=STEP, max_steps=MAX_STEPS, n_lookahead=N_LOOKAHEAD,
          both_ways=True, align_exp=1.0, dyadic_fields=True,
          intensity_exp=INTENSITY_EXP, turn="sample", rings=RELAX_RINGS):
    """Walk cone-search streamlines from `seeds` along `seed_dirs`.

    vol        (Z, Y, X) image the support is read from
    field      DenseField, or a (3, Z, Y, X) array for the single-direction case
    seeds      (S, 3) start positions, (z, y, x)
    seed_dirs  (S, 3) start directions
    floor      support below which a walker stops -- it has left the structure
    both_ways  walk each seed forwards and backwards. The field is NEMATIC: e3
               and -e3 are the same fibre, so a seed has no intrinsic direction
               and taking only one would sample half of each tract.

    TWO MEANINGS OF alpha, selected by `turn`
    -----------------------------------------
    "sample"  (default, unchanged) alpha is the ANGULAR STEP in radians. The
              cone is enumerated at that resolution and the walker jumps to the
              single best-supported candidate. theta is then quantised in units
              of alpha -- `cone_weights` builds round(theta/alpha) rings -- so
              the two parameters are not independent and settings sharing a ring
              count are the identical cone.

    "relax"   alpha is a FRACTION in [0, 1]. The cone is sampled at a fixed
              resolution (`rings`, not alpha), the mean orientation of the field
              inside it is computed, that mean is clamped onto the theta cone,
              and the walker rotates from its current heading toward it,
              stopping at fraction alpha of the way. alpha = 0 never turns;
              alpha = 1 turns by min(angle-to-mean, theta).

              The clamp is load-bearing, not defensive: the cone bounds where
              the field is READ, not what is found there, so an unclamped mean
              turned by 89.87 deg at theta = 17.19 deg on the curved phantom.
              See `_slerp`.

    The difference is not cosmetic. "sample" picks ONE direction by argmax, so
    it is as sharp as its discretisation and no sharper, and its derivative in
    theta is exactly zero. "relax" uses EVERY vector in the cone and is smooth
    in all three parameters by construction, and it removes the theta/alpha ring
    degeneracy entirely because alpha no longer controls the sampling.

    Returns a list of (M, 3) paths.
    """
    if turn not in ("sample", "relax"):
        raise ValueError(f"turn must be 'sample' or 'relax', got {turn!r}")
    if turn == "relax" and not 0.0 <= alpha <= 1.0:
        raise ValueError(f"turn='relax' needs alpha in [0, 1], got {alpha}; "
                         f"it is a fraction of the turn, not an angle")
    vec1, vec2 = _fields(field)
    if dyadic_fields:
        vec1 = dyadic(vec1)
        vec2 = None if vec2 is None else dyadic(vec2)
    shape = vol.shape
    s = np.asarray(seeds, float)
    d = _unit(np.asarray(seed_dirs, float), axis=1)
    if both_ways:
        s = np.concatenate([s, s])
        d = np.concatenate([d, -d])
    ok = np.isfinite(d).all(axis=1) & (np.linalg.norm(d, axis=1) > 0.5)
    s, d = s[ok], d[ok]
    N = len(s)
    if N == 0:
        return []

    wv, wa, wb = (cone_sample_weights(theta, rings) if turn == "relax"
                  else cone_weights(theta, alpha))
    hist = np.full((max_steps + 1, N, 3), np.nan)
    hist[0] = s
    nstep = np.zeros(N, int)
    live = np.arange(N)
    x, v = s.copy(), d.copy()

    for it in range(max_steps):
        a, b = _basis(v)
        dirs = _unit(wv[None, :, None] * v[:, None, :]
                     + wa[None, :, None] * a[:, None, :]
                     + wb[None, :, None] * b[:, None, :], axis=2)
        if turn == "relax":
            # Every vector in the cone votes, then the heading moves a fraction
            # alpha of the way there. No argmax, so nothing depends on which
            # candidate happened to be enumerated.
            m = _mean_director(vec1, vec2, x, dirs, lookahead, n_lookahead,
                               v, dyadic_fields)
            v_new = _slerp(v, m, alpha, max_angle=theta)
            _sup, gate = support(vol, vec1, vec2, x, v_new[:, None, :],
                                 lookahead, n=n_lookahead,
                                 align_exp=align_exp,
                                 dyadic_fields=dyadic_fields,
                                 intensity_exp=intensity_exp)
            # Gated on the direction actually taken, exactly as below.
            alive = gate[:, 0] >= floor
            if not alive.any():
                break
            v = v_new
        else:
            sup, gate = support(vol, vec1, vec2, x, dirs, lookahead,
                                n=n_lookahead, align_exp=align_exp,
                                dyadic_fields=dyadic_fields,
                                intensity_exp=intensity_exp)

            k = np.argmax(sup, axis=1)
            rows = np.arange(len(k))
            # The walk ends on the brightness-coupled gate at the direction it
            # actually chose, never on the score -- otherwise `intensity_exp`
            # would silently change what "off the end of a fibre" means too.
            alive = gate[rows, k] >= floor
            if not alive.any():
                break
            v = dirs[rows, k]
        x = x + step * v
        inside = np.ones(len(x), bool)
        for c in range(3):
            inside &= (x[:, c] >= 0) & (x[:, c] < shape[c])
        alive &= inside

        hist[it + 1, live[alive]] = x[alive]
        nstep[live[alive]] = it + 1
        live, x, v = live[alive], x[alive], v[alive]
        if not len(live):
            break

    return [hist[:nstep[i] + 1, i] for i in range(N) if nstep[i] >= 1]


def _fields(field):
    """(vec1, vec2) from a DenseField, or (e3, None) from a bare array."""
    if hasattr(field, "vec1"):
        v2 = field.vec2 if field.n_multi else None
        return np.asarray(field.vec1, float), (None if v2 is None
                                               else np.asarray(v2, float))
    f = np.asarray(field, float)
    if f.shape[0] != 3:
        raise ValueError(f"expected a DenseField or (3, Z, Y, X), got {f.shape}")
    return f, None


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------

def seeds_from(field, max_seeds=5000, mask=None, second=True, jitter=0.0,
               rng=None):
    """Seed positions and directions from a DenseField.

    Oriented voxels seed along vec1, and -- where Stage 2 resolved two -- a
    second seed along vec2. Without that second family a crossing is only ever
    entered along one of its fibres, and the tract density at the junction can
    never reach 2 no matter what the tracker does.

    `max_seeds` is a COMPUTE BUDGET, not a parameter to optimise: more seeds
    means more columns in the design matrix, a larger feasible set for the
    weight fit, so the residual can essentially only fall. `weight_fit` records
    that argument in full. It is expressed as a count rather than a stride so
    that the same setting means the same cost on a 48^3 phantom and a 96^3 crop
    -- a fixed stride that gives 1,400 seeds on the phantom gives 50,000 on i74.

    ONE BUDGET ON LOCATIONS, NOT ONE PER FAMILY
    -------------------------------------------
    The seed LOCATIONS are chosen once by striding the oriented voxels; a
    location that carries two directions then emits both. So a two-direction
    voxel contributes exactly 2 seeds and a one-direction voxel exactly 1 --
    which is what makes the tract density mean what `tract_density` says it
    means.

    This replaces a per-family budget, under which each family was strided
    INDEPENDENTLY. Because the two-direction set is far smaller, it received
    nearly the same ABSOLUTE seed count as the whole oriented set and was
    therefore massively over-sampled per voxel. Measured on preprocessed i74
    (64^3 at the volume centre):

        1-direction   249,325 voxels    1,933 seeds      7.75 per 1000 voxels
        2-direction     8,332 voxels    1,732 seeds    207.87 per 1000 voxels

    -- 26.8x over-seeded. The TDI at those voxels was then reporting SEED
    density, not fibre density: the median smoothed TDI ratio between two- and
    one-direction voxels came out at 13.91 where the model requires 2. With one
    budget on locations that ratio measures 2.09, and coverage of tissue rises
    from 64.6% to 81.1% at the same total seed count.

    The old scheme was not arbitrary -- it was guarding a real problem, that
    seeding two-direction voxels at FULL density does not scale (51,444 such
    voxels on i74 produced 53,322 seeds against a budget of 2,000, because a
    nearest-patch fill spreads each routed patch over its whole neighbourhood,
    so "two-direction voxel" counts patch volume, not junctions). Striding the
    locations solves that without the bias: the second family is subsampled by
    the same stride as the first, so it can never exceed the location budget.

    The total returned is therefore max_seeds + (chosen locations carrying a
    second direction), i.e. between 1x and 2x max_seeds rather than a hard 2x.
    """
    n_dirs = field.n_dirs

    keep = n_dirs >= 1
    if mask is not None:
        keep &= np.asarray(mask, bool)
    idx = np.argwhere(keep)
    # Stride the FLAT list, not each axis: a per-axis stride samples a lattice
    # and can systematically miss whole fibres lying between its planes.
    # Deterministic, so two loss evaluations see the same seeds.
    if max_seeds and len(idx) > max_seeds:
        idx = idx[::int(np.ceil(len(idx) / max_seeds))]
    z, y, x = idx[:, 0], idx[:, 1], idx[:, 2]
    pos = [idx.astype(float)]
    dirs = [field.vec1[:, z, y, x].T]

    if second and field.n_multi:
        # The SAME locations, not a separately strided set. `keep` already
        # carried the mask, so no second masking is needed here.
        m = n_dirs[z, y, x] >= 2
        if m.any():
            pos.append(idx[m].astype(float))
            dirs.append(field.vec2[:, z[m], y[m], x[m]].T)

    p = np.concatenate(pos)
    d = np.concatenate(dirs)
    if jitter > 0:
        rng = rng or np.random.default_rng(0)
        p = p + rng.normal(scale=jitter, size=p.shape)
        good = np.ones(len(p), bool)
        for c in range(3):
            good &= (p[:, c] >= 0) & (p[:, c] < field.shape[c])
        p, d = p[good], d[good]
    return p, d


# ---------------------------------------------------------------------------

def selftest(verbose=True):
    """Agreement with the scalar tracker, and the two-direction degradation.

    The point of the first check is that the speed work above changed no
    physics: same seeds, same parameters, same paths. The second checks the
    claim the docstring makes about `max` -- that a field with vec2 all zero
    tracks identically to one with no vec2 at all.
    """
    import time
    import cone_search as CS
    from dense_field import DenseField

    rng = np.random.default_rng(0)
    n = 24
    vol = np.zeros((n, n, n), np.float32)
    vol[:, n // 2, n // 2] = 100.0          # one fibre along z
    from scipy.ndimage import gaussian_filter
    vol = gaussian_filter(vol, 1.2)
    vol = vol / vol.max() * 100.0
    e3 = np.zeros((3, n, n, n))
    e3[0] = 1.0
    F = DenseField.from_e3(e3)

    seeds = np.array([[4.0, n / 2, n / 2], [6.0, n / 2, n / 2]])
    sdir = np.array([[1.0, 0.0, 0.0]] * 2)
    kw = dict(alpha=0.2, lookahead=4.0, theta=0.3, floor=1.0)

    # dyadic_fields=False for this comparison ONLY: it is the wrong estimator
    # (see `dyadic`) but it is the one cone_search.track implements, and the
    # claim under test here is that the vectorisation changed no arithmetic.
    t0 = time.time()
    new = track(vol, F, seeds, sdir, dyadic_fields=False, intensity_exp=1.0,
                **kw)
    t_new = time.time() - t0
    t0 = time.time()
    old = CS.track(vol, e3, seeds, kw["alpha"], kw["lookahead"], kw["theta"],
                   kw["floor"])
    t_old = time.time() - t0

    ok = len(new) == len(old)
    dmax = 0.0
    if ok:
        for a, b in zip(sorted(new, key=lambda p: (len(p), p[0, 0])),
                        sorted(old, key=lambda p: (len(p), p[0, 0]))):
            if len(a) != len(b):
                ok = False
                break
            dmax = max(dmax, float(np.abs(a - b).max()))
    if verbose:
        print(f"  vectorised vs scalar: {len(new)} vs {len(old)} paths, "
              f"max |dx| {dmax:.2e}, {t_old / max(t_new, 1e-9):.1f}x faster  "
              f"{'OK' if ok and dmax < 1e-9 else '*** FAIL ***'}")
    fails = [] if (ok and dmax < 1e-9) else ["vectorised tracker disagrees "
                                             "with cone_search.track"]

    # Force the two-direction code path with a vec2 that is all zero. Leaving
    # n_dirs at 1 would take the `vec2 is None` shortcut instead and the check
    # would pass without ever evaluating the `max`.
    F2 = F.copy()
    F2.vec2 = np.zeros_like(F2.vec1)
    F2.n_dirs[:] = 2
    assert F2.n_multi > 0
    base = track(vol, F, seeds, sdir, intensity_exp=1.0, **kw)
    same = track(vol, F2, seeds, sdir, intensity_exp=1.0, **kw)
    ok2 = len(same) == len(base) and all(
        np.allclose(a, b) for a, b in zip(same, base))
    if verbose:
        print(f"  all-zero vec2 tracks identically: "
              f"{'OK' if ok2 else '*** FAIL ***'}")
    if not ok2:
        fails.append("an all-zero vec2 changed the tracks")

    # The dyadic must agree with |d . v| exactly where the signs happen to be
    # coherent -- a straight fibre with a uniform e3 is that case, so the two
    # estimators may not differ here at all. Where they DO differ is the point,
    # and `single_fibre_check` is what measures it.
    d3 = max((np.abs(a - b).max() if a.shape == b.shape else np.inf)
             for a, b in zip(sorted(base, key=lambda p: (len(p), p[0, 0])),
                             sorted(new, key=lambda p: (len(p), p[0, 0]))))
    ok3 = np.isfinite(d3) and d3 < 1e-9
    if verbose:
        print(f"  dyadic == componentwise on a sign-coherent field: "
              f"max |dx| {d3:.2e}  {'OK' if ok3 else '*** FAIL ***'}")
    if not ok3:
        fails.append("dyadic and componentwise disagree where the field has "
                     "no sign flips -- one of them is wrong")
    return fails


if __name__ == "__main__":
    import sys
    print("dense_tracker self-test")
    f = selftest()
    print("  " + ("ALL PASS" if not f else "FAIL: " + "; ".join(f)))
    sys.exit(1 if f else 0)
