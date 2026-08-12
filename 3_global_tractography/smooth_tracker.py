"""
smooth_tracker.py -- a cone tracker whose parameters are real numbers.

Why this exists
---------------
`dense_tracker` has no derivative with respect to its parameters, and not as a
numerical inconvenience -- structurally. Three discrete operations stand between
the parameters and the loss:

    cone_weights   theta enters ONLY as int(round(theta / alpha)), a ring COUNT.
                   theta never reaches a floating-point quantity, so d/dtheta is
                   exactly 0 between quantisation boundaries and undefined on
                   them. At the default alpha = 0.2 the seven-rung theta ladder
                   yields four distinct cones, and 0.15 / 0.20 / 0.25 / 0.30 are
                   bit-identical.
    argmax         the heading is selected from a finite candidate set, so L and
                   alpha act only by flipping which index wins.
    floor          walkers die on a hard threshold, in whole steps.

This module replaces each with a smooth counterpart, changing no physics:

    FIXED QUADRATURE    candidate directions are a fixed set covering a cap out
                        to TILT_MAX, built once. The cone WIDTH no longer sets
                        how many candidates exist; it sets how they are WEIGHTED.
    ANGULAR RESOLUTION  alpha keeps its meaning -- how finely the tracker can
                        tell two directions apart -- but stops being a sample
                        COUNT. The evidence is smoothed across the candidate set
                        by a kernel of angular width alpha before the choice is
                        made. Coarse alpha blurs nearby directions together,
                        which is exactly what coarse sampling did; fine alpha
                        resolves them. It is now a real number.

                        This also gives alpha a genuine optimum for the first
                        time. Under the old reading, finer was always at least
                        as good and merely cost compute, so a search on it ran
                        to the edge of its ladder. Here fine alpha also stops
                        averaging away noise in the support, so there is a
                        trade-off -- too coarse to steer against too sharp to be
                        stable -- and a descent on it means something.
    ANGULAR PRIOR       w(tilt) = exp(-tilt^2 / (2 theta^2)). theta is now a real
                        number that moves the weights continuously. There is no
                        ring count to round.
    SOFTMAX             p propto exp(score / tau) * w, and the new heading is the
                        p-weighted mean direction. As tau -> 0 this returns the
                        argmax, so the hard tracker is the zero-temperature
                        limit of this one rather than a different model.
    SIGMOID SURVIVAL    a walker carries a weight in [0, 1] that decays by
                        sigmoid((gate - floor) / beta) each step instead of
                        being killed. Tract length becomes continuous.

What is NOT claimed
-------------------
Smoothness is not accuracy. This tracker answers a relaxed question, and at
finite tau its streamlines are not the hard tracker's. tau and beta are
relaxation knobs, not physics: they should be small enough that the answer is
stable, and `temperature_check` in `gradient_optimise` is what measures whether
it is.

Frame: positions and directions are (z, y, x), as everywhere else.
"""

import numpy as np

import dense_tracker as DT

EPS = 1e-12

STEP = 1.0
MAX_STEPS = 60
N_LOOKAHEAD = 4

TILT_MAX = 1.20        # radians; the widest turn the quadrature can express
# Quadrature resolution -- ACCURACY, not a parameter, and deliberately finer
# than any alpha the descent will visit. If it were not, alpha could not blur
# below the spacing of the candidates it is blurring over and the smallest
# alphas would all behave identically -- the same quantisation this module
# exists to remove.
TILT_STEP = 0.15
TAU = 0.02             # softmax temperature on the support score
BETA = 0.05            # sigmoid width on the survival gate, in units of floor
MIN_SURVIVAL = 1e-3    # drop walkers whose weight has decayed below this


def quadrature(tilt_max=TILT_MAX, tilt_step=TILT_STEP):
    """Fixed candidate directions over a cap, as frame weights plus their tilts.

    Returns (w_v, w_a, w_b, tilt, sep) where `sep` is the K x K matrix of
    ANGLES BETWEEN CANDIDATES -- what the alpha kernel is built on. Because the
    frame is orthonormal, the cosine between two candidates is just the dot
    product of their (w_v, w_a, w_b) coefficients, so no direction has to be
    materialised to get it.

    Built ONCE and reused at every step of every walker, for every parameter
    value -- which is the whole point: the candidate set must not depend on
    theta or alpha, or those would again be changing an integer.
    """
    wv, wa, wb, tl = [1.0], [0.0], [0.0], [0.0]
    n_ring = max(int(round(tilt_max / tilt_step)), 1)
    for i in range(1, n_ring + 1):
        tilt = i * tilt_step
        n_phi = max(int(round(2 * np.pi * np.sin(tilt) / tilt_step)), 1)
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            wv.append(np.cos(tilt))
            wa.append(np.sin(tilt) * np.cos(phi))
            wb.append(np.sin(tilt) * np.sin(phi))
            tl.append(tilt)
    C = np.stack([np.asarray(wv), np.asarray(wa), np.asarray(wb)], axis=1)
    sep = np.arccos(np.clip(C @ C.T, -1.0, 1.0))
    return (np.asarray(wv), np.asarray(wa), np.asarray(wb), np.asarray(tl), sep)


def resolution_kernel(sep, alpha):
    """Row-normalised angular blur of width `alpha` over the candidate set.

    This is what alpha DOES: it decides how finely two directions can be told
    apart. Smoothing the evidence over neighbours within alpha reproduces the
    effect of having sampled that coarsely, without changing how many samples
    there are -- so alpha stays a real number and stays differentiable.

    Row-normalised so the blur redistributes evidence rather than adding or
    removing it; an unnormalised kernel would make coarse alpha look uniformly
    better simply by summing more terms.
    """
    K = np.exp(-(sep ** 2) / (2.0 * max(float(alpha), EPS) ** 2))
    return K / (K.sum(axis=1, keepdims=True) + EPS)


def track_soft(vol, field, seeds, seed_dirs, lookahead, theta, alpha, floor,
               tau=TAU, beta=BETA, step=STEP, max_steps=MAX_STEPS,
               n_lookahead=N_LOOKAHEAD, quad=None, both_ways=True,
               return_ids=False, turn="sample"):
    """Smooth streamlines. Returns (positions, directions, weights).

    positions   (M, 3)  every step's midpoint, all walkers concatenated
    directions  (M, 3)  the unit heading taken at that step
    weights     (M,)    the walker's survival weight when it took that step

    A step with weight 0.2 is a fifth of a step, not a fifth of a fibre: the
    weight expresses how confident the walk is that it is still on structure.
    Any loss reading these must weight by it, or a tract that has effectively
    ended still votes at full strength.

    `return_ids` adds a fourth output, (M,) walker index, and exists for ONE
    reason: every loss here is a sum over steps and does not care which walker a
    step came from, but a tractogram written to disk does. Without an identity
    the concatenated steps cannot be reassembled into polylines, and the file a
    viewer opens would have to come from a SECOND tracking run -- which is not
    the tractogram the loss scored. See `as_streamlines`.

    Ids are assigned BEFORE the `both_ways` duplication is undone, so id < N is
    the forward half of seed `id` and id >= N is the backward half of seed
    `id - N`; `as_streamlines` uses that to rejoin the two halves into one tract.
    """
    if turn not in ("sample", "relax"):
        raise ValueError(f"turn must be 'sample' or 'relax', got {turn!r}")
    if turn == "relax" and not 0.0 <= alpha <= 1.0:
        raise ValueError(f"turn='relax' needs alpha in [0, 1], got {alpha}; "
                         f"it is a fraction of the turn, not an angle")

    vec1, vec2 = DT._fields(field)
    T1 = DT.dyadic(vec1)
    T2 = None if vec2 is None else DT.dyadic(vec2)
    wv, wa, wb, tilt, sep = quad if quad is not None else quadrature()
    # In "relax" mode alpha is a turn fraction, not an angular resolution, so
    # there is no resolution kernel to build -- the evidence is averaged over
    # the whole cone instead of being blurred and then argmax-ed.
    Kres = None if turn == "relax" else resolution_kernel(sep, alpha)

    s = np.asarray(seeds, float)
    d = DT._unit(np.asarray(seed_dirs, float), axis=1)
    n_seed = len(s)
    if both_ways:
        s = np.concatenate([s, s])
        d = np.concatenate([d, -d])
    ok = np.isfinite(d).all(axis=1) & (np.linalg.norm(d, axis=1) > 0.5)
    x, v = s[ok].copy(), d[ok].copy()
    wid = np.arange(len(s))[ok]
    if not len(x):
        empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
        return empty + (np.zeros(0, np.int64), n_seed) if return_ids else empty

    surv = np.ones(len(x))
    shape = vol.shape
    # Angular prior in log space, so it adds to the softmax logits directly.
    logw = -(tilt ** 2) / (2.0 * max(float(theta), EPS) ** 2)

    P, D, W, IDS = [], [], [], []
    for _ in range(max_steps):
        a, b = DT._basis(v)
        dirs = DT._unit(wv[None, :, None] * v[:, None, :]
                        + wa[None, :, None] * a[:, None, :]
                        + wb[None, :, None] * b[:, None, :], axis=2)
        if turn == "relax":
            # Every vector in the cone votes through the dyadic mean, then the
            # heading moves a fraction alpha of the way there, clamped to theta.
            # This path has no softmax and no argmax at all -- it is smooth in
            # (L, theta, alpha) by construction rather than by relaxation, so
            # tau does not enter it.
            m = DT._mean_director(T1, T2, x, dirs, float(lookahead),
                                  n_lookahead, v, True)
            v_new = DT._slerp(v, m, alpha, max_angle=float(theta))
            _s, gate = DT.support(vol, T1, T2, x, v_new[:, None, :],
                                  float(lookahead), n=n_lookahead,
                                  dyadic_fields=True, intensity_exp=0.0)
            g_bar = gate[:, 0]
        else:
            score, gate = DT.support(vol, T1, T2, x, dirs, float(lookahead),
                                     n=n_lookahead, dyadic_fields=True,
                                     intensity_exp=0.0)

            # alpha first: blur the evidence to the tracker's angular
            # resolution, BEFORE any choice is made. Applied to the score and
            # the gate alike, since both read the same evidence.
            score = score @ Kres.T
            gate = gate @ Kres.T

            # softmax over candidates, with the angular prior. Subtracting the
            # row max is the standard overflow guard and changes nothing.
            logits = score / max(float(tau), EPS) + logw[None, :]
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True) + EPS

            v_new = DT._unit(np.einsum("nk,nkc->nc", p, dirs), axis=1)
            g_bar = np.einsum("nk,nk->n", p, gate)

        x_mid = x + 0.5 * step * v_new
        x = x + step * v_new
        # Survival decays smoothly instead of the walk being cut.
        surv = surv * _sigmoid((g_bar - floor) / max(float(beta) * floor, EPS))

        inside = np.ones(len(x), bool)
        for c in range(3):
            inside &= (x[:, c] >= 0) & (x[:, c] < shape[c])
        P.append(x_mid[inside])
        D.append(v_new[inside])
        W.append(surv[inside])
        IDS.append(wid[inside])

        live = inside & (surv > MIN_SURVIVAL)
        if not live.any():
            break
        x, v, surv, wid = x[live], v_new[live], surv[live], wid[live]

    if not P:
        empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
        return empty + (np.zeros(0, np.int64), n_seed) if return_ids else empty
    out = (np.concatenate(P), np.concatenate(D), np.concatenate(W))
    if return_ids:
        return out + (np.concatenate(IDS), n_seed)
    return out


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(z, -60.0, 60.0)))


def align_loss(pos, dirs, w, field):
    """Survival-weighted 1 - <(d . v)^2>. The objective validated in
    `objective_search`, read off a smooth tractogram.

    Weighting by survival is what keeps this from being gamed the obvious way.
    Without it a parameter setting that kills every walker after two steps would
    report near-perfect alignment on the two steps that survived.
    """
    if not len(pos):
        return 1.0
    vec1, vec2 = DT._fields(field)
    a2 = _align2(DT.dyadic(vec1), pos, dirs)
    if vec2 is not None:
        a2 = np.maximum(a2, _align2(DT.dyadic(vec2), pos, dirs))
    tot = float(w.sum())
    if tot <= EPS:
        return 1.0
    return float(1.0 - (w * a2).sum() / tot)


def _align2(T, pts, dirs):
    out = np.zeros(len(pts))
    for (i, j, wt), comp in zip(DT._DYAD, T):
        out += wt * DT._sample(comp, pts) * dirs[:, i] * dirs[:, j]
    return np.clip(out, 0.0, 1.0)


def as_tdi(pos, dirs, w, shape, step=STEP, n_keep=None):
    """The smooth output rasterised DIRECTLY, without going via `as_paths`.

    Each step is a chord of length `step`; splatting its two endpoints with half
    the mass each is exactly what `rasterise` does to the two-point segment
    `as_paths` would have built, but vectorised over all steps at once. Saves the
    ~11,000-element Python loop that dominated every image-domain evaluation.
    """
    from tract_density import rasterise_points
    keep = np.ones(len(pos), bool) if n_keep is None else (w >= float(n_keep))
    p, d = pos[keep], dirs[keep]
    if not len(p):
        return np.zeros(tuple(shape))
    pts = np.concatenate([p - 0.5 * d, p + 0.5 * d])
    m = np.full(len(pts), 0.5 * float(step))
    return rasterise_points(pts, shape, m)


def as_streamlines(pos, dirs, w, ids, n_seed, n_keep=None, min_pts=4):
    """The concatenated steps reassembled into POLYLINES, one per seed.

    `as_paths` deliberately emits one two-point segment per step, because that is
    the quantity every loss in this repo is defined on. A viewer needs the
    opposite: connected curves. This is that reassembly, and it is only ever for
    export -- nothing scores these.

    Two halves, one tract. `track_soft(both_ways=True)` launches each seed
    forward and backward, so seed i produces walkers i and i + n_seed. Written
    out separately they appear as two tracts meeting end-to-end at every seed,
    which doubles the apparent tract count and puts a spurious endpoint in the
    middle of every fibre. The backward half is reversed and prepended instead,
    so one seed yields one curve through it.

    Steps are appended in time order by `track_soft` and walkers keep their
    relative order within each append, so a STABLE sort on the id recovers each
    walker's path in the order it was walked. An unstable sort would shuffle the
    points of a tract into a tangle that still looks like a tractogram.

    min_pts drops walkers that died almost immediately: a 2-point stub carries no
    orientation a viewer can show and there are many of them.
    Returns (streamlines, weights) with positions in (z, y, x) voxel units.
    """
    keep = np.ones(len(pos), bool) if n_keep is None else (w >= float(n_keep))
    p, ww, ii = pos[keep], w[keep], np.asarray(ids)[keep]
    if not len(p):
        return [], np.zeros(0)
    order = np.argsort(ii, kind="stable")
    p, ww, ii = p[order], ww[order], ii[order]
    bounds = np.flatnonzero(np.diff(ii)) + 1

    halves = {}
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(ii)]):
        halves[int(ii[a])] = (p[a:b], ww[a:b])

    lines, weights = [], []
    for sid in range(int(n_seed)):
        fwd, bwd = halves.get(sid), halves.get(sid + int(n_seed))
        chunks = ([bwd[0][::-1]] if bwd is not None else []) + \
                 ([fwd[0]] if fwd is not None else [])
        if not chunks:
            continue
        pts = np.concatenate(chunks)
        if len(pts) < int(min_pts):
            continue
        lines.append(np.ascontiguousarray(pts, float))
        # One weight per tract, from the survival it carried. The mean, not the
        # final value: survival decays monotonically, so the last step's weight
        # describes the tail rather than the tract.
        weights.append(float(np.concatenate(
            [c for c in ((bwd[1] if bwd is not None else None),
                         (fwd[1] if fwd is not None else None))
             if c is not None]).mean()))
    return lines, np.asarray(weights, float)


def as_paths(pos, dirs, w, n_keep=None):
    """The smooth output as a point cloud a path-based scorer can read.

    `theta_validation.tract_error` walks (M, 3) arrays and differences them, so
    handing it these positions directly would measure the difference between
    unrelated walkers. Instead each STEP is emitted as its own two-point
    segment, which is exactly the quantity the angular error is defined on.
    Segments below `n_keep` survival are dropped so a decayed tail does not
    dominate an unweighted median.
    """
    keep = np.ones(len(pos), bool) if n_keep is None else (w >= float(n_keep))
    p, d = pos[keep], dirs[keep]
    return [np.stack([q - 0.5 * dd, q + 0.5 * dd]) for q, dd in zip(p, d)]
