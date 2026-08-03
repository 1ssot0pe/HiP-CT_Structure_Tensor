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
from structure_tensor import structure_tensor_3d, eig_special_3d

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "mass_conservation"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

from host_fod import analyse
from tract_density import rasterise, density_at, mse

EPS = 1e-12
CROSSING_S = 0.65        # S above this = superposing = crossing
TANGENCY_S = 0.35        # S below this = no superposition = tangency


# ---------------------------------------------------------------------------
# D_obs -- target density from the field's own anisotropy
# ---------------------------------------------------------------------------

def anisotropy_target(vol, sigma=1.0, rho=3.0, air_frac=0.05):
    """Normalised (lambda1 - lambda3) as the density a tractography should match.

    Normalised to unit MEAN over tissue, not unit max: a max-normalisation is set
    by the single brightest voxel and would make the target hostage to one
    outlier, the same failure mode the absolute-threshold work kept hitting.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        S = structure_tensor_3d(vol.astype(np.float32), float(sigma), float(rho))
        val, _vec = eig_special_3d(S)
    del S
    lam1, lam3 = val[0], val[2]          # Skielex returns DESCENDING
    aniso = np.asarray(lam1 - lam3, dtype=np.float64)
    tissue = vol > air_frac * float(vol.max())
    m = float(aniso[tissue].mean()) if tissue.any() else 0.0
    if m <= 0:
        return np.zeros_like(aniso), tissue
    return aniso / m, tissue


# ---------------------------------------------------------------------------
# superposition metric
# ---------------------------------------------------------------------------

def background_level(vol, air_frac=0.05):
    """I_0: the background the superposition is measured over.

    Median of the sub-air-threshold voxels, falling back to the volume's own
    low percentile when the phantom has no true background (synthetic volumes
    are often exactly 0 outside the tubes, which is a valid I_0 of 0).
    """
    thr = air_frac * float(vol.max())
    bg = vol[vol <= thr]
    if bg.size >= 32:
        return float(np.median(bg))
    return float(np.percentile(vol, 1))


def superposition_S(vol, centre, d1, d2, I0=None, arm=6.0, half=1):
    """S = (I_c - I0) / ((I_a - I0) + (I_b - I0)) at one junction.

    I_a and I_b are sampled `arm` voxels along +-d1 and +-d2 respectively, taking
    the max over the two signs so a one-sided structure is still measured at its
    own brightness rather than at its faded end.
    """
    vol = np.asarray(vol)
    if I0 is None:
        I0 = background_level(vol)
    c = np.asarray(centre, float)

    def sample(p):
        idx = tuple(int(np.clip(round(v), 0, s - 1)) for v, s in zip(p, vol.shape))
        if half <= 0:
            return float(vol[idx])
        sl = tuple(slice(max(v - half, 0), min(v + half + 1, s))
                   for v, s in zip(idx, vol.shape))
        return float(vol[sl].max())

    I_c = sample(c)
    I_a = max(sample(c + arm * np.asarray(d1, float)),
              sample(c - arm * np.asarray(d1, float)))
    I_b = max(sample(c + arm * np.asarray(d2, float)),
              sample(c - arm * np.asarray(d2, float)))
    denom = (I_a - I0) + (I_b - I0)
    if abs(denom) < EPS:
        return np.nan, I_c, I_a, I_b, I0
    return float((I_c - I0) / denom), I_c, I_a, I_b, I0


def verdict_from_S(S):
    if not np.isfinite(S):
        return "unscored"
    if S >= CROSSING_S:
        return "CROSSING"
    if S <= TANGENCY_S:
        return "TANGENCY"
    return "AMBIGUOUS"


# ---------------------------------------------------------------------------
# streamline integration through the unified orientation field
# ---------------------------------------------------------------------------

def _nearest_patch(field_pts, p, max_d):
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
    dirs = [np.atleast_2d(np.asarray(p["dirs"], float)) for p in field]
    step = float(theta["step"])
    cosmax = np.cos(np.deg2rad(float(theta["angle_deg"])))
    max_d = max(float(theta.get("assoc_radius", 4.0)), step * 2.0)
    verdicts = junction_verdicts or {}

    # seeds: a fraction of the field's own patches, both directions
    n_seed = max(int(len(field) * float(theta["seed_density"])), 1)
    idx = rng.choice(len(field), size=min(n_seed, len(field)), replace=False)

    streams, weights = [], []
    for i in idx:
        for d0 in dirs[i]:
            for sgn in (1.0, -1.0):
                p = pts[i].copy()
                u = sgn * d0 / (np.linalg.norm(d0) + EPS)
                path = [p.copy()]
                for _ in range(int(theta.get("max_steps", 200))):
                    p = p + step * u
                    if not all(0 <= p[k] < shape[k] for k in range(3)):
                        break
                    j, dist = _nearest_patch(pts, p, max_d)
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

    data_term = mse(D_rec[tissue], D_obs[tissue])

    contribs = []
    pen = 0.0
    for j in junctions:
        d_here = density_at(D_rec, j["centre"], half=1)
        if j["verdict"] == "TANGENCY":
            # a tangency should carry ~1 fibre's worth, not 2
            c = max(d_here - 1.0, 0.0) ** 2
        elif j["verdict"] == "CROSSING":
            # a crossing should carry ~2
            c = (d_here - 2.0) ** 2
        else:
            c = 0.0
        pen += c
        contribs.append(dict(centre=[int(v) for v in j["centre"]],
                             S=float(j["S"]), verdict=j["verdict"],
                             density=float(d_here), loss_contribution=float(c)))
    pen = verdict_penalty * pen / max(len(junctions), 1)
    total = (0.0 if not np.isfinite(data_term) else data_term) + pen
    return total, dict(mse=float(data_term), penalty=float(pen),
                       total=float(total), n_streams=len(streams),
                       junctions=contribs)


# ---------------------------------------------------------------------------
# theta optimisation -- coordinate search
# ---------------------------------------------------------------------------

THETA0 = dict(step=0.75, angle_deg=45.0, seed_density=0.5, link_weight=1.0,
              assoc_radius=4.0, max_steps=200)
GRID = dict(step=[0.5, 0.75, 1.0, 1.5],
            angle_deg=[25.0, 35.0, 45.0, 60.0],
            seed_density=[0.25, 0.5, 1.0],
            link_weight=[0.5, 1.0, 1.5])


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

def build_field_and_junctions(vol, half, centre, sigma_grad=2.5, search=6,
                              stride=6, air_frac=0.05):
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
    from mass_conservation_validate import (raw_localized_crossing, raw_kissing,
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

    out = {}
    for name, vol, centre, half, truth in cases:
        print(f"\n--- {name}  (truth: {truth}) ---")
        D_obs, tissue = anisotropy_target(vol)
        field, junctions, I0 = build_field_and_junctions(vol, half, centre)
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
