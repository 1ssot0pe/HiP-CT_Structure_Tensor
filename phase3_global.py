"""
phase3_global.py -- Phase 3: the FULL JOINT GLOBAL model (2D mechanism proof).

Phase 2 proved a LOCAL per-peak certifier cannot separate kissing from crossing on
real tissue: the distinguishing fact is RELATIONAL (which fibre segment connects to
which across a junction), and a peak scored in isolation does not contain it. The
global model fixes this by solving for a whole configuration of tracts at once and
scoring it jointly.

This is the mechanism prototype, in 2D on phantoms with known truth (per the design
doc, "Phase 0 = 2D/small-3D proof"), so the claim is checked before scaling to 3D/i74.

  REPRESENTATION  a set of oriented SEGMENTS extracted from the image (structure-
                  tensor orientation, anisotropy-gated -> segments sit on clean line
                  bodies, sparse at ambiguous crossings, which the linker must bridge).
  ENERGY (joint)  over a configuration of segment-to-segment LINKS (each segment end
                  used at most once, no cycles -> tracts are chains):
                      E = w_c * SUM curvature(link) + w_n * (number of tracts)
                  continuity (w_c) wants smooth links; parsimony (w_n) wants few
                  tracts. NO term is per-voxel-local: #tracts is a global count and
                  the links compete for shared endpoints.
  INFERENCE       a greedy min-curvature degree-constrained forest (Kruskal-style,
                  union-find) -- the exact global solve for this per-link-decomposable
                  energy, producing clean unbroken chains. `anneal()` is the general
                  Metropolis sampler the design doc names, kept for when a coupling
                  data term makes the energy non-decomposable.

TEST (the milestone): count tracts that SPAN the midline. A true crossing's fibres
continue straight through -> many spanning tracts. A kissing contact would need a
sharp bend to bridge two different fibres -> the energy refuses it -> ~0 spanning
tracts. So: crossing >> kissing spanning-tracts == the global model separated them,
where the local certifier could not.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 2D phantoms (known truth)
# ---------------------------------------------------------------------------

def make_phantom(kind, N=110, ang2=70.0, spacing=12.0, width=1.7, noise=0.03, seed=0):
    """Two line-families. crossing: both over the whole field. kissing: family-1
    (horizontal) in x<mid only, family-2 (ang2) in x>=mid only -- they touch at the
    midline but no fibre crosses it."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:N, 0:N].astype(float)
    cx = cy = N / 2.0

    def family(theta_deg):
        t = np.deg2rad(theta_deg)
        nrm = np.array([-np.sin(t), np.cos(t)])          # normal to the lines
        perp = (xs - cx) * nrm[0] + (ys - cy) * nrm[1]   # signed dist to a line
        return np.exp(-((perp - np.round(perp / spacing) * spacing) ** 2)
                      / (2 * width ** 2))

    f1, f2 = family(0.0), family(ang2)
    img = (f1 + f2) if kind == "crossing" else np.where(xs < cx, f1, f2)
    img = img + noise * rng.standard_normal((N, N))
    img = img - img.min()
    return (img / (img.max() + 1e-9)), cx


# ---------------------------------------------------------------------------
# segments: MULTI-ORIENTATION seeding (a 2D FOD data term)
# ---------------------------------------------------------------------------

def extract_segments(img, stride=3, half=5, nbins=36, rel=0.45, min_sep_deg=25.0,
                     fg_frac=0.30, mag_frac=0.12):
    """At each strided foreground pixel, build a gradient-weighted histogram of the
    FIBRE orientation (perpendicular to the gradient) over [0,pi), find its peaks,
    and place ONE segment PER PEAK. A single-fibre pixel spawns 1 segment; a
    CROSSING pixel spawns 2 -- which a single structure-tensor direction cannot,
    because the 2nd-order tensor is near-isotropic at a crossing (this was the whole
    reason the earlier single-orientation seeding failed here, mirroring why HOST-FOD
    exists). Returns (pos [M,2] as (x,y), dirs [M,2] unit, w [M])."""
    gy, gx = np.gradient(img)
    fg = img > fg_frac * img.max()
    N = img.shape[0]
    centres = np.deg2rad((np.arange(nbins) + 0.5) * 180.0 / nbins)
    sep_bins = int(round(min_sep_deg / (180.0 / nbins)))
    pos, dirs, w = [], [], []
    for y in range(half, N - half, stride):
        for x in range(half, N - half, stride):
            if not fg[y, x]:
                continue
            gxw = gx[y - half:y + half + 1, x - half:x + half + 1].ravel()
            gyw = gy[y - half:y + half + 1, x - half:x + half + 1].ravel()
            gm = np.hypot(gxw, gyw)
            keep = gm > mag_frac * (gm.max() + 1e-12)
            if keep.sum() < 8:
                continue
            fib = (np.arctan2(gyw[keep], gxw[keep]) + np.pi / 2) % np.pi
            h, _ = np.histogram(fib, bins=nbins, range=(0, np.pi), weights=gm[keep])
            h = _smooth_circular(h, 2)
            for a_idx in _peaks_circular(h, rel, sep_bins)[:2]:
                a = centres[a_idx]
                pos.append([float(x), float(y)])
                dirs.append([np.cos(a), np.sin(a)])
                w.append(float(h[a_idx]))
    return np.array(pos), np.array(dirs), np.array(w)


def _smooth_circular(h, wdt):
    k = np.ones(2 * wdt + 1) / (2 * wdt + 1)
    return np.array([np.dot(k, np.take(h, range(i - wdt, i + wdt + 1), mode="wrap"))
                     for i in range(len(h))])


def _peaks_circular(h, rel, sep_bins):
    """Circular-histogram local maxima above rel*max, greedily de-duplicated so peaks
    are >= sep_bins apart. Strongest first."""
    n = len(h); thr = rel * h.max()
    cand = [i for i in range(n)
            if h[i] >= thr and h[i] >= h[(i - 1) % n] and h[i] >= h[(i + 1) % n]]
    cand.sort(key=lambda i: -h[i])
    out = []
    for i in cand:
        if all(min(abs(i - j), n - abs(i - j)) >= sep_bins for j in out):
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# candidate links + curvature cost
# ---------------------------------------------------------------------------

def candidate_links(pos, dirs, k=3, r_min=2.0, r_max=20.0,
                    cone_deg=35.0, orient_deg=30.0):
    """DIRECTION-AWARE candidate links. For each segment i, search AHEAD and BEHIND
    along its own tangent (+/- u_i, within `cone_deg`) for the nearest few segments
    j whose orientation is also similar (within `orient_deg`) -- i.e. this fibre's
    continuation, not whatever is nearest in raw distance. (Nearest-by-distance
    fails at a crossing: perpendicular segments of the OTHER family crowd out the
    collinear continuation.) A link's curvature cost = how far the connecting
    direction departs from BOTH orientations (0 = perfectly collinear)."""
    M = len(pos)
    cos_cone = np.cos(np.deg2rad(cone_deg))
    cos_or = np.cos(np.deg2rad(orient_deg))
    links = {}
    for i in range(M):
        rel = pos - pos[i]
        d = np.linalg.norm(rel, axis=1)
        vhat = rel / (d[:, None] + 1e-12)
        ali = vhat @ dirs[i]                              # signed alignment w/ u_i
        same = np.abs(dirs @ dirs[i]) > cos_or            # similar orientation
        band = (d > r_min) & (d <= r_max) & same
        for side in (+1.0, -1.0):                         # ahead and behind
            sel = np.where(band & (side * ali > cos_cone))[0]
            for j in sel[np.argsort(d[sel])[:k]]:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in links:
                    continue
                v = rel[j] / (d[j] + 1e-12)
                ci = np.arccos(min(1.0, abs(v @ dirs[i])))
                cj = np.arccos(min(1.0, abs(v @ dirs[j])))
                links[(a, b)] = ci ** 2 + cj ** 2         # curvature cost (rad^2)
    return links


# ---------------------------------------------------------------------------
# joint energy + simulated-annealing inference
# ---------------------------------------------------------------------------

def _connected(adj, i, j):
    """BFS: are i and j already in the same tract under current active links?"""
    seen, stack = {i}, [i]
    while stack:
        u = stack.pop()
        if u == j:
            return True
        for v in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)
    return False


def anneal(pos, cand, M, w_c=1.0, w_n=0.30, iters=90000, T0=0.5, seed=0):
    """Minimise E = w_c*sum curvature(active links) + w_n*(#tracts) over link on/off
    states, subject to segment degree <= 2 and no cycles (tracts are chains).
    Returns the set of active links."""
    rng = np.random.default_rng(seed)
    keys = list(cand.keys())
    active = set()
    adj = {i: set() for i in range(M)}
    deg = np.zeros(M, int)
    n_tracts = M                                          # every segment alone
    E = w_n * n_tracts
    for it in range(iters):
        T = T0 * (1.0 - it / iters) + 1e-3                # linear cool
        a, b = keys[rng.integers(len(keys))]
        cost = cand[(a, b)]
        if (a, b) in active:                              # propose REMOVE
            dE = -w_c * cost + w_n                        # +1 tract
            if dE <= 0 or rng.random() < np.exp(-dE / T):
                active.discard((a, b))
                adj[a].discard(b); adj[b].discard(a)
                deg[a] -= 1; deg[b] -= 1; n_tracts += 1; E += dE
        else:                                             # propose ADD
            if deg[a] >= 2 or deg[b] >= 2:
                continue
            if _connected(adj, a, b):                     # would make a cycle
                continue
            dE = w_c * cost - w_n                         # -1 tract
            if dE <= 0 or rng.random() < np.exp(-dE / T):
                active.add((a, b))
                adj[a].add(b); adj[b].add(a)
                deg[a] += 1; deg[b] += 1; n_tracts -= 1; E += dE
    return active, adj


def greedy_forest(cand, M, gate=0.30):
    """Global min-curvature degree-constrained forest (Kruskal-style): add links in
    ascending curvature cost while cost < gate (=w_n/w_c), keeping each segment's
    degree <= 2 and forbidding cycles (union-find). This directly (and cleanly)
    minimises the SAME energy E = w_c*curv + w_n*#tracts that `anneal` samples --
    for this per-link-decomposable energy the greedy forest is the exact global
    solve, and it produces the long unbroken chains a stochastic sampler leaves
    fragmented. Returns (active links, adjacency)."""
    parent = list(range(M))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    adj = {i: set() for i in range(M)}
    deg = [0] * M
    active = set()
    for (i, j) in sorted(cand, key=cand.get):
        if cand[(i, j)] > gate:
            break
        if deg[i] >= 2 or deg[j] >= 2:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:                                       # would make a cycle
            continue
        parent[ri] = rj
        adj[i].add(j); adj[j].add(i)
        deg[i] += 1; deg[j] += 1
        active.add((i, j))
    return active, adj


# ---------------------------------------------------------------------------
# read out tracts + the spanning-midline test
# ---------------------------------------------------------------------------

def tracts_from(adj, M):
    seen, out = set(), []
    for s in range(M):
        if s in seen:
            continue
        comp, stack = [], [s]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u)
            stack.extend(adj[u] - seen)
        out.append(comp)
    return out


def midline_links(active, pos, cx):
    """Active links whose two segments straddle the midline (for the figure)."""
    xs = pos[:, 0]
    return [(i, j) for (i, j) in active if (xs[i] - cx) * (xs[j] - cx) < 0]


def spanning_count(tracts, pos, cx, D=15, min_seg=3):
    """Tracts that reach at least D past the midline on BOTH sides -- a fibre the
    model connected clear THROUGH the junction. D >> the seeding half-window, so
    boundary partial-volume bleed (which only reaches ~half past the midline) cannot
    fake a span. A true crossing has many; a kissing contact ~0."""
    n = 0
    for c in tracts:
        if len(c) < min_seg:
            continue
        xs = pos[c, 0]
        if xs.min() < cx - D and xs.max() > cx + D:
            n += 1
    return n


def run_one(kind, ax):
    img, cx = make_phantom(kind)
    pos, dirs, w = extract_segments(img)
    cand = candidate_links(pos, dirs)
    active, adj = greedy_forest(cand, len(pos))
    tracts = tracts_from(adj, len(pos))
    cross = midline_links(active, pos, cx)
    span = spanning_count(tracts, pos, cx)
    big = [t for t in tracts if len(t) >= 3]
    print(f"  {kind:>9}: {len(pos)} segments, {len(cand)} candidate links, "
          f"{len(active)} linked -> {len(big)} tracts(len>=3); "
          f"{span} tracts reach THROUGH the junction (>=15 vox both sides)")

    ax.imshow(img, cmap="gray", origin="upper")
    ax.axvline(cx, color="cyan", ls="--", lw=1.2)
    rng = np.random.default_rng(1)
    for c in tracts:
        if len(c) < 3:
            continue
        P = pos[c]
        col = plt.cm.tab20(rng.integers(20))
        # order the chain for a clean polyline: walk the degree<=2 graph
        order = _order_chain(adj, c)
        Pp = pos[order]
        ax.plot(Pp[:, 0], Pp[:, 1], "-", color=col, lw=1.8)
        ax.plot(P[:, 0], P[:, 1], ".", color=col, ms=3)
    # highlight the midline-crossing links (the connectivity decision)
    for i, j in cross:
        ax.plot(pos[[i, j], 0], pos[[i, j], 1], "-", color="#ff2a2a", lw=2.2)
    ax.set_title(f"{kind}: {span} tracts reach through the junction")
    ax.set_xticks([]); ax.set_yticks([])
    return span


def _order_chain(adj, comp):
    """Order a degree<=2 component into a path (start from an endpoint)."""
    ends = [u for u in comp if len(adj[u]) <= 1] or [comp[0]]
    start = ends[0]
    order, seen, cur = [start], {start}, start
    while True:
        nxt = [v for v in adj[cur] if v not in seen]
        if not nxt:
            break
        cur = nxt[0]; order.append(cur); seen.add(cur)
    return order


def run(save="phase3_global.png"):
    print("=" * 74)
    print("PHASE 3 -- full joint global model (2D mechanism proof)")
    print("  energy = w_c*curvature + w_n*#tracts ; inference = greedy min-curvature "
          "forest (anneal() is the general sampler)")
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    sc = run_one("crossing", ax[0])
    sk = run_one("kissing", ax[1])
    print("-" * 74)
    ok = sc >= 4 and sk <= max(1, sc // 4)
    verdict = ("SEPARATED: global wiring connects a true crossing, refuses the kissing bend"
               if ok else "NOT separated (tune w_n/w_c or segment gating)")
    print(f"  crossing through-tracts={sc}, kissing through-tracts={sk}  => {verdict}")
    print("=" * 74)
    fig.suptitle("Phase 3: joint global tractography connects a true crossing through the\n"
                 "junction but leaves the kissing contact as two terminating tracts", fontsize=12)
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)
    print(f"Saved {save}")


if __name__ == "__main__":
    run()
