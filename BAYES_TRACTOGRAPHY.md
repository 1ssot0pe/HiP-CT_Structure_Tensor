# Scoping: global / Bayesian tractography for the "too complex" fibres

**Motivation.** The per-voxel pipeline outputs three buckets: single fibre,
clean 2-way crossing (HOST), and **"too complex"** (3+ fibres, narrow-angle
crossings, kissing-vs-crossing). The last bucket is unresolvable *locally*
because a single voxel lacks the evidence. Global tractography reframes the
question from **"what is the orientation at this voxel?"** to **"what set of
tracts across the whole volume best explains the data?"** — and the missing
evidence lives in the neighbours.

This is the same move diffusion-MRI made (local FOD estimation → global
tractography). HOST-FOD is the structure-tensor twin of the dMRI FOD, so the
machinery transfers.

---

## 1. The model

A Bayesian model has three parts. `posterior ∝ likelihood × prior`.

### Representation (what a "state" is)
Two candidate parameterisations, from simplest to most powerful:

- **(A) Per-voxel discrete orientation field.** Each voxel carries 0/1/2
  orientations (from HOST). State = the field. A smoothness/agreement prior over
  neighbours. Inference = MRF labelling (ICM / graph-cuts / belief propagation).
  *This is the principled, full-strength version of `refined_field.py`'s crude
  `spatial_regularize`.* Cheapest; adds spatial coherence but does NOT resolve
  connectivity (kissing vs crossing).

- **(B) Explicit line-segment / "particle" set** (dMRI "spin-glass" / Gibbs
  tracking). The volume is populated with short oriented segments that (i) want to
  sit on real structure and (ii) want to connect end-to-end into smooth tracts.
  State = the segment set (variable number). Inference = reversible-jump MCMC or
  simulated annealing (birth/death/move). This is the version that **resolves
  connectivity** — kissing vs crossing falls out of the continuity term.

**Recommendation:** prototype the *mechanism* with (A)-style continuity now, design
toward (B) for the connectivity payoff.

### Likelihood (data term) — two options
- **(L1) FOD-matching.** Score how well the proposed tracts reproduce the **HOST
  FODs** at each voxel. *Does not need the raw-image noise model* — HOST is the
  "observation". This is what dMRI global tractography uses, and it lets us
  **start now** (half-map only tightens the FOD uncertainty weight).
- **(L2) Generative forward model.** Given tracts → predict the **image** (via
  LIC, already in `reconstruction.py`) → compare to the observed image. This is
  the honest generative version but **needs a calibrated noise model** (the
  half-map) for the residual term to be meaningful.

### Prior
- **Continuity:** tracts are smooth (low curvature / bending energy). *This is
  the term that disambiguates kissing vs crossing.*
- **Parsimony (Occam):** penalise the number of tracts/fibres — RJMCMC
  birth/death, so complexity is only invoked when the data demands it.
- **Spatial coherence:** neighbouring orientations agree.

**Risk to name explicitly:** a strong prior can *hallucinate* continuity the data
doesn't support. The prior weight must be set against a real likelihood — which is
why the noise model (half-map) matters even here.

---

## 2. Phased plan (what needs the half-map, what doesn't)

| Phase | Scope | Needs half-map? |
|---|---|---|
| **0 — mechanism** | 2D/small-3D proof that global continuity resolves a locally-ambiguous junction (crossing, then kissing). Data term = image/FOD; prior = curvature. | No |
| **1 — 3D on phantoms** | Rep (A)→(B), MRF/segment inference, validated on the `host_geometry_test` crossing **and kissing** phantoms (known truth). Milestone: correctly separate kissing from crossing. **✓ MET (wide-angle scope)** — gated energy rule separates θ₂∈[70,90°], tracking the fundamental narrow-crossing limit; narrow = genuinely ambiguous. See Phase-1c/1d note. | No (FOD data term, L1) |
| **2 — real i74, FOD likelihood** | Run L1 on i74 tissue; compare the global field to the per-voxel one; does it demote spurious crossings and connect real tracts? **◐ STARTED, local certifier REJECTED** — both straightness (kissing demote 27%) and tight-cone min-reach (13%) confirm true crossings but miss kissing. Local per-peak certification is insufficient; kissing/crossing on real data needs the full joint global model (Phase 3). See Phase-2 note. | Weakly (FOD weight) |
| **2.5 — joint global model** | The segment/particle representation (option B) with continuity + parsimony over ALL tracts at once — the model Phase 2 showed is required. **✓ MECHANISM PROVEN in 2D** (`phase3_global.py`): separates kissing from crossing where the local certifier could not. See Phase-2.5 note. | No |
| **3 — generative, honest** | L2 forward model (LIC) + **calibrated noise likelihood**; full posterior / uncertainty. | **Yes** — the residual term is meaningless without the measured noise |

So: **Phases 0–1 can start immediately**; the half-map is required only to make
Phase 3 (the honest generative posterior) rigorous — and, per the noise thread,
to keep the prior from silently running the show.

### Phase-1 result (kissing/crossing separation — MET)

Three prototypes, each isolating what the next one needs:

- **`phase1_kissing_crossing.py`** — a *local greedy* tracker (HOST fibre-ness in a
  curvature cone). Honest limit: it **merges** the kissing bundles — at the boundary
  it bends onto bundle B (net turn 63°) instead of terminating. Local steps cannot
  separate the two cases → motivates a whole-tract objective.
- **`phase1b_global.py`** — a *global* energy `E = −data + λ·curv`. Showed **global
  parsimony** (don't double-count structure another tract already explains) rejects
  the spurious TURN, where a curvature limit alone does not. *But* it picked
  TERMINATE for **both** phantoms — so it removed the wrong merge yet still did **not
  separate** kissing from crossing.
- **`phase1c_directional.py`** — closes the gap. Two fixes: (i) a **directional**
  data term = HOST fibre-ness *along the tract's own tangent*, **normalised by the
  local peak** (`f_along/f_max ∈ [0,1]`) so it is **contrast-invariant** — the same
  rank-based lesson as the [[host-gate-decision]] (never trust an absolute level);
  (ii) a third **TRAVERSE** hypothesis (straight across the whole volume), which is
  the *correct* answer for a true crossing. Result over three hypotheses
  (TERMINATE / TRAVERSE / TURN): **KISSING → TERMINATE, CROSSING → TRAVERSE —
  separated**, robust over the Occam length-price `MU ∈ [0.31, 0.61]`.

**The concrete requirement handed to the full model:** it takes **both** a
directional, contrast-invariant data term (kills a straight traverse where no fibre
supports it — the kissing case) **and** global cross-tract parsimony (kills a turn
onto structure another tract already owns). The normalised discriminant is
`f(+x)/f_max = 0.43` (kissing, only d2's tail leaks into +x) vs `0.92` (crossing,
a real d1 fibre). Neither ingredient alone suffices.

- **`phase1d_generalize.py`** — generalisation test (crossing-angle sweep + a
  3-family junction). It first exposed that the Phase-1c energy rule was **overfit to
  70°** (fixed `MU` + a *linear* discount `(1-|t·d2|)` that haircut real d1 support
  at ~55°, plus an x<0/x≥0 split that miscounted the +y TURN leg at 90°), then
  **fixed it**: the parsimony discount became a hard **angular gate** (a segment
  earns nothing only when within ~20° of another tract's direction, position-
  independent). With the gate the energy rule now separates θ₂∈**[70,90°]**,
  matching the fundamental limit below — no longer overfit. The honest quantity
  underneath is the **forward-support discriminant** `disc = mean_{x>0} f(+x)/f_max`:

  | θ₂ | 25° | 45° | 55° | 70° | 90° |
  |----|-----|-----|-----|-----|-----|
  | kissing disc | 0.99 | 0.86 | 0.75 | 0.45 | 0.18 |
  | crossing disc | 1.00 | 0.97 | 0.96 | 0.92 | 0.88 |

  Crossing keeps `disc≈0.9` at every angle (a real d1 fibre persists); kissing
  decays. **Separable only for wide crossings (θ₂ ≳ 70°, gap ≥ 0.25); narrow
  crossings are genuinely ambiguous** — a d1 seed heading into nearly-parallel d2
  fibres sees the same forward support whether or not they connect. This is the
  same narrow-crossing resolution limit as the [[host-gate-decision]] "too-complex"
  bucket. The 3-family junctions point correctly (crossing-3 `disc=0.95`,
  kissing-3 `disc=0.46`), so the SIGNAL — not a tuned threshold — is what carries.

**Status:** BOTH the local discriminant AND the fixed (angular-gate) global energy
rule now generalise to the honest wide-angle limit [70,90°]; the gate was the fix.
Resolving *narrow* crossings is a genuine resolution limit (nearly-parallel fibres),
needing true whole-tract connectivity (does an individual fibre LINE continue across
the boundary?) — deferred to the full model, and may itself be resolution-bound.
**Phase 1 is closed at wide-angle scope; proceeding to Phase 2 (FOD likelihood on
real i74) carrying that scope as a documented caveat.**

### Phase-2 first result (real i74 — `phase2_i74_demote.py`)

Applies the generalising Phase-1 signal (FORWARD SUPPORT) as a per-crossing
**certifier** on real tissue: for every voxel HOST `classify` calls a 2-way
crossing, walk both peaks as continuity streamlines and test whether each traces a
**coherent, persistent line**. Answers the design-doc question "does global
continuity demote spurious crossings?".

- **Reach saturated — the phantom threshold did NOT transfer.** In dense lung tissue
  a continuity walk almost always finds *some* in-cone fibre above `R_STOP`, so
  "reach" hit the cap for ~every peak (100% confirmed = non-discriminating). This is
  the **same synthetic→real transfer failure** as the [[host-gate-decision]]
  ("absolute λ3 does not transfer"). Reach measures tissue density, not peak validity.
- **Straightness discriminates.** Switching the certifier to streamline
  **straightness** = |end-to-end displacement| / path length (real through-fibre ≈1,
  a wandering/spurious peak low) spreads cleanly: weaker-peak straightness median
  **0.58** (IQR [0.22, 0.83]) vs stronger-peak **0.87**. That asymmetry is the signal
  — the dominant peak is a real line, the second often is not.
- **Result:** on a 112³ crop, 20 HOST crossings, **13/20 = 65%** have a weak peak
  that fails the straightness test, 35% confirmed.

- **VALIDATION (`phase2b_validate.py`) — the certifier is NOT validated as a
  kissing/crossing test.** On known-truth phantoms: TRUE crossings **100% confirmed**
  (weaker straightness median 0.90 ✓), but KISSING boundary contacts only **27%
  demoted** (median 0.86 ✗). Reason: at a kissing contact the wrong-side peak still
  traces a *locally straight* partial-volume line and dies only on one side;
  straightness is an end-to-end ratio dominated by the long real side, so it stays
  high. **Straightness discriminates wandering / noise-driven peaks — which is what
  spread on i74 — but NOT kissing.** So the i74 65% must be read as "65% have a
  noise-like non-straight weak peak", *not* "65% are kissing-not-crossing".

**What this teaches — and the follow-up, also negative.** The kissing signal is
**directional asymmetry** (the peak continues far one way, dies the other), which a
single straightness scalar throws away. So I tried (i) `min(reach_forward,
reach_backward)` with a TIGHT (20°) cone as the certifier — a one-sided peak *should*
score low. **It also failed:** on the kissing phantom it demotes only **13%** (median
weaker min-reach 25 vox), while still confirming true crossings (100%, min-reach 22).
A one-sided kissing peak still reaches far enough on its real side, and HOST flags
boundary voxels at varied offsets, so min-reach stays high. **Both local per-peak
certifiers (straightness 27%, min-reach 13%) fail to catch kissing** — threshold/cone
tweaks do not rescue them.

**Firm conclusion:** a *local, per-peak* certifier **cannot** separate kissing from
crossing on realistic tissue; it can only flag noise-like peaks. Honest kissing/
crossing separation on real data needs the **full joint global model** — parsimony
over all tracts at once (does an individual fibre LINE connect across a junction?),
not one peak scored in isolation. That is the joint global model below.

### Phase-2.5 result — the joint global model works (2D mechanism, `phase3_global.py`)

Built the model Phase 2 proved is required: solve for a whole configuration of tracts
at once and score it jointly, so the *relational* fact (which segment connects to
which) is in the objective instead of thrown away.

- **Representation:** oriented segments from a 2D **multi-orientation** seeder (a
  gradient-orientation histogram, peaks → one segment per fibre direction; a single
  structure-tensor direction is near-isotropic at a crossing and cannot seed it —
  the same reason HOST-FOD exists, rediscovered here as a bug then fixed).
- **Energy (joint):** `E = w_c·Σ curvature(links) + w_n·(#tracts)` over segment-to-
  segment links (degree ≤ 2, acyclic → tracts are chains). Continuity wants smooth
  links; parsimony wants few tracts. Neither term is per-voxel-local.
- **Inference:** greedy min-curvature degree-constrained forest (Kruskal + union-
  find) — the exact global solve for this per-link-decomposable energy; `anneal()`
  keeps a Metropolis sampler for the general (coupled) case. **Direction-aware
  candidate links** (search along each segment's own tangent, not nearest-by-
  distance) were essential — raw nearest-neighbour links let the perpendicular
  family crowd out each fibre's continuation at a crossing.

**Result (test = tracts that reach ≥15 vox through the junction on both sides):**

| θ₂ | 45° | 60° | 70° | 90° |
|----|-----|-----|-----|-----|
| crossing through-tracts | 1 | 7 | 12 | 15 |
| kissing through-tracts | **0** | **0** | **0** | **0** |

**Kissing is 0 at every angle** — the global model *never* fabricates a through-
connection (zero false crossings), exactly the failure the local Phase-2 certifiers
could not avoid (they passed 73–87% of kissing). A true crossing is connected
through; its through-count scales with angle and goes weak at 45° (=1) — the same
narrow-crossing resolution limit as Phase 1 — but it fails **safely** (under-connects
rather than hallucinates). So the joint global model **separates kissing from
crossing where the local per-peak certifier could not** — the Phase-2 conclusion,
now demonstrated constructively.

**Next:** (i) scale the 2D mechanism to 3D and run on real i74 (reuse HOST FODs as
the multi-orientation seeder); (ii) the fully-honest Phase 3 then adds the L2
generative term + the **calibrated half-map noise likelihood**, so the prior is
pinned against real data and cannot silently hallucinate continuity.

---

## 3. What is already in place (reuse, don't rebuild)

- **HOST FODs** — `host_fod.py` (the per-voxel data term / L1 observation).
- **Forward model** — `reconstruction.py`'s LIC literally maps a direction field
  → image texture: the generative operator L2 needs.
- **Crossing + kissing phantoms with ground truth** — `host_geometry_test.py`
  (`make_kissing_phantom`, `make_true_crossing_same_angles`) — the Phase-1
  validation set already exists.
- **Spatial-regularisation seed** — `refined_field.spatial_regularize` (the
  baby version of the continuity prior).
- **Noise model (Phase 3)** — `half_map_analysis.py` T1, ready for when the data
  arrives.

---

## 4. Key risks & open decisions

1. **Prior hallucination** (above) — mitigate by validating on kissing phantoms
   where an over-eager continuity prior would *wrongly* merge the bundles.
2. **Inference cost** at volume scale (MCMC/annealing over a large state). Mitigate:
   gate with the cheap pipeline (only run global inference in the "too complex"
   regions), coarse-to-fine, and GPU.
3. **Validation without ground truth on real data** — same limitation as
   everywhere; lean on phantoms (Phase 1) + self-consistency + the half-map.
4. **Rep (A) vs (B)** — (A) is a quick win (spatial coherence) but does not buy
   the connectivity payoff; (B) does but is a real build. Decide after Phase 0.

---

## 5. Immediate next step (this scoping)

Phase-0 proof-of-concept: show on a 2D phantom that **global continuity traverses
a junction the per-voxel structure tensor cannot resolve** — `bayes_tracto_scope.py`.
It demonstrates the core "global > local" claim before committing to the full build.
