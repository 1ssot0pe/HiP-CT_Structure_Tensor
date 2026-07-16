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
| **1 — 3D on phantoms** | Rep (A)→(B), MRF/segment inference, validated on the `host_geometry_test` crossing **and kissing** phantoms (known truth). Milestone: correctly separate kissing from crossing. | No (FOD data term, L1) |
| **2 — real i74, FOD likelihood** | Run L1 on i74 tissue; compare the global field to the per-voxel one; does it demote spurious crossings and connect real tracts? | Weakly (FOD weight) |
| **3 — generative, honest** | L2 forward model (LIC) + **calibrated noise likelihood**; full posterior / uncertainty. | **Yes** — the residual term is meaningless without the measured noise |

So: **Phases 0–1 can start immediately**; the half-map is required only to make
Phase 3 (the honest generative posterior) rigorous — and, per the noise thread,
to keep the prior from silently running the show.

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
