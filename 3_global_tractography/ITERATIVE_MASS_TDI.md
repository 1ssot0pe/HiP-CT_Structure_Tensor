# Iterative tract-density + mass-conservation pipeline — DESIGN (not yet built)

Status: **design agreed, no code written yet.** Paused 2026-07-30.

## The goal

Make global tractography an **iterative** process driven by tract density, and use
**mass conservation as the scoring metric** at every iteration. For each identified
junction patch, iterate over the possible wiring combinations and score them; after
N iterations kissing and crossing separate.

## Decision taken

**Substrate: 3D, on the already-validated mass-conservation phantoms.** Chosen over
(a) extending the 2D `phase3_global.py` machinery and (b) going straight to i74.

Rationale: `mass_conservation_validate.py`'s ratio is 3D and already proven
(catch 100% / false-positive 0%), so the metric arrives pre-validated instead of
being reimplemented in 2D where that proof would not transfer. The phantoms carry
known truth, so "kissing and crossing are separated after N iterations" is
measurable rather than asserted. Real i74 was rejected for now because of the
recorded calibration selection bias (only ~30% of candidates get scored, and the
unscored ones sit in significantly denser tissue, p < 0.001) — that would confound
whether the loop itself is working.

## The core idea

At a junction patch HOST flags with 2 peaks, the four half-fibre stubs entering the
patch can be wired a small number of ways. Each wiring **predicts a tract density at
the centre**; mass conservation **measures** that same number from intensity.

| hypothesis | wiring | predicted centre density |
|---|---|---|
| `CROSS` | N–S and E–W (both centrelines through) | 2 |
| `TURN_A` | N–E and S–W | ~1 |
| `TURN_B` | N–W and S–E | ~1 |
| `TERMINATE` | one pair joined, others end | ~1 |

Measured side: `mass_ratio = I(centre) / single-fibre calibration`
(`mass_conservation_validate.local_calibration`, which takes the median peak
intensity of nearby patches HOST confidently calls "single").

**Score per hypothesis = residual between its predicted centre density and the
measured mass ratio.** Per the brief, mass conservation is *the* metric — curvature
enters only as a proposal prior, not as part of the score.

## Why it must be iterative (not one-shot)

The two sides are coupled, so this is a fixed point:

- the single-fibre **calibration** depends on which nearby voxels are genuinely
  single-fibre, which is what the tractography is deciding;
- the **density at a junction** depends on whether upstream junctions were resolved
  the right way.

One pass computes each from a guess of the other. Iterating lets them agree.

## Loop (one iteration)

1. **Sample** streamlines probabilistically from the HOST fibre-ness FOD (angular
   dispersion around each peak), with junction transitions drawn from the current
   per-patch hypothesis weights.
2. **Rasterise** to a TDI (track-density image) volume — the pipeline artifact the
   brief calls "tract densities".
3. **Measure** the mass-conservation ratio at each patch centre, recalibrating from
   voxels the current TDI says are single-fibre.
4. **Score** each hypothesis by its density-vs-ratio residual.
5. **Update** the per-patch hypothesis posterior (softmax of −residual²/2σ²).
6. Repeat until converged or N iterations.

**Readout:** per-patch posterior mass on `CROSS` vs the turn hypotheses, against the
phantoms' known labels. The success criterion is separation of those two
populations, and how many iterations it takes.

## Pieces to reuse (do not reimplement)

| need | where |
|---|---|
| mass ratio + local calibration | `mass_conservation/mass_conservation_validate.py` — `local_calibration`, `_patch_at`, `CROSS_HALF=14`, `KISS_HALF=8`, `THRESH=1.5` |
| labelled 3D phantoms | same file — `raw_localized_crossing`, `raw_kissing` |
| noise (synthetic + real half-map) | same file — `anoise`, `real_noise` (needs `lung_A/B.npy`) |
| patch detection / FOD peaks | `core/host_fod.py` — `analyse`, `classify`, `patch_gradients`, `directional_energy` |
| a streamline stepper to adapt | `phase1_kissing_crossing.py` — `track3d`, `local_fibreness` (greedy; needs stochastic sampling added) |
| joint energy / link inference | `phase3_global.py` — `greedy_forest`, `anneal`, `candidate_links` (2D; the *pattern* transfers) |
| sphere directions | `core/fod_dipy.py` — `SPHERE` |

## Planned new files

- `3_global_tractography/tract_density.py` — streamline → TDI rasteriser (the density primitive)
- `3_global_tractography/iterative_mass_tractography.py` — the loop above
- a validation entry point reporting separation vs known labels per iteration

## Open questions for when this resumes

- Number of streamline samples per iteration vs cost — `local_fibreness` calls
  `patch_gradients` per step and is the bottleneck; likely needs a precomputed
  fibre-ness field on a strided grid rather than on-the-fly evaluation.
- Whether `TERMINATE` needs to be separate from the turn hypotheses given both
  predict ~1.
- Convergence criterion and how to detect oscillation between coupled patches.
