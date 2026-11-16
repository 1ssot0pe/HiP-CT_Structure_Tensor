# Global tractography

Answers the question the per-voxel HOST classifier can't: **is a flagged crossing
a REAL crossing** (two fibres genuinely pass through) **or a kissing/spurious one**
(two fibres touch, or noise faked a second peak)? A per-voxel look can't tell them
apart — you have to follow the fibres and see whether both actually continue. HOST
over-calls crossings on i74; this stage demotes the ones that don't hold up.

Shared phantom + kernel code (`crossing_shape_3d.py`, `kissing_phantom.py`,
`host_fod.py`, `noise_floor.py`, `reconstruction.py`) lives in **`core/`** —
this stage and `mass_conservation/` both use it. Files here put the repository
root on `sys.path` at the top and import `from core.<module> import ...`.

> **Phases 1–3 below are history.** They record how the crossing/kissing
> question was answered; most of their scripts were removed once the findings
> were written up (see [Archived](#archived)). Phase 1 and Phase 2 survive
> under new names: `fibre_continuity.py` and `peak_straightness.py`.

# Terms (read first)

- **Crossing vs kissing** — a *crossing* has both fibre centrelines passing
  through the same point (both continue out the other side). A *kissing* pair
  just touches at closest approach — neither centreline goes through, and each
  fibre turns away. Locally they look identical; globally they don't.
- **Continuity tracker / streamline** — start on a fibre and step along it,
  following the local direction. Through a real crossing it keeps going; at a
  kissing boundary it runs out of support and stops.
- **Forward support / straightness** — how well a candidate peak keeps finding a
  collinear fibre ahead of it. A real fibre traces a straight, persistent line; a
  spurious peak wanders or dies quickly.
- **Over-calling** — the per-voxel HOST classifier labelling more crossings than
  are real (noise-driven second peaks), which this stage exists to correct.
- **Joint global model** — instead of certifying each peak locally, fit *all*
  fibre segments and their links at once and keep the globally-consistent set.
  This is what finally separated kissing from crossing where local rules failed.
- **Generative likelihood (noise-whitened)** — each hypothesis (crossing vs
  kissing) predicts a clean image; score it by the residual **whitened by the
  measured noise** (`noise_model/`). The noise-calibrated version stays accurate
  where an un-calibrated one gets overconfident.
- **Gap sweep** — shrink the distance between two kissing bundles and find where
  they stop being separable (the resolution limit).

# The pipeline, in order

1. **The two phantoms** — `core/crossing_shape_3d.py` + `core/kissing_phantom.py`
   build a true crossing and a kissing pair with **known** geometry, so every
   method below can be checked against truth before touching real data.
   `core/localized_phantoms.py` builds the two the **live pipeline**
   validates against: a *localized* crossing (second family bounded to a zone,
   so single-family tissue exists in the same volume) and a curved kissing
   pair. These used to sit in `mass_conservation/`, which made this stage
   depend on a downstream folder.

2. **Phase 1 — can a tracker tell them apart?** — `fibre_continuity.py`
   (originally `phase1_kissing_crossing.py`).
   A continuity tracker traverses a real crossing but terminates at a kissing
   boundary. **Closed for wide angles**; narrow crossings stay unresolvable (a
   stated limit). Also the `local_fibreness` / `V` primitives Phase 2 builds on.

3. **Phase 2 — demote spurious crossings on real i74** — `peak_straightness.py`,
   originally `phase2_i74_demote.py` (forward-support straightness certifier). Walk both peaks of each HOST
   crossing; a real one traces a straight persistent line, a spurious one
   wanders. It demotes ~76% of HOST crossings — but a *local* per-peak certifier
   proved insufficient, which motivated Phase 3. Exports `collect_crossings`,
   which all five `mass_conservation/` i74 scripts import.

4. **Phase 3 — the joint global model** — consolidated into
   `archive/phase3.py`. The original scripts were `phase3_global.py` (the model),
   `phase3_i74.py` (on real i74), `phase3_validate3d.py` (3D phantom, kissing=0
   at every angle), `phase3_realnoise.py` (**real noise degrades true-crossing
   confirm 89%→57%**) and `phase3_generative.py` (the noise-calibrated generative
   likelihood — selects the true geometry 100%, beats a white-noise model at
   high noise).

5. **Stage 3 — two steps, and nothing more.**

   | | | |
   |---|---|---|
   | **3a** field | `field_refine.py` *(archived)* | Frank-Oseen smooth the director field |
   | **3b** the input | `dense_field.py` | `DenseField`: Stage 1's e3 at every voxel, HOST's peaks painted over it where the gate routed |
   | **3c** tracts | `dense_tracker.py` → `dense_optimise.py` → `tract_density.py` | cone search over that field; superposition S decides crossing vs kissing; coordinate descent on (lookahead, θ) |
   | **3d** the loop | `smooth_tracker.py` → `gradient_optimise.py` → `tdi_mse.py` → `descent_loop.py` | true gradient descent on (L, θ, α), scored by MSE(preprocessed image, normalised TDI), with every output written for a viewer |

   Driven by `run_pipeline.py`; 3a is opt-in with `--refine`. **3d** is its own
   entry point, `descent_loop.py`.

   **3b — what Stage 3 actually reads.** A *dense* field, not the patch list.
   `field_from_gated(..., host_only=True)` used to drop every non-routed patch —
   987 of 1331 on i74, i.e. every single-fibre measurement — so tracts existed
   only in `assoc_radius` bubbles around the 344 HOST patches, and `host_support`
   had to shrink the loss support to match. Now `vec1` carries e3 everywhere and
   `vec2` carries HOST's second peak only where there is one, so the loss is
   scored over the whole tissue mask. `vec2` and the labels remain *patch*-
   resolution evidence upsampled by nearest-centre fill — stated in
   `dense_field.py`, not quietly reported as per-voxel.

   > **Two findings from building the tracker, both measured on the crossing
   > phantom, both standing gates now.**
   >
   > *Brightness must not choose the turn.* Scoring candidate directions by
   > `intensity × |d·e3|` makes the walker follow the intensity ridge, not the
   > director: tracts left e3 by a median **12.7°** in voxels carrying exactly
   > one direction, where there is nothing to decide, while e3 is coherent to
   > **1.4°** voxel-to-voxel. The deviation is *flat* in the cone's angular
   > sampling (11.98° at α=0.4 through 12.93° at α=0.025, a 16× refinement), so
   > it is the support function and not the discretisation. Dropping intensity
   > from the direction choice — it still sets the floor that ends a walk — gives
   > **5.7°**, with crossing traversal (0.104) and kissing bridge (0.000)
   > **unchanged**. `single_fibre_check.py`.
   >
   > *Directors interpolate as `v vᵀ`.* 13.2% of adjacent voxel pairs are
   > sign-opposed, so componentwise trilinear sampling of `vec1` returned |v|
   > with median 0.78 and 31.8% of reads below 0.5 — the alignment evidence was
   > being cancelled before any exponent could sharpen it. The outer product has
   > no sign to cancel and `dᵀT d = (d·v)²` is exact. 
   
   > Same seam mechanism as the
   > 3a sign-alignment note above in frank-oseen, removed at the source.

   **3a — the whole field.** An orientation field has no head or tail, so its
   correct smoothing energy is Frank-Oseen, not a Gaussian blur. 
   
   Measured:
   angular error 9.10 → 8.21° at noise 0.3, tractography loss −1.3% (crossing)
   and −3.0% (kissing). Gaussian blurring of the same field makes it *worse*
   (15.99°), because componentwise blurring averages across the sign ambiguity;
   the unit-norm projection here does not. One constant, not three — equal
   K matched or beat the unequal white-matter prior at every noise level, so
   that prior is unsupported by this data. Off by default: the gain is real but
   small, and it costs 35–80 iterations of dense 3D vector calculus.

   > **Sign alignment must never leave the function that needs it.** Aligning
   > director signs is required only so finite differences see a consistent
   > field. Returning the aligned field cost the tractography **+62%**
   > (crossing) and **+161%** (kissing): BFS forces one global sign convention
   > on a field that admits none, manufacturing sign *seams*, and consumers
   > that sample by trilinear interpolation average across a seam and get ≈0.
   > `refine()` restores the input's signs on the way out. The energy is even
   > in **n**, so this is free.

   **3b — crossing vs kissing.** Superposition S only:
   `S = (I_c−I₀)/((I_a−I₀)+(I_b−I₀))`, decided at 0.75. Brightness *adds* where
   two fibres genuinely pass through; at a tangential kiss neither centreline
   does, so it can never exceed one fibre's worth.

   > **A geometric alternative was built and falsified.** Pairing the four
   > half-branches at a junction by minimum bending energy — "a crossing is the
   > pairing that stays straight, a kiss is the pairing that turns" — scored
   > **0/20 on kisses** at every arm radius and noise level, and grew *more*
   > confident as it got more room. The premise is false for a *tangential*
   > kiss: the two curves are locally parallel there, so bridging the gap is
   > perfectly straight while the true continuation must curve around its own
   > bulge. Minimum bending energy is therefore guaranteed to pick the false
   > merge. This is why brightness beats geometry here, and it is the same
   > mechanism behind Phase 2's local certifier failing on real i74.

   **3d — the optimisation loop.** `descent_loop.py` is the entry point and runs
   three things that used to be separate: a real gradient descent on the search
   cone, an objective scored against the raw greyscale, and the export of
   everything a viewer needs to check the result.

   *Why the descent is possible at all.* `dense_tracker` has **no derivative**
   in θ or α — θ reaches it only through `int(round(θ/α))`, a ring count, so
   `d/dθ` is exactly zero. `smooth_tracker` replaces the three discrete
   operations (ring count, argmax, hard floor) with a fixed quadrature plus an
   angular-resolution kernel, a softmax, and a sigmoid survival weight. The
   hard tracker is its τ→0 limit, not a different model. `gradient_optimise`
   measures the hard finite difference as its own control and prints it coming
   back at zero.

   *The objective.* `tdi_mse.py`. Both sides are put on [0, 1] before they are
   compared — the image by percentile clip + rescale (`core/preprocess.py`), the TDI
   by `log10(1 + D)` + rescale — so the residual is a plain mean square with no
   fitted scale and no fitted intercept. That fixes the scale mismatch that made
   the previous raw number meaningless (0.327 against a `mean(I²)` of 0.314);
   it does **not** fix the standing result in `cone_objective`. So every report
   carries `decompose`, which splits the loss exactly into

   > `MSE = (mean TDI − mean I)²  +  var TDI + var I − 2 cov(TDI, I)`
   >  ⟵ offset ⟶      ⟵ pattern ⟶

   and the run states what fraction of its own loss is the sparsity offset. If
   `offset_frac` is near 1 the descent is minimising **coverage**, not tracking
   quality, and the direction it moved means nothing. Stating that from the
   loop's own numbers is the point — it is not left to be inferred from where
   the descent ended.

   *Both field variants, always.* The loop runs twice on one setup: `vec1` only,
   and the full `vec1 + vec2` HOST field. Seeds are drawn **once from the full
   field and shared**, because `_dirs_at` seeds along `vec2` wherever Stage 2
   resolved one — letting each variant draw its own would give them different
   seed counts in different places and the difference between the two descents
   would be partly a difference in seeding.

   *Exports.* `viz_export.py`, on one affine so everything overlays: the
   preprocessed image, the normalised TDI and their **residual** as NIfTI; the
   tractogram as `.trk` (TrackVis, not `.tck` — a `.trk` carries dimensions,
   voxel size and voxel-to-RAS in its own header, which is the whole game for a
   crop out of a larger volume); the loss over (L, θ, α) as a NIfTI whose axes
   are parameters; a comparison figure whose joint histogram is the panel that
   decides the question; and the descent path plus the cone geometry as an
   interactive self-contained HTML.

   > **What the loop found (2026-08-11).** The normalisation does what it was
   > for and does not rescue the objective.
   >
   > *It works on a phantom.* Curved bundle, 4 iterations: loss 0.0422 → 0.0325,
   > Pearson r **+0.744 → +0.826**, offset 22% → 28%, and θ moved **up**
   > (0.30 → 0.35 rad). It does not collapse to the no-turn cone.
   >
   > *On real i74 it is a coverage term.* 48³ at centre 384: loss 0.2927 →
   > 0.2891, but `decompose` splits the endpoint as **offset 83.9%, pattern
   > 16.1%**, with r = **+0.034** and 19.8% coverage. 64³: offset **87.0%**,
   > r = +0.034. The image is bright over ~98% of tissue and a TDI is zero over
   > ~85% of it, so `log10(1+D)` cannot close the gap — zeros stay zero.
   >
   > *The global minimum is still degenerate.* The 5³ sweep on the 48³ chunk
   > minimises at **θ = 1.72°, the bottom of the range**, flagged automatically
   > as `ON A BOUNDARY`. The descent's own endpoint (θ = 8.3°, 0.28914) sits
   > *above* that grid minimum (0.28089) — it converged into a local basin, not
   > to the loss's actual preference. Note the whole 5³ grid spans only
   > **0.2809–0.2935**, i.e. 4.3% of the value: the surface is shallow, and what
   > slope it has points at no-turn. Same structure as `objective_search`'s
   > result for `mse_raw`, now measured with the scale mismatch removed rather
   > than fitted away.
   >
   > **`vec1` vs `vec1+vec2` needs the right chunk.** Two-direction voxels are
   > rare and *location*-specific on i74 — measured **zero** at centres
   > 300/450/520/600 (r=24), 450 (r=32) and 300 (r=**40**), but **8332** at
   > centre 384 r=32. A bigger chunk does not help. At a zero chunk the two
   > variants are the same array and return **bit-identical** TDIs and
   > streamlines, which reads exactly like two methods agreeing — hence the
   > explicit `n_multi == 0` warning in `main`. Caveat on the non-zero chunk:
   > FA seeding gates at FA > 0.7 and a crossing has *lower* FA, so only 3 of
   > 153 seeds start on a two-direction voxel; walkers can still switch onto
   > `vec2` in transit, but a small variant difference is partly a seeding
   > effect and not evidence that `vec2` is inert.

   > **Two meanings of α, selected by `turn`** (`dense_tracker.track`,
   > `smooth_tracker.track_soft`, `descent_loop --turn`).
   >
   > `turn="sample"` *(default, unchanged)* — α is the **angular step** in
   > radians. The cone is enumerated at that resolution and the walker jumps to
   > the single best-supported candidate. θ is then quantised in units of α
   > (`round(θ/α)` rings), so the two are not independent.
   >
   > `turn="relax"` — α is a **fraction in [0,1]**. The cone is sampled at a
   > fixed resolution (`RELAX_RINGS`, *not* α), the mean orientation of the
   > field inside it is taken, and the walker rotates from its current heading
   > toward that mean, stopping at fraction α of the way. α=0 never turns; α=1
   > turns by `min(angle-to-mean, θ)`. This removes the θ/α ring degeneracy
   > entirely and has no argmax, so it is smooth in all three parameters by
   > construction rather than by relaxation.
   >
   > *The mean is over outer products, not vectors.* e3 is a director, and 13.2%
   > of adjacent voxel pairs are sign-opposed, so a componentwise mean cancels
   > v against −v. `_mean_director` sums `v vᵀ` and takes the principal
   > eigenvector — the standard mean for axial data — then signs it against the
   > current heading.
   >
   > *The θ clamp is load-bearing.* It is tempting to argue a mean of vectors
   > sampled inside a θ cone must lie within θ. **That is false and measurably
   > so:** the cone bounds *where the field is read*, not *what is found there*.
   > Unclamped, the turn reached **89.87° at θ = 17.19°** on the curved phantom.
   > With the clamp the turn is exactly `α·θ` — verified in both trackers.
   >
   > *α is reported in its own units.* `descent_loop.fmt_alpha` prints degrees
   > for `sample` and a bare fraction for `relax`; printing a fraction in degrees
   > reports 0.587 as "33.64 deg", which looks like an angle and is not one.

   > **The tractogram written out is the one the loss scored.** `track_soft`
   > concatenates every walker's steps, because every loss here is a sum over
   > steps and does not care which walker a step came from — but a polyline
   > does. `return_ids` carries a walker index through so `as_streamlines` can
   > reassemble it; without that the file a viewer opens would have to come from
   > a *second* tracking run. The reassembly needs a **stable** sort (steps are
   > appended in time order and walkers keep their relative order within each
   > append) and it rejoins the `both_ways` halves, reversing the backward one —
   > written out separately they appear as two tracts meeting end-to-end at
   > every seed, doubling the tract count and putting a spurious endpoint in the
   > middle of every fibre.

6. **Kissing gap — the resolution limit** — `kissing_gap_sweep.py` +
   `test_kissing_phantom.py`. Global tractography stays clean down to a gap of
   **~5× the fibre radius**; below that the phantom itself fuses — a resolution
   limit, not a bug.

7. **Whole-pipeline capstone** — `brain_phantom_pipeline.py`. Runs every stage
   at once on a 3D all-features phantom (single / crossing / kissing / complex)
   at realistic SNR, against known labels.

# How to run

From the repository root, with the environment from the [main README](../README.md#setup) active, e.g. `python -u 3_global_tractography/run_pipeline.py`.
Phantom-based scripts need no real data; the i74 ones need the volume.

| Script | Needs | What it shows |
|---|---|---|
| **`run_pipeline.py`** | phantom (`--i74` for real) | **the end-to-end Stage 1→2→3 run**; exits non-zero on any FAIL. `--tracker patch` for the old walker, `--alpha-check` to test the cone's angular sampling |
| **`descent_loop.py`** | i74 volume | **the 3d loop**: gradient descent on (L, θ, α) against MSE(preprocessed image, normalised TDI), both field variants, all viewer exports. `--sweep-n 5` adds the loss volume, `--smooth` the scale-matched variant |
| `tdi_mse.py` | nothing | self-test: the loss decomposition is exact, `log10` normalisation keeps zeros at zero and is monotone, the image transform is a no-op on an already-[0,1] volume |
| `viz_export.py` | — | library: NIfTI / `.trk` / comparison figure / parameter volume / interactive HTML, all on one affine |
| `descent_diagnose.py` | i74 volume | why a descent stopped early: h-convergence of the finite difference, and a line scan through the converged point |
| `param_volume.py` | i74 volume | the (L, θ, α) loss volume as NIfTI, on the `align` objective |
| `tract_density.py` | nothing | self-test: a tract crossing a voxel edge-to-edge deposits 1.0 regardless of step size; a crossing reads 2.0 |
| `test_frame_convention.py` | nothing | **standing gate**: the `(x,y,z)`→`(z,y,x)` reversal, at the tensor, in `DenseField`, and end-to-end |
| `dense_tracker.py` | nothing | self-test: the vectorised tracker is bit-identical to `cone_search.track`, 15–171× faster |
| `single_fibre_check.py` | phantom | **standing gate**: tracts follow e3 where e3 is the only direction on offer |

`peak_straightness.py` and `fibre_continuity.py` are here for `mass_conservation/`,
not for this stage — see the reach-in note below.

## Archived

`archive/` holds the Stage 3 scripts that are not part of the descent loop —
`objective_search`, `phase3`, `kissing_gap_sweep`, `brain_phantom_pipeline`,
`gradient_surface`, `descent_mse`, `descent_curve`, `s_calibration`,
`loss_surface`, `loss_sensitivity`, `param_volume_figure`, `field_before_after`,
`test_kissing_phantom`, `toy_demo`. **The findings are recorded above; only the
code moved.** `archive/README.md` says what each one established and how to
re-run it. Nothing outside `archive/` imports anything inside it.

Deleted in August 2026, with the findings recorded above and in
[BAYES_TRACTOGRAPHY.md](../docs/archive/BAYES_TRACTOGRAPHY.md): `phase3_global`,
`phase3_i74`, `phase3_validate3d`, `phase3_realnoise`, `phase3_generative`,
`phase1b_global`, `phase1c_directional`, `phase1d_generalize`,
`phase2b_validate`, `phase3_calibrated`, and the figure-only scripts
`*_arrows.py` / `phantom_overview.py`. To recover one, find its last version in
git history, e.g. `git log --all --oneline -- '*phase3_generative.py'`.

# Shared in `core/`, and the one downstream reach-in

Kernels in `core/`: `crossing_shape_3d`, `kissing_phantom`, `host_fod`,
`host_geometry_test`, `noise_floor`, `reconstruction`, `fod_dipy`.
The `mass_conservation/` stage reaches *into* this folder for
`fibre_continuity` and `peak_straightness` (it reuses those certifiers) —
a one-way downstream→upstream dependency, via `sys.path.insert`.
