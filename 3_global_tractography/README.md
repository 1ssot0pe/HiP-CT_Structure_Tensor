# Global tractography

Answers the question the per-voxel HOST classifier can't: **is a flagged crossing
a REAL crossing** (two fibres genuinely pass through) **or a kissing/spurious one**
(two fibres touch, or noise faked a second peak)? A per-voxel look can't tell them
apart — you have to follow the fibres and see whether both actually continue. HOST
over-calls crossings on i74; this stage demotes the ones that don't hold up.

Shared phantom + kernel code (`crossing_shape_3d.py`, `kissing_phantom.py`,
`host_fod.py`, `noise_floor.py`, `reconstruction.py`) stays at the **repo root** —
this stage and `mass_conservation/` both use it. Files here reach root via a
`sys.path.insert` at the top.

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

1. **The two phantoms** — `crossing_shape_3d.py` + `kissing_phantom.py` *(root)*
   build a true crossing and a kissing pair with **known** geometry, so every
   method below can be checked against truth before touching real data.
   `localized_phantoms.py` *(root)* builds the two the **live pipeline**
   validates against: a *localized* crossing (second family bounded to a zone,
   so single-family tissue exists in the same volume) and a curved kissing
   pair. These used to sit in `mass_conservation/`, which made this stage
   depend on a downstream folder.

2. **Phase 1 — can a tracker tell them apart?** — `phase1_kissing_crossing.py`.
   A continuity tracker traverses a real crossing but terminates at a kissing
   boundary. **Closed for wide angles**; narrow crossings stay unresolvable (a
   stated limit). Also the `local_fibreness` / `V` primitives Phase 2 builds on.

3. **Phase 2 — demote spurious crossings on real i74** — `phase2_i74_demote.py`
   (forward-support straightness certifier). Walk both peaks of each HOST
   crossing; a real one traces a straight persistent line, a spurious one
   wanders. It demotes ~76% of HOST crossings — but a *local* per-peak certifier
   proved insufficient, which motivated Phase 3. Exports `collect_crossings`,
   which all five `mass_conservation/` i74 scripts import.

4. **Phase 3 — the joint global model** — `phase3_global.py` (the model),
   `phase3_i74.py` (on real i74), `phase3_validate3d.py` (3D phantom, kissing=0
   at every angle), `phase3_realnoise.py` (**real noise degrades true-crossing
   confirm 89%→57%**), `phase3_generative.py` (the noise-calibrated generative
   likelihood — selects the true geometry 100%, beats a white-noise model at
   high noise).

5. **Stage 3 — two steps, and nothing more.**

   | | | |
   |---|---|---|
   | **3a** field | `field_refine.py` | Frank-Oseen smooth the director field |
   | **3b** topology + tracts | `self_optimising_tractography.py` → `tract_density.py` | superposition S decides crossing vs kissing, then the tracts are fitted |

   Driven by `run_pipeline.py`; 3a is opt-in with `--refine`.

   **3a — the whole field.** An orientation field has no head or tail, so its
   correct smoothing energy is Frank-Oseen, not a Gaussian blur. Measured:
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

6. **Kissing gap — the resolution limit** — `kissing_gap_sweep.py` +
   `test_kissing_phantom.py`. Global tractography stays clean down to a gap of
   **~5× the fibre radius**; below that the phantom itself fuses — a resolution
   limit, not a bug.

7. **Whole-pipeline capstone** — `brain_phantom_pipeline.py`. Runs every stage
   at once on a 3D all-features phantom (single / crossing / kissing / complex)
   at realistic SNR, against known labels.

# How to run

From the repo root, with `$PY` set to the interpreter in your venv (see `PIPELINE.md` § 0 — the venv lives **outside** this folder, because it is OneDrive-synced), e.g. `$PY -u 3_global_tractography\run_pipeline.py`.
Phantom-based scripts need no real data; the i74 ones need the volume.

| Script | Needs | What it shows |
|---|---|---|
| **`run_pipeline.py`** | phantom (`--i74` for real) | **the end-to-end Stage 1→2→3 run**; exits non-zero on any FAIL |
| `phase1_kissing_crossing.py` | phantom | tracker separates kissing from crossing |
| `phase2_i74_demote.py` | i74 volume | straightness demotes ~76% of HOST crossings |
| `phase3_i74.py` | i74 volume | joint global tractography on real tissue |
| `phase3_realnoise.py` | phantom + `lung_A/B.npy` | the 89%→57% real-noise degradation |
| `phase3_generative.py` | phantom | noise-calibrated model selection |
| `kissing_gap_sweep.py` | phantom | the ~5× fibre-radius resolution limit |
| `brain_phantom_pipeline.py` | phantom | whole-pipeline capstone (5/5 at realistic SNR) |

Deleted in the `simple` pass (findings recorded above and in
`BAYES_TRACTOGRAPHY.md`): `phase1b_global`, `phase1c_directional`,
`phase1d_generalize` (superseded Phase-1 extensions), `phase2b_validate`,
`phase3_calibrated` (the scalar-gate negative), and the figure-only scripts
`*_arrows.py` / `phantom_overview.py`.

# Shared at repo root, and the one downstream reach-in

Kernels at root: `crossing_shape_3d.py`, `kissing_phantom.py`, `host_fod.py`,
`host_geometry_test.py`, `noise_floor.py`, `reconstruction.py`, `fod_dipy.py`.
The `mass_conservation/` stage reaches *into* this folder for
`phase1_kissing_crossing` and `phase2_i74_demote` (it reuses those certifiers) —
a one-way downstream→upstream dependency, via `sys.path.insert`.
