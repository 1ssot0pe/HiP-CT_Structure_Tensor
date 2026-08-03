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

2. **Phase 1 — can a tracker tell them apart?** — `phase1_kissing_crossing.py`
   (+ `phase1b_global` / `phase1c_directional` / `phase1d_generalize`
   extensions). A continuity tracker traverses a real crossing but terminates at
   a kissing boundary. **Closed for wide angles**; narrow crossings stay
   unresolvable (a stated limit).

3. **Phase 2 — demote spurious crossings on real i74** — `phase2_i74_demote.py`
   (forward-support straightness certifier) + `phase2b_validate.py`. Walk both
   peaks of each HOST crossing; a real one traces a straight persistent line, a
   spurious one wanders. It demotes ~76% of HOST crossings — but a *local*
   per-peak certifier proved insufficient, which motivated Phase 3.

4. **Phase 3 — the joint global model** — `phase3_global.py` (the model),
   `phase3_i74.py` (on real i74), `phase3_validate3d.py` (3D phantom, kissing=0
   at every angle), `phase3_realnoise.py` (**real noise degrades true-crossing
   confirm 89%→57%**), `phase3_calibrated.py` (honest negative — a scalar gate
   can't pin the over-calling magnitude), `phase3_generative.py` (the
   noise-calibrated generative likelihood — selects the true geometry 100%,
   beats a white-noise model at high noise).

5. **Kissing gap — the resolution limit** — `kissing_gap_sweep.py` +
   `test_kissing_phantom.py`. Global tractography stays clean down to a gap of
   **~5× the fibre radius**; below that the phantom itself fuses — a resolution
   limit, not a bug.

6. **Whole-pipeline capstone** — `brain_phantom_pipeline.py`. Runs every stage
   at once on a 3D all-features phantom (single / crossing / kissing / complex)
   at realistic SNR, against known labels.

7. **Figures** — `*_arrows.py` (recovered-orientation arrow overlays for the
   kissing/gap-sweep/capstone phantoms) and `phantom_overview.py` (phantom
   gallery).

# How to run

From the repo root, e.g. `.venv\Scripts\python -u global_tractography\phase3_i74.py`.
Phantom-based scripts need no real data; the i74 ones need the volume.

| Script | Needs | What it shows |
|---|---|---|
| `phase1_kissing_crossing.py` | phantom | tracker separates kissing from crossing |
| `phase2_i74_demote.py` | i74 volume | straightness demotes ~76% of HOST crossings |
| `phase3_i74.py` | i74 volume | joint global tractography on real tissue |
| `phase3_realnoise.py` | phantom + `lung_A/B.npy` | the 89%→57% real-noise degradation |
| `phase3_generative.py` | phantom | noise-calibrated model selection |
| `kissing_gap_sweep.py` | phantom | the ~5× fibre-radius resolution limit |
| `brain_phantom_pipeline.py` | phantom | whole-pipeline capstone (5/5 at realistic SNR) |

# Shared at repo root, and the one downstream reach-in

Kernels at root: `crossing_shape_3d.py`, `kissing_phantom.py`, `host_fod.py`,
`host_geometry_test.py`, `noise_floor.py`, `reconstruction.py`, `fod_dipy.py`.
The `mass_conservation/` stage reaches *into* this folder for
`phase1_kissing_crossing` and `phase2_i74_demote` (it reuses those certifiers) —
a one-way downstream→upstream dependency, via `sys.path.insert`.
