# Optimising Structure Tensors

UCL MPBE Summer 2026.

Reads fibre orientation out of HiP-CT volumes. The core problem: a structure
tensor returns **one** direction per voxel, but where two fibres cross there is
no single correct direction. This pipeline finds those places, and decides which
of them are real crossings rather than fibres that merely touch.

```powershell
.\run_all.ps1                  # regression check -> phantoms -> real volume
.\run_all.ps1 -Quick -SkipI74  # ~1 min smoke test, no data needed
```

Setup and the full step-by-step are in [PIPELINE.md](PIPELINE.md).

## The four steps

| Step | Question | Output |
|---|---|---|
| **0** Load | What are we measuring? | `vol (z,y,x)` |
| **1** Scale | How wide a region should we measure over? | σ, ρ per 32³ block |
| **2** Gate | Can one direction be trusted here? | orientation field + routing mask |
| **3** Resolve | Real crossing, or fibres just touching? | S + verdict per junction |

## Logic flow

Each step asks one question. The **decisions** are what the pipeline is; the
boxes are just where they happen.

```
                    reconstruction.load_volume()      real scan, no truth
                    localized_phantoms.raw_*()        phantom, known truth
                                   │
                                   ▼   vol (z, y, x) float32

┌─ STEP 1 · At what scale do we measure? ─────────────  1_sigma_rho_selection
│
│    A structure tensor has two knobs, and the tissue sets both:
│
│      first peak of the Lindeberg response  →  σ    how wide a gradient
│      where angular drift flattens          →  ρ    how far to average
│
│    σ too small measures noise; too large blurs neighbouring fibres.
│    ρ too small leaves direction jittery; too large averages across fibres.
│    Picked per 32³ block, because the best σ varies ~1–6 across a volume.
│
│    ? does the block hold tissue    block mean ≥ 5% of the volume max
│      ├─ yes → pick (σ, ρ) for it, and use that
│      └─ no  → BACKGROUND. skipped before any scale pick is attempted;
│               those patches fall back to σ = ρ = 0.85  ⚠ WARNS
│
│    A sparse volume can fail this on every block — then Step 1 never ran,
│    and the whole run is on one fixed scale. Cost, measured: ~20° of
│    angular error on the kissing phantom.
└──────────────────────────────────────────────────────────────────────────
                                   │
                                   ▼   sigma_map, rho_map

┌─ STEP 2 · Can ONE direction be trusted here? ───────  2_host_classification
│
│    A tensor gives one direction per voxel. Where two fibres cross, that
│    answer is not imprecise — it is meaningless. Find those places, and
│    spend the expensive method only there.
│
│    per patch (half = 14):   host_fod.analyse(σ_grad = 4.0)
│                               → c_s   how much it has NO dominant direction
│                               → FA    how strongly directional it is
│
│    ? c_s ≤ FA  AND  fewer than 3 peaks        ← both from the same
│      │                                          eigenvalues, so no
│      │                                          threshold to tune
│      ├─ yes → SINGLE.  keep Step 1's e3, one direction, done
│      │
│      └─ no  → ROUTE TO HOST-FOD, at a different scale:
│               analyse(σ_grad = 2.5) → up to 2 directions
│               classify() → single, crossing, dispersed,
│                            disorganised, complex
│
│    Three scales, not interchangeable:  σ≈1.0 for direction,
│    4.0 for the gate, 2.5 for peaks. At 1.0 the gate never fires;
│    at 4.0 the second peak is already smoothed away.
└──────────────────────────────────────────────────────────────────────────
                                   │
                                   ▼   GatedField → results/unified_field_*.npz

┌─ STEP 3 · Real crossing, or fibres just touching? ── 3_global_tractography
│
│  (a) MEASURE — from the image, once per junction. Brightness adds:
│
│        S = (I_c − I₀) ⁄ ((I_a − I₀) + (I_b − I₀))
│
│        S ≈ 1.0   both fibres deposit here      →  CROSSING
│        S ≈ 0.5   only one is really here       →  TANGENCY
│        decision at 0.75, ±0.10 reported as AMBIGUOUS
│
│  (b) FIT — θ = (step, angle, seed density, link weight)
│
│        ┌───────────────────────────────────────────────────────┐
│        │                                                       │
│        ▼                                                       │
│      integrate streamlines through the field                   │
│        └─ S is a WALL here: a path may run ALONG a tangency,   │
│           never THROUGH it                                     │
│        │                                                       │
│        ▼                                                       │
│      rasterise → D_rec    arc-length weighted, so density 2    │
│        │                  literally means two fibres           │
│        ▼                                                       │
│      L = MSE(D_rec, D_obs)        does it explain the image?   │
│          + verdict penalty        does it obey S?              │
│        │                            CROSSING wants density 2   │
│        │                            TANGENCY wants density ≤ 1 │
│        ▼                                                       │
│      coordinate descent — one knob at a time, keep what helps ─┘
│        (no gradient: streamlines terminate discretely, so L is
│         flat, then jumps. There is no slope to follow.)
│
│      stop when a full sweep improves nothing
└──────────────────────────────────────────────────────────────────────────
                                   │
                                   ▼

     results/pipeline_results.json    S + verdict per junction, fitted θ
     results/unified_field_*.npz      the orientation field
     results/field_*.png              greyscale + orientations, to check by eye
     stdout                           PASS / FAIL, or a verdict with no
                                      correctness claim on real data
```

**Where the loop closes.** S is measured once from the image and never revised.
θ adapts to S; S never adapts to θ. So a wrong verdict is optimised *toward*,
not away from — the honest limit of the current design.

## What actually runs

Traced from a real run — 13 modules, not the 56 in the repo:

```
run_pipeline.py                                  the only entry point
 ├─ reconstruction.py            load_volume         → vol
 │  or localized_phantoms.py     raw_* phantoms      → vol + known truth
 │
 ├─ sigma_rho_gated_field.py                       STEP 1 + 2
 │   ├─ sigma_rho_adaptive.py → pick_sigma_rho.py → orthogonality_residual.py
 │   └─ host_fod.py → two_orientation_test.py → crossing_shape_3d.py
 │                                                 → kissing_phantom.py
 ├─ self_optimising_tractography.py                STEP 3
 │   └─ tract_density.py
 └─ render_field.py                                figures
```

Each run writes `results/field_*.png` — the greyscale image with the recovered
orientations drawn on top, so a run can be checked by eye, not just read.

Everything else in the repo is either a library for one of these, or a
standalone experiment recording a finding. Nothing is orphaned.

## Root engine

Shared by every folder. The count is how many tracked files import it.

| File | Function | In → Out |
|---|---|---|
| `reconstruction.py` (18) | data gateway; volume path + voxel size live here | TIFF folder, `crop` → `vol` |
| `orthogonality_residual.py` (6) | the plain structure tensor | `(vol, σ, ρ)` → `e3` |
| `crossing_shape_3d.py` (15) | tube/coord primitives, `shape_measures`, `knee_sigma` | params → `vol`; `(vol,σ,ρ)` → c_l/c_p/c_s |
| `kissing_phantom.py` (5) | curved bundles that touch but never cross | `(gap, curv)` → `vol` |
| `localized_phantoms.py` (4) | the two phantoms the pipeline validates against | params → `vol` + truth |
| `two_orientation_test.py` (3) | `patch_gradients` primitive | `(patch, σ_grad)` → scatter `G` |
| **`host_fod.py` (18)** | **the HOST-FOD kernel** | `analyse(patch, σ_grad)` → metrics; `classify(m)` → label |
| `noise_floor.py` (8) | N-matrix / power-spectrum library | `(I, σ)` → `N` |
| `host_geometry_test.py` (6) | labelled phantom suite | params → `vol` + truth |
| `selfcalib_threshold.py` (4) | self-calibrating c_s guard | `(vol,σ,ρ)` → c_s field |
| `model_ladder.py` (2) | c_s calibration against known geometries | known dirs → c_s ladder |
| `fod_dipy.py` (1) | dipy FOD cross-check | `patch` → dipy peaks |

`classify` returns `single` / `crossing` / `dispersed` / `disorganised` /
`complex`. `complex` means the method gave up — filters selecting `== "single"`
correctly exclude it.

## Folders

| | |
|---|---|
| `1_sigma_rho_selection/` | choosing σ and ρ, and the evidence they are right |
| `2_host_classification/` | the routing gate and its alternatives |
| `3_global_tractography/` | crossing vs kissing; the Step-3 loop; the entry point |
| `mass_conservation/` | the same question answered by brightness instead |
| `noise_model/` | measuring the noise, so "real or noise?" gets a number |

Each folder has a README with its ordered walkthrough and what each experiment
found. Files reach the root engine via a `sys.path.insert`, so any script runs
from anywhere.

## Status

| Step | Runs on any volume? | Note |
|---|---|---|
| 1 σ, ρ | yes (edit `reconstruction.py`) | ground-truth-free |
| 2 gate | yes | self-calibrates; no tuned threshold |
| 3 resolve | yes | mechanism-validated on phantoms |
| noise | needs even/odd half-maps | check `corr(A,B)` first |

Main open items: the noise-calibrated likelihood is validated as a model
selector but not yet wired into link-scoring; `D_obs = λ1 − λ3` cannot tell one
bright fibre from two dim crossing ones. Full caveats in [PIPELINE.md](PIPELINE.md);
the record of what was tried and falsified is in
[archive/BAYES_TRACTOGRAPHY.md](archive/BAYES_TRACTOGRAPHY.md).
