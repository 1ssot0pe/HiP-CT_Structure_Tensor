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
