# Optimising Structure Tensors

UCL MPBE Summer 2026.

Reads fibre orientation out of HiP-CT volumes, decides where the one-direction
structure tensor can be trusted, and resolves the rest with a global
tractography that optimises itself against the image.

## The live path

One chain runs end to end. Everything else is a library it uses, or recorded
evidence for why it looks the way it does.

```
run_pipeline.py --i74            ← the only entry point
 ├─ reconstruction.load_volume                      (raw volume)
 ├─ sigma_rho_gated_field.build_gated_field         STAGE 1+2
 │   ├─ sigma_rho_adaptive.compute_adaptive_map → pick_sigma_rho → orthogonality_residual
 │   └─ host_fod.analyse / classify → fod_dipy → two_orientation_test
 │                                    → selfcalib_threshold → noise_floor
 │                                    → crossing_shape_3d
 ├─ self_optimising_tractography (S3)               STAGE 3
 │   └─ tract_density.rasterise
 └─ mass_conservation_validate.raw_localized_crossing / raw_kissing   (phantoms)
```

```
$PY 3_global_tractography/run_pipeline.py            # phantoms, known truth
$PY 3_global_tractography/run_pipeline.py --i74      # real volume
```

`$PY` is the Python in the project venv, which lives **outside this repo**
(currently `%USERPROFILE%\venvs\hipct-structure-tensor`) because this folder is
OneDrive-synced and sync churn corrupts an in-repo venv. Setup is in
`PIPELINE.md` § 0; dependencies are the 9 in `requirements.txt`.

## What each stage does

```
raw volume V (z,y,x float32)
        │
        ▼
╔═ STAGE 1 ═════════════════════════════════════════════════╗
║ sigma_rho_adaptive.compute_adaptive_map(vol, block=32)    ║
║   → sigma_map, rho_map, zs/ys/xs   (per 32³ block)        ║
║ block_scale(centre) → (σ, ρ) for each patch               ║
║ tensor_fields(vol, σ, ρ) → val (λ desc), vec (e3)         ║
╚═══════════════════════════════════════════════════════════╝
        │  per-block (σ,ρ) + dense e3
        ▼
╔═ STAGE 2 ═ sigma_rho_gated_field.build_gated_field ════════╗
║ per patch (half=14, grid anchored on volume centre):      ║
║   analyse(patch, σ_grad=4.0, order=6) → c_s, FA           ║
║                                                            ║
║   ┌── c_s ≤ FA ──→ single: dirs = [e3 from Stage 1]       ║
║   └── c_s >  FA ──→ analyse(patch, σ_grad=2.5, order=6)   ║
║                     → dirs (≤2), classify() → label       ║
╚═══════════════════════════════════════════════════════════╝
        │  GatedField: centre, dirs, n_dirs, gated, label,
        │              c_s, FA, sigma, rho
        ├──→ gated.save("unified_field_*.npz")   ← the artifact
        ▼
╔═ STAGE 3 ═════════════════════════════════════════════════╗
║ S3.orientation_field(vol)          → e3 (z,y,x order)     ║
║ S3.field_from_gated(gated, vol, e3)                       ║
║   · reverses every dir (x,y,z) → (z,y,x)   ← the join     ║
║   · multi-dir patches → superposition_S() → S, verdict    ║
║   → field[], junctions[], I_0                             ║
║                                                            ║
║ S3.anisotropy_target(vol) → D_obs = norm(λ1−λ3), tissue   ║
║                                                            ║
║ S3.optimise(...)  coordinate descent over θ:              ║
║   loss(θ) → integrate(field, θ, verdicts)  ← S gates      ║
║                                    links here              ║
║           → rasterise() → D_rec                            ║
║           → mse(D_rec, D_obs) + verdict penalty            ║
║   → θ, loss trace, per-junction S/verdict/contribution     ║
╚═══════════════════════════════════════════════════════════╝
```

Three scales are in play and they are **not** interchangeable — Stage 1's σ
(~0.85–1.0 vox) for the single-fibre direction, 4.0 for the routing gate, 2.5
for extracting peaks. At σ=1.0 the gate never fires; at 4.0 the peaks that
justified gating are already smoothed away. `sigma_rho_gated_field.py` records
the sweep.

## Folders

| | |
|---|---|
| `1_sigma_rho_selection/` | pick the two structure-tensor scales |
| `2_host_classification/` | route each patch: cheap answer, or HOST-FOD |
| `3_global_tractography/` | real crossing vs kissing; the Stage-3 loop |
| `mass_conservation/` | the independent brightness check on the same question |
| `noise_model/` | measure the noise so "real or noise?" gets a number |

Each has its own README with the ordered walkthrough. Files reach the root
engine via a `sys.path.insert` at the top, so any script runs from anywhere.

## Root engine

Shared by every folder, in dependency order. The count is how many tracked files
import it.

| File | Function | In → Out |
|---|---|---|
| `reconstruction.py` (24) | data gateway; `DEFAULT_VOLUME_DIR` / `VOXEL_SIZE_UM` live here | TIFF folder, `crop=(a,b)` → `vol (z,y,x)` |
| `orthogonality_residual.py` (8) | the plain structure tensor | `(vol, σ, ρ)` → `e3` field |
| `crossing_shape_3d.py` (16) | crossing/single phantoms, `shape_measures`, `onset_sigma`, `knee_sigma` | params → `vol`; `(vol,σ,ρ)` → c_l/c_p/c_s |
| `selfcalib_threshold.py` (5) | self-calibrating c_s guard, `sphericity_field` | `(vol,σ,ρ)` → c_s field |
| `two_orientation_test.py` (4) | `patch_gradients` primitive | `(patch, σ_grad)` → scatter `G` |
| `fod_dipy.py` (3) | `SPHERE` + dipy FOD cross-check | `patch` → dipy peaks |
| **`host_fod.py` (23)** | **the HOST-FOD kernel** | `analyse(patch, σ_grad, order=6)` → metrics `m`; `classify(m)` → label |
| `noise_floor.py` (10) | N-matrix / power-spectrum library | `(I, σ)` → `N`; `structure_tensor_field(subtract_N=)` |
| `host_geometry_test.py` (13) | labelled phantom suite | params → `vol` + truth |
| `kissing_phantom.py` (6) | `build_kissing(same_size, gap, curv)` | → `vol` |
| `model_ladder.py` (3) | c_s calibration against known geometries | lazy, `--validate` only |

`classify` returns `single` / `crossing` / `dispersed` / `disorganised` /
`complex`. `complex` means the method gave up — downstream filters selecting
`== "single"` correctly exclude it.

## Status

| Stage | Runs on any volume? | Notes |
|---|---|---|
| 1 σ, ρ | yes (edit `reconstruction.py`) | ground-truth-free, certified |
| 2 HOST + gate | yes | gate self-calibrates per volume |
| 3 tractography | yes | mechanism-validated on phantoms |
| noise floor | needs even/odd half-maps | check `corr(A,B)` first |

**Caveats.** σ and ρ are not auto-threaded into every script — read them from
Stage 1 and set the `SIGMA`/`RHO` constants where relevant. The noise stages
assume A and B are registered. Voxel size is hardcoded per script; keep it
consistent. i74 is brain and has **no half-maps**, so every "real noise" result
borrows lung noise as a proxy — only the isotropic spectral shape is verified to
transfer, the anisotropy is not.

See `PIPELINE.md` for the runbook on a new volume, and `BAYES_TRACTOGRAPHY.md`
for the record of what was tried, what worked, and what was falsified.
