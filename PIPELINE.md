# Running the pipeline on a new volume

A practical runbook. Stages A–C are wired together into one driver; the noise
stages (D–F) need **two half-reconstructions** (even/odd projection subsets) of
the same scan.

All commands below are written as `$PY <script>`, where `$PY` is the Python
inside the project venv. Set it once per shell — see § 0.

---

## 0. Requirements

### The venv lives OUTSIDE this repo

This folder is inside OneDrive. A venv is ~400 MB of many small files, and
leaving it here means OneDrive re-syncs it constantly — which is what corrupted
the previous in-repo `.venv` (a `pip install` was interrupted partway through
the large binary wheels, leaving a half-populated environment that imported
`numpy` but not `scipy`, `dipy` or `matplotlib`).

Create it somewhere unsynced. The current one is at
`%USERPROFILE%\venvs\hipct-structure-tensor`:

```powershell
# PowerShell — one-time setup
python -m venv $env:USERPROFILE\venvs\hipct-structure-tensor
$PY = "$env:USERPROFILE\venvs\hipct-structure-tensor\Scripts\python.exe"
& $PY -m pip install -r requirements.txt
```

```bash
# Git Bash — one-time setup
python -m venv "$USERPROFILE/venvs/hipct-structure-tensor"
PY="$USERPROFILE/venvs/hipct-structure-tensor/Scripts/python.exe"
$PY -m pip install -r requirements.txt
```

Then in each new shell, just re-set `$PY` to that path.

### Dependencies

`requirements.txt` is the **9 packages the pipeline actually imports** (numpy,
scipy, matplotlib, tifffile, structure-tensor, dipy, nibabel, tqdm,
scikit-image), derived from the import statements — not a `pip freeze` dump.
It installs in one step and pulls ~35 packages with transitive deps.

`requirements-dev.txt` holds the other 176 from the original environment, needed
only for the Jupyter notebook, the Human Organ Atlas download tooling
(`hoa-tools`, `gcsfs`, `zarr`, `dask`), and lint/type tools. Install it only if
you want those:

```
$PY -m pip install -r requirements-dev.txt
```

### Data

- Your volume as a **folder of single-slice TIFFs** (sorted by filename), or a
  multipage `.tif` / `.npy` / NIfTI.
- Voxel size in µm.

---

## 1. Point the code at your volume (the one required edit)

Almost every stage loads the volume through `reconstruction.load_volume()`,
which defaults to a hardcoded path. Edit **`reconstruction.py`** (top of file):

```python
DEFAULT_VOLUME_DIR = Path(r"C:\path\to\your\volume_folder")   # folder of *.tif slices
VOXEL_SIZE_UM      = 16.5                                      # your voxel size (µm)
```

`load_volume(folder, crop=(a, b))` carves a central cube `[a:b]` **on all three
axes** (RAM-bound — the full stack can be very large). Use a crop for a first
pass, `crop=None` for the whole stack.

That single edit propagates to stages A–C. Stages D–F take their inputs
explicitly.

---

## 2. Stages

### A–C. The end-to-end run

Stages 1, 2 and 3 now share data — one command runs all three:

```
$PY 3_global_tractography/run_pipeline.py            # phantoms first
$PY 3_global_tractography/run_pipeline.py --i74 --radius 48
```

The phantom form validates against known truth (a localised crossing → truth
CROSSING, a kissing pair → truth TANGENCY) and should be run before trusting the
real form. Flags: `--radius` (half-width of the cube), `--stride` (patch grid
spacing), `--block` (Stage-1 block size), `--iters` (Stage-3 iterations),
`--centre z y x`.

Outputs `unified_field_*.npz` (the orientation-field artifact),
`pipeline_results.json` / `pipeline_i74_results.json`, and per-junction
S / verdict lines. Smaller `--radius`, larger `--stride` = faster, coarser.

### Running a stage on its own

| | |
|---|---|
| **A** σ, ρ only | `1_sigma_rho_selection/sigma_rho_global.py` (one box → ACCEPTED / FLAGGED / REJECTED) or `sigma_rho_adaptive.py` (per-block map, current best) |
| **B** per-patch labels | `2_host_classification/visualize_i74_orientations.py` for the glyph overlay. API: `host_fod.analyse(patch)` → peaks; `host_fod.classify(m)` → label |
| **C** tractography alone | `3_global_tractography/phase3_i74.py --crop 96 --stride 4` |

> **i74-specific assumption:** `visualize_i74_orientations.py` loads via
> `selfcalib_threshold.load_interior()`, which reads the volume as **uint8** and
> carves a fixed interior box. For a **float32** reconstruction (e.g. the lung),
> casting to uint8 would destroy it — load your volume as float and run
> `adaptive_triage.triage_patch_v2(patch)` directly over your own patch grid
> instead. `host_fod.analyse/classify` are dtype-agnostic.

---

### Noise stages (need even/odd half-reconstructions)

You need **two folders**: the volume reconstructed from even-angle projections
(A) and from odd-angle projections (B). The signal is shared; `n = (A − B)/√2`
is pure noise.

### D. Measure the noise floor

General entry — works directly on two folders:

```
$PY noise_model/half_map_analysis.py --a PATH_EVEN --b PATH_ODD --vox 16.5 --crop 256
```

Outputs `halfmap_T1..T4_*.png` + `HALFMAP_RESULTS.md`: T1 = noise character
(white vs correlated, anisotropy), T2 = does `c_s` complexity survive the
measured noise, T4 = resolution. `--synthetic` runs a self-test with known
answers.

> **Large, untiled TIFFs** (e.g. 8000² float32): reading full slices is costly.
> Use the caching path instead — set the two folder paths at the top of
> **`noise_model/build_lung_slab.py`** and pick an output dir, then:
> ```
> LUNG_SLAB_DIR=/scratch $PY noise_model/build_lung_slab.py     # caches lung_A/B.npy
> LUNG_SLAB_DIR=/scratch $PY noise_model/run_lung_halfmap.py    # T1–T4 on a tissue block
> LUNG_SLAB_DIR=/scratch $PY noise_model/run_lung_multiblock.py # robustness over blocks
> ```
> It prints `corr(A,B)` — this must be high (well-registered) before trusting
> anything downstream.

### E. Noise correction (subtract the measured N)

```
LUNG_SLAB_DIR=/scratch $PY noise_model/run_lung_subtractN.py
```

Subtracts the measured **anisotropic** N from the structure tensor before
re-reading shape, and reports the crossing/complexity fractions uncorrected vs
corrected (and vs a plain white floor, to show the anisotropy matters).

### F. Calibrated decision (noise-whitened likelihood)

```
LUNG_SLAB_DIR=/scratch $PY 3_global_tractography/phase3_generative.py
```

Scores each geometry by a generative likelihood whitened by the measured noise
spectrum: selects the true fibre configuration and reports a confidence that
shrinks as noise rises. **Status:** validated as a model-*selector*; not yet
wired inside the tractography link-scoring (the one open build).

---

## 3. Minimal recipe

```
# once: edit reconstruction.py -> your folder + voxel size
$PY 3_global_tractography/run_pipeline.py           # validate on phantoms
$PY 3_global_tractography/run_pipeline.py --i74     # then the real volume

# if you have even/odd half-maps:
$PY noise_model/half_map_analysis.py --a EVEN --b ODD --vox 16.5 --crop 256
# (large untiled TIFFs -> use build_lung_slab.py + run_lung_halfmap.py instead)
```

---

## 4. Status of each stage

| Stage | Runs on any volume? | Notes |
|---|---|---|
| A σ,ρ | ✅ (edit reconstruction.py) | ground-truth-free, certified |
| B HOST + gate | ✅ | gate threshold self-calibrates per volume |
| C tractography | ✅ | mechanism-validated on phantoms |
| D noise floor | needs even/odd half-maps | general CLI; check `corr(A,B)` first |
| E N-subtraction | needs the measured N (D) | uses the cached slab |
| F calibrated decision | needs the measured N (D) | model-selector; not yet inside inference |

**Caveats.** σ, ρ are not auto-threaded into every downstream script — read them from A
and set the `SIGMA`/`RHO` constants where relevant. The noise stages assume A and B are
registered (same reconstruction geometry, different projection subsets). Voxel size is
hardcoded per script (`VOXEL_SIZE_UM` / `--vox` / `VOX_UM`); keep it consistent.
