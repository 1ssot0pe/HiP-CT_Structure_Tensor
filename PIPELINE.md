# Running the pipeline on a new volume

A practical runbook. This is a **research pipeline** — a sequence of scripts, not one
CLI. Stages A–C run on any volume after one config edit; the noise stages (D–F) need
**two half-reconstructions** (even/odd projection subsets) of the same scan.

All commands use the project venv:

```
.venv/Scripts/python.exe <script> …      # Windows
```

---

## 0. Requirements

- The `.venv` in this repo (numpy, scipy, tifffile, matplotlib, dipy, nibabel).
- Your volume as a **folder of single-slice TIFFs** (sorted by filename), or a
  multipage `.tif` / `.npy` / NIfTI.
- Voxel size in µm.

---

## 1. Point the code at your volume (the one required edit)

Almost every stage loads the volume through `reconstruction.load_volume()`, which
defaults to a hardcoded path. Edit **`reconstruction.py`** (top of file):

```python
DEFAULT_VOLUME_DIR = Path(r"C:\path\to\your\volume_folder")   # folder of *.tif slices
VOXEL_SIZE_UM      = 16.5                                      # your voxel size (µm)
```

`load_volume(folder, crop=(a, b))` carves a central cube `[a:b]` on all three axes
(RAM-bound — the full stack can be very large). Use a crop for a first pass, `crop=None`
for the whole stack.

That single edit propagates to stages A, B, C. Stages D–F take their inputs explicitly
(see below).

---

## 2. Stages

### A. Pick the structure-tensor scales (σ, ρ)

```
.venv/Scripts/python.exe sigma_rho_final.py
```

Prints a recommended **σ** (gradient scale) and **ρ** (integration scale) in voxels and
µm, from the Lindeberg criterion cross-checked by FSC sharpness and gradient
orthogonality. **Note the recommended σ, ρ** — later stages have them as constants you
may want to match (e.g. `SIGMA, RHO` in the analysis scripts).

### B. Per-voxel orientation + HOST gate (single / crossing / complex)

The cheap structure tensor gives one direction per voxel; HOST-FOD is run only where
that model fails. The per-patch classifier is `adaptive_triage.triage_patch_v2(patch)`
(local Lindeberg → HOST-FOD → label), gated by the self-calibrating `c_s` threshold
(`selfcalib_threshold.py`). To render the labelled orientation overlay on a slice:

```
.venv/Scripts/python.exe visualize_i74_orientations.py
```

API for your own driver: `host_fod.analyse(patch)` → dict with peaks; `host_fod.classify(m)`
→ `"single" | "crossing" | "complex" | "disorganised"`.

> **i74-specific assumption:** `visualize_i74_orientations.py` loads via
> `selfcalib_threshold.load_interior()`, which reads the volume as **uint8** and carves a
> fixed interior box. For a **float32** reconstruction (e.g. the lung), casting to uint8
> would destroy it — load your volume as float and run `triage_patch_v2(patch)` directly
> over your own patch grid instead. `host_fod.analyse/classify` are dtype-agnostic.

### C. Global tractography (resolve real crossings vs touching fibres)

Seeds segments from the HOST orientations, links collinear neighbours, assembles tracts:

```
.venv/Scripts/python.exe phase3_i74.py --crop 96 --stride 4
```

Outputs `phase3_i74.png` (z-projection + tracts) and a printed confirmed/demoted count
for the HOST-flagged crossings. Smaller `--crop` / larger `--stride` = faster, coarser.

---

### Noise stages (need even/odd half-reconstructions)

You need **two folders**: the volume reconstructed from even-angle projections (A) and
from odd-angle projections (B). The signal is shared; `n = (A − B)/√2` is pure noise.

### D. Measure the noise floor (T1–T4)

General entry — works directly on two folders:

```
.venv/Scripts/python.exe half_map_analysis.py --a PATH_EVEN --b PATH_ODD --vox 16.5 --crop 256
```

Outputs `halfmap_T1..T4_*.png` + `HALFMAP_RESULTS.md`: T1 = noise character (white vs
correlated, anisotropy), T2 = does `c_s` complexity survive the measured noise, T4 =
resolution. `--synthetic` runs a self-test with known answers.

> **Large, untiled TIFFs** (e.g. 8000² float32): reading full slices is costly. Use the
> caching path instead — set the two folder paths at the top of **`build_lung_slab.py`**
> and pick an output dir, then:
> ```
> LUNG_SLAB_DIR=/scratch .venv/Scripts/python.exe build_lung_slab.py     # caches lung_A/B.npy
> LUNG_SLAB_DIR=/scratch .venv/Scripts/python.exe run_lung_halfmap.py    # T1–T4 on a tissue block
> LUNG_SLAB_DIR=/scratch .venv/Scripts/python.exe run_lung_multiblock.py # robustness over blocks
> ```
> It prints `corr(A,B)` — this must be high (well-registered) before trusting anything.

### E. Noise correction (subtract the measured N)

```
LUNG_SLAB_DIR=/scratch .venv/Scripts/python.exe run_lung_subtractN.py
```

Subtracts the measured **anisotropic** N from the structure tensor before re-reading
shape, and reports the crossing/complexity fractions uncorrected vs corrected (and vs a
plain white floor, to show the anisotropy matters).

### F. Calibrated decision (noise-whitened likelihood)

```
LUNG_SLAB_DIR=/scratch .venv/Scripts/python.exe phase3_generative.py
```

Scores each geometry by a generative likelihood whitened by the measured noise spectrum:
selects the true fibre configuration and reports a confidence that shrinks as noise
rises. **Status:** validated as a model-*selector*; not yet wired inside the
tractography link-scoring (the one open build).

---

## 3. Minimal recipe

```
# once: edit reconstruction.py -> your folder + voxel size
.venv/Scripts/python.exe sigma_rho_final.py                 # A: (σ, ρ)
.venv/Scripts/python.exe visualize_i74_orientations.py      # B: per-voxel labels
.venv/Scripts/python.exe phase3_i74.py --crop 96 --stride 4 # C: tracts

# if you have even/odd half-maps:
.venv/Scripts/python.exe half_map_analysis.py --a EVEN --b ODD --vox 16.5 --crop 256   # D
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
