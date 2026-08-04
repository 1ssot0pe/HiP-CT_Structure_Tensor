# The pipeline, step by step

What runs, in order, with the file, input and output of each step. The file
lists below are **measured**, not assumed: they are the modules Python actually
loads during a run.

```
$PY 3_global_tractography/run_pipeline.py            # phantoms, known truth
$PY 3_global_tractography/run_pipeline.py --i74      # real volume
```

`$PY` is the venv interpreter — see § 0. Results go to `results/` at the repo
root. The phantom form exits non-zero if any case fails.

---

## 0. Setup

### The venv lives OUTSIDE this repo

This folder is inside OneDrive. A venv is ~400 MB of small files; leaving it
here means OneDrive re-syncs it constantly, which is what corrupted the previous
in-repo `.venv` (an interrupted install left an environment with `numpy` but not
`scipy` or `dipy`).

```powershell
# PowerShell — one time
python -m venv $env:USERPROFILE\venvs\hipct-structure-tensor
$PY = "$env:USERPROFILE\venvs\hipct-structure-tensor\Scripts\python.exe"
& $PY -m pip install -r requirements.txt
```

```bash
# Git Bash — one time
python -m venv "$USERPROFILE/venvs/hipct-structure-tensor"
PY="$USERPROFILE/venvs/hipct-structure-tensor/Scripts/python.exe"
$PY -m pip install -r requirements.txt
```

`requirements.txt` is the 9 packages the code imports, derived from the import
statements. `requirements-dev.txt` adds the notebook, the Human Organ Atlas
download tooling and lint tools — install only if you need them.

### Point the code at your volume

`reconstruction.py`, top of file:

```python
DEFAULT_VOLUME_DIR = Path(r"C:\path\to\your\volume_folder")   # folder of *.tif slices
VOXEL_SIZE_UM      = 16.5
```

`load_volume(crop=(a, b))` carves a central cube `[a:b]` **on all three axes**.
That one edit feeds every step below.

---

## Step 0 — Get a volume

| | |
|---|---|
| **Question** | What are we measuring? |
| **File** | `reconstruction.py` — real data · `localized_phantoms.py` — synthetic |
| **In** | folder of TIFF slices, `crop` · or phantom geometry parameters |
| **Out** | `vol` — float32 array indexed `(z, y, x)` |

Phantoms come first because they have known truth. `raw_localized_crossing`
builds two fibre families that genuinely cross inside a bounded zone;
`raw_kissing` builds two curved bundles that touch and turn away. Locally these
look identical — that is the whole problem the pipeline exists to solve.

---

## Step 1 — Choose the measurement scales (σ, ρ)

| | |
|---|---|
| **Question** | How wide a region should we measure over? |
| **Files** | `1_sigma_rho_selection/sigma_rho_adaptive.py` → `pick_sigma_rho.py` → `orthogonality_residual.py` |
| **In** | `vol`, `block=32` |
| **Out** | `sigma_map`, `rho_map`, block edges `zs, ys, xs` |

The structure tensor has two knobs and both have a right answer set by the
tissue, not by taste:

- **σ**, the differentiation scale — too small measures noise, too large blurs
  neighbouring fibres together. Correct when it matches the fibre's own width.
  Found as the *first* peak of the Lindeberg response (first, not highest, so a
  slow background gradient cannot pose as the fibre scale).
- **ρ**, the integration scale — too small leaves the direction jittery, too
  large averages across genuinely different fibres. Found where angular drift
  flattens.

Picked **per 32³ block**, because the best σ varies ~1–6 across a real volume, so
one global value is wrong nearly everywhere.

> If you see `[stage 1] WARNING: only N% of blocks certified`, the run fell back
> to a single fixed σ and this step contributed nothing.

---

## Step 2 — Decide where one direction is enough

| | |
|---|---|
| **Question** | Can the cheap one-direction answer be trusted here? |
| **Files** | `2_host_classification/sigma_rho_gated_field.py` → `host_fod.py` → `two_orientation_test.py`, `crossing_shape_3d.py` |
| **In** | `vol`, the (σ, ρ) maps, patch half-width 14, `stride` |
| **Out** | `GatedField` → `results/unified_field_*.npz` |

A structure tensor returns one direction per voxel. Where two fibres cross,
there is no single correct direction, so the answer is not merely imprecise —
it is meaningless. This step finds those places and spends the expensive method
only there.

Per patch:

1. `host_fod.analyse(patch, σ_grad=4.0)` → **c_s** (sphericity, "no dominant
   direction") and **FA** (anisotropy, "strongly directional").
2. **c_s ≤ FA** → single fibre. Keep Step 1's `e3`.
   Otherwise → `analyse(patch, σ_grad=2.5)` → up to 2 directions, `classify()`
   → `single` / `crossing` / `dispersed` / `disorganised` / `complex`.

Both quantities come from the same three eigenvalues of the same patch, so the
rule is local and has **no threshold to tune**. It replaced a fixed cutoff that
called only ~3% of real tissue single, implausibly few; this gives ~29%.

Three different σ are in play and they are not interchangeable — at Stage 1's
σ (~1.0) the gate never fires at all, and at 4.0 the second peak that justified
gating is already smoothed away:

| σ | job |
|---|---|
| ~0.85–1.0 (Step 1, per block) | the single-fibre direction |
| 4.0 | the routing decision |
| 2.5 | extracting peak directions |

---

## Step 3 — Resolve which flagged crossings are real

| | |
|---|---|
| **Question** | Do two fibres genuinely cross here, or merely touch? |
| **Files** | `3_global_tractography/self_optimising_tractography.py` → `tract_density.py` |
| **In** | `GatedField`, `vol` |
| **Out** | per-junction S + verdict, fitted θ, `results/pipeline_results.json` |

A single voxel cannot answer this: crossing and kissing look the same up close.
You have to follow the fibres and see whether both continue. Two independent
signals are used:

**Superposition (local, physical).** At a real crossing both fibres deposit
intensity at the same point, so it is roughly their sum. At a kissing point
neither centreline passes through, so it can never exceed about one fibre.
`superposition_S` reads that ratio; `S ≳ 0.75` → CROSSING, low → TANGENCY.

**Global consistency.** Streamlines are integrated through the field and
rasterised into a reconstructed density `D_rec` (`tract_density.rasterise`,
trilinear splatting so density 2 literally means two fibres). This is compared
against `D_obs`, taken from the image's own anisotropy `λ1 − λ3`. Coordinate
descent over θ = (step, angle threshold, seed density, link weight) minimises
`L = mse(D_rec, D_obs) + verdict penalty`.

> **The join.** `host_fod` returns directions as `(x, y, z)`; Step 3 indexes
> positions as `(z, y, x)`. `field_from_gated` reverses every direction. Getting
> this wrong does not crash — every streamline just walks a mirrored path.
> `test_frame_convention.py` guards it.

---

## What comes out

| Output | Meaning |
|---|---|
| `results/unified_field_*.npz` | the orientation field: per patch, its directions, label, c_s, FA, σ, ρ |
| `results/pipeline_results.json` | per-junction S and verdict, fitted θ, loss trace |
| stdout | `PASS` / `FAIL` per phantom; on real data, verdict with no correctness claim |

On phantoms the crossing scores S ≈ 1.0 → CROSSING, and the kissing pair
produces no junction at all — the gate never routes it, so no crossing is
invented. Both are passes.

On real i74 every scored junction comes back TANGENCY (S median ≈ 0.44),
consistent with the independent tractography and mass-conservation findings that
HOST over-calls crossings on this data.

---

## Optional: measuring the noise

Needs **two half-reconstructions** (even/odd projection subsets) of one scan.
Their difference `n = (A − B)/√2` cancels the shared signal and leaves pure
noise — the only clean way to see noise on its own.

| Step | Command | Result |
|---|---|---|
| Cache the slabs | `$PY noise_model/build_lung_slab.py` | `lung_A/B.npy` (gitignored) |
| Measure | `$PY noise_model/half_map_analysis.py --a EVEN --b ODD --vox 16.5 --crop 256` | noise is **correlated** (slope ≈ −3.5, not white's −5) and **anisotropic** (λ1/λ3 ≈ 1.5–1.9) |
| Correct | `$PY noise_model/run_lung_subtractN.py` | subtracts the measured N from the tensor |
| Decide | `$PY 3_global_tractography/phase3_generative.py` | noise-whitened likelihood; selects the true geometry |

Check the printed `corr(A, B)` first — if the two halves are not well
registered, nothing downstream means anything.

---

## Caveats

- **σ and ρ are not auto-threaded.** Steps 1–3 share them; standalone scripts
  have their own `SIGMA` / `RHO` constants you must set to match.
- **Voxel size is per-script** (`VOXEL_SIZE_UM` / `--vox` / `VOX_UM`).
- **i74 has no half-maps.** It is brain; every "real noise" result borrows lung
  noise as a proxy. Only the isotropic spectral shape is verified to transfer —
  the anisotropy is not.
- **`D_obs = λ1 − λ3` is a proxy, not truth.** It cannot distinguish one bright
  fibre from two dim crossing ones.
- **The generative likelihood is validated as a model selector**, but is not yet
  inside the tractography link-scoring. That is the one open build.
