<div align="center">

<h1>Optimising the Structure Tensor Analysis of<br>White Matter Fibres in Synchrotron Tomography</h1>

Isabella Hezhao · Andrew Keenlyside

<i>Department of Medical Physics and Biomedical Engineering, University College London · 2026</i>

[![CI](https://github.com/1ssot0pe/HiP-CT_Structure_Tensor/actions/workflows/ci.yml/badge.svg)](https://github.com/1ssot0pe/HiP-CT_Structure_Tensor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24289711.svg)](https://doi.org/10.5281/zenodo.24289711)

[Pipeline walkthrough](docs/PIPELINE.md) | [Work log](docs/WORKLOG.md) | [Release & full outputs](https://github.com/1ssot0pe/HiP-CT_Structure_Tensor/releases) | [Citation](#citation)

</div>

A pipeline for reading white-matter fibre orientation from Hierarchical
Phase-Contrast Tomography (HiP-CT) volumes. The structure tensor gives one
orientation per voxel and has two scale parameters, the differentiation scale
**σ** and the integration scale **ρ**, which are usually set by hand. This
pipeline sets them from the data, checking the result against tractography.

## Key result

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/stage4_rho_sweep_dark.png">
    <img src="docs/figures/stage4_rho_sweep.png" width="760" alt="Median angular error against the integration scale ρ. The curve is U-shaped with its minimum of 20.2° at ρ = 3.79; Stage 1's pick, ρ = 1.41, sits on the steep left side near 30°.">
  </picture>
</p>

On a 48³-voxel chunk of the i74 brain volume (20 µm voxels), a plain structure
tensor best reproduces the tractography-refined field at **σ = 1.00,
ρ = 3.79**. That holds for every seed count from 1,000 to 5,000 and for both
MSE and MAE as the Stage 3 objective. It is wider than Stage 1's
data-driven pick of ρ = 1.41.

<sub>Line: the 5,000-seed run. Band: range over the five seed counts. Beyond
ρ ≈ 6 the 4ρ Gaussian window exceeds the chunk's half-width. Drawn from
[`results/seed_rho_curves.json`](results/seed_rho_curves.json) by
[`docs/figures/make_stage4_rho_sweep.py`](docs/figures/make_stage4_rho_sweep.py);
a vector PDF sits alongside.</sub>

**Caveats.** ρ is located only to the grid spacing (~20%). Smaller 24³ chunks
put the minimum between ρ = 1.8 and 3.1, so the optimum varies by region. About
20° of median angular error remains at the optimum. See
[4_optimising_parameters/README.md](4_optimising_parameters/README.md).

## Method

| Stage | Question | Output |
|---|---|---|
| **1 · Scale** <br> [`1_sigma_rho_selection/`](1_sigma_rho_selection/) | How wide a region should the tensor measure over? | σ, ρ per 32³ block (Lindeberg scale selection) |
| **2 · Gate** <br> [`2_host_classification/`](2_host_classification/) | Can one orientation be trusted here? | per-voxel field: `vec1` everywhere, `vec2` at resolved crossings (higher-order structure tensor) |
| **3 · Resolve** <br> [`3_global_tractography/`](3_global_tractography/) | Real crossing, or fibres just touching? | crossing/kissing verdict per junction; optimised tractogram; refined field |
| **4 · Optimise** <br> [`4_optimising_parameters/`](4_optimising_parameters/) | Which single (σ, ρ) reproduces the refined field? | best (σ, ρ) and its angular error |

Two supporting studies give independent evidence:
[`mass_conservation/`](mass_conservation/) (crossing vs kissing from
brightness alone) and [`noise_model/`](noise_model/) (imaging noise measured
from half-reconstructions). Inputs, outputs and the logic of each stage are in
[docs/PIPELINE.md](docs/PIPELINE.md).

<details>
<summary><b>Stage 2 variants</b>: per-patch (default) or per-voxel</summary>

<br>

| Variant | Entry point | Decides | Gate | Use for |
|---|---|---|---|---|
| **Standard** | `3_global_tractography/run_pipeline.py` | per patch (stride-6 grid) | c_s > FA | the validated end-to-end pipeline |
| **Per-voxel, redesigned** | `2_host_classification/run_dense_stage12.py` | every voxel, Gaussian window | λ3/λ2 | the best per-voxel decision on real data |
| **Per-voxel, patch-exact** | `2_host_classification/export_nifti.py` | every voxel, box window | c_s > FA | the standard pipeline's answer at full resolution, as MRtrix-style NIfTI (kissing phantom) |

The per-voxel variants cost more compute. The patch-exact variant was
contributed by Andrew Keenlyside. How the three differ is in
[2_host_classification/README.md](2_host_classification/README.md#part-d--per-voxel-variants).

</details>

## Quick start

1. Create an environment (Python 3.13) and install the requirements.

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

   If the repository is in a synced folder (OneDrive, Dropbox), create the
   environment outside it: syncing thousands of small files can corrupt an
   install. `requirements-dev.txt` adds notebooks, Atlas download tools and
   formatters.

2. Run the checks with a known answer. These need no data and are what CI runs.

   ```bash
   python 3_global_tractography/test_frame_convention.py
   python 3_global_tractography/run_pipeline.py --radius 16 --iters 1
   ```

3. Run on real data (see [Data](#data)): Stages 1–3, then the Stage 3
   optimisation loop and Stage 4.

   ```bash
   python 3_global_tractography/run_pipeline.py --i74

   python 3_global_tractography/descent_loop.py --radius 24 --seeds 5000 \
       --turn relax --refine-field --variant vec1 --outdir results/seed_5000
   python 4_optimising_parameters/sigma_rho_match.py --run results/seed_5000
   ```

On Windows, `.\run_all.ps1` runs the checks and the real-data step in order and
stops at the first failure (`-Quick -SkipI74` for a one-minute smoke test). It
expects a venv at `%USERPROFILE%\venvs\hipct-structure-tensor`; pass `-Python`
to use another. Outputs go to `results/`; only small JSON summaries are tracked
in git, and the full outputs of the recorded runs are attached to the
[release](https://github.com/1ssot0pe/HiP-CT_Structure_Tensor/releases).

## Data

Real-data steps use HiP-CT volumes from the
[Human Organ Atlas](https://human-organ-atlas.esrf.fr/); no data is
redistributed here. Everything else runs on synthetic phantoms with known
ground truth.

| Volume | Used by | Set with |
|---|---|---|
| i74 brain, sensitivity chunk (`*.tif` slices, 20 µm) | Stages 1–4, `mass_conservation/` | `HIPCT_VOLUME_DIR` (default `data/i74/`) |
| LADAF-2022-13 left lung, two half-reconstructions (16.547 µm) | `noise_model/` | `LUNG_HALF_A_DIR`, `LUNG_HALF_B_DIR` |

<details>
<summary><b>Using your own dataset</b></summary>

<br>

A volume is a folder of single-slice `*.tif` files, chosen with two
environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `HIPCT_VOLUME_DIR` | folder of `*.tif` slices | `data/i74` |
| `HIPCT_VOXEL_UM` | isotropic voxel size in µm (written into NIfTI and `.trk` headers) | `20` |

A relative `HIPCT_VOLUME_DIR` is resolved from the repository root. Keep each
dataset in its own folder under `data/` (ignored by git), or use an absolute
path for large volumes kept elsewhere.

```powershell
# PowerShell, this terminal only
$env:HIPCT_VOLUME_DIR = "data/lung_overview"
$env:HIPCT_VOXEL_UM   = "16.547"
python 3_global_tractography/run_pipeline.py --i74
```

```bash
# bash, this terminal only
export HIPCT_VOLUME_DIR=data/lung_overview HIPCT_VOXEL_UM=16.547
python 3_global_tractography/run_pipeline.py --i74
```

To make a setting permanent on Windows, run
`setx HIPCT_VOLUME_DIR "D:\scans\brain_01"` once and open a new terminal (VS
Code needs a full restart). Real-data scripts analyse a cube cut from the
volume (`--centre`, `--radius`; `load_volume` defaults to `crop=(200, 456)`),
so check it fits a smaller volume. The `--i74` flag means "real data" and reads
whatever `HIPCT_VOLUME_DIR` points at.

</details>

## Repository layout

```
core/                      shared kernels: volume loading, structure tensor,
                           HOST-FOD, phantoms, noise library
1_sigma_rho_selection/     Stage 1
2_host_classification/     Stage 2
3_global_tractography/     Stage 3 (archive/ holds superseded experiments)
4_optimising_parameters/   Stage 4
mass_conservation/         supporting study: brightness-based crossing test
noise_model/               supporting study: half-map noise measurement
notebooks/                 initial structure tensor exploration
docs/                      pipeline walkthrough, work log, figures
results/                   JSON summaries of recorded runs
tools/                     repository checks used by CI
```

Each folder's README walks through its scripts in order and records what each
experiment found, including approaches that were ruled out (collected in
[docs/archive/BAYES_TRACTOGRAPHY.md](docs/archive/BAYES_TRACTOGRAPHY.md)).

## Limitations

- Stages 1–3 are validated on phantoms with known truth. Real data has no
  ground truth, so real-data results are measurements, not verified answers.
- The Stage 4 optimum is consistent across seed counts and metrics on one chunk
  but varies between regions.
- The track-density target `λ1 − λ3` cannot tell one bright fibre from two
  dim crossing ones.
- Crossings of three or more fibres are labelled `complex` and not resolved.
- i74 has no half-reconstructions, so its noise model is borrowed from lung
  data; only the isotropic spectral shape is shown to transfer.

## Citation

If you use this code, please cite:

```bibtex
@software{hezhao2026structuretensor,
  author    = {Hezhao, Isabella and Keenlyside, Andrew},
  title     = {Optimising the Structure Tensor Analysis of White Matter
               Fibres in Synchrotron Tomography},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24289711},
  url       = {https://doi.org/10.5281/zenodo.24289711}
}
```

This DOI always resolves to the latest release; each release also has its own
(v1.0.0: [10.5281/zenodo.24289712](https://doi.org/10.5281/zenodo.24289712)).
GitHub's "Cite this repository" button gives other formats from
[CITATION.cff](CITATION.cff). If you use Human Organ Atlas data, also follow the
Atlas's citation guidance for those datasets.

## License

Code: [MIT](LICENSE). The licence covers this repository's code only, not the
HiP-CT data it analyses.
