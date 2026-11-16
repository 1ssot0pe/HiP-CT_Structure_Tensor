# Optimising the Structure Tensor Analysis of White Matter Fibres in Synchrotron Tomography

[![CI](https://github.com/1ssot0pe/Optimising-the-Structure-Tensor-Analysis-of-White-Matter-Fibres-in-Synchrotron-Tomography/actions/workflows/ci.yml/badge.svg)](https://github.com/1ssot0pe/Optimising-the-Structure-Tensor-Analysis-of-White-Matter-Fibres-in-Synchrotron-Tomography/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A pipeline for reading white-matter fibre orientation from Hierarchical
Phase-Contrast Tomography (HiP-CT) volumes, built during a UCL Medical Physics
and Biomedical Engineering summer project (2026).

The structure tensor gives one orientation per voxel and has two scale
parameters: the differentiation scale **σ** and the integration scale **ρ**.
Both are usually set by hand. This pipeline:

1. chooses σ and ρ from the data,
2. finds where one orientation per voxel is not enough (fibre crossings) and
   resolves up to two orientations there,
3. checks those orientations with tractography, telling real crossings from
   fibres that only touch, and refines the orientation field, and
4. finds the single (σ, ρ) whose plain structure tensor best reproduces the
   refined field.

## Key result

On a 48³-voxel chunk of the i74 brain volume (20 µm voxels), the refined field
is best reproduced at **σ = 1.00, ρ = 3.79**. This holds for every seed count
from 1,000 to 5,000 and for both MSE and MAE as the Stage 3 objective. Stage 1's
data-driven pick on the same chunk was ρ = 1.41.

![Median angular error against ρ for five seed counts; every curve has its minimum at ρ = 3.79](docs/figures/stage4_rho_sweep.png)

Caveats: ρ is located only to the grid spacing (~20%). Smaller 24³ chunks put
the minimum between ρ = 1.8 and 3.1, so the optimum varies by region. About 20°
of median angular error remains at the optimum. Details are in
[4_optimising_parameters/README.md](4_optimising_parameters/README.md).

## The four stages

| Stage | Question | Folder | Output |
|---|---|---|---|
| **1** Scale | How wide a region should the tensor measure over? | [`1_sigma_rho_selection/`](1_sigma_rho_selection/) | σ, ρ per 32³ block (Lindeberg scale selection) |
| **2** Gate | Can one orientation be trusted here? | [`2_host_classification/`](2_host_classification/) | per-voxel field: `vec1` everywhere, `vec2` at resolved crossings (higher-order structure tensor) |
| **3** Resolve | Real crossing, or fibres just touching? Which tracking parameters explain the image? | [`3_global_tractography/`](3_global_tractography/) | crossing/kissing verdict per junction; optimised tractogram; refined field |
| **4** Optimise | Which single (σ, ρ) reproduces the refined field? | [`4_optimising_parameters/`](4_optimising_parameters/) | best (σ, ρ) and its angular error |

Two supporting studies give independent evidence:
[`mass_conservation/`](mass_conservation/) (crossing vs kissing from
brightness alone) and [`noise_model/`](noise_model/) (the imaging noise,
measured from half-reconstructions).

The step-by-step walkthrough, with inputs, outputs and the logic flow of each
stage, is in [docs/PIPELINE.md](docs/PIPELINE.md).

## Setup

Python 3.13. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements-dev.txt` adds the notebook kernel, the Human Organ Atlas download
tools and the formatters; it is only needed for those.

> If the repository sits in a synced folder (OneDrive, Dropbox), create the
> virtual environment outside it. Syncing thousands of small files can corrupt
> an environment mid-install. `run_all.ps1` expects one at
> `%USERPROFILE%\venvs\hipct-structure-tensor` by default; pass `-Python` to
> use another.

## Data

The real-data steps use HiP-CT volumes from the
[Human Organ Atlas](https://human-organ-atlas.esrf.fr/). No data is
redistributed here.

| Volume | Used by | Point the code at it with |
|---|---|---|
| i74 brain, sensitivity chunk (folder of `*.tif` slices, 20 µm) | Stages 1–4, `mass_conservation/` | `HIPCT_VOLUME_DIR` (default `data/i74/`) |
| LADAF-2022-13 left lung, two half-reconstructions (16.547 µm) | `noise_model/` | `LUNG_HALF_A_DIR`, `LUNG_HALF_B_DIR` |

Everything else runs on synthetic phantoms with known ground truth and needs no
download.

## Running

```bash
# Checks with a known answer -- no data needed (this is what CI runs)
python 3_global_tractography/test_frame_convention.py
python 3_global_tractography/run_pipeline.py --radius 16 --iters 1

# Stages 1-3 on real data
python 3_global_tractography/run_pipeline.py --i74

# Stage 3 optimisation loop, then Stage 4
python 3_global_tractography/descent_loop.py --radius 24 --seeds 5000 \
    --turn relax --refine-field --variant vec1 --outdir results/seed_5000
python 4_optimising_parameters/sigma_rho_match.py --run results/seed_5000
```

On Windows, `.\run_all.ps1` runs the checks and the real-data step in order and
stops at the first failure (`-Quick -SkipI74` for a one-minute smoke test).

Outputs go to `results/`. Only small JSON summaries are tracked in git. The
full outputs of the recorded runs (tractograms, fields, interactive viewers) are
attached to the GitHub release.

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
docs/                      pipeline walkthrough, work log, report draft, figures
results/                   JSON summaries of recorded runs
tools/                     repository checks used by CI
```

Each folder's README walks through its scripts in order and records what each
experiment found, including approaches that were tried and ruled out.

## Status and limitations

- Stages 1–3 are validated on phantoms with known truth. On real data there is
  no ground truth, so results are reported as measurements, not as correct
  answers.
- The Stage 4 optimum is consistent across seed counts and metrics on one chunk
  but varies between regions (see above).
- The track-density target `λ1 − λ3` cannot tell one bright fibre from two
  dim crossing ones.
- Crossings of three or more fibres are labelled `complex` and not resolved.
- i74 has no half-reconstructions, so its noise model is borrowed from lung
  data; only the isotropic spectral shape is shown to transfer.

The work log is in [docs/WORKLOG.md](docs/WORKLOG.md). Approaches that were
tried and ruled out are recorded in
[docs/archive/BAYES_TRACTOGRAPHY.md](docs/archive/BAYES_TRACTOGRAPHY.md).

## Citation

If you use this code, please cite it using [CITATION.cff](CITATION.cff) (GitHub
shows a "Cite this repository" button). If you use Human Organ Atlas data,
also follow the Atlas's own citation guidance for the datasets you use.

## Licence

Code: [MIT](LICENSE). The licence covers this repository's code only, not the
HiP-CT data it analyses.
