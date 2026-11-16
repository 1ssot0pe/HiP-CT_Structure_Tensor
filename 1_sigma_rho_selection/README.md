# Sigma & rho selection

Picks two scales the structure tensor needs on i74: 
**sigma** (the differentiation scale — how wide a gradient to measure)
**rho** (the integration scale — how far to average the tensor before reading a direction).

Self-contained folder: files import each other flat
(`from pick_sigma_rho import ...`) and reach the shared kernels in `core/`
(`core.reconstruction`, `core.orthogonality_residual`) via a `sys.path.insert`
of the repository root at the top of each file. Run any script from anywhere.

# Terms 

- **Voxel** — one 3D pixel. On i74 each voxel is a 20 µm cube.
- **Structure tensor** — the tool that reads the local fibre direction at each
  voxel by looking at how brightness changes around that point. It has two
  tuning knobs: sigma and rho.
- **Sigma (σ)** — the *measuring* scale: how wide a region to look across when
  measuring the brightness change. Too small measures noise; too large blurs
  neighbouring fibres together. One "right" σ matches the fibre's own width.
- **Rho (ρ)** — the *averaging* scale: after measuring, how far to average
  before committing to a direction. Too small leaves the direction noisy; too
  large averages across genuinely different fibres.
- **Fibre direction (e3)** — the way a fibre points at a voxel. It comes out of
  the structure tensor as the direction along which brightness changes *least*
  (nothing changes as you move *along* a fibre).
- **Phantom** — a synthetic test image where we already know the true fibre
  widths and directions. Used to check a method before trusting it on real
  data, which has no answer key.
- **Lindeberg response** — a score computed at each σ that peaks when σ matches
  the fibre width. It is corrected (`σ² × mean of the two largest structure-
  tensor eigenvalues`) so it doesn't simply favour the smallest σ; the σ where
  it peaks is the fibre scale. Named after the scale-space theory it comes from.
- **Angular drift** — how much the recovered direction *changes* when you nudge
  ρ. A big change means ρ is too small (still jittery); a flat stretch (a
  "plateau") means the direction has settled. The ρ at the plateau is the pick.
- **FSC (Fourier Shell Correlation)** — a way to measure how fine the *real*
  detail goes without an answer key: build the same picture twice with
  different random noise, then check from coarse to fine how far the two still
  agree. Real structure agrees to fine detail; noise stops agreeing early.
- **Discretisation floor** — the smallest σ that still makes sense on a pixel
  grid (~0.85 voxels here). Below it the gradient maths breaks down, so any
  pick under it is rejected on principle.

# The pipeline, in order

1. **Prove it in 2D** — `toy2d/`. Test the two rules (Lindeberg response → σ,
   angular drift → ρ) on a 2D phantom with known directions. They track the
   truth; a naive alternative didn't. Now a **standalone experiment**: it used
   to be dragged into every pipeline run because `core/crossing_shape_3d.py`
   imported `coherence_sigma.knee_sigma`. That function is copied into
   `crossing_shape_3d.py`, so nothing outside this folder imports `toy2d/`.

2. **The 3D core** — `pick_sigma_rho.py`. What every later step calls.
   **σ** = the *first* peak of the Lindeberg response (first, not highest, so a
   slow background gradient can't pose as the fibre scale). **ρ** = where
   angular drift bottoms out.

3. **Check σ on clean phantoms** — `lindeberg_sigma_phantoms.py`. On
   known-width tubes the peak lands at the right width. Also gives
   `refined_peak_sigma` (reads the peak *between* grid points), used by steps
   4-6. *Caveat:* solid, sharp-walled cylinders report their **wall** scale,
   not their diameter.

4. **One (σ, ρ) for a real box** — `sigma_rho_global.py`. Runs the rules on one
   i74 box, checks the floor, reports **ACCEPTED / FLAGGED / REJECTED**.
   *FSC was tried as a second opinion and dropped:* it scores "sharpest", but
   sharpest is won by over-smoothing, so on a known-answer phantom it chose the
   wrong (σ, ρ). Now it's **Lindeberg + floor only**.

5. **A different (σ, ρ) per region** — `sigma_rho_adaptive.py`
   **(current best)**. One global pick drifts (best σ runs ~1-6 across the
   volume), so this picks per block and stitches a direction map, each block at
   its own scale.

6. **Stress-tests** — `sigma_rho_robustness.py`. (a) *Noise adaptivity* — an
   earlier "σ ignores noise" scare was a broken test (it froze ρ); with ρ free
   there's no bug. (b) *Region drift* — the finding behind step 5.
   (c) *High-pass* — a peak-finder sanity check.

7. **A scale per voxel** — `lindeberg_feature_size.py` (map) +
   `render_feature_size_map.py` (figure). The voxel-continuous version of step
   5: a Lindeberg scale pick at every voxel, giving a feature-size heat map in
   µm. Three settings were each necessary: the *first* interior maximum (not the
   global one), a σ grid reaching down to 0.5, and pooling the response at 2σ
   before peak-finding. On a 256³ i74 patch: 94% of tissue certified, median
   feature size 57 µm (σ\* ≈ 0.70). Calibrated on known-width tubes
   (`--calibrate`, σ\* = 0.578 × width, R² 0.987); re-calibrate after changing
   the σ grid or pooling. An experiment, not a pipeline step: step 5's block
   map is still what Stage 2 uses.

8. **Does the choice of ρ matter?** — `rho_range_sensitivity.py`. On a 96³
   i74 core, e3 at ρ = 2.0 and ρ = 3.5 (the credible range at the time) differ
   by a median **15.5°**, and the ρ ladder from 0.85 to 5.0 never plateaus. So
   ρ, not σ, is the scale that moves the field. This is what Stage 4 picks up.
   Recorded output: `rho_range_sensitivity_results.json`. *The script cannot
   currently run:* it imports helpers from `adaptive_sigma_benefit.py`, which
   was never committed and has been lost.

# How to run

From the repository root, with the environment from the [main README](../README.md#setup) active, e.g. `python -u 1_sigma_rho_selection/sigma_rho_global.py`.

| Script | Needs | Output |
|---|---|---|
| `lindeberg_sigma_phantoms.py` | nothing (synthetic) | `lindeberg_sigma_phantoms.png` |
| `sigma_rho_global.py` | i74 volume | `sigma_rho_global.png` |
| `sigma_rho_adaptive.py` | i74 volume | `sigma_rho_adaptive_map.png` |
| `sigma_rho_robustness.py` | i74 volume + `lung_A.npy` / `lung_B.npy` | 3 PNGs |
| `lindeberg_feature_size.py` | i74 volume (`--calibrate`: nothing) | per-voxel σ\* map + `_meta.json`; calibration in `lindeberg_feature_size_calib.json` |
| `render_feature_size_map.py` | output of the above | feature-size heat map PNG |
| `rho_range_sensitivity.py` | i74 volume + the lost `adaptive_sigma_benefit.py` | `rho_range_sensitivity_results.json` (recorded) |

# Downstream consumers outside this folder

`2_host_classification/adaptive_triage.py` uses `pick_sigma_rho` for a
*different* job — a per-patch scale for HOST-FOD, not the volume-wide pick here.
`2_host_classification/sigma_rho_gated_field.py` uses
`sigma_rho_adaptive.compute_adaptive_map` — that is the live Stage 1 → Stage 2
link. Both reach in via `sys.path.insert`.

Nothing outside this folder imports `toy2d/` any more.
