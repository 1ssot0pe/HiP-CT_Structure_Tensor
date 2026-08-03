# σ, ρ selection for structure-tensor fibre analysis on HiP-CT i74

**Deliverable (Goal A).** A defensible, ground-truth-free recommendation for the
structure-tensor smoothing scale σ and integration scale ρ on the i74 volume
(20 µm/voxel), plus an honest account of how far it can be *certified*.

## Recommendation

| Parameter | Value | Basis |
|---|---|---|
| **σ** | **≈ 1.0 voxel (~20 µm)** | Lindeberg γ-normalised response — clean single interior peak at σ≈0.9; at the discretisation floor |
| **ρ** | **≈ 2–3.5 voxels** | Lindeberg ρ-stability plateau → 3.5; the (failed) FSC optimum → 2.0. Loosely pinned. |

Hard lower bound: **σ ≥ 0.85 voxel** — below this the discrete Gaussian
derivative stops being a faithful derivative (noise_floor.py T1, measured).

## Evidence

- **Lindeberg (primary, phantom-validated).** The γ-normalised response
  `σ²·mean(plane energy)` peaks at the fibre's own scale (validated: σ ≈ 0.87×width
  on clean tubes). On real i74 it gives a **sharp single interior peak at σ≈0.9**
  (not a ramp), and the ρ-stability plateau gives ρ≈3.5. This method *escapes*
  the over-smoothing trap below because γ-normalisation is a scale-SELECTION
  principle, not a smoothness reward.

- **Discretisation floor (hard bound).** σ ≥ 0.85 vox. The Lindeberg pick sits
  right at it.

## Key finding: the analysis is resolution-limited

The optimal σ is pinned at the **finest resolvable scale**. There is no "sweet
spot" comfortably above the floor because the fibre texture is ~1 voxel — at
20 µm/voxel the **imaging resolution, not the choice of σ, is the binding
constraint**. A corollary caveat: at σ≈1 we cannot cleanly separate genuine
finest-scale fibre structure from high-frequency content, which ties back to the
noise/resolution questions.

## Why there is no independent unsupervised certifier (yet)

We tried to corroborate the Lindeberg pick with an independent *output-quality*
metric. Three principled candidates were built and run through the same phantom
trust-gate (does the metric's optimum coincide with the supervised
angular-error optimum?). All three fail, for one root cause:

| Metric | Trust-gate result | Why it fails |
|---|---|---|
| FSC of noise-LIC reconstructions | corr −0.10; optimum at large σ,ρ | scores field *coherence* → rewards smoothness → over-smooths |
| Cross-scale orthogonality residual | corr −0.88; optimum at large σ | same over-smoothing trap |
| Tensor-field reproducibility (noise-resampling) | corr **−0.90** (right sign); optimum at small σ | correct *structure*, but its optimum is set by the resampling **noise level** |

**Conclusion.** Any self-consistency metric rewards smoothness and over-smooths.
The one metric with the right structure (reproducibility) needs the data's real
noise level to calibrate — the exact quantity that cannot be measured without a
**half-reconstruction** (see RESULTS.md / the noise-floor experiment). So a
rigorous, independent σ,ρ certifier is blocked by the *same* linchpin as the
"is the complexity real?" question.

**Until a half-reconstruction is available**, the recommendation rests on
Lindeberg (phantom-validated scale selection) + the measured discretisation
floor. That is a defensible pick, not a two-method certification.

## Reproduce

- `python sigma_rho_selection/sigma_rho_global.py` — single global Lindeberg
  (σ, ρ) pick on real i74 + discretisation-floor check (`sigma_rho_global.png`).
  (FSC corroboration was removed after it failed its trust gate — see
  `sigma_rho_selection/README.md`.)
- `python sigma_rho_selection/sigma_rho_adaptive.py` — per-region adaptive pick
  (current best; the single global pick drifts across tissue).
- `python quality_metric.py` — the cross-scale residual trust-gate
  (`quality_metric_validate.png`); `tensor_reproducibility` / `sweep_reproducibility`
  for the reproducibility metric.
