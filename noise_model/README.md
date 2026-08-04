# Noise model

Measures what the imaging **noise** actually looks like, so the later stages can
answer "is this crossing real, or is it noise?" with a number instead of a guess.
The whole reason this stage exists: i74 has pervasively high sphericity (lots of
voxels look like crossings), and we need to know how much of that is genuine
structure versus the structure tensor reacting to noise.

The noise *library* (`noise_floor.py`) stays at the **repo root** — the
tractography stage imports it too. The files here are the experiments that use
it. Each reaches the root libs via a `sys.path.insert` at the top.

# Terms (read first)

- **Half-map** — two independent reconstructions A and B of the *same* sample.
  Their difference `n = (A − B)/√2` cancels the shared signal and leaves **pure
  noise** — the only clean way to see the noise on its own.
- **Noise floor (N)** — the structure tensor you'd measure on pure noise. It adds
  to the true tensor as a **bias** (not something more averaging can remove), so
  it can inflate sphericity and manufacture fake crossings.
- **White noise** — noise with equal energy at all spatial frequencies. Its
  structure-tensor trace falls off as σ⁻⁵ with the smoothing scale; a shallower
  falloff means the noise is **not** white.
- **Correlated (low-frequency) noise** — noise whose energy sits mostly at low
  frequencies (ring/cupping artifacts). Falloff slope ≈ −3.5, not −5.
- **Anisotropic noise** — noise with a preferred direction (λ1/λ3 > 1). A single
  fibre plus directional noise can fake a second peak → a spurious crossing.
- **Power spectrum / slope** — how noise energy is spread across frequencies; its
  log-log slope is the fingerprint that tells white from correlated.
- **c_s (sphericity)** — `3·λ3 / (λ1+λ2+λ3)`, the contrast-invariant 0–1 "no
  dominant direction" score. The quantity this stage checks for noise inflation.
- **Cross-organ proxy** — i74 is **brain** and has **no half-maps**, so its own
  noise can't be measured directly. Every "real noise" test borrows **lung**
  half-map noise as a stand-in — an assumption this stage tries to falsify.

# The pipeline, in order

1. **Is the high sphericity just white noise?** — `run_noise_floor.py` (uses the
   root `noise_floor.py` lib). Runs the T1–T9 battery on i74: does correcting for
   a white-noise floor collapse the high-c_s fraction? **Result: white-noise form
   FALSIFIED** — c_s barely moves, so the complexity is not a white-noise
   artifact. But the *correlated* form couldn't be tested on i74 (no half-maps;
   the air is clipped to 0, so there's no signal-free region to measure).

2. **Get real noise from half-maps** — `build_lung_slab.py`. i74 has no
   half-maps, but a **lung** dataset does. This reads its two reconstructions,
   crops matched tissue blocks, and caches `lung_A.npy` / `lung_B.npy` so
   `n = (A − B)/√2` is available as genuine measured noise.

3. **Measure that noise directly** — `half_map_analysis.py` (the T1–T4 library) +
   `run_lung_halfmap.py` (one best-SNR block) + `run_lung_multiblock.py`
   (robustness across blocks). **Result: the noise is CORRELATED (slope ≈ −3.5,
   not white's −5) and ANISOTROPIC (λ1/λ3 ≈ 1.5–1.9)** — exactly the ring/cupping
   form that was untestable on i74, now measured and confirmed.

4. **De-bias the tensor with it** — `run_lung_subtractN.py`. Subtract the full
   measured N from the structure tensor to remove the noise bias continuously
   (uses the whole anisotropic covariance, no fragile stable-axis assumption).

5. **Does lung noise transfer to brain i74?** — `i74_noise_transfer_falsify.py`
   (the honest caveat, this session). With no i74 half-maps the transfer can only
   be *falsified*, not proven: it compares i74's high-frequency spectral tail to
   lung noise's. **Result: SURVIVES** (slopes −4.04 vs −4.01) — but the match is
   almost certainly the shared HiP-CT reconstruction roll-off, so only the
   *isotropic* spectral shape transfers; the anisotropy stays unverifiable. See
   the cross-organ caveat in `BAYES_TRACTOGRAPHY.md`.

# How to run

From the repo root, with `$PY` set to the interpreter in your venv (see `PIPELINE.md` § 0 — the venv lives **outside** this folder, because it is OneDrive-synced), e.g. `$PY -u noise_model\run_lung_multiblock.py`.

| Script | Needs | Note |
|---|---|---|
| `run_noise_floor.py` | i74 volume | the T1–T9 white-noise-floor test |
| `build_lung_slab.py` | the lung half-map TIFFs | regenerates `lung_A/B.npy` (gitignored) |
| `run_lung_halfmap.py` | `lung_A/B.npy` | single-block T1–T4 |
| `run_lung_multiblock.py` | `lung_A/B.npy` | robustness across blocks |
| `run_lung_subtractN.py` | `lung_A/B.npy` | subtract-N de-biasing |
| `i74_noise_transfer_falsify.py` | i74 volume + `lung_A/B.npy` | cross-organ falsification |

`lung_A.npy` / `lung_B.npy` are large and **gitignored** — rebuild them with
`build_lung_slab.py` rather than committing them.

# Shared at repo root

`noise_floor.py` (the N-matrix / power-spectrum library, used here and by the
tractography stage), plus `reconstruction.py` and `orthogonality_residual.py`.
Files here import them via `sys.path.insert` to root.

`debias.py` (the analysis-time de-biasing layer) was deleted in the `simple`
pass — it was never imported by anything. `noise_floor.structure_tensor_field`
still takes `subtract_N=`, which is the mechanism it wrapped.
