# Mass conservation

A second, **independent** way to tell a real crossing from a kissing pair —
using brightness instead of tractography. Physics: at a real crossing both fibre
centrelines pass through the point, so the intensity there is roughly the **sum**
of both fibres (mass conserved, ratio ≈ 2×). At a kissing pair's closest
approach *neither* centreline goes through, so the intensity can never exceed
about **one** fibre's value (ratio < 1×) — no matter how small the gap.

This stage reuses the HOST crossing certifiers from `global_tractography/`
(`phase2_i74_demote`, `phase1_kissing_crossing`) and the phantom/kernel code at
the **repo root** (`crossing_shape_3d.py`, `kissing_phantom.py`, `host_fod.py`,
`reconstruction.py`). Files here reach both via a `sys.path.insert` at the top.

# Terms (read first)

- **Superposition** — where two fibres genuinely cross, their intensities add.
  That additivity is the signature this stage looks for.
- **Mass-conservation ratio** — `intensity at the ambiguous point / a local
  single-fibre calibration`. ≈2 means "looks like two fibres summed" (a real
  crossing); ≤1 means "looks like at most one fibre" (kissing / spurious). The
  threshold is a physically-motivated 1.5.
- **Local calibration** — the single-fibre brightness to divide by, taken from
  nearby patches HOST confidently calls "single" (no ground truth needed).
- **Fire fraction** — the fraction of HOST-flagged crossings whose ratio clears
  the threshold, i.e. the fraction that look genuinely superposed.
- **Selection bias** — the calibration needs a nearby confidently-single patch;
  in dense tissue there often isn't one, so those candidates get skipped — which
  can bias *which* crossings even get scored.

# The pipeline, in order

1. **Which brightness metric separates crossing from kissing?** —
   `crossing_vs_kissing_metrics.py`. A cheap first test of candidate metrics on
   phantoms; the mass-conservation ratio wins.

2. **Prove it on phantoms** — `mass_conservation_validate.py`. A proper
   train/test sweep across
   crossing angles, tube widths, kissing gaps, and both synthetic *and* real
   measured noise. **Result: catch = 100% (real crossings flagged), false-positive
   = 0% (kissing never flagged)** at the fixed 1.5 threshold — the metric
   generalises.

   > **This file has a second job.** Its phantom builders
   > (`raw_localized_crossing`, `raw_kissing`, and the `CROSS_*` constants) are
   > imported by `3_global_tractography/run_pipeline.py`,
   > `self_optimising_tractography.py` and
   > `2_host_classification/sigma_rho_gated_field.py`. It is a **library the live
   > pipeline depends on**, not only an experiment — do not delete it as
   > evidence.

3. **Apply it to real i74** — `real_i74_mass_conservation.py`. Run the
   phantom-validated ratio on actual HOST 2-peak crossing voxels. (First checked
   that bright fibre cores don't clip at 255 — they don't — so mass conservation
   isn't broken by saturation.) **Result: 0% fire fraction** — every scored real
   crossing looks single-branch, not superposed, echoing the tractography
   demotion finding.

4. **Is that 0% trustworthy? — the bias check** — `i74_calibration_bias_check.py`.
   Only ~30% of candidates got a calibration. **The unscored ones sit in
   significantly denser, more complex tissue (p < 0.001)** — exactly where real
   crossings are most likely. So the clean 0% is measuring the *easy* subset; the
   result is biased.

5. **Fix attempts** — `i74_directional_calibration.py` (calibrate by walking
   along the fibre axis — only recovered ~10% of the dense candidates) and
   `i74_global_density_calibration.py` (calibrate from a density-matched pool
   anywhere in the volume). The dense-tissue coverage limit proved hard to close.

6. **Do the two independent metrics agree?** — `i74_metric_agreement.py`. Checks
   whether the mass-conservation ratio and the tractography straightness
   certifier demote the *same* voxels (not just similar overall rates) — the real
   test of whether they're detecting the same thing.

# How to run

From the repo root, with `$PY` set to the interpreter in your venv (see `PIPELINE.md` § 0 — the venv lives **outside** this folder, because it is OneDrive-synced), e.g. `$PY -u mass_conservation\mass_conservation_validate.py`.

| Script | Needs | What it shows |
|---|---|---|
| `crossing_vs_kissing_metrics.py` | phantom | candidate-metric shortlist |
| `mass_conservation_validate.py` | phantom + `lung_A/B.npy` | the 100% / 0% phantom proof |
| `real_i74_mass_conservation.py` | i74 volume | 0% fire fraction on real crossings |
| `i74_calibration_bias_check.py` | i74 volume | the selection-bias finding |
| `i74_directional_calibration.py` / `i74_global_density_calibration.py` | i74 volume | the two fix attempts |
| `i74_metric_agreement.py` | i74 volume | per-voxel agreement of the two metrics |

# Dependencies

Reaches *up* to the repo root (`crossing_shape_3d`, `kissing_phantom`, `host_fod`,
`reconstruction`) and *into* `3_global_tractography/` (`phase2_i74_demote`,
`phase1_kissing_crossing`) — a one-way downstream→upstream reach via
`sys.path.insert`. All five i74 scripts here import
`phase2_i74_demote.collect_crossings`, so those two files cannot be treated as
tractography-only evidence. `lung_A/B.npy` is gitignored; rebuild it with
`noise_model/build_lung_slab.py`.
