# HOST classification & the sphericity gate

The stage after σ/ρ selection. The structure tensor gives one direction per voxel,
so it cannot describe a place where two fibres cross. This folder decides, for each
patch, whether the cheap one-direction answer can be trusted, and runs the
expensive multi-fibre method (HOST-FOD) only where it cannot.

The engine (`host_fod.py`, `selfcalib_threshold.py`, `model_ladder.py`) lives in
`core/`, because the tractography code uses it too. The files in this folder are
the gate and classification logic built on top. They import the engine as
`from core.host_fod import ...`, after a `sys.path.insert` of the repository root
at the top of each file.

# Terms

Voxel / patch — a voxel is one 3D pixel (20 µm on i74). A patch is a small cube
around it. Everything below is measured on a patch, not a single voxel.

Gradient scatter matrix — inside a patch, take the image gradient at each voxel,
drop the near-flat ones, and sum their outer products. Its three eigenvalues
λ1 ≥ λ2 ≥ λ3 describe the shape of the spread of gradient directions. Every
quantity below is read off these three numbers.

HOST-FOD — Higher-Order Structure Tensor Fibre-Orientation Distribution. An
order-6 polynomial on the sphere that can have several minima, one per fibre, so
unlike the ordinary tensor it can resolve a crossing. Accurate but expensive.

Fibre-ness / ODF — `f(u) = max(E) − E(u)`, large in directions fibres point along.
Its peaks are the candidate fibres.

The five classes — `single` (one fibre), `crossing` (two separated fibres),
`dispersed` (a direction exists but is smeared out), `disorganised` (no direction
at all: noise or empty), `complex` (three or more peaks, past what this method can
resolve, so no direction is claimed).

Sphericity, c_s — `3·λ3 / (λ1+λ2+λ3)`, between 0 and 1. How much the patch has no
dominant direction. Low = clean single fibre; high = crossing, dispersed or noise.

Fractional anisotropy, FA — between 0 and 1, from the same three eigenvalues. How
unequal they are, i.e. how strongly the patch prefers some directions. High =
strongly directional.

Why both — c_s and FA summarise the same eigenvalues differently. c_s depends on
λ3 alone; FA uses the spread of all three. So FA separates patches that share a
c_s value, which is what makes the step 3 rule work with no tuned threshold.

Contrast-invariant — unchanged if the whole patch gets brighter or dimmer. c_s and
FA are ratios, so both are. A raw eigenvalue like λ3 is not.

Multi-scale persistence — a real fibre peak reappears across a range of σ; a noise
peak appears at only one σ.

Spatial regularisation — a real orientation agrees with its neighbours; an
isolated direction with no neighbour support is dropped.

# Part A — what runs, per patch, in order

**Which of these is actually live.** There are two per-patch routes in this
folder and they pick their scale differently:

| | `sigma_rho_gated_field.build_gated_field` | `adaptive_triage.triage_patch_v2` |
|---|---|---|
| Called by | **`3_global_tractography/run_pipeline.py`** — the pipeline | `visualize_i74_orientations.py` only |
| σ for the gate | certified `gate_sigma_grad = 5.0` | per-patch Lindeberg sweep |
| σ for directions | certified `dir_sigma_grad = 2.5` | the same per-patch pick |
| Direction for a single patch | Stage 1's e3 at the block's certified (σ, ρ) | HOST's own peak |

Both gate scales used to be module constants (`GATE_SIGMA_GRAD = 4.0`,
`DIR_SIGMA_GRAD = 2.5`, `GATE_HALF = 14`) justified by a sweep table in the
module docstring. They are now measured by
`1_sigma_rho_selection/certify_gate_scales.py` and read from `gate_scales.json`.
Re-deriving them moved one: the certification adds a **single-fibre control**
the old table lacked, without which "maximise the crossing's c_s − FA" is
monotone in σ_grad and simply walks to the largest scale swept. Against all
three cases — crossing must fire, kissing and single must not — the gate scale
comes out at **5.0**, at a genuine interior turnover (margin +0.598 at σ=4,
+0.961 at σ=5, +0.850 at σ=6). `half=14` and `dir_sigma_grad=2.5` are reproduced
exactly, and half=14 turns out to be the *only* window that both fires the gate
and lets HOST decompose the crossing into two peaks.

So the **live** path is `sigma_rho_gated_field`, on certified scales. Step 1 below
describes `adaptive_triage`'s per-patch Lindeberg pick, which is *not* in the
pipeline — it is the more principled version, kept for the overlay figure and
as the obvious upgrade path. Steps 2–4 are shared: both call the same
`host_fod.analyse` / `classify` and apply the same c_s ≤ FA rule.

`sigma_rho_gated_field.py`'s own docstring records why the fixed scales are
what they are: at Stage 1's σ (~0.85–1.0) the gate never fires at all, and at
4.0 the second peak that justified gating is already smoothed away — so the
gate scale and the direction scale genuinely cannot be the same number.

## Step 1 — pick the scale for this patch *(adaptive_triage only, not live)*

- In: the raw patch, a cube of image voxels.
- Does: a Lindeberg sweep over σ on this patch alone; take the first interior peak.
- Out: sigma_grad — one number for this patch.
- Code: `pick_local_sigma_grad`, reusing `pick_sigma_rho` from `1_sigma_rho_selection`.

HOST-FOD only works when σ_grad matches the local fibre width, so this is picked
per patch, not once for the volume. The live path uses fixed scales instead —
see the table above.

## Step 2 — measure the patch, once

- In: the raw patch, plus sigma_grad from step 1.
- Does: build the gradient scatter matrix and the order-6 ODF.
- Out: one metrics dict `m`, holding `n_peaks`, `sep` (angle between the top two
  peaks), `amp2` (their balance), `aniso` (how far the ODF is from flat), `c_s`,
  `FA`, and `dirs` (the peak directions).
- Code: `host_fod.analyse(patch, sigma_grad)`.

Everything after this reads `m`. The image is not touched again.

## Step 3 — decide whether the cheap answer is trustworthy

- In: `c_s` and `FA` from `m` (step 2).
- Does: compare them.
- Out: a route, plus a label if the fast path is taken.

> A patch is a single fibre when c_s ≤ FA.
> Otherwise it is handed to HOST-FOD's own classifier in step 4.

In words: trust the one-direction answer when the patch is more directional than
it is shapeless.

Both sides come from the same eigenvalues of the same patch, so the rule is local
and has no threshold to tune. It replaced a fixed cutoff (c_s ≤ 0.15) which called
only ~3% of real i74 tissue single, implausibly few; c_s ≤ FA gives ~29%. The two
classes overlap in c_s, so a c_s threshold alone cannot reproduce this split — FA
is adding information.

Written in **three** places, because there are three ways in:

1. `host_fod.classify` — the single-vs-dispersed branch (the reference).
2. `_is_single` in `adaptive_triage.py` — the fast path.
3. `sigma_rho_gated_field.build_gated_field` — **the live gate**.

All three must carry the same `n_peaks >= 3` guard, because 2 and 3 both return
"single" *without* calling `classify()` and would otherwise hand a 3+-peak patch
back as a clean single fibre. Copy 3 was missing that guard: the live Stage 2
tested only `c_s <= fa`, so a patch the method has given up on could contribute
one confident direction to the tractography field. Now fixed — such patches go
down the HOST branch and get labelled by `classify()` like everything else.

Measured on real i74 before the fix (48³ at the volume centre, half=14,
stride=6): **0 of 34** gate-passing patches had `n_peaks >= 3`, so this was a
latent inconsistency rather than an active fault — in line with the ~0.5% rate
recorded at the bottom of this file. Three copies of one rule is the underlying
smell; a shared helper would be the real fix.

Passing a number as `cs_single_cutoff` / `cs_single` restores the old fixed
threshold; that is an override for comparison, not the default.

## Step 4 — HOST-FOD's class, for patches that reached it

- In: the same `m` from step 2. Nothing is recomputed.
- Out: one of the five class labels, plus the directions that justify it.
- Code: `host_fod.classify(m)`.

Tested in this order:

| order | test | label |
|---|---|---|
| 1 | ODF nearly flat (`aniso < 0.9`) | `disorganised` |
| 2 | exactly 2 peaks, balanced (`amp2 ≥ 0.4`) and separated (`sep ≥ 30°`) | `crossing` |
| 3 | 3 or more peaks | `complex` |
| 4 | otherwise, c_s ≤ FA | `single` |
| 5 | otherwise | `dispersed` |

Three or more peaks is past the validated two-fibre ceiling, and is as likely to be
one peak splitting as a real triple crossing, so it gets its own label. `complex`
means the method has given up here. Downstream filters that select `== "single"`
therefore exclude it.

## Step 5 — remove crossings that should not have been called

- In: labels and directions for a whole grid of patches, plus `m` at several σ.
- Does: keep only peaks that survive multi-scale persistence and spatial
  regularisation, then re-derive the class from the survivors.
- Out: revised labels for the grid.
- Code: `refined_field.py`.

On real i74 the crossing class is over-called: noise-driven second peaks appear at
only the largest σ and sit alone in space. These two filters make each peak earn
its place.

## A note on cost

In `triage_patch`, `analyse()` runs before the c_s ≤ FA test, because c_s and FA
come out of it. So step 3 does not save the cost of step 2 — it decides which
answer to believe. The saving is in `triage_patch_v2`, where a patch that passes as
single skips the expensive sweep of σ through HOST-FOD.

# Part B — two different gates

Two things in this folder are both called "the gate". They answer different
questions.

| | c_s ≤ FA (step 3) | λ3 fire-fraction (`refined_field.run_i74_gated`) |
|---|---|---|
| Question | Is the cheap single-fibre answer trustworthy? | Where is it worth spending HOST at all? |
| Input | c_s and FA, from the patch's own eigenvalues | λ3 alone, at a fixed σ_grad = 3.0 |
| Threshold | none, the two quantities compare directly | a percentile over the volume; ~65% of patches fire |
| Contrast-invariant | yes | no, λ3 grows with local brightness |
| Status | the live per-patch rule | a coarse volume-level screen |

The λ3 gate uses a percentile rather than an absolute cutoff for a measured reason:
the threshold fitted on phantoms lands off the end of
the real i74 distribution, so only the fire fraction transfers between datasets,
not the number. `gate_threshold.json` records both; the absolute value is
kept only as a diagnostic.

# Part C — the evidence behind those choices

These do not run in the pipeline. They are why it looks the way it does.

1. Does gating make sense on this data? — `core/selfcalib_threshold.py`. Checks
   the real i74 c_s histogram has a low-c_s bulk and a separable high-c_s tail with
   a valley between. If it were one smooth smear, no c_s gate could work. It also
   derives a threshold from the histogram's shape: that is the override in step 3,
   not the default.

2. What are the high-c_s voxels? — `core/model_ladder.py`. Calibrates c_s against
   known structures (1 fibre; 2 fibres at 30/60/90°; 3 coplanar; dispersed cones;
   pure noise) at matched σ, ρ and noise, then overlays the real i74 distribution to
   read off what high-c_s tissue most resembles.

3. Where does the λ3 gate fail? — `gate_battery.py`. On phantoms with known
   regions: because λ3 is absolute, under a contrast gradient it fires on bright
   single-fibre tissue and misses faint crossings.

4. Which cheap signal is defensible? — `gate_protocol.py`. Train/test on disjoint
   phantom seeds, comparing λ3, c_s at one σ, median c_s and minimum c_s across σ,
   scored by AUC and a fixed-recall operating point. Finding: no absolute cutoff
   transfers between datasets; set the operating point by fire fraction. A two-tier
   gate was fitted and tested here and did not beat the single-tier gate.

5. What does it look like on real tissue? — `visualize_i74_orientations.py`. Runs
   `triage_patch_v2` over a grid of patches on one i74 slice and draws a glyph per
   patch. No ground truth exists, so this is a qualitative map of where the pipeline
   is confident, where it resolves a crossing, and where it gives up.

# How to run

From the repository root, with the environment from the [main README](../README.md#setup) active, e.g. `python -u 2_host_classification/gate_protocol.py`.

| Script | Needs | What it does |
|---|---|---|
| `adaptive_triage.py` | nothing (synthetic) | the live per-patch path on the HOST geometry phantoms, v1 vs v2 |
| `gate_battery.py` | nothing (synthetic) | stress-tests the λ3 gate across known phantoms |
| `gate_protocol.py` | nothing (synthetic) | train/test comparison of cheap gate signals; writes `gate_threshold.json` |
| `refined_field.py` | i74 for `i74` / `gated` | `validate` (default), `i74`, or `gated` as the first argument |
| `visualize_i74_orientations.py` | i74 volume | orientation glyphs over one real slice |

# Engine in `core/`

`host_fod.py` (`analyse` / `classify`), `host_geometry_test.py` (phantom suite),
`selfcalib_threshold.py` (the c_s guard), `model_ladder.py` (c_s calibration),
`fod_dipy.py` (dipy cross-check), `two_orientation_test.py` (`patch_gradients`,
used by both gates).

# Fixed while writing this document

1. `gate_threshold.json` was orphaned by the folder rename. `GATE_FILE` resolves
   next to the script, but the only copy sat at the repo root, so `load_gate()`
   never found it and silently refitted the gate on every call. The file has been
   moved into this folder; the recorded fit (thr 0.3965, 96% train catch, 64.8%
   fire fraction) now loads instead of being regenerated.

2. The `complex` label existed only in a comment. `host_fod.classify` said a patch
   with three or more peaks was labelled complex, but the branch was never written:
   such patches fell through to the c_s vs FA rule and came back `single` whenever
   c_s ≤ FA. That is a filter, not a display string — `mass_conservation` builds
   its calibration pool from `classify(m) == "single"` — so patches the method had
   given up on were entering a clean-single-fibre pool. Measured on real i74 before
   the fix: 3+ peaks is rare (1 of 196 sampled patches, 0.5%) but that one patch
   was labelled `single`, about 1.4% of the single pool at that sampling. One
   observation, so treat the rate as indicative, not precise. The branch is now
   implemented, and the same guard was added to `adaptive_triage._is_single`,
   because the fast path returns `single` without calling `classify()` and would
   otherwise bypass it.
