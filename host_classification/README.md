# HOST classification & the sphericity gate

The stage after σ/ρ selection. Given the structure tensor at each voxel, decide
**what's there** — one fibre, a crossing, or something unresolvable — and run the
expensive multi-fibre method only where it's actually needed.

The shared engine (`host_fod.py`, `selfcalib_threshold.py`, `model_ladder.py`)
lives at the **repo root**, because the downstream tractography code uses it too
(same reason `reconstruction.py` stayed out of `sigma_rho_selection/`). The files
*in this folder* are the gate + classification logic built on top. Moved files
reach the root engine via a `sys.path.insert` at the top.

# Terms (read first)

- **Voxel / patch** — a voxel is one 3D pixel (20 µm on i74); a patch is a small
  cube around it that the tensor is measured on.
- **Structure tensor eigenvalues (λ1 ≥ λ2 ≥ λ3)** — the three numbers the tensor
  produces. A clean single fibre has one small eigenvalue (λ3, along the fibre)
  and two big ones; the messier the local structure, the larger λ3 grows.
- **HOST-FOD** — Higher-Order Structure Tensor Fibre-Orientation Distribution.
  The ordinary tensor gives **one** direction, so it can't see a crossing. The
  higher-order (order-6) version is a polynomial on the sphere that can have
  **several** minima — one per fibre — so it resolves crossings. Accurate but
  expensive.
- **Fibre-ness / ODF** — `f(u) = max(E) − E(u)`, high in directions fibres point.
  Its peaks are the fibres; how many peaks and how sharp gives the class.
- **The four classes** — HOST-FOD labels each patch: **single** (one sharp
  peak), **crossing** (≥2 separated peaks), **dispersed** (one broad peak), or
  **disorganised** (no direction — isotropic/noise).
- **Sphericity (c_s)** — `3·λ3 / (λ1+λ2+λ3)`, a **contrast-invariant** 0–1 score
  of how "no dominant direction" a voxel is. Low = clean single fibre; high =
  crossing, dispersed, or noise. The natural cheap signal for the gate.
- **The gate** — HOST-FOD is too expensive to run everywhere, so a **cheap**
  test decides where to run it: skip patches that are confidently single, run
  HOST only on the ambiguous ones.
- **λ3 gate** — using the *absolute* smallest eigenvalue λ3 as that cheap test.
  Simpler than c_s but **not** contrast-invariant (λ3 also grows with local
  brightness/texture), which is the flaw the gate files probe.
- **Fire-fraction** — the tuning rule that survived: set the gate by the
  *fraction* of voxels it fires on across the volume, not by an absolute λ3
  cutoff (which doesn't transfer between regions).
- **Triage** — per patch, take the cheap fast-path if it's confidently single,
  otherwise hand it to HOST-FOD.
- **Multi-scale persistence** — a real fibre peak recurs across a range of σ; a
  noise-driven spurious peak shows up at only one σ. Keep peaks that persist.
- **Spatial regularisation** — a real orientation agrees with its neighbours; an
  isolated peak with no neighbour support is dropped.

# The pipeline, in order

1. **Classify a patch** — `host_fod.py` *(root)*. HOST-FOD: build the
   higher-order tensor, turn it into a fibre-ness ODF, read off the peaks, and
   label the patch **single / crossing / dispersed / disorganised**. This is the
   accurate-but-expensive engine everything else gates or cleans up.

2. **Read the sphericity** — `selfcalib_threshold.py` + `model_ladder.py`
   *(root)*. `selfcalib_threshold` checks the real i74 c_s histogram actually
   has a separable low-c_s bulk and high-c_s tail (if it didn't, gating would be
   pointless) and sets the threshold from the data's own shape — no ground truth
   needed. `model_ladder` calibrates c_s against known phantoms (1 fibre, 2
   fibres at 30/60/90°, dispersed cones, noise) to read what real high-c_s
   tissue most resembles.

3. **Gate + triage** — `adaptive_triage.py`. For each patch: pick the local σ
   (reusing the Lindeberg picker from `sigma_rho_selection`), then triage — if
   c_s is low, trust the cheap single-fibre answer; otherwise run HOST-FOD and
   trust its own class. Runs the expensive method only where it's needed.

4. **Which cheap gate is defensible?** — `gate_battery.py` → `gate_protocol.py`.
   `gate_battery` stress-tests the λ3 gate across phantoms with known
   crossing/single regions and shows its weakness: λ3 is *absolute*, so under a
   contrast gradient it fires on bright single-fibre tissue and misses faint
   crossings. `gate_protocol` then does a train/test comparison of cheap signals
   (absolute λ3 vs normalised c_s vs multi-scale median c_s). **Finding:** no
   single absolute cutoff transfers — use the contrast-invariant c_s set by
   **fire-fraction**, not an absolute λ3 threshold.

5. **Demote over-called crossings** — `refined_field.py`. On real i74 the
   crossing class is over-called: noise-driven second peaks (appearing at only
   the largest σ, spatially isolated) look like crossings. Two filters make each
   peak earn its place — **multi-scale persistence** (must recur across σ) and
   **spatial regularisation** (must agree with neighbours) — then re-derive the
   class from the survivors.

*(Superseded exploration removed in the cleanup: the cheap-scalar-localiser
attempts `coarse_to_fine.py` / `label_scaffold.py` / `crossing_trigger_test.py`
established the negative result "no cheap per-patch scalar reliably isolates
crossings", now folded into step 4; plus `selfcal_gate.py`, `phantom_gated_test.py`,
`premise_check.py`, `noise_axis_gate.py`, `host_geometry_report.py`, `ced.py`.)*

# How to run

From the repo root, e.g. `.venv\Scripts\python -u host_classification\gate_protocol.py`.
All four are runnable and phantom-based (no i74 volume needed unless noted).

| Script | What it does |
|---|---|
| `adaptive_triage.py` | gate + triage, tested on the HOST geometry phantoms |
| `gate_battery.py` | stress-tests the λ3 gate across known phantoms |
| `gate_protocol.py` | train/test comparison of cheap gate signals |
| `refined_field.py` | persistence + spatial filters that demote spurious crossings |

# Shared engine at repo root (used here and by downstream tractography)

`host_fod.py` (HOST-FOD `analyse`/`classify`), `host_geometry_test.py` (phantom
suite), `selfcalib_threshold.py` (the c_s guard), `model_ladder.py` (c_s
calibration), `fod_dipy.py` (dipy cross-check). Files here import them via a
`sys.path.insert` to the repo root; `visualize_i74_orientations.py` (root)
reaches *into* this folder for `adaptive_triage`.
