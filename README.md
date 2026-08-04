# Optimising Structure Tensors 
UCL MPBE Summer 2026

[Folders]
- global_tractography
- host classification
- mass_conservation
- noise model
- results
- sigma_rho_selection

[Root files]

# Core I/O & tensor

reconstruction.py (18 imp) — loads the i74 TIFF volume, sampling/cropping helpers. The data gateway.
orthogonality_residual.py (7) — the structure tensor itself (fibre_direction, eigen-decomposition).

# HOST engine (the classifier kernel host_classification/ is built on)
host_fod.py (20) — the higher-order structure-tensor FOD (analyse/classify). Most-used file in the repo.
host_geometry_test.py (13) — the phantom suite (single/crossing/kissing/complex) with known truth.
selfcalib_threshold.py (5) — the self-calibrating sphericity (c_s) guard.
model_ladder.py (2) — calibrates c_s against known structures. (host_fod imports it.)
fod_dipy.py (2) — dipy FOD cross-check. (host_fod imports it.)
two_orientation_test.py (3) — patch_gradients primitive. (host_fod imports it.)

# Phantom builders (shared by tractography and mass conservation)
crossing_shape_3d.py (15) — crossing phantom + knee_sigma.
kissing_phantom.py (6) — curved kissing-bundle builder.

# Noise & quality
noise_floor.py (7) — the N-matrix / power-spectrum library (used by noise_model and tractography).
resolution_sweep.py (1) — FSC resolution utility.

# The odd one out
visualize_i74_orientations.py (0) — a standalone figure script (renders the triage output overlay)."


raw volume V (z,y,x float32)
        │
        ▼
╔═ STAGE 1 ═════════════════════════════════════════════════╗
║ sigma_rho_adaptive.compute_adaptive_map(vol, block=32)    ║
║   → sigma_map, rho_map, zs/ys/xs   (per 32³ block)        ║
║ block_scale(centre) → (σ, ρ) for each patch               ║
║ tensor_fields(vol, σ, ρ) → val (λ desc), vec (e3)         ║
╚═══════════════════════════════════════════════════════════╝
        │  per-block (σ,ρ) + dense e3
        ▼
╔═ STAGE 2 ═ sigma_rho_gated_field.build_gated_field ════════╗
║ per patch (half=14, grid anchored on volume centre):      ║
║   analyse(patch, σ_grad=4.0, order=6) → c_s, FA           ║
║                                                            ║
║   ┌── c_s ≤ FA ──→ single: dirs = [e3 from Stage 1]       ║
║   └── c_s >  FA ──→ analyse(patch, σ_grad=2.5, order=6)   ║
║                     → dirs (≤2), classify() → label       ║
╚═══════════════════════════════════════════════════════════╝
        │  GatedField: centre, dirs, n_dirs, gated, label,
        │              c_s, FA, sigma, rho
        ├──→ gated.save("unified_field_*.npz")   ← the artifact
        ▼
╔═ STAGE 3 ═════════════════════════════════════════════════╗
║ S3.orientation_field(vol)          → e3 (z,y,x order)     ║
║ S3.field_from_gated(gated, vol, e3)                       ║
║   · reverses every dir (x,y,z) → (z,y,x)   ← the join     ║
║   · multi-dir patches → superposition_S() → S, verdict    ║
║   → field[], junctions[], I_0                             ║
║                                                            ║
║ S3.anisotropy_target(vol) → D_obs = norm(λ1−λ3), tissue   ║
║                                                            ║
║ S3.optimise(...)  coordinate descent over θ:              ║
║   loss(θ) → integrate(field, θ, verdicts)  ← S gates      ║
║                                    links here              ║
║           → rasterise() → D_rec                            ║
║           → mse(D_rec, D_obs) + verdict penalty            ║
║   → θ, loss trace, per-junction S/verdict/contribution     ║
╚═══════════════════════════════════════════════════════════╝
