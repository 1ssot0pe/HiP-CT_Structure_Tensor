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