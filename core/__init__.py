"""Shared kernels used by every pipeline stage.

The numbered stage folders (``1_sigma_rho_selection/`` ... ``4_optimising_parameters/``)
are script folders, not packages; each puts the repository root on ``sys.path``
and imports from here, e.g. ``from core.host_fod import analyse, classify``.

Modules
-------
Data and I/O
    reconstruction          load_volume, tissue_mask, voxel size, progress helpers
    preprocess              image preprocessing applied right after load_volume
    nifti_io                write (z, y, x) arrays and vector fields as NIfTI

Structure tensor
    tensor                  the structure tensor, computed in one place
    orthogonality_residual  plain tensor: (vol, sigma, rho) -> e3
    host_fod                higher-order structure tensor (HOST) peaks + classify
    two_orientation_test    patch_gradients primitive
    selfcalib_threshold     self-calibrating sphericity (c_s) guard
    fod_dipy                dipy FOD cross-check

Phantoms (known ground truth)
    crossing_shape_3d       tube primitives, shape measures, knee_sigma
    kissing_phantom         curved bundles that touch but never cross
    localized_phantoms      the two phantoms run_pipeline.py validates against
    host_geometry_test      labelled phantom suite
    model_ladder            c_s calibration against known geometries

Noise
    noise_floor             noise power-spectrum / N-matrix library

Modules with a demo under ``if __name__ == "__main__"`` are run from the
repository root as ``python -m core.<module>``.
"""
