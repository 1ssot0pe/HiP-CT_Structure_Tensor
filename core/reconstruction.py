"""
Single-eigenvector image reconstruction from a structure-tensor principal-
direction field, via 3D Line Integral Convolution (LIC).

Why LIC:
    The structure tensor *measures* local orientation from the asymmetry of
    intensity correlation (smooth along a fibre, choppy across it). LIC does the
    inverse: it *manufactures* that same anisotropic correlation by smearing
    white noise along streamlines of the direction field. Direction becomes
    texture, using none of the real image's greyscale values -- which is what
    makes the downstream resolution loss ground-truth-free.

Pipeline position:
    input -> structure tensor -> eigendecomposition -> principal eigenvector
    field -> [THIS: LIC reconstruction] -> resolution estimation (FSC / PSD)

Scope: reconstruction only. No FSC / no (sigma, rho) loop here.
"""

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


# ----------------------------------------------------------------------------
# Progress bar (shared by every (sigma, rho) sweep in the package)
# ----------------------------------------------------------------------------

def iter_progress(items, total=None, desc="progress", enable=True):
    """Yield from `items` while showing a live progress bar.

    Uses tqdm if it is installed; otherwise falls back to a minimal
    carriage-return bar with an ETA -- no third-party dependency required.
    Pass enable=False to silence it (e.g. inside another progress context).
    """
    seq = items if total is not None else list(items)
    total = total if total is not None else len(seq)
    if not enable or total == 0:
        yield from seq
        return
    try:
        from tqdm import tqdm
        yield from tqdm(seq, total=total, desc=desc)
        return
    except Exception:
        pass
    start = time.time()
    for k, it in enumerate(seq, 1):
        yield it
        frac = k / total
        elapsed = time.time() - start
        eta = elapsed / frac - elapsed if frac > 0 else 0.0
        bar = ("#" * int(28 * frac)).ljust(28)
        sys.stdout.write(f"\r{desc} [{bar}] {k}/{total} "
                         f"{100 * frac:5.1f}%  {elapsed:5.1f}s elapsed, ~{eta:5.1f}s left")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def grid_cells(n_sigma, n_rho):
    """(i, j) index pairs for a sigma x rho grid, row-major. Shared by sweeps so
    each cell can be fed through iter_progress with a known total."""
    return list(product(range(n_sigma), range(n_rho)))


# ----------------------------------------------------------------------------
# Real HiP-CT volume (the actual data these drafts target)
# ----------------------------------------------------------------------------
# Subject i74, sensitivity chunk, 20 um/voxel isotropic. The stack is a folder
# of single-slice TIFFs; stacking gives an (Nz, Ny, Nx) uint8 volume. RAM is the
# binding constraint (see notebook), so we truncate to a cube by default.

DEFAULT_VOLUME_DIR = Path(
    r"C:\Users\issyh\Downloads\sensitivity_chunk_IC_768\sensitivity_chunk_IC_768"
)
VOXEL_SIZE_UM = 20.0  # isotropic voxel edge, subject i74

# Fraction of the volume max below which a voxel is treated as air. It is one
# constant because it was three: `sigma_rho_gated_field.build_gated_field`,
# `self_optimising_tractography.anisotropy_target` and `background_level` each
# carried their own `air_frac=0.05` default and each recomputed the mask. Three
# copies of a threshold is three chances for them to disagree, and the mask they
# produce decides what counts as tissue in the loss, in the gate, and in I_0.
AIR_FRAC = 0.05


def tissue_mask(vol, air_frac=AIR_FRAC):
    """Boolean tissue mask: everything above `air_frac` of the volume max.

    Computed ONCE per run and passed down, rather than re-derived in each stage.
    Relative to the volume's own maximum, not an absolute level -- the same
    rank-not-level lesson the gate work recorded, where an absolute intensity
    threshold never transferred between regions.
    """
    vol = np.asarray(vol)
    return vol > float(air_frac) * float(vol.max())


def load_volume(folder=DEFAULT_VOLUME_DIR, crop=(200, 456), dtype=float,
                preprocess=True, clip_pct=(1.0, 99.0), mask_air=True,
                verbose=False):
    """Load and stack the TIFF slice folder into a 3D volume.

    folder : directory of *.tif slices (one slice per file, sorted by name).
    crop   : (start, stop) applied identically to all three axes to carve a cube
             (RAM-bound: full volume is ~768^3). Pass None to keep the full stack.
    dtype  : output dtype for the RAW path. See the warning below -- it is
             deliberately ignored when `preprocess` is on.
    preprocess : clip to `clip_pct` of this volume's own histogram, then
             rescale onto [0, 1]. See `preprocess.py` for what each step is for.
    mask_air : compute the clip percentiles over FOREGROUND only, so the large
             air peak at 0 does not drag p_lo down to a no-op. On lung data
             this is the difference between the clip doing something and doing
             nothing, which is why it defaults on here even though
             `preprocess_volume` itself defaults it off.

    Returns the volume as an (Nz, Ny, Nx) array.

    WHY `dtype` IS IGNORED WHEN PREPROCESSING
    -----------------------------------------
    Preprocessing outputs floats in [0, 1]. Callers in this repo ask for
    `dtype=np.uint8` (a memory-efficient raw read) and then cast up to float32
    themselves. Applying that uint8 cast to a [0, 1] float volume would collapse
    every voxel to 0 or 1 and destroy the data silently -- the volume would still
    have the right shape and the pipeline would still run. So the cast is skipped
    on the preprocessed path and float64 is returned; a later `.astype(float32)`
    by the caller is harmless.
    """
    import tifffile

    folder = Path(folder)
    files = sorted(folder.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"no *.tif slices found in {folder}")
    volume = np.stack([tifffile.imread(f) for f in files])
    if crop is not None:
        a, b = crop
        volume = volume[a:b, a:b, a:b]
    if preprocess:
        from core.preprocess import preprocess_volume
        return preprocess_volume(volume, clip_pct=clip_pct, mask_air=mask_air,
                                 normalise=True, verbose=verbose)
    return volume.astype(dtype)


# ----------------------------------------------------------------------------
# Sampling helpers (trilinear, boundary = clamp so streamlines never wrap)
# ----------------------------------------------------------------------------

def _sample_scalar(volume, coords):
    # coords: (3, N) index-space positions -> (N,) interpolated values.
    # mode='nearest' clamps out-of-bounds samples to the edge (never wraps).
    return map_coordinates(volume, coords, order=1, mode="nearest")


def _sample_vector(field, coords):
    # field: (Nx, Ny, Nz, 3); coords: (3, N) -> (N, 3)
    comps = [map_coordinates(field[..., c], coords, order=1, mode="nearest")
             for c in range(3)]
    return np.stack(comps, axis=-1)


# ----------------------------------------------------------------------------
# Streamline weighting
# ----------------------------------------------------------------------------

def _streamline_weights(length, kernel):
    # weights[k] is the weight of the sample k steps from the seed voxel.
    k = np.arange(length + 1, dtype=float)
    if kernel == "boxcar":
        w = np.ones_like(k)
    elif kernel == "hanning":
        # 1 at the centre, tapering smoothly to 0 at the streamline ends.
        # A taper suppresses the ringing a hard boxcar edge would add to the
        # reconstruction's power spectrum -- relevant because the loss reads
        # exactly that spectrum.
        w = 0.5 * (1.0 + np.cos(np.pi * k / max(length, 1)))
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    return w


# ----------------------------------------------------------------------------
# Streamline integration (vectorised over all voxels, loops over steps only)
# ----------------------------------------------------------------------------

def _integrate(field, source, start_pos, weights, step_size, length, sign):
    """
    Walk one half-streamline (forward if sign=+1, backward if sign=-1) from
    every seed simultaneously, accumulating weighted samples of `source`.

    field:     (Nx,Ny,Nz,3) unit-vector field
    source:    scalar volume being smeared (noise for LIC, image for baseline)
    start_pos: (N,3) seed positions in index space
    returns:   (acc, wsum) each (N,), the weighted sum and summed weights
    """
    pos = start_pos.copy()
    N = pos.shape[0]
    acc = np.zeros(N)
    wsum = np.zeros(N)
    prev_dir = None

    for k in range(1, length + 1):
        v = _sample_vector(field, pos.T)                 # (N,3)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / (norm + 1e-12)                           # guard isotropic voxels

        if prev_dir is None:
            step_dir = sign * v                          # seed the walk direction
        else:
            # Eigenvector sign is arbitrary (v == -v). Flip v to keep travelling
            # the same way, otherwise the streamline reverses on itself.
            d = np.sum(v * prev_dir, axis=1)
            flip = np.where(d < 0.0, -1.0, 1.0)[:, None]
            step_dir = v * flip

        pos = pos + step_size * step_dir
        prev_dir = step_dir

        w = weights[k]
        acc += w * _sample_scalar(source, pos.T)
        wsum += w

    return acc, wsum


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def reconstruct_from_eigenvector(eigenvector_field,
                                 length=10,
                                 step_size=0.5,
                                 kernel="hanning",
                                 method="lic",
                                 input_image=None,
                                 seed=None):
    """
    Reconstruct a synthetic greyscale volume from a principal-eigenvector field.

    length / step_size / kernel set the spatial-frequency content of the output
    -- the exact quantity the downstream FSC loss measures. They are exposed
    (not hard-coded) because they will later be swept alongside (sigma, rho).

    method="lic":       smear seeded white noise along fibre streamlines
                        (ground-truth-free: no real greyscale used).
    method="gaussian":  smear the ORIGINAL image along the same streamlines.
                        Fast sanity check only. NOTE: this reuses real greyscale
                        values, so it is NOT a ground-truth-free reconstruction
                        -- do not treat its resolution as independent evidence.

    Reminder on rho: the eigenvector_field is only well-defined when the tensor
    was rho-windowed (rho > 0). At rho = 0 the 3D tensor is rank-1 and the
    principal direction is undefined; the field is then noise wearing an
    orientation, which this function will faithfully turn into texture. Exclude
    rho = 0 upstream.
    """
    field = np.asarray(eigenvector_field, dtype=float)
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError("eigenvector_field must have shape (Nx,Ny,Nz,3)")
    shape = field.shape[:3]

    if method == "lic":
        rng = np.random.default_rng(seed)
        source = rng.standard_normal(shape)
    elif method == "gaussian":
        if input_image is None:
            raise ValueError("method='gaussian' requires input_image")
        source = np.asarray(input_image, dtype=float)
        if source.shape != shape:
            raise ValueError("input_image shape must match the field")
    else:
        raise ValueError(f"unknown method {method!r}")

    weights = _streamline_weights(length, kernel)

    gx, gy, gz = np.indices(shape)
    start = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(float)

    # Seed voxel (offset 0) sampled once, directly.
    w0 = weights[0]
    acc = w0 * source.ravel()
    wsum = np.full(start.shape[0], w0)

    fwd_acc, fwd_w = _integrate(field, source, start, weights, step_size, length, +1)
    bwd_acc, bwd_w = _integrate(field, source, start, weights, step_size, length, -1)

    acc += fwd_acc + bwd_acc
    wsum += fwd_w + bwd_w

    out = (acc / wsum).reshape(shape)
    return out


# ----------------------------------------------------------------------------
# Boundary cropping (apply identically to reconstruction AND the input before
# any resolution comparison -- both have corrupted edges)
# ----------------------------------------------------------------------------

def crop_region(volume, margin):
    """Crop `margin` voxels from every face. Return the cropped array.

    Rationale: structure-tensor convolution and LIC streamlines both corrupt the
    edges with spurious high frequencies, exactly the band FSC is sensitive to.
    Following Gray et al., use margin ~ 2/3 * (largest kernel radius in play).
    """
    m = int(margin)
    if m <= 0:
        return volume
    s = tuple(slice(m, dim - m) for dim in volume.shape[:3])
    return volume[s]


def suggested_margin(sigma, rho, length, step_size):
    """A rough crop margin: covers the ST kernel support and the LIC reach."""
    st_radius = 3.0 * (sigma + rho)          # ~3 sigma of the Gaussian kernels
    lic_reach = length * step_size
    return int(np.ceil((2.0 / 3.0) * max(st_radius, lic_reach)))


# ----------------------------------------------------------------------------
# Gray-style phantom (parameterised by fibre width), with ground truth retained
# ----------------------------------------------------------------------------

def make_phantom(shape=(96, 96, 96), width=1.5, period=8.0,
                 noise_sigma=0.02, seed=0):
    """
    Three regions stacked along x:
        [0]        straight fibres along +z
        [1]        crossing fibres (+z superimposed with +x)
        [2]        isotropic noise

    Fibres are Gaussian 'tubes' on a periodic 2D lattice in the plane
    perpendicular to the fibre axis (width = PSF size, period = spacing). The
    smallest-gradient direction inside a tube lattice is the tube axis, so the
    structure tensor should recover it.

    Returns (volume, truth) where truth is (Nx,Ny,Nz,3): the unit fibre
    direction in the straight region, and NaN where direction is ambiguous
    (crossing) or undefined (noise). Retained so the unsupervised resolution
    loss can later be checked against Gray-style supervised angular error.

    Reason in units of `width`: sigma, rho, length, step_size should all be set
    relative to fibre width, not absolute voxels (Gray: 'structure size, not
    pixels'). Gray's 3D sweet spot ~ sigma, rho in 0.5-1.0 x width.
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = shape
    x, y, z = np.indices(shape).astype(float)

    def tubes(a, b):
        # periodic Gaussian tubes on the (a,b) lattice
        da = ((a + period / 2) % period) - period / 2
        db = ((b + period / 2) % period) - period / 2
        return np.exp(-(da**2 + db**2) / (2 * width**2))

    vol = np.zeros(shape)
    truth = np.full(shape + (3,), np.nan)

    x0, x1 = Nx // 3, 2 * Nx // 3
    ax = np.arange(Nx)

    straight = ax < x0
    crossing = (ax >= x0) & (ax < x1)
    noisy = ax >= x1

    # straight: fibres along z -> pattern in (x,y), broadcast over z
    fz = tubes(x, y)
    vol[straight] = fz[straight]
    truth[straight] = np.array([0.0, 0.0, 1.0])

    # crossing: z-fibres + x-fibres (pattern in (y,z)), left ambiguous in truth
    fx = tubes(y, z)
    vol[crossing] = np.maximum(fz[crossing], fx[crossing])

    # noise region
    vol[noisy] = rng.random(vol[noisy].shape)

    # light global noise + normalise
    vol = vol + noise_sigma * rng.standard_normal(shape)
    vol = (vol - vol.min()) / (np.ptp(vol) + 1e-12)
    return vol, truth


# ----------------------------------------------------------------------------
# Minimal standalone structure tensor -- FOR THIS DRAFT'S DEMO ONLY.
# In the repo, replace with run_structure_tensor(...) and the existing
# smallest-eigenvalue eigenvector extraction.
# ----------------------------------------------------------------------------

def _demo_principal_direction(volume, sigma, rho):
    g = np.gradient(gaussian_filter(volume, sigma))
    Ix, Iy, Iz = g
    def win(a):
        return gaussian_filter(a, rho)
    Sxx, Syy, Szz = win(Ix*Ix), win(Iy*Iy), win(Iz*Iz)
    Sxy, Sxz, Syz = win(Ix*Iy), win(Ix*Iz), win(Iy*Iz)

    S = np.empty(volume.shape + (3, 3))
    S[..., 0, 0] = Sxx; S[..., 1, 1] = Syy; S[..., 2, 2] = Szz
    S[..., 0, 1] = S[..., 1, 0] = Sxy
    S[..., 0, 2] = S[..., 2, 0] = Sxz
    S[..., 1, 2] = S[..., 2, 1] = Syz

    vals, vecs = np.linalg.eigh(S)          # ascending eigenvalues
    principal = vecs[..., 0]                # SMALLEST eigenvalue's eigenvector
    return principal


def _run_on_volume(volume, sigma=1.5, rho=5.5, length=8, step_size=0.5):
    """LIC-reconstruct a real (ground-truth-free) volume at one (sigma, rho).

    Uses the draft's self-consistent structure tensor (_demo_principal_direction),
    whose eigenvector components are already in the volume's [axis0,axis1,axis2]
    index basis -- the same basis reconstruct_from_eigenvector steps along -- so
    no z/y/x reordering is needed here. (sigma=1.5, rho=5.5 are the values used in
    the notebook's see_visualisations run.)
    """
    volume = np.asarray(volume, dtype=float)
    field = _demo_principal_direction(volume, sigma=sigma, rho=rho)

    recon = reconstruct_from_eigenvector(field, length=length, step_size=step_size,
                                         kernel="hanning", method="lic", seed=1)
    base = reconstruct_from_eigenvector(field, length=length, step_size=step_size,
                                        method="gaussian", input_image=volume)

    m = suggested_margin(sigma=sigma, rho=rho, length=length, step_size=step_size)
    print("volume       ", volume.shape, "range",
          round(volume.min(), 3), round(volume.max(), 3))
    print("field        ", field.shape)
    print("lic recon    ", recon.shape, "std", round(recon.std(), 3))
    print("gaussian base", base.shape, "std", round(base.std(), 3))
    print("crop margin  ", m, "-> cropped", crop_region(recon, m).shape)
    return recon, base


if __name__ == "__main__":
    # Small phantom smoke test (always runs -- no data needed).
    vol, truth = make_phantom(shape=(48, 48, 48), width=1.5, period=8.0)
    field = _demo_principal_direction(vol, sigma=1.0, rho=1.5)

    recon = reconstruct_from_eigenvector(field, length=8, step_size=0.5,
                                         kernel="hanning", method="lic", seed=1)
    base = reconstruct_from_eigenvector(field, length=8, step_size=0.5,
                                        method="gaussian", input_image=vol)

    m = suggested_margin(sigma=1.0, rho=1.5, length=8, step_size=0.5)
    recon_c = crop_region(recon, m)

    print("=== phantom smoke test ===")
    print("phantom     ", vol.shape, "range", round(vol.min(), 3), round(vol.max(), 3))
    print("field        ", field.shape)
    print("lic recon    ", recon.shape, "std", round(recon.std(), 3))
    print("gaussian base", base.shape, "std", round(base.std(), 3))
    print("crop margin  ", m, "-> cropped", recon_c.shape)

    # Sanity: in the straight (+z) region the reconstruction should vary less
    # along z (smeared) than across x. Compare mean |gradient| per axis there.
    sx, sy, sz = np.gradient(recon[:16])   # straight third
    print("straight-region mean|grad|  x:%.4f  y:%.4f  z:%.4f  (z should be smallest)"
          % (np.abs(sx).mean(), np.abs(sy).mean(), np.abs(sz).mean()))

    # Real HiP-CT volume -- runs only if the TIFF folder is present on this machine.
    if DEFAULT_VOLUME_DIR.exists():
        print("\n=== real volume (i74 sensitivity chunk) ===")
        # Smaller crop than the notebook's 256^3 keeps the LIC demo quick; widen
        # via load_volume(crop=(200, 456)) for the full truncated cube.
        volume = load_volume(crop=(200, 328))   # 128^3
        _run_on_volume(volume, sigma=1.5, rho=5.5, length=8, step_size=0.5)
    else:
        print(f"\n[real volume skipped: {DEFAULT_VOLUME_DIR} not found]")
