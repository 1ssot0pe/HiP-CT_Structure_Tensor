"""
param_volume.py -- the (L, theta, alpha) loss volume on real i74, as a NIfTI.

Two things, on one i74 chunk, with one set of seeds:

  1. SWEEP    the smooth align loss on an n x n x n grid over (L, theta, alpha),
              written as a 3D NIfTI so it can be opened in NiiVue.
  2. DESCEND  a gradient descent on the same three parameters, from a neutral
              start, so where the descent lands can be compared against the
              volume it was walking on.

Both use `smooth_tracker`, because the hard tracker has no gradient in theta or
alpha -- those derivatives are exactly zero, which `gradient_optimise` measures
as its control.

The NIfTI's axes are PARAMETERS, not space
------------------------------------------
Voxel (i, j, k) holds the loss at (L_i, theta_j, alpha_k). The grids are
log-spaced, because all three are positive scale parameters spanning an order of
magnitude and a linear grid would spend most of its samples at the coarse end.

The affine therefore maps index -> log10(parameter), which makes NiiVue's
crosshair readout meaningful: the number it shows on each axis is the log10 of
the parameter, and the spacing is uniform in that coordinate. It is NOT
millimetres and nothing here is anatomical. A sidecar JSON carries the actual
parameter value at every index so the mapping never has to be reconstructed by
hand.

Run
---
    python param_volume.py                    # n=8, radius 24, ~est printed
    python param_volume.py --n 6 --seeds 100  # quicker
    python param_volume.py --no-descent       # sweep only
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import smooth_tracker as ST                                  # noqa: E402
from cone_descent import stage2, seed, FLOOR_FRAC            # noqa: E402
from gradient_optimise import Problem, descend, hard_gradient  # noqa: E402

# Grid extents. theta's lower end sits below the quadrature step on purpose --
# that is the degenerate corner where the cone cannot turn, and a volume that
# excluded it would hide the one feature the surface is known to have.
L_RANGE = (2.0, 12.0)
THETA_RANGE = (0.03, 0.60)
ALPHA_RANGE = (0.05, 0.40)


def _bar_eta(done, total, t0, width=28):
    frac = done / max(total, 1)
    n = int(round(frac * width))
    el = time.time() - t0
    eta = (el / max(done, 1)) * (total - done)
    return (f"[{'#' * n}{'-' * (width - n)}] {frac:5.1%}  {done}/{total}  "
            f"elapsed {el / 60:5.1f}m  remaining {eta / 60:5.1f}m")


def load_chunk(centre=384, radius=24, verbose=True):
    from reconstruction import load_volume
    lo, hi = centre - radius, centre + radius
    vol = load_volume(crop=(lo, hi), dtype=np.uint8).astype(np.float32)
    if verbose:
        print(f"  i74 chunk {vol.shape} at ({centre},{centre},{centre}), "
              f"mean {vol.mean():.1f}")
    return vol, (lo, lo, lo)


def sweep(prob, n=8, verbose=True):
    """Loss on the (L, theta, alpha) grid. Returns (Ls, thetas, alphas, Z)."""
    Ls = np.exp(np.linspace(*np.log(L_RANGE), n))
    ths = np.exp(np.linspace(*np.log(THETA_RANGE), n))
    als = np.exp(np.linspace(*np.log(ALPHA_RANGE), n))
    Z = np.zeros((n, n, n), np.float32)
    tot, t0, done = n ** 3, time.time(), 0
    for i, L in enumerate(Ls):
        for j, th in enumerate(ths):
            for k, al in enumerate(als):
                Z[i, j, k] = prob.f(np.log([L, th, al]))
                done += 1
                if verbose and (done % max(n, 1) == 0 or done == tot):
                    print("    " + _bar_eta(done, tot, t0), flush=True)
    return Ls, ths, als, Z


def save_volume(Ls, ths, als, Z, outdir, tag="i74"):
    """Write the loss volume as NIfTI, with an index -> log10(param) affine.

    The two ANGLES are written in DEGREES. Radians are what the tracker
    computes in and degrees are what the numbers get read in, and a cone width
    of 0.255 says far less to a reader than 14.6 degrees. Because the grids are
    log-spaced, the conversion is a pure translation of the affine's offset --
    log10(deg) = log10(rad) + log10(180/pi) -- so the voxel spacing is untouched
    and the volume itself does not have to be recomputed or resampled.

    L stays in VOXELS. It is a length, not an angle.
    """
    import nibabel as nib

    ths_deg, als_deg = np.degrees(ths), np.degrees(als)

    # Uniform in log10, by construction of the grids.
    def _step(v):
        lg = np.log10(v)
        return float(lg[1] - lg[0]) if len(v) > 1 else 1.0

    A = np.diag([_step(Ls), _step(ths_deg), _step(als_deg), 1.0])
    A[:3, 3] = [np.log10(Ls[0]), np.log10(ths_deg[0]), np.log10(als_deg[0])]

    img = nib.Nifti1Image(np.ascontiguousarray(Z, np.float32), A)
    # The units are not spatial. Marking them "unknown" is more honest than
    # letting a viewer print millimetres for a lookahead in voxels.
    img.header.set_xyzt_units("unknown")
    img.header["descrip"] = b"align loss; x=log10(L vox) y=log10(theta deg) " \
                            b"z=log10(alpha deg)"
    p = os.path.join(outdir, f"param_volume_{tag}.nii.gz")
    nib.save(img, p)

    side = dict(
        axes=["L (voxels)", "theta (degrees)", "alpha (degrees)"],
        note="voxel (i,j,k) = align loss at (L[i], theta_deg[j], alpha_deg[k]);"
             " affine maps index -> log10(parameter). ANGLES ARE IN DEGREES."
             " Not millimetres, not anatomical.",
        L=[float(v) for v in Ls],
        theta_deg=[float(v) for v in ths_deg],
        alpha_deg=[float(v) for v in als_deg],
        theta_rad=[float(v) for v in ths], alpha_rad=[float(v) for v in als],
        loss_min=float(Z.min()), loss_max=float(Z.max()))
    ps = os.path.join(outdir, f"param_volume_{tag}.json")
    with open(ps, "w") as f:
        json.dump(side, f, indent=2)
    return p, ps


def main(centre=384, radius=24, n=8, n_seeds=150, seeding="fa",
         do_descent=True, L0=4.0, theta0=0.15, alpha0=0.15, max_iters=40,
         outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 78)
    print(f"PARAMETER VOLUME on i74 -- {n}x{n}x{n} over (L, theta, alpha)")
    print("=" * 78)
    # Stage 1-2 cost scales with the VOLUME, while a tracking pass scales with
    # walkers x candidates x steps and barely notices it. On a large crop the
    # setup can therefore dominate the whole run, so it is timed separately --
    # otherwise a long silence before the first progress bar is indist-
    # inguishable from a hang.
    t_setup = time.time()
    vol, origin = load_chunk(centre, radius)
    print(f"  [setup] loaded in {time.time() - t_setup:.1f} s; "
          f"{vol.size / 1e6:.1f} M voxels", flush=True)
    t0 = time.time()
    dense, D_obs, tissue, FA, _sc = stage2(vol, verbose=False)
    print(f"  [setup] stage 1-2 (scales, tensor, HOST) in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    s, sd = seed(dense, tissue, FA, seeding, n=n_seeds)
    floor = FLOOR_FRAC * float(vol.max())
    print(f"  {seeding} seeding: {len(s)} seeds; "
          f"{int(tissue.sum())} tissue voxels; "
          f"setup total {(time.time() - t_setup) / 60:.1f} min")

    prob = Problem(vol, dense, s, sd, floor)

    # Calibrate on one real evaluation, so the estimate printed before a long
    # sweep is measured rather than guessed.
    t0 = time.time()
    prob.f(np.log([L0, theta0, alpha0]))
    per = time.time() - t0
    # ~8.7 tracking passes per descent iteration: 6 for the central difference
    # on three parameters, plus the line search.
    n_desc = int(round(8.7 * max_iters)) if do_descent else 0
    print(f"\n  one tracking pass: {per:.2f} s")
    print(f"  ESTIMATE: sweep {n ** 3} passes ~{n ** 3 * per / 60:.1f} min"
          + (f"  +  descent ~{n_desc} passes ~{n_desc * per / 60:.1f} min"
             if do_descent else "")
          + f"  =  ~{(n ** 3 + n_desc) * per / 60:.1f} min total\n")

    Ls, ths, als, Z = sweep(prob, n=n)
    k = np.unravel_index(int(np.argmin(Z)), Z.shape)
    print(f"\n  grid minimum {Z[k]:.6f} at L={Ls[k[0]]:.3f} "
          f"theta={ths[k[1]]:.4f} alpha={als[k[2]]:.4f}")

    p, ps = save_volume(Ls, ths, als, Z, outdir)
    print(f"  wrote {p}")
    print(f"  wrote {ps}   (parameter value at every index)")

    out = dict(centre=centre, radius=radius, n=n, n_seeds=int(len(s)),
               volume=list(vol.shape), seconds_per_pass=per,
               L=[float(v) for v in Ls], theta=[float(v) for v in ths],
               alpha=[float(v) for v in als],
               theta_deg=[float(v) for v in np.degrees(ths)],
               alpha_deg=[float(v) for v in np.degrees(als)],
               grid_min=dict(loss=float(Z[k]), L=float(Ls[k[0]]),
                             theta=float(ths[k[1]]), alpha=float(als[k[2]]),
                             theta_deg=float(np.degrees(ths[k[1]])),
                             alpha_deg=float(np.degrees(als[k[2]]))))

    if do_descent:
        print(f"\n  [control] hard-tracker finite difference at the start point")
        gh = hard_gradient(vol, dense, s, sd, floor, L0, theta0, alpha0)
        for i, nm in enumerate(("d/dlogL    ", "d/dlogtheta", "d/dlogalpha")):
            print(f"      {nm} = {gh[i]:+.3e}"
                  + ("   EXACTLY ZERO" if gh[i] == 0.0 else ""))
        print(f"\n  [descent] from L={L0:g} theta={theta0:g} alpha={alpha0:g}")
        L, th, al, trace = descend(prob, L0=L0, theta0=theta0, alpha0=alpha0,
                                   max_iters=max_iters)
        print(f"\n  descent  loss {trace[-1]['loss']:.6f} at L={L:.3f} "
              f"theta={th:.4f} alpha={al:.4f}")
        print(f"  grid min loss {Z[k]:.6f} at L={Ls[k[0]]:.3f} "
              f"theta={ths[k[1]]:.4f} alpha={als[k[2]]:.4f}")
        gap = trace[-1]["loss"] - float(Z[k])
        print(f"  descent - grid = {gap:+.6f}  "
              + ("(descent found a better point; it is not restricted to grid "
                 "nodes)" if gap < 0 else
                 "(grid found a better point: the descent stopped short of it)"))
        out["descent"] = dict(L=L, theta=th, alpha=al,
                              theta_deg=float(np.degrees(th)),
                              alpha_deg=float(np.degrees(al)), trace=trace,
                              hard_gradient=[float(v) for v in gh],
                              n_passes=prob.n_eval)

    pj = os.path.join(outdir, "param_volume_run.json")
    with open(pj, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  wrote {pj}")
    print("=" * 78)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="(L, theta, alpha) loss volume on i74, as NIfTI, + descent")
    ap.add_argument("--centre", type=int, default=384)
    ap.add_argument("--radius", type=int, default=24)
    ap.add_argument("--n", type=int, default=8, help="grid points per axis")
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--seeding", choices=("fa", "random"), default="fa")
    ap.add_argument("--no-descent", action="store_true")
    ap.add_argument("--L0", type=float, default=4.0)
    ap.add_argument("--theta0", type=float, default=0.15)
    ap.add_argument("--alpha0", type=float, default=0.15)
    ap.add_argument("--max-iters", type=int, default=40,
                    help="descent iteration cap; each costs ~8.7 passes")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    main(centre=a.centre, radius=a.radius, n=a.n, n_seeds=a.seeds,
         seeding=a.seeding, do_descent=not a.no_descent, L0=a.L0,
         theta0=a.theta0, alpha0=a.alpha0, max_iters=a.max_iters,
         outdir=a.outdir)
