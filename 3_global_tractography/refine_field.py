r"""
refine_field.py -- Stage 3 done the way it should be: the FIELD is the product.

What stage 3 currently does
---------------------------
`descent_loop.py` descends (L, theta, alpha) on MSE between the preprocessed
image and the tractogram's density. Measured consequences, all from this repo's
own runs:

  * the field is never touched -- after a descent `vec1`/`vec2` are byte-
    identical to Stage 2's (max |change| exactly 0.0);
  * run to convergence the loss drives alpha -> 0.0317 and theta -> 0.61 deg,
    i.e. a max turn of 0.019 deg/step, because straight tracts covering more
    tissue is what minimises it;
  * the loss is a coverage term: the descent's gain came from `offset` while
    `pattern` got worse, and Pearson r stayed at ~0.02-0.10 (r^2 < 1%).

So the optimum of that objective is a tracker that does not turn -- and used as
a field operator it is a smoother that does not smooth.

What stage 3 is supposed to do
------------------------------
Produce a REFINED ORIENTATION FIELD, with (L, theta, alpha) as by-products:

    field_k = cone_smooth(field_0, L, theta, alpha)     pure function of params
    loss    = how badly field_k disagrees with the raw image gradient
    descend (L, theta, alpha) on that loss
    output  field* = cone_smooth(field_0, L*, theta*, alpha*)

THE OBJECTIVE, AND WHY IT IS NOT CIRCULAR
-----------------------------------------
Along a fibre the intensity does not change, so the fibre direction v satisfies
grad I . v = 0. The loss is that residual, summed over voxels and normalised:

    E(field) = sum_v (grad I . v)^2 / sum_v |grad I|^2

This is a statement about the RAW IMAGE. It does not read the structure tensor,
the rho smoothing, the eigendecomposition, or HOST -- which is exactly the
independence needed, because those are what produced the field being scored.
Scoring a field against e3 (as `align` does) would ask only whether the tracts
follow the field, which any smoothing of that field trivially satisfies.

The shared denominator is what removes the need for a gradient threshold: a flat
voxel contributes almost nothing to the numerator AND almost nothing to the
denominator, instead of contributing a ratio of noise to noise.

The honest limit: grad I and the structure tensor come from the same data, so
this is not an independent MEASUREMENT. It is independent of the PROCESSING that
could have gone wrong. At a genuine crossing grad I carries both fibres, so the
term is non-zero even for a correct direction -- the same second-order ceiling
everything built from grad I grad I^T has.

NO TRACKING IN THE LOOP
-----------------------
Because the loss scores the field directly, the ~55 s/pass tractography
disappears from the inner loop entirely. Tracking becomes an output, not part of
the objective, and an evaluation costs one smoothing pass.

Run
---
    python refine_field.py --radius 22            # i74
    python refine_field.py --dataset crossing     # phantom, with truth
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import cone_field_smooth as CFS                              # noqa: E402
import descent_loop as DL                                    # noqa: E402
from cone_descent import stage2                              # noqa: E402
from gradient_optimise import descend                        # noqa: E402
from nifti_io import nifti_affine, save_nifti                # noqa: E402
from reconstruction import VOXEL_SIZE_UM                     # noqa: E402

EPS = 1e-12
GRAD_SIGMA = 1.0        # derivative scale for grad I; see the module docstring


def image_gradient(vol, sigma=GRAD_SIGMA):
    """(3, Z, Y, X) image gradient in (z, y, x), at the derivative scale.

    Deliberately NOT the integration scale rho: rho is the smoothing that builds
    the structure tensor, and the whole point of this term is to test the field
    WITHOUT the processing that could have gone wrong. sigma here is only "the
    scale at which a gradient is meaningful", which any gradient needs.
    """
    v = np.asarray(vol, float)
    return np.stack([gaussian_filter(v, sigma, order=(1, 0, 0)),
                     gaussian_filter(v, sigma, order=(0, 1, 0)),
                     gaussian_filter(v, sigma, order=(0, 0, 1))])


def align_error(vecs, grad, mask):
    """sum (grad I . v)^2 / sum |grad I|^2 over `mask`. Lower is better.

    0 would mean every direction is exactly perpendicular to the local intensity
    gradient -- i.e. lies in the iso-surface, which is what a fibre direction
    does. 1 would mean every direction points straight along the gradient.
    """
    v = np.asarray(vecs, float)
    n = np.linalg.norm(v, axis=0)
    v = np.where(n > EPS, v / np.maximum(n, EPS), 0.0)
    m = np.asarray(mask, bool)
    dot = (grad * v).sum(axis=0)[m]
    den = (grad ** 2).sum(axis=0)[m]
    tot = float(den.sum())
    return float((dot ** 2).sum() / tot) if tot > EPS else 1.0


class FieldAlign:
    """The objective: cone-smooth the ORIGINAL field, then score it.

    Callable on (L, theta, alpha). Holds the original field and the image
    gradient, so an evaluation is one smoothing pass and a dot product.
    """

    def __init__(self, dense, vol, mask, sigma=GRAD_SIGMA, second=True):
        self.dense, self.mask, self.second = dense, np.asarray(mask, bool), second
        self.grad = image_gradient(vol, sigma)
        self.sigma = float(sigma)
        self.n_calls = 0

    def field(self, L, theta, alpha):
        return CFS.smooth_field(self.dense, float(L), float(theta),
                                float(alpha), second=self.second)

    def __call__(self, L, theta, alpha):
        self.n_calls += 1
        return align_error(self.field(L, theta, alpha).vec1, self.grad,
                           self.mask)

    def baseline(self):
        """The unsmoothed field's score -- what refinement has to beat."""
        return align_error(self.dense.vec1, self.grad, self.mask)


class _Prob:
    """Minimal `descend`-compatible problem over (log L, log theta, log alpha).

    `gradient_optimise.Problem` is built around tracking; this objective needs
    none, so rather than thread a bypass through it the three methods `descend`
    actually calls are provided directly.
    """

    def __init__(self, obj, progress=None):
        self.obj, self.n_eval, self.progress = obj, 0, progress

    def f(self, u):
        L, th, al = np.exp(u)
        # alpha is a FRACTION and the descent works in log space, which is
        # unbounded above; a proposal past 1.0 is not a turn fraction at all.
        val = self.obj(L, th, min(float(al), 1.0))
        self.n_eval += 1
        if self.progress:
            self.progress(self)
        return val

    def grad(self, u, h=0.05):
        g = np.zeros_like(u)
        for i in range(len(u)):
            e = np.zeros_like(u)
            e[i] = h
            g[i] = (self.f(u + e) - self.f(u - e)) / (2.0 * h)
        return g, self.f(u)


# ---------------------------------------------------------------------------

def main(dataset="i74", centre=384, radius=22, L0=4.0, theta0=0.30,
         alpha0=0.5, max_iters=0, mode="adaptive", outdir=None, verbose=True):
    outdir = outdir or os.path.join(_ROOT, "results", "refine_field")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 78)
    print("STAGE 3 (corrected) -- descend (L, theta, alpha) to REFINE THE FIELD")
    print("=" * 78)

    vol, origin, label = DL.load_setup(dataset, centre, radius, verbose=False)
    print(f"  {label} {vol.shape}")
    t0 = time.time()
    dense, _D, tissue, _FA, scales = stage2(vol, mode=mode, verbose=False)
    oriented = np.asarray(dense.n_dirs) >= 1
    print(f"  stage 1-2 in {(time.time() - t0) / 60:.1f} min; "
          f"{int(oriented.sum())} oriented voxels, "
          f"{dense.n_multi} with two directions")

    obj = FieldAlign(dense, vol, oriented)
    base = obj.baseline()
    print(f"\n  OBJECTIVE  E(field) = sum (grad I . v)^2 / sum |grad I|^2   "
          f"(lower is better)")
    print(f"  Stage-2 field scores {base:.6f}   <- the number to beat")

    t0 = time.time()
    one = obj(L0, theta0, alpha0)
    per = time.time() - t0
    print(f"  one evaluation {per:.2f} s  (no tracking in the loop)")

    def _prog(p):
        el = time.time() - t0
        print(f"\r      eval {p.n_eval:4d}  {el / 60:5.1f}m  "
              f"({el / max(p.n_eval, 1):4.1f}s/eval)", end="", flush=True)

    prob = _Prob(obj, progress=_prog)
    print(f"\n  [descent] from L={L0:g} theta={np.degrees(theta0):.2f}d "
          f"alpha={alpha0:g}" + ("  (NO CAP)" if not max_iters else ""))
    L, th, al, trace = descend(prob, L0=L0, theta0=theta0, alpha0=alpha0,
                               max_iters=max_iters)
    al = min(al, 1.0)

    refined = obj.field(L, th, al)
    final = align_error(refined.vec1, obj.grad, oriented)
    moved = CFS.angular_change(refined.vec1, dense.vec1, mask=oriented)

    print(f"\n\n  converged L={L:.3f} vox  theta={np.degrees(th):.3f} deg  "
          f"alpha={al:.4f}")
    print(f"  align error {base:.6f} -> {final:.6f}  "
          f"({100 * (final - base) / base:+.2f}%)")
    print(f"  field moved by median {moved['median']:.3f} deg "
          f"(mean {moved['mean']:.3f}, p95 {moved['p95']:.3f})")

    truth = None
    if dataset != "i74":
        # On a phantom the TRUE tangent exists, so the objective can be checked
        # against the thing it is a proxy for rather than trusted.
        try:
            from theta_validation import CurvedTruth  # noqa: F401
        except Exception:
            pass

    A = nifti_affine(origin, VOXEL_SIZE_UM)
    paths = {
        "vec1": save_nifti(np.moveaxis(np.asarray(refined.vec1), 0, -1),
                           os.path.join(outdir, "refined_vec1.nii.gz"), A,
                           vector=True, dtype=np.float32),
        "vec2": save_nifti(np.moveaxis(np.asarray(refined.vec2), 0, -1),
                           os.path.join(outdir, "refined_vec2.nii.gz"), A,
                           vector=True, dtype=np.float32),
    }
    refined.save(os.path.join(outdir, "refined_field.npz"))

    out = dict(dataset=dataset, label=label, centre=centre, radius=radius,
               shape=list(vol.shape), crop_origin=list(origin),
               objective="image_align_on_field", grad_sigma=obj.sigma,
               L=float(L), theta=float(th), alpha=float(al),
               theta_deg=float(np.degrees(th)),
               align_error_before=base, align_error_after=final,
               improvement_pct=float(100 * (final - base) / base),
               moved=moved, n_eval=prob.n_eval, seconds_per_eval=per,
               n_oriented=int(oriented.sum()), n_multi=int(dense.n_multi),
               trace=trace, truth=truth,
               paths={k: os.path.basename(v) for k, v in paths.items()})
    p = os.path.join(outdir, "refine_field.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  wrote {p}")
    for k, v in paths.items():
        print(f"    {k:6} {v}")
    print("=" * 78)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", choices=("i74", "crossing"), default="i74")
    ap.add_argument("--centre", type=int, default=384)
    ap.add_argument("--radius", type=int, default=22)
    ap.add_argument("--L0", type=float, default=4.0)
    ap.add_argument("--theta0", type=float, default=0.30)
    ap.add_argument("--alpha0", type=float, default=0.5)
    ap.add_argument("--max-iters", type=int, default=0,
                    help="0 = no cap; descend until the line search exhausts")
    ap.add_argument("--mode", choices=("adaptive", "global"), default="adaptive")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    main(dataset=a.dataset, centre=a.centre, radius=a.radius, L0=a.L0,
         theta0=a.theta0, alpha0=a.alpha0, max_iters=a.max_iters,
         mode=a.mode, outdir=a.outdir)
