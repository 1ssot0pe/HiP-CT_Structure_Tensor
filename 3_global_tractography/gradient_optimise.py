
"""
gradient_optimise.py -- actual gradient descent on the cone parameters.

Coordinate descent was not a stylistic choice; it was forced. The hard tracker's
loss is piecewise constant in (L, theta), so no derivative exists to follow --
`smooth_tracker` explains exactly which three operations destroy it. With those
relaxed, (L, theta) move the loss continuously and a gradient descent is
available for the first time.

The gradient is estimated by CENTRAL FINITE DIFFERENCES, not autodiff. The
relaxation is what makes the derivative exist; the estimator is a separate
choice, and with two parameters a central difference costs four tracking passes
per step against the substantial rewrite an autodiff framework would need. The
first thing `main` does is prove the relaxation was necessary: it measures the
same finite difference on the HARD tracker, where it comes back exactly zero.

Descent in LOG SPACE. L and theta are positive scale parameters spanning an
order of magnitude, so a step that suits L = 8 is far too large at L = 2, and an
additive step can propose a negative theta. Working in (log L, log theta) makes
the step multiplicative and keeps both positive with no clipping.

Backtracking line search, because the loss is smooth but not convex and its
curvature is unknown. A fixed learning rate here would either crawl or overshoot
into the flat region where every direction looks equally good.

Run
---
    python gradient_optimise.py                 # curved phantom, with truth
    python gradient_optimise.py --temperature-check
"""

import argparse
import json
import os
import sys

import itertools

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "1_sigma_rho_selection"),
           os.path.join(_ROOT, "2_host_classification")):
    sys.path.insert(0, _p)

import dense_tracker as DT                                  # noqa: E402
import smooth_tracker as ST                                 # noqa: E402
import cone_objective as CO                                 # noqa: E402
from cone_descent import stage2, seed, FLOOR_FRAC           # noqa: E402
from theta_validation import CurvedTruth, field_error, tract_error  # noqa: E402

EPS = 1e-12
FD_STEP = 0.05          # finite-difference step, in log space
MAX_ITERS = 40
LR0 = 0.30              # initial log-space step length
LR_MIN = 1e-3
TOL = 1e-6              # stop when a full line search cannot improve by this


# ---------------------------------------------------------------------------

class Problem:
    """The loss, as a function of (L, theta), with everything else fixed."""

    def __init__(self, vol, dense, seeds, sdirs, floor, tau=ST.TAU,
                 beta=ST.BETA, max_steps=ST.MAX_STEPS, objective="align",
                 ctx=None, turn="sample"):
        self.vol, self.dense = vol, dense
        self.seeds, self.sdirs, self.floor = seeds, sdirs, floor
        self.tau, self.beta, self.max_steps = tau, beta, max_steps
        self.quad = ST.quadrature()
        self.n_eval = 0
        self.progress = None        # set by the caller; see `f`
        # Optional FIELD OPERATOR: (L, theta, alpha) -> a refined field to track
        # through. When set, the parameters are scored by the field they
        # produce, not only by how a walker steps, and the refined field becomes
        # the run's deliverable. See `cone_field_smooth`.
        self.field_op = None
        # WHAT alpha MEANS -- see `dense_tracker.track`. "sample" is the angular
        # resolution (radians); "relax" is the fraction of the turn taken toward
        # the cone's mean orientation, in [0, 1]. The descent works in LOG space
        # on all three parameters, which is fine for both: a fraction in (0, 1]
        # is still a positive scale parameter. But the two are not comparable
        # numbers, so a run must record which one it used.
        self.turn = turn
        # WHICH objective drives the descent. "align" is the validated one;
        # "mse" exists so the old objective can be given the wheel rather than
        # only being read off a path that something else chose -- a passive
        # read-out says what MSE thought of the align descent's route, never
        # where MSE would have gone.
        #
        # A CALLABLE is also accepted, taking the smooth tractogram
        # (pos, dirs, w) and returning a scalar. New objectives are a research
        # question here, not a fixed menu, and each one added as a string
        # literal is another branch in `f` that the descent machinery has to
        # know about. `tdi_mse` is the first to use this.
        if not callable(objective) and objective not in ("align", "mse",
                                                         "image_mse"):
            raise ValueError(f"objective must be 'align', 'mse', 'image_mse' or"
                             f" a callable, got {objective!r}")
        if objective in ("mse", "image_mse") and ctx is None:
            raise ValueError("objective='mse' needs a cone_objective.Context")
        self.objective, self.ctx = objective, ctx

    def field_at(self, L, theta, alpha):
        """The field to track through at these parameters.

        Identity unless `field_op` is set. Kept separate from `track` so the
        refined field can be recovered for the optimum WITHOUT re-running the
        descent -- the whole point of smoothing from the original field every
        time is that this is a pure function of the parameters.
        """
        return self.dense if self.field_op is None else self.field_op(
            float(L), float(theta), float(alpha))

    def track(self, L, theta, alpha):
        return ST.track_soft(self.vol, self.field_at(L, theta, alpha),
                             self.seeds, self.sdirs,
                             lookahead=float(L), theta=float(theta),
                             alpha=float(alpha), floor=self.floor,
                             tau=self.tau, beta=self.beta,
                             max_steps=self.max_steps, quad=self.quad,
                             turn=self.turn)

    def f(self, u):
        """Loss at u = (log L, log theta, log alpha), by the chosen objective."""
        L, th, al = np.exp(u)
        pos, dirs, w = self.track(L, th, al)
        self.n_eval += 1
        # A tracking pass at 5000 seeds costs ~55 s, and an iteration is 8-12 of
        # them, so without this the run is silent for eight minutes at a stretch
        # and cannot be distinguished from a hang. Reported per PASS because the
        # pass is the unit of work; the iteration bar in `descend` is too coarse
        # to show anything is alive.
        if self.progress is not None:
            self.progress(self)
        if callable(self.objective):
            return float(self.objective(pos, dirs, w))
        if self.objective == "align":
            return ST.align_loss(pos, dirs, w, self.dense)
        # Same tractogram, scored a different way. Each step becomes its own
        # two-point segment so every objective always sees identical tracts.
        import cone_objective as _CO
        if self.objective == "image_mse":
            # Rasterise directly -- image_mse only ever needs the TDI, and the
            # per-step path list is the slow way to get it.
            return _CO.image_mse(
                ST.as_tdi(pos, dirs, w, self.vol.shape, n_keep=0.5), self.ctx)
        return _CO.data_mse(ST.as_paths(pos, dirs, w, n_keep=0.5), self.ctx)

    def both(self, u):
        """(align, mse) at u -- for reporting a descent against the objective
        it did NOT optimise, without re-tracking."""
        import cone_objective as _CO
        L, th, al = np.exp(u)
        pos, dirs, w = self.track(L, th, al)
        return (float(ST.align_loss(pos, dirs, w, self.dense)),
                float(_CO.data_mse(ST.as_paths(pos, dirs, w, n_keep=0.5),
                                   self.ctx)) if self.ctx is not None
                else float("nan"))

    def grad(self, u, h=FD_STEP):
        """Central finite difference. Returns (gradient, f(u))."""
        g = np.zeros_like(u)
        for i in range(len(u)):
            e = np.zeros_like(u)
            e[i] = h
            g[i] = (self.f(u + e) - self.f(u - e)) / (2.0 * h)
        return g, self.f(u)


def hard_gradient(vol, dense, seeds, sdirs, floor, L, theta, alpha,
                  h=FD_STEP):
    """The same finite difference on the HARD tracker, for comparison.

    This is the control for the whole module. If it does not come back at or
    near zero, the relaxation was unnecessary and a gradient descent could have
    been run on the original tracker all along.
    """
    ctx = CO.Context(vol, dense, np.zeros(vol.shape),
                     np.ones(vol.shape, bool), rho=1.0)

    def f(Lv, tv, av):
        paths = DT.track(vol, dense, seeds, sdirs, alpha=float(av),
                         lookahead=float(Lv), theta=float(tv), floor=floor)
        return CO.alignment_error(paths, ctx)

    u = np.log([L, theta, alpha])
    g = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        g[i] = (f(*np.exp(u + e)) - f(*np.exp(u - e))) / (2.0 * h)
    return g


# ---------------------------------------------------------------------------

def _bar(done, total, width=24):
    """Progress bar. A descent converges when it converges, so `total` is the
    ITERATION CAP and the bar shows worst-case progress -- an early finish
    stops it short rather than filling it, which is the honest display for a
    loop with no fixed length."""
    frac = min(max(done / max(total, 1), 0.0), 1.0)
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + f"] {frac:4.0%}"


def descend(prob, L0=4.0, theta0=0.30, alpha0=0.15, max_iters=MAX_ITERS,
            lr0=LR0, verbose=True, on_accept=None):
    """Gradient descent with backtracking. Returns (L, theta, alpha, trace).

    Every trace entry carries WALL-CLOCK TIMING as well as parameters:
    `seconds` for that iteration, `t_cum` since the descent started, `n_eval`
    (cumulative tracking passes) and `passes` (this iteration's share).

    Timing is recorded per ITERATION rather than divided out afterwards from a
    total, because the cost per iteration is not constant and the variation is
    the informative part: an iteration is one central difference (6 passes, a
    fixed cost) plus a backtracking line search whose length depends on how
    badly the first step overshot. An iteration that took four halvings is
    visibly more expensive than one that took none, and a mean pass time hides
    exactly that. `seconds / passes` is therefore reported too -- it should be
    roughly flat, and if it is not, the cost is coming from somewhere other than
    tracking.
    """
    import time as _time

    u = np.log([float(L0), float(theta0), float(alpha0)])
    t_start = _time.time()
    n_prev = prob.n_eval
    f0 = prob.f(u)
    trace = [dict(iter=-1, L=float(np.exp(u[0])), theta=float(np.exp(u[1])),
                  alpha=float(np.exp(u[2])), loss=float(f0), grad_norm=None,
                  lr=None, seconds=float(_time.time() - t_start),
                  t_cum=float(_time.time() - t_start),
                  n_eval=int(prob.n_eval), passes=int(prob.n_eval - n_prev))]
    lr = lr0
    if verbose:
        print(f"    start  L={np.exp(u[0]):6.3f} theta={np.exp(u[1]):6.4f} "
              f"alpha={np.exp(u[2]):6.4f}  loss {f0:.6f}"
              f"  [{trace[0]['seconds']:.2f} s]")

    # max_iters falsy (0 or None) = NO CAP: run until the descent stops itself,
    # which it does on a zero gradient or an exhausted line search. Both are
    # properties of the surface, not of a budget, so an uncapped run is the only
    # one whose endpoint is a statement about the loss rather than about when we
    # ran out of patience. The r22 MSE run hit its cap of 12 while still
    # improving, so its endpoint was the cap.
    _iters = itertools.count() if not max_iters else range(max_iters)
    for it in _iters:
        t_it, n_prev = _time.time(), prob.n_eval
        g, f_here = prob.grad(u)
        gn = float(np.linalg.norm(g))
        if gn < EPS:
            if verbose:
                print("    gradient is zero -- the loss is flat here")
            break
        step = g / gn                       # normalised: lr sets the length

        # Backtracking: halve until the move actually helps.
        acc, lr_try = False, lr
        while lr_try >= LR_MIN:
            u_try = u - lr_try * step
            f_try = prob.f(u_try)
            if f_try < f_here - TOL:
                u, f0, acc = u_try, f_try, True
                break
            lr_try *= 0.5

        # ONE side effect per ACCEPTED iteration, never per evaluation.
        # `prob.f` is called ~6 times for the central difference and once per
        # line-search trial, so a hook that fired on every evaluation would act
        # a variable number of times per iteration -- a backtracking search that
        # halved four times would apply it four extra times, making the result
        # depend on the optimiser's internals rather than on the data. Firing
        # only here means exactly one update per step the descent actually took.
        if acc and on_accept is not None:
            on_accept(u, it)
        dt = float(_time.time() - t_it)
        trace.append(dict(iter=it, L=float(np.exp(u[0])),
                          theta=float(np.exp(u[1])),
                          alpha=float(np.exp(u[2])), loss=float(f0),
                          grad_norm=gn, lr=float(lr_try), seconds=dt,
                          t_cum=float(_time.time() - t_start),
                          n_eval=int(prob.n_eval),
                          passes=int(prob.n_eval - n_prev)))
        if verbose:
            print(f"    {_bar(it + 1, max_iters) if max_iters else f'[it {it + 1:3d}]'} it {it:2d} "
                  f"L={np.exp(u[0]):6.3f} theta={np.exp(u[1]):6.4f} "
                  f"alpha={np.exp(u[2]):6.4f}  loss {f0:.6f}  "
                  f"|g| {gn:.4f}  lr {lr_try:.4f}  ({prob.n_eval} passes, "
                  f"{dt:5.1f} s)", flush=True)
        if not acc:
            if verbose:
                print("    line search exhausted -- converged")
            break
        # Grow the step a little after a success, so a long smooth run is not
        # paid for one halving at a time.
        lr = min(lr_try * 1.6, lr0)

    return (float(np.exp(u[0])), float(np.exp(u[1])), float(np.exp(u[2])),
            trace)


def temperature_check(prob, L, theta, alpha, taus=(0.04, 0.02, 0.01),
                      max_iters=5):
    """Does the ANSWER depend on the relaxation? Re-descend as tau falls.

    tau is not physics, so the question is whether it changes the conclusion.
    Re-READING the loss at fixed parameters cannot answer that: the loss value
    is guaranteed to move with tau, because tau sets how concentrated the
    softmax is and therefore how much the averaged heading is smoothed. A large
    drift in the VALUE is expected and means nothing.

    What matters is whether the OPTIMUM moves. So the descent is run again at
    each tau, from the same start, and the resulting parameters are compared.
    `dense_optimise.alpha_convergence` makes the identical argument about the
    cone's angular sampling, and for the identical reason.
    """
    rows = []
    for i, t in enumerate(taus):
        p2 = Problem(prob.vol, prob.dense, prob.seeds, prob.sdirs, prob.floor,
                     tau=t, beta=prob.beta, max_steps=prob.max_steps,
                     objective=prob.objective, ctx=prob.ctx)
        L2, th2, al2, _tr = descend(p2, L0=L, theta0=theta, alpha0=alpha,
                                    max_iters=max_iters, verbose=False)
        rows.append(dict(tau=float(t), L=L2, theta=th2, alpha=al2))
        print(f"      {_bar(i + 1, len(taus))} tau={t:.3f} -> L={L2:6.3f} "
              f"theta={th2:6.4f} alpha={al2:6.4f}", flush=True)

    # Spread of each parameter across tau, relative to its own mean. The
    # optimum is what has to be stable, not the number attached to it.
    out = {}
    worst = 0.0
    for k in ("L", "theta", "alpha"):
        v = np.array([r[k] for r in rows], float)
        rel = float((v.max() - v.min()) / max(abs(v.mean()), EPS))
        out[k] = rel
        worst = max(worst, rel)
        print(f"      {k:6} spread across tau: {rel:.1%}")
    print(f"      -> optimum is {'STABLE' if worst < 0.15 else 'tau-DEPENDENT'}"
          f" (worst {worst:.1%})")
    return dict(rows=rows, spread=out, worst=float(worst),
                stable=bool(worst < 0.15))


# ---------------------------------------------------------------------------

def main(curv=0.028, radius=2.0, ymax=24.0, noise=0.0, n_seeds=250,
         seeding="fa", L0=4.0, theta0=0.30, alpha0=0.15, check_tau=False,
         outdir=None):
    outdir = outdir or os.path.join(_ROOT, "results")
    os.makedirs(outdir, exist_ok=True)

    shape = (20, int(2 * ymax + 16), 40)
    truth = CurvedTruth(shape, curv=curv, radius=radius, ymax=ymax)
    vol = truth.volume(noise=noise)
    print("=" * 78)
    print("GRADIENT DESCENT on (L, theta, alpha) -- smooth tracker")
    print(f"  curved bundle {shape}, curv={curv:g}, noise={noise:g}")
    print("=" * 78)

    dense, D_obs, tissue, FA, _sc = stage2(vol, verbose=False)
    fe = field_error(dense.vec1, truth)
    s, sd = seed(dense, tissue, FA, seeding, n=n_seeds)
    floor = FLOOR_FRAC * float(vol.max())
    print(f"  {len(s)} seeds; field floor {fe['median']:.2f} deg\n")

    # ---- the control: is the hard tracker's gradient really zero? ----------
    # Reported PER PARAMETER, not as a norm. The two behave differently and
    # collapsing them hides the point. theta reaches the tracker only through
    # int(round(theta / alpha)), so its derivative is identically zero. L does
    # enter continuously -- it sets the lookahead sample positions -- but only
    # ever acts THROUGH the argmax, so what a finite difference returns for it
    # is a jump quotient across a decision flip, not a derivative: halve the
    # step and it does not halve, it changes arbitrarily or vanishes.
    print("  [control] central finite difference on the HARD tracker")
    gh = hard_gradient(vol, dense, s, sd, floor, L0, theta0, alpha0)
    gh_half = hard_gradient(vol, dense, s, sd, floor, L0, theta0, alpha0,
                            h=FD_STEP / 2)
    for i, nm in enumerate(("d/dlogL     ", "d/dlogtheta ", "d/dlogalpha ")):
        tag = ("EXACTLY ZERO -- no derivative exists" if gh[i] == 0.0
               else f"non-zero, but at h/2 it is {gh_half[i]:+.3e}: a jump "
                    f"quotient across an argmax flip, not a derivative")
        print(f"      {nm} = {gh[i]:+.3e}   -> {tag}")

    prob = Problem(vol, dense, s, sd, floor)
    g0, f0 = prob.grad(np.log([L0, theta0, alpha0]))
    print(f"\n  [smooth]  same point: d/dlogL = {g0[0]:+.4f}  "
          f"d/dlogtheta = {g0[1]:+.4f}  d/dlogalpha = {g0[2]:+.4f}  "
          f"|g| = {np.linalg.norm(g0):.4f}")
    print("      -> a descent direction exists in all three\n")

    L, th, al, trace = descend(prob, L0=L0, theta0=theta0, alpha0=alpha0)

    # ---- what the answer is actually worth, against the truth -------------
    pos, dirs, w = prob.track(L, th, al)
    te = tract_error(ST.as_paths(pos, dirs, w, n_keep=0.5), truth)
    print(f"\n  converged: L={L:.3f} theta={th:.4f} alpha={al:.4f} after "
          f"{prob.n_eval} tracking passes")
    print(f"  TRUE angular error at the optimum: {te['median']:.2f} deg "
          f"(field floor {fe['median']:.2f} deg, stray {te['stray_frac']:.1%})")

    tau_info = None
    if check_tau:
        print("\n  [temperature] does the OPTIMUM depend on the relaxation?")
        tau_info = temperature_check(prob, L, th, al)

    out = dict(shape=list(shape), curv=curv, noise=noise, n_seeds=int(len(s)),
               field_error=fe, hard_gradient=[float(v) for v in gh],
               smooth_gradient=[float(v) for v in g0],
               L=L, theta=th, alpha=al, trace=trace,
               n_tracking_passes=prob.n_eval,
               true_angular_error=te, temperature=tau_info)
    p = os.path.join(outdir, "gradient_optimise.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote {p}")
    print("=" * 78)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="gradient descent on the cone")
    ap.add_argument("--curv", type=float, default=0.028)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=250)
    ap.add_argument("--L0", type=float, default=4.0)
    ap.add_argument("--theta0", type=float, default=0.30)
    ap.add_argument("--alpha0", type=float, default=0.15)
    ap.add_argument("--temperature-check", action="store_true")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    main(curv=a.curv, noise=a.noise, n_seeds=a.seeds, L0=a.L0,
         theta0=a.theta0, alpha0=a.alpha0, check_tau=a.temperature_check,
         outdir=a.outdir)
