"""
model_ladder.py

Calibrate sphericity c_s against a ladder of known synthetic structures, then
overlay the REAL i74 c_s distribution to read off what the real high-c_s tissue
most resembles. Answers: are real high-c_s voxels 2-fibre crossings (a fixel
model would help), higher-order/dispersed (needs a full FOD), or noise-like
(no orientation recoverable at this resolution)?

Ladder (all measured at the SAME sigma=4, rho=8 the real data was analysed at,
with matched noise so noise-inflation is comparable):
  1 fibre; 2 fibres at 30/60/90 deg; 3 coplanar fibres; dispersed cones
  (half-angle 20/40 deg); pure noise.

Reuses _tube_family/_coords (crossing_shape_3d), sphericity_field +
load_interior (selfcalib_threshold). Only new code: an n-direction phantom
builder and the dispersed-cone direction sampler.
"""

import numpy as np

from core.crossing_shape_3d import _tube_family, _coords, _noise_volume
from core.selfcalib_threshold import sphericity_field, load_interior

EPS = 1e-12
SPACINGS = [12.0, 17.0, 21.0, 25.0, 29.0]   # per-family, avoid a shared period


def make_multifibre(shape=(96, 96, 96), dirs=((1, 0, 0),), width=1.5,
                    noise_sigma=0.3, seed=0):
    """Sum of straight Gaussian-tube families along the given unit `dirs`."""
    coords = _coords(shape)
    vol = np.zeros(shape, np.float32)
    for i, d in enumerate(dirs):
        d = np.asarray(d, float); d = d / (np.linalg.norm(d) + EPS)
        vol = vol + _tube_family(coords, d, SPACINGS[i % len(SPACINGS)], width)
    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        vol = vol + noise_sigma * rng.standard_normal(shape).astype(np.float32)
    vol = (vol - vol.min()) / (np.ptp(vol) + EPS)
    return vol.astype(np.float32)


def _inplane(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([np.cos(t), np.sin(t), 0.0])


def _cone_dirs(kappa_deg, n, seed):
    """n directions sampled within a cone of half-angle kappa about the x-axis
    (a dispersed / fanning single population)."""
    rng = np.random.default_rng(seed)
    ct = np.cos(np.deg2rad(kappa_deg))
    dirs = []
    for _ in range(n):
        u = rng.uniform(ct, 1.0)
        phi = rng.uniform(0, 2 * np.pi)
        st = np.sqrt(max(1 - u * u, 0.0))
        dirs.append([u, st * np.cos(phi), st * np.sin(phi)])
    return dirs


def _models():
    """Return an ordered list of (label, dirs) for the ladder."""
    return [
        ("1 fibre",        [_inplane(20)]),
        ("2 fibre 30",     [_inplane(5), _inplane(35)]),
        ("2 fibre 60",     [_inplane(-10), _inplane(50)]),
        ("2 fibre 90",     [_inplane(-25), _inplane(65)]),
        ("3 fibre coplanar", [_inplane(0), _inplane(60), _inplane(120)]),
        ("dispersed 20",   _cone_dirs(20, 6, seed=1)),
        ("dispersed 40",   _cone_dirs(40, 6, seed=2)),
    ]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SIGMA, RHO, NOISE = 4.0, 8.0, 0.3
    print("=" * 70)
    print(f"MODEL LADDER -- c_s at sigma={SIGMA}, rho={RHO}, noise={NOISE}")

    rows = []
    for label, dirs in _models():
        cs = np.concatenate([
            sphericity_field(make_multifibre(dirs=dirs, noise_sigma=NOISE, seed=s),
                             SIGMA, RHO).ravel()
            for s in (0, 1)])
        cs = cs[np.isfinite(cs)]
        rows.append((label, np.median(cs), np.percentile(cs, 25), np.percentile(cs, 75)))
        print(f"  {label:18s} c_s median={np.median(cs):.3f} "
              f"IQR=[{np.percentile(cs,25):.3f},{np.percentile(cs,75):.3f}]")
    # noise reference
    cs_n = np.concatenate([sphericity_field(_noise_volume((96, 96, 96), s), SIGMA, RHO).ravel()
                           for s in (0, 1)])
    rows.append(("pure noise", np.median(cs_n), np.percentile(cs_n, 25), np.percentile(cs_n, 75)))
    print(f"  {'pure noise':18s} c_s median={np.median(cs_n):.3f}")

    # real distribution
    real = load_interior()
    cs_real = sphericity_field(real, SIGMA, RHO).ravel()
    cs_real = cs_real[np.isfinite(cs_real)]
    rp = np.percentile(cs_real, [10, 25, 50, 75, 90])
    print("-" * 70)
    print(f"  REAL i74  c_s  10/25/50/75/90 = " + " ".join(f"{v:.3f}" for v in rp))
    two_fib_max = max(r[1] for r in rows if r[0].startswith("2 fibre"))
    frac_above_2f = float((cs_real > two_fib_max).mean())
    print(f"  max 2-fibre model c_s median = {two_fib_max:.3f}")
    print(f"  fraction of REAL voxels above the 2-fibre ceiling = {frac_above_2f:.1%}")
    print("=" * 70)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(10, 5.2))
    labels = [r[0] for r in rows]
    meds = [r[1] for r in rows]
    lo = [r[1] - r[2] for r in rows]; hi = [r[3] - r[1] for r in rows]
    xpos = np.arange(len(rows))
    ax.errorbar(xpos, meds, yerr=[lo, hi], fmt="o", ms=8, capsize=4,
                color="tab:blue", label="synthetic model c_s (median, IQR)")
    # real distribution as horizontal bands
    ax.axhspan(rp[1], rp[3], color="tab:red", alpha=0.15, label="real i74 c_s IQR")
    ax.axhline(rp[2], color="tab:red", lw=2, label=f"real median={rp[2]:.2f}")
    ax.axhline(rp[4], color="tab:red", ls="--", lw=1, label=f"real 90th pct={rp[4]:.2f}")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("sphericity c_s"); ax.set_ylim(0, 1)
    ax.set_title("c_s ladder: what synthetic structure does real high-c_s tissue match?")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig("model_ladder.png", dpi=120)
    print("  saved model_ladder.png")
