"""
coherence_phantoms.py

Renders the exact phantoms (same parameters + seeds) used by coherence_sigma.py's
Layer A-E runs, into a single gallery PNG so the inputs behind the results are
visible. Purely a visualisation helper -- reuses phantom.make_phantom and
coherence_sigma._noise_image, generates no new data.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phantom import make_phantom
from coherence_sigma import _noise_image

SHAPE = (200, 200)


def _show(ax, img, title):
    ax.imshow(img, cmap="gray", origin="upper")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


if __name__ == "__main__":
    # Exact inputs per layer (parameters + seeds match coherence_sigma.py).
    a_fibre, _ = make_phantom(SHAPE, theta_deg=30.0, spacing=12.0, width=1.2,
                              noise_sigma=0.0)
    a_noise = _noise_image(SHAPE, seed=0)
    b_img, _ = make_phantom(SHAPE, theta_deg=30.0, spacing=12.0, width=1.2,
                            noise_sigma=0.35, seed=1)
    c_img, _ = make_phantom(SHAPE, theta_deg=30.0, spacing=12.0, width=1.2,
                            noise_sigma=0.5, seed=2)
    d_img, _ = make_phantom(SHAPE, theta_deg=20.0, spacing=12.0, width=1.2,
                            theta2_deg=110.0, spacing2=17.0, width2=1.2,
                            noise_sigma=0.35, seed=3)
    e_noises = (0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
    e_imgs = [make_phantom(SHAPE, theta_deg=30.0, spacing=12.0, width=1.2,
                           noise_sigma=ns, seed=4)[0] for ns in e_noises]

    fig, axes = plt.subplots(2, 6, figsize=(16, 5.6))

    # Top row: Layers A-D (5 panels, last cell blank).
    top = [
        (a_fibre, "Layer A: clean fibre 30 deg"),
        (a_noise, "Layer A: pure noise"),
        (b_img,   "Layer B: noisy 30 deg (noise=0.35)"),
        (c_img,   "Layer C: noisy 30 deg (noise=0.5)"),
        (d_img,   "Layer D: crossing 20/110 (noise=0.35)"),
    ]
    for ax, (img, title) in zip(axes[0], top):
        _show(ax, img, title)
    axes[0][5].axis("off")

    # Bottom row: Layer E noise sweep (single 30 deg phantom, rising noise).
    for ax, ns, img in zip(axes[1], e_noises, e_imgs):
        _show(ax, img, f"Layer E: 30 deg, noise={ns}")

    fig.suptitle("Phantoms used in coherence_sigma.py Layers A-E "
                 "(200x200, width=1.2, spacing=12)")
    fig.tight_layout()
    fig.savefig("coherence_phantoms.png", dpi=120)
    print("Saved coherence_phantoms.png")
