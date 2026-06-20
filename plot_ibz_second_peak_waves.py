"""Plot IBZ wavevector grid with >50% second-peak highlights."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
PT_DIR = ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt"
OUT = ROOT / "INFERENCE/second_peak_ibz_map.png"

# Overlap (c_test ∩ b_test), both >50%
OVERLAP = {
    0, 12, 312, 24, 324, 300, 13, 11, 1, 23, 2, 22, 10, 14, 3, 8, 21, 16, 9, 15,
    7, 17, 18, 6, 4, 20, 19, 5, 37,
}
# b_test only >50%
B_ONLY = {62, 237, 262, 275, 287, 299, 301, 323}


def main() -> None:
    kxy = torch.load(PT_DIR / "wavevectors_full.pt", map_location="cpu", weights_only=False)[0].numpy()
    n_wv = kxy.shape[0]
    pi = np.pi
    kx_n = kxy[:, 0] / pi
    ky_n = kxy[:, 1] / pi

    overlap = np.array(sorted(OVERLAP), dtype=int)
    b_only = np.array(sorted(B_ONLY), dtype=int)
    other = np.array([i for i in range(n_wv) if i not in OVERLAP and i not in B_ONLY], dtype=int)

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)

    # IBZ rectangle [-π, π] × [0, π]  (symmetry_type='none', a=1)
    rect = Rectangle((-1, 0), 2, 1, fill=False, linewidth=2, edgecolor="black", linestyle="-", zorder=1)
    ax.add_patch(rect)

    # Full grid
    ax.scatter(
        kx_n[other], ky_n[other], s=18, c="#b0b0b0", alpha=0.7, linewidths=0,
        label=f"Other wavevectors ({len(other)})", zorder=2,
    )
    ax.scatter(
        kx_n[b_only], ky_n[b_only], s=70, c="#9b59b6", edgecolors="black", linewidths=0.4,
        label=f"b_test only >50% ({len(b_only)})", zorder=4,
    )
    ax.scatter(
        kx_n[overlap], ky_n[overlap], s=55, c="#e74c3c", edgecolors="black", linewidths=0.35,
        label=f"Overlap >50% c∩b ({len(overlap)})", zorder=5,
    )

    # ky=0 axis (Γ–X path)
    ky0_mask = np.abs(ky_n) < 1e-3
    ax.plot(kx_n[ky0_mask], ky_n[ky0_mask], color="#3498db", linewidth=1.5, alpha=0.5, zorder=3, label="ky=0 (Γ–X)")

    # kx=0 axis
    kx0_mask = np.abs(kx_n) < 1e-3
    ax.plot(kx_n[kx0_mask], ky_n[kx0_mask], color="#2ecc71", linewidth=1.5, alpha=0.5, zorder=3, label="kx=0")

    # Annotate specials
    labels = {
        12: "Γ\nw12",
        0: "w0",
        24: "w24",
        312: "w312",
        324: "w324",
        300: "w300",
        37: "w37",
    }
    for w, txt in labels.items():
        ax.annotate(
            txt,
            (kx_n[w], ky_n[w]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="black",
        )

    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.06, 1.08)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"$k_x / \pi$")
    ax.set_ylabel(r"$k_y / \pi$")
    ax.set_title(
        "IBZ wavevector grid (325 points)\n"
        "Red = >50% second-peak in both c_test and b_test  |  "
        "Purple = b_test only"
    )
    ax.legend(loc="center", bbox_to_anchor=(2 / 3, 0.5), fontsize=8, framealpha=0.92)

    # Axis ticks in π units
    ax.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels([r"$-\pi$", r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$",
                        r"$0$", r"$+\frac{\pi}{4}$", r"$+\frac{\pi}{2}$", r"$+\frac{3\pi}{4}$", r"$+\pi$"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([r"$0$", r"$+\frac{\pi}{4}$", r"$+\frac{\pi}{2}$", r"$+\frac{3\pi}{4}$", r"$+\pi$"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
