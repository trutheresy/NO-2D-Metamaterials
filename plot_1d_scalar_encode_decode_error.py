"""
Regenerate scalar Gabor encode--decode figures (from eigenfrequency_encoding_tests.ipynb).

Writes:
  FIGURES/eigenfrequency_encoding.png
  FIGURES/eigenvalue_encode_decode_error_test.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from NO_utilities import embed_eigenfrequency_wavelet, extract_eigenfrequency_from_wavelet

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "FIGURES"

S_MIN = 1.0
S_MAX = 8000.0
N_ENCODING_EXAMPLES = 12
N_ERROR_TEST = 2000

# Notebook defaults were title=7; +4, then +2, then +1.
TITLE_FS = 14


def _format_s(s: float) -> str:
    return f"s={s:.1f}"


def plot_encoding_examples(out_path: Path) -> None:
    s_values = np.logspace(np.log10(S_MIN), np.log10(S_MAX), N_ENCODING_EXAMPLES)
    gabor_images = []
    for s in s_values:
        img, _, _ = embed_eigenfrequency_wavelet(s)
        gabor_images.append(img)

    # 4 rows x 6 columns via nested gridspec so row1-2 (within-pair) and
    # row2-3 (between blocks) can each be tightened independently.
    # Horizontal and those vertical gaps are 50% of the prior layout
    # (wspace 0.02 -> 0.01; within-pair / between-block hspace halved).
    n_cols = N_ENCODING_EXAMPLES // 2
    fig = plt.figure(figsize=(12, 8))
    outer = fig.add_gridspec(
        2,
        1,
        hspace=0.12,  # row2-3 gap between blocks (~50% of prior)
        left=0.06,
        right=0.99,
        top=0.97,
        bottom=0.02,
    )
    axes = np.empty((4, n_cols), dtype=object)
    for half in range(2):
        inner = outer[half].subgridspec(
            2,
            n_cols,
            hspace=0.08,  # row1-2 / row3-4 within each spatial-spectral pair
            wspace=0.01,  # 50% of prior 0.02
        )
        for row in range(2):
            for col in range(n_cols):
                axes[2 * half + row, col] = fig.add_subplot(inner[row, col])

    for i in range(N_ENCODING_EXAMPLES):
        half = i // n_cols
        col = i % n_cols
        spatial_ax = axes[2 * half, col]
        fft_ax = axes[2 * half + 1, col]

        s_label = _format_s(s_values[i])
        spatial_ax.imshow(gabor_images[i], cmap="RdBu_r", vmin=-1, vmax=1)
        spatial_ax.axis("off")
        spatial_ax.set_title(s_label, fontsize=TITLE_FS)

        F_mag = np.abs(
            np.fft.fftshift(np.fft.fft2(gabor_images[i] - np.mean(gabor_images[i])))
        )
        fft_ax.imshow(np.log1p(F_mag), cmap="viridis")
        fft_ax.axis("off")

    # Row labels on the left edge: spatial above, spectral below, per block.
    for half in range(2):
        for row_offset, row_label in ((0, "Spatial"), (1, "Spectral")):
            ax = axes[2 * half + row_offset, 0]
            ax.text(
                -0.12,
                0.5,
                row_label,
                transform=ax.transAxes,
                fontsize=TITLE_FS,
                rotation=90,
                va="center",
                ha="center",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_encode_decode_error(out_path: Path) -> None:
    s_original = np.logspace(np.log10(S_MIN), np.log10(S_MAX), N_ERROR_TEST)
    s_reconstructed = np.zeros(N_ERROR_TEST)

    for i, s in enumerate(s_original):
        img, _, _ = embed_eigenfrequency_wavelet(s)
        s_rec, _, _ = extract_eigenfrequency_from_wavelet(img)
        s_reconstructed[i] = s_rec

    rel_error = (s_reconstructed - s_original) / s_original * 100

    # Two square panels: width ~= 2 * height (was 14x5; shortened to 10x5).
    label_fs = 14
    title_fs = 15
    legend_fs = 12
    tick_fs = 12

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(s_original, s_original, "b-", linewidth=1.5, label="Original s")
    ax1.plot(
        s_original,
        s_reconstructed,
        "r.",
        markersize=2,
        alpha=0.6,
        label="Reconstructed s",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Original s", fontsize=label_fs)
    ax1.set_ylabel("s value", fontsize=label_fs)
    ax1.set_title("Original vs Reconstructed s", fontsize=title_fs)
    ax1.legend(fontsize=legend_fs)
    ax1.tick_params(axis="both", labelsize=tick_fs)
    ax1.grid(True, alpha=0.3)
    ax1.set_box_aspect(1)

    ax2.plot(s_original, np.abs(rel_error), "g.", markersize=2, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("s value", fontsize=label_fs)
    ax2.set_ylabel("|Relative error| (%)", fontsize=label_fs)
    ax2.set_title("Relative Error vs s", fontsize=title_fs)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color="r", linestyle="--", alpha=0.4, label="1% threshold")
    ax2.legend(fontsize=legend_fs)
    ax2.tick_params(axis="both", labelsize=tick_fs)
    ax2.set_box_aspect(1)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    abs_err = np.abs(rel_error)
    print(f"Wrote {out_path}")
    print(f"Reconstruction error over {N_ERROR_TEST} log-spaced s values [{S_MIN}, {S_MAX}]:")
    print(f"  Mean |error|:   {np.mean(abs_err):.4f} %")
    print(f"  Median |error|: {np.median(abs_err):.4f} %")
    print(f"  Max |error|:    {np.max(abs_err):.4f} %")
    print(f"  Std |error|:    {np.std(abs_err):.4f} %")
    print(f"  Fraction < 1%:  {np.mean(abs_err < 1.0) * 100:.1f} %")


def main() -> None:
    plot_encoding_examples(OUT_DIR / "eigenfrequency_encoding.png")
    plot_encode_decode_error(OUT_DIR / "eigenvalue_encode_decode_error_test.png")


if __name__ == "__main__":
    main()
