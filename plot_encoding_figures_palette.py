"""
Regenerate wavelet encoding palette figures using NO_utilities embeddings.

Writes formatted figures to PLOTS/encoding_figures_palette/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, FixedLocator

import NO_utilities as NU

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "PLOTS" / "encoding_figures_palette"

# Larger type; leave headroom so multi-line kx/ky titles do not collide.
TITLE_FONTSIZE = 20
CBAR_FONTSIZE = 16
CBAR_WIDTH = 0.012
CBAR_GAP = 0.01
CBAR_N_TICKS = 6  # always this many, evenly spaced, nice round values
# Mantissas for nice steps f×10^k (richer set → finer ticks when span sits between 0.25 and 0.5).
_NICE_FRAC = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def ibz_half_plane_wavevectors(n_kx: int = 25, n_ky: int = 13, a: float = 1.0) -> np.ndarray:
    """Asymmetric IBZ half-plane used in the paper / 2d-dispersion-py ('none')."""
    X, Y = np.meshgrid(
        np.linspace(-np.pi / a, np.pi / a, n_kx),
        np.linspace(0.0, np.pi / a, n_ky),
    )
    return np.column_stack([X.ravel(), Y.ravel()])


def wavevector_title(kx: float, ky: float, prefix: str | None = None) -> str:
    """Two-line kx / ky subtitle so larger fonts do not collide between subplots."""
    body = f"$k_x$={kx:.3f}\n$k_y$={ky:.3f}"
    if prefix:
        return f"{prefix}\n{body}"
    return body


def _nice_step(raw: float) -> float:
    """Smallest nice step (1/2/2.5/5×10^k) >= raw."""
    if raw <= 0 or not np.isfinite(raw):
        return 1.0
    exp = np.floor(np.log10(raw))
    frac = raw / (10.0**exp)
    for f in _NICE_FRAC:
        if frac <= f + 1e-12:
            return float(f * 10.0**exp)
    return float(10.0 * 10.0**exp)


def _next_nice_step(step: float) -> float:
    """Next larger nice step after ``step``."""
    exp = np.floor(np.log10(step))
    frac = step / (10.0**exp)
    for f in _NICE_FRAC:
        if f > frac + 1e-12:
            return float(f * 10.0**exp)
    return float(10.0 * 10.0**exp)


def _clim_and_ticks(
    arrays: list[np.ndarray], *, symmetric: bool, n_ticks: int = CBAR_N_TICKS
) -> tuple[tuple[float, float], np.ndarray]:
    """
    Shared color limits plus exactly ``n_ticks`` evenly spaced nice tick values.

    Clim is snapped outward so endpoints and step are nice (e.g. 0.25, 0.50).
    """
    n_int = n_ticks - 1
    if symmetric:
        v = float(max(np.max(np.abs(a)) for a in arrays))
        if v == 0.0:
            v = 1.0
        step = _nice_step((2.0 * v) / n_int)
        half = (n_int // 2) * step
        while half + 1e-15 < v:
            step = _next_nice_step(step)
            half = (n_int // 2) * step
        clim = (-half, half)
        ticks = np.linspace(-half, half, n_ticks)
    else:
        vmin = float(min(np.min(a) for a in arrays))
        vmax = float(max(np.max(a) for a in arrays))
        if vmin == vmax:
            vmax = vmin + 1.0
        step = _nice_step((vmax - vmin) / n_int)
        lo = np.floor(vmin / step) * step
        hi = lo + n_int * step
        while hi + 1e-15 < vmax:
            step = _next_nice_step(step)
            lo = np.floor(vmin / step) * step
            hi = lo + n_int * step
        clim = (float(lo), float(hi))
        ticks = np.linspace(lo, hi, n_ticks)
    return clim, ticks


def _apply_shared_clim_and_ticks(
    images: list, arrays: list[np.ndarray], *, symmetric: bool
) -> np.ndarray:
    clim, ticks = _clim_and_ticks(arrays, symmetric=symmetric)
    for im in images:
        im.set_clim(*clim)
    return ticks


def _add_colorbar_matching_images(
    fig: plt.Figure,
    mappable,
    image_axes: list[plt.Axes],
    image_mappables: list,
    ticks: np.ndarray,
    *,
    fontsize: float = CBAR_FONTSIZE,
) -> None:
    """
    Place a free-standing colorbar (does not shrink any subplot) whose height
    matches one image pixel grid, centered on the image block.
    Uses a fixed set of evenly spaced nice ticks (same count on every figure).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [
        im.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        for im in image_mappables
    ]
    block_y0 = min(b.y0 for b in bboxes)
    block_y1 = max(b.y1 for b in bboxes)
    x_right = max(b.x1 for b in bboxes)
    h = bboxes[0].height
    y0 = 0.5 * (block_y0 + block_y1) - 0.5 * h
    cbar_ax = fig.add_axes([x_right + CBAR_GAP, y0, CBAR_WIDTH, h])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.ax.minorticks_off()
    cbar.locator = FixedLocator(ticks)
    cbar.formatter = FormatStrFormatter("%.2f")
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=fontsize, which="major")


def _layout(fig: plt.Figure, *, hspace: float = 0.35) -> None:
    # Leave room on the right for a free-standing colorbar (no subplot shrinks).
    fig.subplots_adjust(
        left=0.02,
        right=0.88,
        bottom=0.08,
        top=0.86,
        wspace=0.22,
        hspace=hspace,
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_1d_band_row(
    embeddings: np.ndarray,
    bands: list[int],
    titles: list[str],
    path: Path,
    *,
    symmetric_clim: bool = True,
) -> None:
    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(16, 3.8))
    images = []
    image_axes = []
    for i, ax in enumerate(axes):
        im = ax.imshow(embeddings[i], cmap="viridis")
        images.append(im)
        image_axes.append(ax)
        ax.axis("off")
        ax.set_box_aspect(1)
        ax.set_title(titles[i], fontsize=TITLE_FONTSIZE)
    _ticks = _apply_shared_clim_and_ticks(images, list(embeddings), symmetric=symmetric_clim)
    _layout(fig)
    _add_colorbar_matching_images(fig, images[0], image_axes, images, _ticks)
    _save(fig, path)


def plot_2d_grid(
    embeddings: np.ndarray,
    pairs: np.ndarray,
    path: Path,
    *,
    ncols: int = 6,
    title_prefix: str | None = None,
    figsize_scale: float = 3.2,
    symmetric_clim: bool = True,
) -> None:
    n = len(pairs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize_scale + 1.0, nrows * (figsize_scale + 0.85)),
    )
    axes_flat = np.atleast_1d(axes).ravel()
    images = []
    image_axes = []
    for i, ax in enumerate(axes_flat):
        if i < n:
            im = ax.imshow(embeddings[i], cmap="viridis")
            images.append(im)
            image_axes.append(ax)
            ax.axis("off")
            ax.set_box_aspect(1)
            kx, ky = pairs[i]
            ax.set_title(
                wavevector_title(float(kx), float(ky), prefix=title_prefix),
                fontsize=TITLE_FONTSIZE,
                pad=6,
            )
        else:
            ax.axis("off")
    _ticks = _apply_shared_clim_and_ticks(
        images, list(embeddings[:n]), symmetric=symmetric_clim
    )
    _layout(fig, hspace=0.75 if nrows > 1 else 0.35)
    _add_colorbar_matching_images(fig, images[0], image_axes, images, _ticks)
    _save(fig, path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bands = np.arange(1, 7)

    # --- 1D band encodings (NO_utilities.embed_integer_wavelet) ---
    wavelet_1d = NU.embed_integer_wavelet(bands, size=32)
    wavelet_1d_fft = np.array(
        [np.fft.fftshift(np.abs(np.fft.fft2(emb))) for emb in wavelet_1d]
    )
    plot_1d_band_row(
        wavelet_1d,
        list(bands),
        [f"Band {b}" for b in bands],
        OUT_DIR / "1D_wavelet_encoding.png",
    )
    plot_1d_band_row(
        np.log1p(wavelet_1d_fft),
        list(bands),
        [f"Band {b}" for b in bands],
        OUT_DIR / "1D_wavelet_encoding_fft.png",
        symmetric_clim=False,
    )

    wavevectors = ibz_half_plane_wavevectors()

    # --- 6-panel 2D (paper Methods style) ---
    # Prefer the wavevectors shown in the existing Methods figure when present on the IBZ grid.
    paper_pairs = np.array(
        [
            [3.141, 2.617],
            [1.833, 0.000],
            [-0.523, 3.141],
            [-2.879, 2.617],
            [-0.262, 0.785],
            [0.785, 0.262],
        ],
        dtype=float,
    )
    pairs_6 = np.array(
        [wavevectors[np.linalg.norm(wavevectors - p, axis=1).argmin()] for p in paper_pairs]
    )
    # NO_utilities expects kx, ky in radians (same as wavevectors_full.pt).
    emb_6 = NU.embed_2const_wavelet(pairs_6[:, 0], pairs_6[:, 1], size=32, verbose=False)
    fft_6 = np.array([np.fft.fftshift(np.abs(np.fft.fft2(emb))) for emb in emb_6])
    plot_2d_grid(emb_6, pairs_6, OUT_DIR / "2D_wavelet_encoding.png", ncols=6)
    plot_2d_grid(
        np.log1p(fft_6),
        pairs_6,
        OUT_DIR / "2D_wavelet_encoding_fft.png",
        ncols=6,
        title_prefix="FFT",
        symmetric_clim=False,
    )

    # --- Larger palette grid (seed 42, 30 samples) ---
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(wavevectors), size=30, replace=False)
    pairs_30 = wavevectors[indices]
    emb_30 = NU.embed_2const_wavelet(pairs_30[:, 0], pairs_30[:, 1], size=32, verbose=False)
    fft_30 = np.array([np.fft.fftshift(np.abs(np.fft.fft2(emb))) for emb in emb_30])
    plot_2d_grid(
        emb_30,
        pairs_30,
        OUT_DIR / "2D_wavelet_encoding_palette.png",
        ncols=6,
        figsize_scale=3.0,
    )
    plot_2d_grid(
        np.log1p(fft_30),
        pairs_30,
        OUT_DIR / "2D_wavelet_encoding_fft_palette.png",
        ncols=6,
        title_prefix="FFT",
        figsize_scale=3.0,
        symmetric_clim=False,
    )

    print(f"\nAll figures written under:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
