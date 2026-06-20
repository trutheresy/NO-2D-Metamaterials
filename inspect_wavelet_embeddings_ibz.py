"""
Recompute 2D wavelet embeddings for all IBZ wavevectors, save arrays, and plot
a 13×25 mosaic aligned with the half-plane grid (symmetry_type='none', 25×13).

Use for visual / numerical inspection of aliasing and patch distinctiveness.

Similarity is cosine similarity on log10|FFT| spectra (translation-invariant).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import NO_utilities as NU

ROOT = Path(__file__).resolve().parent
DEFAULT_PT_DIR = ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt"
OUT_PARENT = ROOT / "PLOTS/wavelet_embedding_inspection"

N_KX = 25
N_KY = 13
PI = math.pi


def k_to_label(val: float) -> str:
    """Format k as 0 or a rational multiple of π (e.g. π/4, −π/2)."""
    r = val / PI
    n = round(r * 12)
    if abs(r - n / 12) > 2e-3:
        return f"{r:.4f}π"
    if n == 0:
        return "0"
    sign = "−" if n < 0 else ""
    n = abs(n)
    g = math.gcd(n, 12)
    n, d = n // g, 12 // g
    if n == 1:
        return f"{sign}π/{d}"
    if n == d:
        return f"{sign}π"
    return f"{sign}{n}π/{d}"


def wave_to_grid(wave: int) -> tuple[int, int]:
    """Map conventional wave index to (i_ky, i_kx) on the 13×25 IBZ meshgrid."""
    i_ky = wave // N_KX
    i_kx = wave % N_KX
    return i_ky, i_kx


def grid_to_plot_row(i_ky: int) -> int:
    """Matplotlib row: ky=0 at bottom, ky=π at top."""
    return N_KY - 1 - i_ky


def format_param_token(val: float | int) -> str:
    s = f"{val:g}"
    return s.replace(".", "p").replace("-", "m")


def output_dir_from_cfg(cfg: dict, parent: Path = OUT_PARENT) -> Path:
    folder = (
        f"fscale{format_param_token(cfg['freq_scale'])}_"
        f"offset{format_param_token(cfg['freq_offset'])}_"
        f"sigma{format_param_token(cfg['sigma_numerator'])}_"
        f"fr{format_param_token(cfg['freq_range'])}_"
        f"kx{cfg['kx_cycles']}_"
        f"ky{cfg['ky_cycles']}"
    )
    return parent / folder


def auto_output_dir(size: int, freq_range: float, **embed_kw) -> Path:
    cfg = NU.embed_2const_wavelet_params(size=size, freq_range=freq_range, **embed_kw)
    return output_dir_from_cfg(cfg)


def load_wavevectors(pt_dir: Path) -> np.ndarray:
    wv_path = pt_dir / "wavevectors_full.pt"
    if not wv_path.exists():
        raise FileNotFoundError(f"Missing {wv_path}")
    kxy = torch.load(wv_path, map_location="cpu", weights_only=False)[0].numpy().astype(np.float64)
    if kxy.shape[0] != N_KX * N_KY:
        raise ValueError(f"Expected {N_KX * N_KY} wavevectors, got {kxy.shape[0]}")
    return kxy


def compute_embeddings(
    kxy: np.ndarray, size: int, freq_range: float, embed_overrides: dict | None = None,
) -> np.ndarray:
    overrides = embed_overrides or {}
    emb = NU.embed_2const_wavelet(
        kxy[:, 0], kxy[:, 1], size=size, freq_range=freq_range, verbose=False, **overrides
    )
    return emb.astype(np.float32)


def fft_log_cosine_similarity_matrix(fft_log_mag: np.ndarray) -> np.ndarray:
    """Cosine similarity on mean-centered flattened log10|FFT| spectra."""
    flat = fft_log_mag.reshape(fft_log_mag.shape[0], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    unit = flat / norms
    return (unit @ unit.T).astype(np.float32)


def mean_offdiag_similarity(sim: np.ndarray) -> float:
    """Mean of all off-diagonal pairwise similarities."""
    off = sim.copy()
    np.fill_diagonal(off, np.nan)
    return float(np.nanmean(off))


def build_distinctiveness_report(sim: np.ndarray, kxy: np.ndarray, threshold: float) -> dict:
    n = sim.shape[0]
    off_diag = sim.copy()
    np.fill_diagonal(off_diag, -np.inf)
    max_off = np.max(off_diag, axis=1)
    argmax_off = np.argmax(off_diag, axis=1)

    near_pairs: list[dict] = []
    for i in range(n):
        j = int(argmax_off[i])
        if max_off[i] >= threshold:
            near_pairs.append(
                {
                    "wave_i": i,
                    "wave_j": j,
                    "cosine": float(max_off[i]),
                    "kx_i": float(kxy[i, 0]),
                    "ky_i": float(kxy[i, 1]),
                    "kx_j": float(kxy[j, 0]),
                    "ky_j": float(kxy[j, 1]),
                }
            )

    # unique unordered pairs above threshold
    flagged: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(sim[i, j])
            if c >= threshold:
                flagged.append({"wave_i": i, "wave_j": j, "cosine": c})

    row_stats = []
    for i_ky in range(N_KY):
        idxs = [i_ky * N_KX + i_kx for i_kx in range(N_KX)]
        sub = sim[np.ix_(idxs, idxs)]
        sub_off = sub.copy()
        np.fill_diagonal(sub_off, np.nan)
        row_stats.append(
            {
                "i_ky": i_ky,
                "ky": float(kxy[idxs[0], 1]),
                "mean_max_offdiag_cos": float(np.nanmax(sub_off, axis=1).mean()),
                "max_offdiag_cos": float(np.nanmax(sub_off)),
            }
        )

    return {
        "similarity_metric": "cosine on mean-centered log10|FFT|",
        "avg_similarity": mean_offdiag_similarity(sim),
        "near_duplicate_threshold": threshold,
        "per_wave_nearest_neighbor": near_pairs,
        "pairs_above_threshold": sorted(flagged, key=lambda d: -d["cosine"]),
        "per_ky_row": row_stats,
        "global_max_offdiag_cos": float(np.max(max_off)),
        "global_mean_max_offdiag_cos": float(np.mean(max_off)),
    }


def save_arrays(
    out_dir: Path,
    embeddings: np.ndarray,
    fft_log_mag: np.ndarray,
    kxy: np.ndarray,
    embed_cfg: dict,
) -> None:
    per_wave = out_dir / "per_wave"
    per_wave.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "embeddings_all.npz",
        embeddings=embeddings,
        fft_log_magnitude=fft_log_mag,
        kx=kxy[:, 0],
        ky=kxy[:, 1],
        wave_indices=np.arange(embeddings.shape[0], dtype=np.int32),
        size=np.int32(embed_cfg["size"]),
        freq_range=np.float32(embed_cfg["freq_range"]),
        freq_scale=np.float32(embed_cfg["freq_scale"]),
        freq_offset=np.float32(embed_cfg["freq_offset"]),
        sigma_numerator=np.float32(embed_cfg["sigma_numerator"]),
        sigma=np.float32(embed_cfg["sigma"]),
        kx_cycles=np.int32(embed_cfg["kx_cycles"]),
        ky_cycles=np.int32(embed_cfg["ky_cycles"]),
    )
    torch.save(
        {
            "embeddings": torch.from_numpy(embeddings),
            "fft_log_magnitude": torch.from_numpy(fft_log_mag),
            "kx": torch.from_numpy(kxy[:, 0].astype(np.float32)),
            "ky": torch.from_numpy(kxy[:, 1].astype(np.float32)),
            "wave_indices": torch.arange(embeddings.shape[0], dtype=torch.int32),
            "embed_config": embed_cfg,
        },
        out_dir / "embeddings_all.pt",
    )

    manifest = []
    for w in range(embeddings.shape[0]):
        i_ky, i_kx = wave_to_grid(w)
        np.save(per_wave / f"w{w:03d}.npy", embeddings[w])
        np.save(per_wave / f"w{w:03d}_fft_log_mag.npy", fft_log_mag[w])
        manifest.append(
            {
                "wave": w,
                "i_kx": i_kx,
                "i_ky": i_ky,
                "plot_row": grid_to_plot_row(i_ky),
                "plot_col": i_kx,
                "kx": float(kxy[w, 0]),
                "ky": float(kxy[w, 1]),
                "kx_label": k_to_label(kxy[w, 0]),
                "ky_label": k_to_label(kxy[w, 1]),
            }
        )

    with open(out_dir / "wave_index_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "grid": {"n_kx": N_KX, "n_ky": N_KY, "layout": "row-major meshgrid, ky=0 bottom"},
                "embed_config": embed_cfg,
                "waves": manifest,
            },
            f,
            indent=2,
        )


def save_similarity_heatmap(out_path: Path, sim: np.ndarray, dpi: int) -> None:
    n = sim.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi, constrained_layout=True)
    im = ax.imshow(sim, cmap="viridis", vmin=-1.0, vmax=1.0, origin="lower", interpolation="nearest")
    ax.set_xlabel("wave index j")
    ax.set_ylabel("wave index i")
    ax.set_title("FFT log-magnitude cosine similarity (325 × 325)")
    tick_step = 25
    ticks = list(range(0, n, tick_step))
    if ticks[-1] != n - 1:
        ticks.append(n - 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compute_fft_magnitude(embeddings: np.ndarray) -> np.ndarray:
    """Linear |FFT| (shifted) per patch."""
    fft = np.fft.fftshift(np.fft.fft2(embeddings), axes=(-2, -1))
    return np.abs(fft).astype(np.float32)


def compute_fft_log_magnitude(fft_mag: np.ndarray) -> np.ndarray:
    """log10 of linear |FFT|."""
    return np.log10(fft_mag + 1e-12).astype(np.float32)


def compute_fft_spectral_energy(fft_mag: np.ndarray) -> np.ndarray:
    """Total spectral energy per patch: sum of |F|² (linear, not log)."""
    return np.sum(fft_mag**2, axis=(-2, -1)).astype(np.float64)


def save_spectral_energy_histogram(out_path: Path, energies: np.ndarray, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi, constrained_layout=True)
    ax.hist(energies, bins=40, color="#3498db", edgecolor="white", linewidth=0.6)
    ax.set_xlabel(r"Total spectral energy  $\sum |F|^2$")
    ax.set_ylabel("Wavevector count")
    ax.set_title(f"FFT spectral energy across {len(energies)} wavevectors (linear |F|, not log)")
    ax.axvline(float(np.mean(energies)), color="#e74c3c", linestyle="--", linewidth=1.2, label=f"mean = {np.mean(energies):.3g}")
    ax.axvline(float(np.median(energies)), color="#2ecc71", linestyle=":", linewidth=1.2, label=f"median = {np.median(energies):.3g}")
    ax.legend(fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_ibz_grid_plot(
    out_path: Path,
    patches: np.ndarray,
    kxy: np.ndarray,
    per_patch_vmax: bool,
    dpi: int,
    suptitle: str,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
) -> None:
    cell = 0.42
    fig, axes = plt.subplots(
        N_KY,
        N_KX,
        figsize=(N_KX * cell, N_KY * cell),
        dpi=dpi,
        squeeze=False,
    )

    if per_patch_vmax:
        global_vmin = global_vmax = None
    elif symmetric:
        global_vmax = max(float(np.abs(patches).max()), 1e-6)
        global_vmin = -global_vmax
    else:
        global_vmin = 0.0
        global_vmax = max(float(patches.max()), 1e-6)

    for w in range(patches.shape[0]):
        i_ky, i_kx = wave_to_grid(w)
        r, c = grid_to_plot_row(i_ky), i_kx
        ax = axes[r, c]
        patch = patches[w]
        if per_patch_vmax:
            if symmetric:
                vmax = max(float(np.abs(patch).max()), 1e-6)
                vmin = -vmax
            else:
                vmin = 0.0
                vmax = max(float(patch.max()), 1e-6)
        else:
            vmin, vmax = global_vmin, global_vmax
        ax.imshow(patch, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{k_to_label(kxy[w, 0])}, {k_to_label(kxy[w, 1])}", fontsize=4.5, pad=1.2)

    fig.suptitle(suptitle, fontsize=9, y=1.01)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94, wspace=0.08, hspace=0.55)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_spatial_ibz_grid_plot(
    out_path: Path,
    embeddings: np.ndarray,
    kxy: np.ndarray,
    per_patch_vmax: bool,
    dpi: int,
) -> None:
    save_ibz_grid_plot(
        out_path,
        embeddings,
        kxy,
        per_patch_vmax=per_patch_vmax,
        dpi=dpi,
        suptitle=(
            "Wavelet embeddings on IBZ grid (25 × 13)\n"
            "bottom row: ky = 0  |  top row: ky = π  |  columns: kx from −π to +π"
        ),
        cmap="RdBu_r",
        symmetric=True,
    )


def save_fft_ibz_grid_plot(
    out_path: Path,
    fft_log_mag: np.ndarray,
    kxy: np.ndarray,
    per_patch_vmax: bool,
    dpi: int,
) -> None:
    save_ibz_grid_plot(
        out_path,
        fft_log_mag,
        kxy,
        per_patch_vmax=per_patch_vmax,
        dpi=dpi,
        suptitle=(
            "2D FFT magnitude (log₁₀|·|) of wavelet embeddings on IBZ grid (25 × 13)\n"
            "bottom row: ky = 0  |  top row: ky = π  |  columns: kx from −π to +π"
        ),
        cmap="magma",
        symmetric=False,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pt-dir", type=Path, default=DEFAULT_PT_DIR, help="Dataset folder with wavevectors_full.pt")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: auto from embed params)")
    p.add_argument("--size", type=int, default=32, help="Embedding patch size")
    p.add_argument("--freq-range", type=float, default=1.0, help="embed_2const_wavelet freq_range")
    p.add_argument(
        "--compare-stored",
        action="store_true",
        help="Compare against waveforms_full.pt if present",
    )
    p.add_argument(
        "--per-patch-vmax",
        action="store_true",
        help="Use per-patch color scale (default: shared global scale)",
    )
    p.add_argument("--dpi", type=int, default=180, help="DPI for IBZ grid PNG")
    p.add_argument(
        "--near-duplicate-cos",
        type=float,
        default=0.95,
        help="Flag pairs with cosine similarity above this threshold",
    )
    return p.parse_args()


def run_inspection(
    *,
    pt_dir: Path = DEFAULT_PT_DIR,
    out_dir: Path | None = None,
    size: int = 32,
    freq_range: float = 1.0,
    embed_overrides: dict | None = None,
    compare_stored: bool = False,
    per_patch_vmax: bool = False,
    dpi: int = 180,
    near_duplicate_cos: float = 0.95,
    quiet: bool = False,
) -> dict:
    """Run full embedding inspection; return distinctiveness report + paths."""
    overrides = embed_overrides or {}
    embed_cfg = NU.embed_2const_wavelet_params(size=size, freq_range=freq_range, **overrides)
    resolved_out = out_dir if out_dir is not None else output_dir_from_cfg(embed_cfg)
    resolved_out.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"Output directory: {resolved_out}")

    kxy = load_wavevectors(pt_dir)
    embeddings = compute_embeddings(kxy, size=size, freq_range=freq_range, embed_overrides=overrides)
    fft_mag = compute_fft_magnitude(embeddings)
    fft_log_mag = compute_fft_log_magnitude(fft_mag)
    spectral_energy = compute_fft_spectral_energy(fft_mag)

    if compare_stored:
        stored_path = pt_dir / "waveforms_full.pt"
        if stored_path.exists():
            stored = torch.load(stored_path, map_location="cpu", weights_only=False).numpy().astype(np.float32)
            diff = np.abs(stored - embeddings)
            compare = {
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "stored_path": str(stored_path),
            }
            if not quiet:
                print(f"Stored vs recomputed: max |diff| = {compare['max_abs_diff']:.6g}")
        else:
            compare = {"error": f"{stored_path} not found"}
            if not quiet:
                print(compare["error"])
        with open(resolved_out / "stored_comparison.json", "w", encoding="utf-8") as f:
            json.dump(compare, f, indent=2)

    save_arrays(resolved_out, embeddings, fft_log_mag, kxy, embed_cfg)

    sim = fft_log_cosine_similarity_matrix(fft_log_mag)
    np.save(resolved_out / "fft_log_cosine_similarity.npy", sim)
    np.save(resolved_out / "fft_log_magnitude_all.npy", fft_log_mag)
    np.save(resolved_out / "fft_spectral_energy.npy", spectral_energy)
    report = build_distinctiveness_report(sim, kxy, near_duplicate_cos)
    report["embed_config"] = embed_cfg
    report["output_dir"] = str(resolved_out)
    with open(resolved_out / "distinctiveness_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    save_similarity_heatmap(resolved_out / "fft_log_cosine_similarity_heatmap.png", sim, dpi=dpi)
    save_spatial_ibz_grid_plot(
        resolved_out / "embeddings_ibz_grid.png", embeddings, kxy, per_patch_vmax=per_patch_vmax, dpi=dpi
    )
    save_fft_ibz_grid_plot(
        resolved_out / "embeddings_fft_ibz_grid.png", fft_log_mag, kxy, per_patch_vmax=per_patch_vmax, dpi=dpi
    )
    save_spectral_energy_histogram(resolved_out / "fft_spectral_energy_histogram.png", spectral_energy, dpi=dpi)

    if not quiet:
        print(f"Wrote {embeddings.shape[0]} embeddings to {resolved_out}")
        print(
            f"  FFT-log global mean max off-diag cosine = {report['global_mean_max_offdiag_cos']:.4f}  "
            f"(max {report['global_max_offdiag_cos']:.4f})"
        )
        n_pairs = len(report["pairs_above_threshold"])
        if n_pairs:
            print(f"  {n_pairs} pairs with cosine >= {near_duplicate_cos}")

    return report


def main() -> None:
    args = parse_args()
    overrides = {}
    run_inspection(
        pt_dir=args.pt_dir,
        out_dir=args.out_dir,
        size=args.size,
        freq_range=args.freq_range,
        embed_overrides=overrides,
        compare_stored=args.compare_stored,
        per_patch_vmax=args.per_patch_vmax,
        dpi=args.dpi,
        near_duplicate_cos=args.near_duplicate_cos,
    )


if __name__ == "__main__":
    main()
