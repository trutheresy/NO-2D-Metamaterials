"""
Recompute band embeddings (default bands 1..6), save arrays, and plot
spatial / FFT mosaics plus FFT-log cosine similarity.

Supports:
  --encoding sinusoidal : embed_band_sinusoidal (paper I_b; default)
  --encoding wavelet    : embed_integer_wavelet (dataset band_fft_full.pt)

Similarity is cosine similarity on log10|FFT| spectra (translation-invariant).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import NO_utilities as NU

ROOT = Path(__file__).resolve().parent
DEFAULT_PT_DIR = ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt"
OUT_PARENT = ROOT / "PLOTS/band_embedding_inspection"

DEFAULT_BANDS = np.arange(1, 7, dtype=np.int32)


def compute_embeddings(
    bands: np.ndarray,
    size: int,
    encoding: str = "sinusoidal",
    freq_range: float = 2.0,
) -> np.ndarray:
    if encoding == "sinusoidal":
        emb = NU.embed_band_sinusoidal(bands, size=size, verbose=False)
    elif encoding == "wavelet":
        emb = NU.embed_integer_wavelet(bands, size=size, freq_range=freq_range)
    else:
        raise ValueError(f"Unknown encoding: {encoding!r} (expected 'sinusoidal' or 'wavelet')")
    return np.asarray(emb, dtype=np.float32)


def fft_log_cosine_similarity_matrix(fft_log_mag: np.ndarray) -> np.ndarray:
    flat = fft_log_mag.reshape(fft_log_mag.shape[0], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    unit = flat / norms
    return (unit @ unit.T).astype(np.float32)


def mean_offdiag_similarity(sim: np.ndarray) -> float:
    off = sim.copy()
    np.fill_diagonal(off, np.nan)
    return float(np.nanmean(off))


def build_distinctiveness_report(sim: np.ndarray, bands: np.ndarray, threshold: float) -> dict:
    n = sim.shape[0]
    off_diag = sim.copy()
    np.fill_diagonal(off_diag, -np.inf)
    max_off = np.max(off_diag, axis=1)
    argmax_off = np.argmax(off_diag, axis=1)

    near_pairs = []
    for i in range(n):
        j = int(argmax_off[i])
        if max_off[i] >= threshold:
            near_pairs.append(
                {
                    "band_i": int(bands[i]),
                    "band_j": int(bands[j]),
                    "cosine": float(max_off[i]),
                }
            )

    flagged = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(sim[i, j])
            if c >= threshold:
                flagged.append(
                    {
                        "band_i": int(bands[i]),
                        "band_j": int(bands[j]),
                        "cosine": c,
                    }
                )

    return {
        "similarity_metric": "cosine on mean-centered log10|FFT|",
        "avg_similarity": mean_offdiag_similarity(sim),
        "near_duplicate_threshold": threshold,
        "per_band_nearest_neighbor": near_pairs,
        "pairs_above_threshold": sorted(flagged, key=lambda d: -d["cosine"]),
        "global_max_offdiag_cos": float(np.max(max_off)),
        "global_mean_max_offdiag_cos": float(np.mean(max_off)),
    }


def compute_fft_magnitude(embeddings: np.ndarray) -> np.ndarray:
    fft = np.fft.fftshift(np.fft.fft2(embeddings), axes=(-2, -1))
    return np.abs(fft).astype(np.float32)


def compute_fft_log_magnitude(fft_mag: np.ndarray) -> np.ndarray:
    return np.log10(fft_mag + 1e-12).astype(np.float32)


def compute_fft_spectral_energy(fft_mag: np.ndarray) -> np.ndarray:
    return np.sum(fft_mag**2, axis=(-2, -1)).astype(np.float64)


def _encoding_label(encoding: str) -> str:
    return "Sinusoidal (paper I_b)" if encoding == "sinusoidal" else "Wavelet (embed_integer_wavelet)"


def save_arrays(
    out_dir: Path,
    embeddings: np.ndarray,
    fft_log_mag: np.ndarray,
    bands: np.ndarray,
    embed_cfg: dict,
) -> None:
    per_band = out_dir / "per_band"
    per_band.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "embeddings_all.npz",
        embeddings=embeddings,
        fft_log_magnitude=fft_log_mag,
        bands=bands.astype(np.int32),
        size=np.int32(embed_cfg["size"]),
    )
    torch.save(
        {
            "embeddings": torch.from_numpy(embeddings),
            "fft_log_magnitude": torch.from_numpy(fft_log_mag),
            "bands": torch.from_numpy(bands.astype(np.int32)),
            "embed_config": embed_cfg,
        },
        out_dir / "embeddings_all.pt",
    )

    manifest = []
    for i, b in enumerate(bands):
        np.save(per_band / f"b{int(b):02d}.npy", embeddings[i])
        np.save(per_band / f"b{int(b):02d}_fft_log_mag.npy", fft_log_mag[i])
        manifest.append({"index": i, "band": int(b)})

    with open(out_dir / "band_index_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"embed_config": embed_cfg, "bands": manifest}, f, indent=2)


def save_similarity_heatmap(out_path: Path, sim: np.ndarray, bands: np.ndarray, dpi: int) -> None:
    n = sim.shape[0]
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=dpi, constrained_layout=True)
    im = ax.imshow(sim, cmap="viridis", vmin=-1.0, vmax=1.0, origin="lower", interpolation="nearest")
    labels = [str(int(b)) for b in bands]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("band j", fontsize=12)
    ax.set_ylabel("band i", fontsize=12)
    ax.set_title(f"FFT log-magnitude cosine similarity ({n} × {n})", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_band_row_plot(
    out_path: Path,
    patches: np.ndarray,
    bands: np.ndarray,
    *,
    suptitle: str,
    cmap: str,
    symmetric: bool,
    per_patch_vmax: bool,
    dpi: int,
) -> None:
    n = patches.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8), dpi=dpi, squeeze=False)
    axes = axes[0]

    if per_patch_vmax:
        global_vmin = global_vmax = None
    elif symmetric:
        global_vmax = max(float(np.abs(patches).max()), 1e-6)
        global_vmin = -global_vmax
    else:
        global_vmin = 0.0
        global_vmax = max(float(patches.max()), 1e-6)

    for i in range(n):
        ax = axes[i]
        patch = patches[i]
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
        ax.set_title(f"Band {int(bands[i])}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_spectral_energy_bars(out_path: Path, energies: np.ndarray, bands: np.ndarray, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=dpi, constrained_layout=True)
    x = np.arange(len(bands))
    ax.bar(x, energies, color="#3498db", edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(b)) for b in bands])
    ax.set_xlabel("Band")
    ax.set_ylabel(r"Total spectral energy  $\sum |F|^2$")
    ax.set_title(f"FFT spectral energy across {len(bands)} bands (linear |F|, not log)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pt-dir", type=Path, default=DEFAULT_PT_DIR, help="Optional dataset folder for --compare-stored")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: auto)")
    p.add_argument(
        "--encoding",
        choices=("sinusoidal", "wavelet"),
        default="sinusoidal",
        help="Band embedding: paper sinusoid (default) or Gabor wavelet",
    )
    p.add_argument("--size", type=int, default=32, help="Embedding patch size")
    p.add_argument("--freq-range", type=float, default=2.0, help="embed_integer_wavelet freq_range (wavelet only)")
    p.add_argument(
        "--bands",
        type=int,
        nargs="+",
        default=None,
        help="Band indices to embed (default: 1 2 3 4 5 6)",
    )
    p.add_argument(
        "--compare-stored",
        action="store_true",
        help="Compare against band_fft_full.pt if present (wavelet encoding only)",
    )
    p.add_argument(
        "--per-patch-vmax",
        action="store_true",
        help="Use per-patch color scale (default: shared global scale)",
    )
    p.add_argument("--dpi", type=int, default=180, help="DPI for PNG figures")
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
    encoding: str = "sinusoidal",
    size: int = 32,
    freq_range: float = 2.0,
    bands: np.ndarray | None = None,
    compare_stored: bool = False,
    per_patch_vmax: bool = False,
    dpi: int = 180,
    near_duplicate_cos: float = 0.95,
    quiet: bool = False,
) -> dict:
    band_arr = DEFAULT_BANDS.copy() if bands is None else np.asarray(bands, dtype=np.int32)
    embed_cfg: dict = {
        "encoding": encoding,
        "size": size,
        "bands": [int(b) for b in band_arr],
    }
    if encoding == "wavelet":
        embed_cfg["freq_range"] = freq_range

    if out_dir is not None:
        resolved_out = out_dir
    elif encoding == "sinusoidal":
        resolved_out = OUT_PARENT / f"sinusoidal_S{size}_b{len(band_arr)}"
    else:
        resolved_out = OUT_PARENT / f"wavelet_S{size}_fr{freq_range:g}_b{len(band_arr)}"

    resolved_out.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"Encoding: {encoding}")
        print(f"Bands: {list(map(int, band_arr))}")
        print(f"Output directory: {resolved_out}")

    embeddings = compute_embeddings(band_arr, size=size, encoding=encoding, freq_range=freq_range)
    fft_mag = compute_fft_magnitude(embeddings)
    fft_log_mag = compute_fft_log_magnitude(fft_mag)
    spectral_energy = compute_fft_spectral_energy(fft_mag)

    if compare_stored:
        if encoding != "wavelet":
            if not quiet:
                print("--compare-stored ignored (only meaningful for --encoding wavelet)")
        else:
            stored_path = pt_dir / "band_fft_full.pt"
            if stored_path.exists():
                stored = torch.load(stored_path, map_location="cpu", weights_only=False).numpy().astype(np.float32)
                # Assume stored rows are bands 1..n in order.
                n = min(stored.shape[0], embeddings.shape[0])
                diff = np.abs(stored[:n] - embeddings[:n])
                compare = {
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "n_compared": int(n),
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

    save_arrays(resolved_out, embeddings, fft_log_mag, band_arr, embed_cfg)

    sim = fft_log_cosine_similarity_matrix(fft_log_mag)
    np.save(resolved_out / "fft_log_cosine_similarity.npy", sim)
    np.save(resolved_out / "fft_log_magnitude_all.npy", fft_log_mag)
    np.save(resolved_out / "fft_spectral_energy.npy", spectral_energy)
    report = build_distinctiveness_report(sim, band_arr, near_duplicate_cos)
    report["embed_config"] = embed_cfg
    report["output_dir"] = str(resolved_out)
    with open(resolved_out / "distinctiveness_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    label = _encoding_label(encoding)
    save_similarity_heatmap(resolved_out / "fft_log_cosine_similarity_heatmap.png", sim, band_arr, dpi=dpi)
    save_band_row_plot(
        resolved_out / "embeddings_band_row.png",
        embeddings,
        band_arr,
        suptitle=f"{label} band embeddings (S={size})",
        cmap="RdBu_r",
        symmetric=True,
        per_patch_vmax=per_patch_vmax,
        dpi=dpi,
    )
    save_band_row_plot(
        resolved_out / "embeddings_fft_band_row.png",
        fft_log_mag,
        band_arr,
        suptitle=f"2D FFT magnitude (log₁₀|·|) of {label.lower()} band embeddings",
        cmap="magma",
        symmetric=False,
        per_patch_vmax=per_patch_vmax,
        dpi=dpi,
    )
    save_spectral_energy_bars(resolved_out / "fft_spectral_energy_bars.png", spectral_energy, band_arr, dpi=dpi)

    if not quiet:
        print(f"Wrote {embeddings.shape[0]} band embeddings to {resolved_out}")
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
    bands = None if args.bands is None else np.asarray(args.bands, dtype=np.int32)
    run_inspection(
        pt_dir=args.pt_dir,
        out_dir=args.out_dir,
        encoding=args.encoding,
        size=args.size,
        freq_range=args.freq_range,
        bands=bands,
        compare_stored=args.compare_stored,
        per_patch_vmax=args.per_patch_vmax,
        dpi=args.dpi,
        near_duplicate_cos=args.near_duplicate_cos,
    )


if __name__ == "__main__":
    main()
