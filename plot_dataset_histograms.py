import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def resolve_pt_folder(folder: Path) -> Path:
    if (folder / "eigenvalue_data_full.pt").exists():
        return folder
    pt_dirs = sorted([p for p in folder.iterdir() if p.is_dir() and p.name.endswith("_pt")])
    if not pt_dirs:
        raise FileNotFoundError(f"No *_pt dataset folder found under: {folder}")
    return pt_dirs[-1]


def save_eigenfrequency_uniform_histogram(
    pt_folder: Path, sample_max: int = 200_000, seed: int = 0
) -> Path:
    """
    Histogram of sampled values from ``eigenfrequency_uniform_full.pt`` (matches
    ``run_generate_pt_folder_stats_all.save_hist`` style used for other tensors).
    """
    path = pt_folder / "eigenfrequency_uniform_full.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    t = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(t):
        raise TypeError(f"Expected tensor in {path}, got {type(t)}")
    flat = t.reshape(-1).detach()
    n = int(flat.numel())
    rng = np.random.default_rng(seed)
    if n <= sample_max:
        idx_t = torch.arange(n, dtype=torch.long)
    else:
        idx = np.sort(rng.choice(n, size=sample_max, replace=False))
        idx_t = torch.from_numpy(idx.astype(np.int64))
    vals = flat[idx_t].to(torch.float64).cpu().numpy()
    finite = vals[np.isfinite(vals)] if vals.size > 0 else np.array([], dtype=np.float64)
    out_png = pt_folder / "hist_eigenfrequency_uniform_full.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    if finite.size > 0:
        bins = max(40, min(220, int(round(math.sqrt(finite.size)))))
        ax.hist(finite, bins=bins, alpha=0.85)
    ax.set_title("eigenfrequency_uniform_full.pt sampled value histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {out_png}")
    return out_png


def save_eigenfrequency_fft_histogram(
    pt_folder: Path, sample_max: int = 200_000, seed: int = 0
) -> Path:
    """Sampled-value histogram for ``eigenfrequency_fft_full.pt`` (stats-script style)."""
    path = pt_folder / "eigenfrequency_fft_full.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    t = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(t):
        raise TypeError(f"Expected tensor in {path}, got {type(t)}")
    flat = t.reshape(-1).detach()
    n = int(flat.numel())
    rng = np.random.default_rng(seed)
    if n <= sample_max:
        idx_t = torch.arange(n, dtype=torch.long)
    else:
        idx = np.sort(rng.choice(n, size=sample_max, replace=False))
        idx_t = torch.from_numpy(idx.astype(np.int64))
    vals = flat[idx_t].to(torch.float64).cpu().numpy()
    finite = vals[np.isfinite(vals)] if vals.size > 0 else np.array([], dtype=np.float64)
    out_png = pt_folder / "hist_eigenfrequency_fft_full.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    if finite.size > 0:
        bins = max(40, min(220, int(round(math.sqrt(finite.size)))))
        ax.hist(finite, bins=bins, alpha=0.85)
    ax.set_title("eigenfrequency_fft_full.pt sampled value histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {out_png}")
    return out_png


def plot_eigenfrequency_hist(pt_folder: Path):
    eigenvalues = torch.load(pt_folder / "eigenvalue_data_full.pt", map_location="cpu", weights_only=False)
    vals = eigenvalues.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=200)
    ax.set_title("Eigenfrequency Commonality")
    ax.set_xlabel("Eigenfrequency")
    ax.set_ylabel("Count")
    fig.tight_layout()

    out_path = pt_folder / "hist_eigenfrequency_commonality.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {out_path}")


def plot_reduced_indices_hists(pt_folder: Path):
    reduced_indices = torch.load(pt_folder / "reduced_indices.pt", map_location="cpu", weights_only=False)
    idx = np.asarray(reduced_indices, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError(f"Expected reduced_indices as Nx3, got shape {idx.shape}")

    labels = ["design_idx (d)", "wavevector_idx (w)", "band_idx (b)"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j in range(3):
        n_unique = int(np.unique(idx[:, j]).size)
        bins = max(n_unique, 1)
        axes[j].hist(idx[:, j], bins=bins)
        axes[j].set_title(labels[j])
        axes[j].set_xlabel("Index value")
        axes[j].set_ylabel("Count")

    fig.suptitle("Reduced Indices Histograms", fontsize=14)
    fig.tight_layout()
    out_path = pt_folder / "hist_reduced_indices_1x3.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot requested histograms for one dataset folder.")
    parser.add_argument("datafolder", type=str, help="Dataset folder or *_pt folder")
    args = parser.parse_args()

    folder = Path(args.datafolder)
    pt_folder = resolve_pt_folder(folder)
    print(f"USING_PT_FOLDER {pt_folder}")

    plot_eigenfrequency_hist(pt_folder)
    plot_reduced_indices_hists(pt_folder)


if __name__ == "__main__":
    main()
