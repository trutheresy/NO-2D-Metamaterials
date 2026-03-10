import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def resolve_pt_folder(folder: Path) -> Path:
    if (folder / "reduced_indices.pt").exists():
        return folder
    pt_dirs = sorted([p for p in folder.iterdir() if p.is_dir() and p.name.endswith("_pt")])
    if not pt_dirs:
        raise FileNotFoundError(f"No pt dataset folder found under: {folder}")
    return pt_dirs[-1]


def load_pt_data(pt_folder: Path):
    geometries = torch.load(pt_folder / "geometries_full.pt", map_location="cpu", weights_only=False)
    wavevectors = torch.load(pt_folder / "wavevectors_full.pt", map_location="cpu", weights_only=False)
    eigenvalues = torch.load(pt_folder / "eigenvalue_data_full.pt", map_location="cpu", weights_only=False)
    waveforms = torch.load(pt_folder / "waveforms_full.pt", map_location="cpu", weights_only=False)
    band_fft = torch.load(pt_folder / "band_fft_full.pt", map_location="cpu", weights_only=False)
    eigenfrequency_fft = torch.load(pt_folder / "eigenfrequency_fft_full.pt", map_location="cpu", weights_only=False)
    displacements = torch.load(pt_folder / "displacements_dataset.pt", map_location="cpu", weights_only=False)
    reduced_indices = torch.load(pt_folder / "reduced_indices.pt", map_location="cpu", weights_only=False)

    if not isinstance(displacements, torch.utils.data.TensorDataset):
        raise TypeError("displacements_dataset.pt is not a TensorDataset")
    if not isinstance(reduced_indices, list):
        raise TypeError("reduced_indices.pt is not a list")
    if len(displacements.tensors) != 4:
        raise ValueError("Expected 4 displacement tensors (xr, xi, yr, yi)")

    return (
        geometries.detach().cpu().numpy(),
        wavevectors.detach().cpu().numpy(),
        eigenvalues.detach().cpu().numpy(),
        waveforms.detach().cpu().numpy(),
        band_fft.detach().cpu().numpy(),
        eigenfrequency_fft.detach().cpu().numpy(),
        [t.detach().cpu().numpy() for t in displacements.tensors],
        reduced_indices,
    )


def make_scalar_image(value: float, size: int = 32) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def find_sample_row(reduced_indices: List[Tuple[int, int, int]], d: int, w: int, b: int) -> int:
    target = (int(d), int(w), int(b))
    for i, tup in enumerate(reduced_indices):
        if tuple(map(int, tup)) == target:
            return i
    raise ValueError(f"(d,w,b)={target} not found in reduced_indices")


def select_unique_geometry_and_band(
    reduced_indices: List[Tuple[int, int, int]], n_select: int
) -> List[Tuple[int, int, int]]:
    pair_to_w = {}
    for tup in reduced_indices:
        d, w, b = map(int, tup)
        key = (d, b)
        if key not in pair_to_w:
            pair_to_w[key] = []
        pair_to_w[key].append(w)

    rng = np.random.default_rng()
    selected: List[Tuple[int, int, int]] = []
    seen_d = set()
    seen_b = set()
    for tup in reduced_indices:
        d, _, b = map(int, tup)
        if d in seen_d or b in seen_b:
            continue
        ws = pair_to_w[(d, b)]
        w = int(rng.choice(ws))
        selected.append((d, w, b))
        seen_d.add(d)
        seen_b.add(b)
        if len(selected) >= n_select:
            break
    if len(selected) < n_select:
        raise RuntimeError(
            f"Could not find {n_select} tuples with unique geometry and unique band. "
            f"Found only {len(selected)}."
        )
    return selected


def plot_one(
    pt_folder: Path,
    datafolder_name: str,
    geometries: np.ndarray,
    wavevectors: np.ndarray,
    eigenvalues: np.ndarray,
    waveforms: np.ndarray,
    band_fft: np.ndarray,
    eigenfrequency_fft: np.ndarray,
    disp_tensors: List[np.ndarray],
    reduced_indices: List[Tuple[int, int, int]],
    d: int,
    w: int,
    b: int,
    out_stem: str,
):
    row = find_sample_row(reduced_indices, d, w, b)
    n_pix = int(geometries.shape[1])

    geom_img = geometries[d].astype(np.float32)
    wv_img = waveforms[w].astype(np.float32)
    band_img = band_fft[b].astype(np.float32)
    ef_img = eigenfrequency_fft[d, w, b].astype(np.float32)

    xr, xi, yr, yi = [arr[row].astype(np.float32) for arr in disp_tensors]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"{datafolder_name} | (d,w,b)=({d},{w},{b}) | sample_row={row}", fontsize=14)

    top = [
        (geom_img, "Geometry"),
        (wv_img, "Encoded Waveform"),
        (band_img, "Encoded Band"),
        (ef_img, "Encoded Eigenfreq"),
    ]
    bottom = [
        (xr, "Disp X Real"),
        (xi, "Disp X Imag"),
        (yr, "Disp Y Real"),
        (yi, "Disp Y Imag"),
    ]

    for j, (img, ttl) in enumerate(top):
        ax = axes[0, j]
        im = ax.imshow(img, origin="lower", cmap="viridis")
        ax.set_title(ttl)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j, (img, ttl) in enumerate(bottom):
        ax = axes[1, j]
        im = ax.imshow(img, origin="lower", cmap="viridis")
        ax.set_title(ttl)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = pt_folder / f"{out_stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot 2x4 panel for one (d,w,b) from PT dataset.")
    parser.add_argument("folder", type=str, help="Dataset folder or *_pt folder")
    parser.add_argument("--d", type=int, default=None, help="Design index")
    parser.add_argument("--w", type=int, default=None, help="Wavevector index")
    parser.add_argument("--b", type=int, default=None, help="Band index")
    parser.add_argument("--first-n-reduced", type=int, default=0, help="Generate N plots from first N reduced_indices tuples")
    parser.add_argument(
        "--first-n-unique-geometries",
        type=int,
        default=0,
        help="Generate N plots with unique geometry and unique band in reduced_indices",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    pt_folder = resolve_pt_folder(folder)
    datafolder_name = pt_folder.name
    geometries, wavevectors, eigenvalues, waveforms, band_fft, eigenfrequency_fft, disp_tensors, reduced_indices = load_pt_data(pt_folder)

    if args.first_n_unique_geometries and args.first_n_unique_geometries > 0:
        selected = select_unique_geometry_and_band(reduced_indices, args.first_n_unique_geometries)

        for i, (d, w, b) in enumerate(selected):
            stem = f"g{d:03d}_wv{w:03d}_b{b}"
            plot_one(
                pt_folder,
                datafolder_name,
                geometries,
                wavevectors,
                eigenvalues,
                waveforms,
                band_fft,
                eigenfrequency_fft,
                disp_tensors,
                reduced_indices,
                d,
                w,
                b,
                stem,
            )
    elif args.first_n_reduced and args.first_n_reduced > 0:
        n = min(args.first_n_reduced, len(reduced_indices))
        for i in range(n):
            d, w, b = map(int, reduced_indices[i])
            stem = f"g{d:03d}_wv{w:03d}_b{b}"
            plot_one(
                pt_folder,
                datafolder_name,
                geometries,
                wavevectors,
                eigenvalues,
                waveforms,
                band_fft,
                eigenfrequency_fft,
                disp_tensors,
                reduced_indices,
                d,
                w,
                b,
                stem,
            )
    else:
        if args.d is None or args.w is None or args.b is None:
            raise ValueError("Provide --d --w --b or use --first-n-reduced")
        stem = f"g{args.d:03d}_wv{args.w:03d}_b{args.b}"
        plot_one(
            pt_folder,
            datafolder_name,
            geometries,
            wavevectors,
            eigenvalues,
            waveforms,
            band_fft,
            eigenfrequency_fft,
            disp_tensors,
            reduced_indices,
            args.d,
            args.w,
            args.b,
            stem,
        )


if __name__ == "__main__":
    main()
