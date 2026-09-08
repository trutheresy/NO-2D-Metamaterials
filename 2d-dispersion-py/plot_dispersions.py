"""
Plot dispersion for full PT datasets (float16-first path).

Expected files in data_dir:
- geometries_full.pt          (N_struct, N_pix, N_pix)
- wavevectors_full.pt         (N_struct, N_wv, 2)
- eigenvalue_data_full.pt     (N_struct, N_wv, N_eig)

Contour points are taken from dataset wavevectors on the p4mm Γ–X–M–Γ path (no
interpolation onto synthetic contour samples).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from plot_dispersion_with_eigenfrequencies_reduced_set import (
    plot_dispersion_on_contour,
    select_p4mm_contour_from_grid,
)


def load_full_pt_dataset(data_dir: Path) -> dict:
    geometries = torch.load(data_dir / "geometries_full.pt", map_location="cpu").to(torch.float16).numpy()
    wavevectors = torch.load(data_dir / "wavevectors_full.pt", map_location="cpu").to(torch.float16).numpy()
    eigenvalue_data = torch.load(data_dir / "eigenvalue_data_full.pt", map_location="cpu").to(torch.float16).numpy()
    return {
        "designs": geometries.astype(np.float16, copy=False),
        "wavevectors": wavevectors.astype(np.float16, copy=False),
        "eigenvalue_data": eigenvalue_data.astype(np.float16, copy=False),
    }


def main(
    data_dir: str,
    n_structs: int | None = None,
    title: str = "",
    output_dir: str | None = None,
) -> None:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = load_full_pt_dataset(data_path)
    designs = data["designs"]  # (N_struct, N_pix, N_pix)
    wavevectors_all = data["wavevectors"]  # (N_struct, N_wv, 2)
    eigenvalues_all = data["eigenvalue_data"]  # (N_struct, N_wv, N_eig)

    n_total = int(designs.shape[0])
    n_plot = n_total if n_structs is None else min(int(n_structs), n_total)

    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        output_dir = Path.cwd() / "PLOTS" / f"{data_path.name}_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    contour_indices, contour_param, contour_info = select_p4mm_contour_from_grid(wavevectors_all)
    print(
        f"Contour grid pts : {contour_info['n_contour_points']} along path "
        f"({contour_info['n_unique_contour_points']} unique k of {wavevectors_all.shape[1]})"
    )

    for struct_idx in range(n_plot):
        d = designs[struct_idx].astype(np.float16, copy=False)
        fig_design = plt.figure(figsize=(6, 5))
        ax_design = fig_design.add_subplot(111)
        im = ax_design.imshow(d, origin="lower", cmap="viridis")
        ax_design.set_title("Geometry (single channel)")
        ax_design.set_xlabel("X")
        ax_design.set_ylabel("Y")
        fig_design.colorbar(im, ax=ax_design, fraction=0.046, pad=0.04)
        design_path = output_dir / "design"
        design_path.mkdir(parents=True, exist_ok=True)
        fig_design.savefig(design_path / f"{struct_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_design)

        frequencies_contour = eigenvalues_all[struct_idx, contour_indices].astype(np.float32, copy=False)

        fig_disp = plt.figure(figsize=(10, 6))
        ax_disp = fig_disp.add_subplot(111)
        plot_dispersion_on_contour(
            ax_disp,
            contour_info,
            frequencies_contour,
            contour_param,
            title=title,
            mark_points=True,
        )
        disp_path = output_dir / "dispersion"
        disp_path.mkdir(parents=True, exist_ok=True)
        fig_disp.savefig(disp_path / f"{struct_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_disp)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot dispersion for full PT datasets")
    parser.add_argument("data_dir", help="Path to full PT dataset directory")
    parser.add_argument("-n", "--n-structs", type=int, default=None, help="Number of structures to plot")
    parser.add_argument("-t", "--title", type=str, default="", help="Custom dispersion plot title (default: blank)")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output folder for plots (default: <cwd>/PLOTS/<name>_full)")
    args = parser.parse_args()
    main(args.data_dir, args.n_structs, args.title, args.output_dir)
