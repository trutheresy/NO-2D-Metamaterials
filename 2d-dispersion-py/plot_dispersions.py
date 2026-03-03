"""
Plot dispersion for full PT datasets (float16-first path).

Expected files in data_dir:
- geometries_full.pt          (N_struct, N_pix, N_pix)
- wavevectors_full.pt         (N_struct, N_wv, 2)
- eigenvalue_data_full.pt     (N_struct, N_wv, N_eig)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.distance import cdist

from plot_dispersion_with_eigenfrequencies_reduced_set import (
    plot_dispersion_on_contour,
    get_IBZ_contour_wavevectors,
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


def main(data_dir: str, n_structs: int | None = None) -> None:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = load_full_pt_dataset(data_path)
    designs = data["designs"]  # (N_struct, N_pix, N_pix)
    wavevectors_all = data["wavevectors"]  # (N_struct, N_wv, 2)
    eigenvalues_all = data["eigenvalue_data"]  # (N_struct, N_wv, N_eig)

    n_total = int(designs.shape[0])
    n_plot = n_total if n_structs is None else min(int(n_structs), n_total)

    output_dir = Path.cwd() / "plots" / f"{data_path.name}_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep wavevector/dispersion logic consistent with existing plotting flow.
    contour_wv, contour_info = get_IBZ_contour_wavevectors(10, 1.0, "p4mm")

    for struct_idx in range(n_plot):
        # Plot single-channel geometry directly.
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

        wavevectors = wavevectors_all[struct_idx].astype(np.float16, copy=False)
        frequencies = eigenvalues_all[struct_idx].astype(np.float16, copy=False)

        frequencies_contour = np.zeros((len(contour_wv), frequencies.shape[1]), dtype=np.float16)
        for eig_idx in range(frequencies.shape[1]):
            # Match MATLAB plot_dispersion.m interpolation pathway:
            # scatteredInterpolant(..., 'linear', 'linear') + nearest fallback outside hull.
            interp = LinearNDInterpolator(
                wavevectors.astype(np.float32),
                frequencies[:, eig_idx].astype(np.float32),
                fill_value=np.nan,
            )
            vals = np.asarray(interp(contour_wv.astype(np.float32)), dtype=np.float32)
            nan_mask = np.isnan(vals)
            if np.any(nan_mask):
                nan_idx = np.where(nan_mask)[0]
                dist = cdist(contour_wv[nan_idx].astype(np.float32), wavevectors.astype(np.float32))
                nearest_idx = np.argmin(dist, axis=1)
                vals[nan_idx] = frequencies[nearest_idx, eig_idx].astype(np.float32)
            frequencies_contour[:, eig_idx] = np.asarray(vals, dtype=np.float16)

        # Intermediate debug saves intentionally disabled in non-debug script.
        # debug_npz = output_dir / "debug" / f"struct_{struct_idx}_plot_arrays.npz"
        # debug_npz.parent.mkdir(parents=True, exist_ok=True)
        # np.savez_compressed(
        #     debug_npz,
        #     wavevectors=wavevectors.astype(np.float16),
        #     frequencies=frequencies.astype(np.float16),
        #     contour_wv=contour_wv.astype(np.float16),
        #     frequencies_contour=frequencies_contour.astype(np.float16),
        # )

        fig_disp = plt.figure(figsize=(10, 6))
        ax_disp = fig_disp.add_subplot(111)
        plot_dispersion_on_contour(
            ax_disp,
            contour_info,
            frequencies_contour.astype(np.float32),
            contour_info["wavevector_parameter"],
            title="Dispersion Relation (Full PT Dataset)",
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
    args = parser.parse_args()
    main(args.data_dir, args.n_structs)
