"""
DEBUG ONLY: save intermediate plotting arrays for parity comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.distance import cdist

from plot_dispersion_with_eigenfrequencies_reduced_set import (
    create_grid_interpolators,
    get_IBZ_contour_wavevectors,
)


def _save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main(data_dir: str, out_dir: str, n_structs: int = 1) -> None:
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    designs = torch.load(data_path / "geometries_full.pt", map_location="cpu").to(torch.float16).numpy()
    wavevectors_all = torch.load(data_path / "wavevectors_full.pt", map_location="cpu").to(torch.float16).numpy()
    eigenvalues_all = torch.load(data_path / "eigenvalue_data_full.pt", map_location="cpu").to(torch.float16).numpy()

    contour_wv, contour_info = get_IBZ_contour_wavevectors(10, 1.0, "p4mm")
    n_plot = min(int(n_structs), int(designs.shape[0]))
    manifest: dict[str, object] = {"n_structs": n_plot, "struct_files": []}

    for struct_idx in range(n_plot):
        wavevectors = wavevectors_all[struct_idx].astype(np.float32, copy=False)
        frequencies = eigenvalues_all[struct_idx].astype(np.float32, copy=False)

        grid_interp, _ = create_grid_interpolators(wavevectors, frequencies, [25, 13])
        points_yx = contour_wv[:, [1, 0]].astype(np.float32)
        frequencies_contour = np.zeros((len(contour_wv), frequencies.shape[1]), dtype=np.float32)
        frequencies_contour_scattered = np.zeros((len(contour_wv), frequencies.shape[1]), dtype=np.float32)
        for eig_idx in range(frequencies.shape[1]):
            frequencies_contour[:, eig_idx] = np.asarray(grid_interp[eig_idx](points_yx), dtype=np.float32)
            interp_sc = LinearNDInterpolator(wavevectors, frequencies[:, eig_idx], fill_value=np.nan)
            vals_sc = np.asarray(interp_sc(contour_wv), dtype=np.float32)
            nan_mask = np.isnan(vals_sc)
            if np.any(nan_mask):
                nan_idx = np.where(nan_mask)[0]
                d = cdist(contour_wv[nan_idx], wavevectors)
                nn = np.argmin(d, axis=1)
                vals_sc[nan_idx] = frequencies[nn, eig_idx]
            frequencies_contour_scattered[:, eig_idx] = vals_sc

        # Save dispersion figure only (no geometry here; this script targets curve parity).
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        for eig_idx in range(frequencies_contour_scattered.shape[1]):
            ax.plot(contour_info["wavevector_parameter"], frequencies_contour_scattered[:, eig_idx], linewidth=2)
        for i in range(contour_info["N_segment"] + 1):
            ax.axvline(i, color="k", linestyle="--", alpha=0.3, linewidth=1)
        ax.set_xlabel("Wavevector Contour Parameter")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title("Dispersion Relation (Python debug)")
        fig.savefig(out_path / f"struct_{struct_idx}_dispersion.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        npz_path = out_path / f"struct_{struct_idx}_plot_debug.npz"
        _save_npz(
            npz_path,
            {
                "wavevectors_raw": wavevectors,
                "frequencies_raw": frequencies,
                "contour_wavevectors": contour_wv.astype(np.float32),
                "contour_parameter": contour_info["wavevector_parameter"].astype(np.float32),
                "frequencies_contour": frequencies_contour,
                "frequencies_contour_scattered": frequencies_contour_scattered,
                "plot_x": contour_info["wavevector_parameter"].astype(np.float32),
                "plot_y": frequencies_contour_scattered,
            },
        )
        manifest["struct_files"].append(str(npz_path))

    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_path), "n_structs": n_plot}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug plotter for full PT datasets.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-structs", type=int, default=1)
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.n_structs)
