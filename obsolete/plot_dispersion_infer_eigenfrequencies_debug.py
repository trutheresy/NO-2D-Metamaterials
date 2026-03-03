"""
DEBUG ONLY: Infer-frequency plotting with intermediate saves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

from plot_dispersion_infer_eigenfrequencies import (
    load_pt_dataset,
    apply_steel_rubber_paradigm_single_channel,
    create_const_dict,
    compute_K_M_matrices,
    reconstruct_frequencies_from_eigenvectors,
    get_IBZ_contour_wavevectors,
    extract_grid_points_on_contour,
)


def _triplet_from_sparse(A):
    if not issparse(A):
        return np.zeros((0, 4), dtype=np.float64)
    coo = A.tocoo()
    return np.column_stack([coo.row, coo.col, coo.data.real, coo.data.imag]).astype(np.float64)


def main(data_dir: str, out_dir: str, n_structs: int = 1) -> None:
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = load_pt_dataset(data_path, original_data_dir=None, require_eigenvalue_data=False)
    designs = data["designs"]
    wavevectors_all = data["wavevectors"]
    eigenvectors = data["eigenvectors"]

    n_plot = min(int(n_structs), int(designs.shape[0]))

    # Keep parameter choices aligned with infer script defaults.
    E_min, E_max = 200e6, 200e9
    rho_min, rho_max = 800, 8000
    nu_min, nu_max = 0.0, 0.5
    t_val = 1.0
    a = 1.0
    N_pix = int(designs.shape[1])
    N_eig = 6

    manifest: dict[str, object] = {"n_structs": n_plot, "struct_files": []}

    for struct_idx in range(n_plot):
        design_param = np.asarray(designs[struct_idx], dtype=np.float16)
        design_normalized = apply_steel_rubber_paradigm_single_channel(
            design_param.astype(np.float32), E_min, E_max, rho_min, rho_max, nu_min, nu_max
        ).astype(np.float16)
        design_for_plot = np.zeros_like(design_normalized, dtype=np.float32)
        design_for_plot[:, :, 0] = E_min + (E_max - E_min) * design_normalized[:, :, 0]
        design_for_plot[:, :, 1] = rho_min + (rho_max - rho_min) * design_normalized[:, :, 1]
        design_for_plot[:, :, 2] = nu_min + (nu_max - nu_min) * design_normalized[:, :, 2]

        wavevectors = np.asarray(wavevectors_all[struct_idx], dtype=np.float16)
        const = create_const_dict(
            design_normalized.astype(np.float32),
            N_pix,
            a=a,
            E_min=E_min,
            E_max=E_max,
            rho_min=rho_min,
            rho_max=rho_max,
            nu_min=nu_min,
            nu_max=nu_max,
            t=t_val,
        )
        K, M = compute_K_M_matrices(const)

        # Explicit T matrices to mirror MATLAB debug outputs and provide infer inputs.
        T_diag = []
        T_data = []
        for wv in wavevectors.astype(np.float32):
            from system_matrices import get_transformation_matrix  # local import to match infer script environment

            T = get_transformation_matrix(wv, const)
            T_data.append(T)
            T_diag.append([int(T.shape[0]), int(T.shape[1])])
        T_diag_arr = np.asarray(T_diag, dtype=np.int64)

        frequencies_recon = reconstruct_frequencies_from_eigenvectors(
            K,
            M,
            T_data,
            eigenvectors,
            wavevectors.astype(np.float32),
            N_eig,
            struct_idx=struct_idx,
            N_struct=designs.shape[0],
            N_pix=N_pix,
            N_ele=1,
        ).astype(np.float16)

        _, contour_info = get_IBZ_contour_wavevectors(10, a, "p4mm")
        contour_wv, frequencies_contour, contour_param = extract_grid_points_on_contour(
            wavevectors.astype(np.float32),
            frequencies_recon.astype(np.float32),
            contour_info,
            a,
            tolerance=2e-3,
        )
        contour_wv = np.asarray(contour_wv, dtype=np.float16)
        frequencies_contour = np.asarray(frequencies_contour, dtype=np.float16)
        contour_param = np.asarray(contour_param, dtype=np.float16)

        # Save final infer dispersion plot points as PNG for visual parity checks.
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        for eig_idx in range(frequencies_contour.shape[1]):
            ax.plot(contour_param.astype(np.float32), frequencies_contour[:, eig_idx].astype(np.float32), linewidth=2)
        ax.set_xlabel("Wavevector Contour Parameter")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title("Dispersion Relation (Python infer debug)")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_path / f"struct_{struct_idx}_dispersion.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        npz_path = out_path / f"struct_{struct_idx}_infer_plot_debug.npz"
        np.savez_compressed(
            npz_path,
            design_raw_single=design_param,
            design_normalized=design_normalized,
            design_for_plot=design_for_plot.astype(np.float32),
            wavevectors_raw=wavevectors,
            frequencies_recon_raw=frequencies_recon,
            K_triplet=_triplet_from_sparse(K),
            M_triplet=_triplet_from_sparse(M),
            T_diag=T_diag_arr,
            contour_wavevectors=contour_wv,
            contour_parameter=contour_param,
            frequencies_contour=frequencies_contour,
            plot_x=contour_param,
            plot_y=frequencies_contour,
        )
        manifest["struct_files"].append(str(npz_path))

    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_path), "n_structs": n_plot}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug infer plotting with intermediate saves.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-structs", type=int, default=1)
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.n_structs)
