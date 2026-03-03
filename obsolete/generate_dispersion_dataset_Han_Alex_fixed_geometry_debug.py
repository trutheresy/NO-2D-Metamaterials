"""
DEBUG ONLY: delete after debugging is complete.

Deterministic fixed-geometry Python generation with optional intermediate saves.
Default dtype policy is float16-first for debug parity profiling.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch


def _add_python_library_path(repo_root: Path) -> None:
    import sys

    py_lib = repo_root / "2d-dispersion-py"
    if str(py_lib) not in sys.path:
        sys.path.insert(0, str(py_lib))


def _base_constants() -> dict:
    n_wv_0 = 25
    return {
        "N_ele": 1,
        "N_pix": 32,
        "N_wv": [n_wv_0, int(np.ceil(n_wv_0 / 2))],
        "N_eig": 6,
        "sigma_eig": 1e-2,
        "a": 1.0,
        "design_scale": "linear",
        "E_min": 200e6,
        "E_max": 200e9,
        "rho_min": 8e2,
        "rho_max": 8e3,
        "poisson_min": 0.0,
        "poisson_max": 0.5,
        "t": 1.0,
        "isUseGPU": False,
        "isUseImprovement": True,
        "isUseSecondImprovement": False,
        "isUseParallel": False,
        "isSaveEigenvectors": True,
        "isSaveKandM": True,
        "isSaveMesh": False,
        "isComputeGroupVelocity": False,
        "isComputeFrequencyDesignSensitivity": False,
        "isComputeGroupVelocityDesignSensitivity": False,
        "symmetry_type": "p4mm",
        "eigenvector_dtype": "double",
    }


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _build_pt_dataset_outputs(
    designs: np.ndarray,
    wavevector_data: np.ndarray,
    eigenvalue_data: np.ndarray,
    eigenvector_data: np.ndarray,
    design_numbers: np.ndarray,
    n_pix: int,
    n_eig: int,
) -> dict:
    # (N_pix, N_pix, 3, N_struct) -> (N_struct, N_pix, N_pix)
    geometries = designs[:, :, 0, :].transpose(2, 0, 1)
    # (N_wv, 2, N_struct) -> (N_struct, N_wv, 2)
    wavevectors = wavevector_data.transpose(2, 0, 1)
    # (N_dof, N_wv, N_eig, N_struct) -> (N_struct, N_wv, N_eig, N_dof)
    eig = eigenvector_data.transpose(3, 1, 2, 0)
    eig_x = eig[..., 0::2].reshape(eig.shape[0], eig.shape[1], eig.shape[2], n_pix, n_pix)
    eig_y = eig[..., 1::2].reshape(eig.shape[0], eig.shape[1], eig.shape[2], n_pix, n_pix)

    reduced_indices_reserved = []
    for d_idx in range(eig.shape[0]):
        for w_idx in range(eig.shape[1]):
            for b_idx in range(eig.shape[2]):
                reduced_indices_reserved.append((d_idx, w_idx, b_idx))

    d_idx = [idx[0] for idx in reduced_indices_reserved]
    w_idx = [idx[1] for idx in reduced_indices_reserved]
    b_idx = [idx[2] for idx in reduced_indices_reserved]
    eig_x_reduced = eig_x[d_idx, w_idx, b_idx]
    eig_y_reduced = eig_y[d_idx, w_idx, b_idx]

    try:
        import NO_utils_multiple  # type: ignore

        waveforms = NO_utils_multiple.embed_2const_wavelet(
            wavevectors[0, :, 0],
            wavevectors[0, :, 1],
            size=n_pix,
        )
        bands_fft = NO_utils_multiple.embed_integer_wavelet(np.arange(1, n_eig + 1), size=n_pix)
    except Exception:
        waveforms = np.zeros((wavevectors.shape[1], n_pix, n_pix), dtype=np.float32)
        bands_fft = np.zeros((n_eig, n_pix, n_pix), dtype=np.float32)

    design_params_array = np.asarray(design_numbers, dtype=np.int64).reshape(-1, 1)

    return {
        "displacements_dataset": torch.utils.data.TensorDataset(
            torch.from_numpy(eig_x_reduced.real).to(torch.float16),
            torch.from_numpy(eig_x_reduced.imag).to(torch.float16),
            torch.from_numpy(eig_y_reduced.real).to(torch.float16),
            torch.from_numpy(eig_y_reduced.imag).to(torch.float16),
        ),
        "reduced_indices": reduced_indices_reserved,
        "geometries_full": torch.from_numpy(geometries).to(torch.float16),
        "waveforms_full": torch.from_numpy(waveforms).to(torch.float16),
        "wavevectors_full": torch.from_numpy(wavevectors).to(torch.float16),
        "band_fft_full": torch.from_numpy(bands_fft).to(torch.float16),
        "design_params_full": torch.from_numpy(design_params_array).to(torch.float16),
        "eigenvalue_data_full": torch.from_numpy(eigenvalue_data.transpose(2, 0, 1)).to(torch.float16),
    }


def run_fixed_geometry_generation_debug(
    geometry_path: Path,
    output_mat_path: Path,
    output_pt_dir: Path,
    debug_dir: Path | None,
    repo_root: Path,
    default_dtype: str = "float16",
) -> dict:
    _add_python_library_path(repo_root)

    from wavevectors import get_IBZ_wavevectors  # type: ignore
    from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt  # type: ignore
    from design_conversion import design_to_explicit, apply_steel_rubber_paradigm  # type: ignore

    if default_dtype not in {"float16", "float32", "float64"}:
        raise ValueError("--default-dtype must be float16, float32, or float64")
    dtype_map = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
    dtype = dtype_map[default_dtype]
    dtype_name = default_dtype

    mat = sio.loadmat(str(geometry_path))
    if "FIXED_DESIGN" not in mat:
        raise KeyError("FIXED_DESIGN not found in geometry .mat file")
    design = np.asarray(mat["FIXED_DESIGN"], dtype=np.float64)
    if design.shape != (32, 32, 3):
        raise ValueError(f"Expected FIXED_DESIGN shape (32, 32, 3), got {design.shape}")

    const = _base_constants()
    design_after_map = apply_steel_rubber_paradigm(design, const)
    design_for_solver = np.asarray(design_after_map, dtype=dtype)
    const["design"] = design_for_solver
    const["wavevectors"] = np.asarray(
        get_IBZ_wavevectors(const["N_wv"], const["a"], "p4mm"), dtype=dtype
    )

    wv, fr, ev, _mesh, K, M, T = dispersion_with_matrix_save_opt(const, const["wavevectors"])
    explicit_props = design_to_explicit(
        np.asarray(design_after_map, dtype=np.float64),
        const["design_scale"],
        const["E_min"],
        const["E_max"],
        const["rho_min"],
        const["rho_max"],
        const["poisson_min"],
        const["poisson_max"],
    )

    output_mat_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "WAVEVECTOR_DATA": np.expand_dims(np.asarray(wv), axis=2),
        "EIGENVALUE_DATA": np.expand_dims(np.real(np.asarray(fr)), axis=2),
        "EIGENVECTOR_DATA": np.expand_dims(np.asarray(ev), axis=3),
        "designs": np.expand_dims(np.asarray(design_for_solver, dtype=dtype), axis=3),
        "E_DATA": np.expand_dims(np.asarray(explicit_props["E"], dtype=np.float64), axis=2),
        "RHO_DATA": np.expand_dims(np.asarray(explicit_props["rho"], dtype=np.float64), axis=2),
        "NU_DATA": np.expand_dims(np.asarray(explicit_props["nu"], dtype=np.float64), axis=2),
        "N_struct": np.array([[1]]),
        "IMAG_TOL": np.array([[1e-3]]),
        "K_DATA": np.array([K], dtype=object),
        "M_DATA": np.array([M], dtype=object),
        "T_DATA": np.asarray(T, dtype=object),
        "SCRIPT_TIME": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "PYTHON_DEFAULT_DTYPE": np.array([dtype_name]),
    }
    sio.savemat(str(output_mat_path), payload, oned_as="column")

    output_pt_dir.mkdir(parents=True, exist_ok=True)
    designs_4d = np.expand_dims(np.asarray(design_for_solver, dtype=dtype), axis=3)
    wv_3d = np.expand_dims(np.asarray(wv, dtype=dtype), axis=2)
    fr_3d = np.expand_dims(np.real(np.asarray(fr, dtype=dtype)), axis=2)
    ev_4d = np.expand_dims(np.asarray(ev), axis=3)
    design_numbers = np.asarray([0], dtype=np.int64)
    pt_outputs = _build_pt_dataset_outputs(
        designs=designs_4d,
        wavevector_data=wv_3d,
        eigenvalue_data=fr_3d,
        eigenvector_data=ev_4d,
        design_numbers=design_numbers,
        n_pix=const["N_pix"],
        n_eig=const["N_eig"],
    )
    torch.save(pt_outputs["displacements_dataset"], output_pt_dir / "displacements_dataset.pt")
    torch.save(pt_outputs["reduced_indices"], output_pt_dir / "reduced_indices.pt")
    torch.save(pt_outputs["geometries_full"], output_pt_dir / "geometries_full.pt")
    torch.save(pt_outputs["waveforms_full"], output_pt_dir / "waveforms_full.pt")
    torch.save(pt_outputs["wavevectors_full"], output_pt_dir / "wavevectors_full.pt")
    torch.save(pt_outputs["band_fft_full"], output_pt_dir / "band_fft_full.pt")
    torch.save(pt_outputs["design_params_full"], output_pt_dir / "design_params_full.pt")
    torch.save(pt_outputs["eigenvalue_data_full"], output_pt_dir / "eigenvalue_data_full.pt")

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        np.save(debug_dir / "design_input.npy", np.asarray(design, dtype=np.float64))
        np.save(debug_dir / "design_after_map.npy", np.asarray(design_after_map, dtype=np.float64))
        np.save(debug_dir / "design_for_solver.npy", np.asarray(design_for_solver, dtype=dtype))
        np.save(debug_dir / "wavevectors.npy", np.asarray(wv, dtype=dtype))
        np.save(debug_dir / "frequencies.npy", np.asarray(fr))
        if K is not None and sp.issparse(K):
            Kcoo = K.tocoo()
            np.save(debug_dir / "K_triplet.npy", np.column_stack([Kcoo.row, Kcoo.col, Kcoo.data.real, Kcoo.data.imag]))
        if M is not None and sp.issparse(M):
            Mcoo = M.tocoo()
            np.save(debug_dir / "M_triplet.npy", np.column_stack([Mcoo.row, Mcoo.col, Mcoo.data.real, Mcoo.data.imag]))
        if T is not None:
            T_diag = []
            for t in T:
                if t is None:
                    continue
                T_diag.append([int(t.shape[0]), int(t.shape[1])])
            np.save(debug_dir / "T_diag.npy", np.asarray(T_diag, dtype=np.int64))
        _save_json(
            debug_dir / "dtype_manifest.json",
            {
                "python_default_dtype": dtype_name,
                "exempt_rows": [
                    "Save reduced sample indices",
                    "Save raw debug dataset object",
                    "Save MATLAB dataset bundle",
                ],
            },
        )

    return {
        "output_mat_path": str(output_mat_path),
        "output_pt_dir": str(output_pt_dir),
        "python_default_dtype": dtype_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-geometry Python debug generator.")
    parser.add_argument("--geometry-mat", required=True, help="Path to .mat with FIXED_DESIGN")
    parser.add_argument("--output-mat", required=True, help="Output .mat file path")
    parser.add_argument("--output-pt-dir", required=True, help="Output directory for PT files")
    parser.add_argument("--debug-dir", default=None, help="Optional directory for intermediate saves")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent), help="Repository root")
    parser.add_argument(
        "--default-dtype",
        choices=["float16", "float32", "float64"],
        default="float16",
        help="Default numeric dtype for Python debug path (float16-first by default).",
    )
    args = parser.parse_args()

    result = run_fixed_geometry_generation_debug(
        geometry_path=Path(args.geometry_mat),
        output_mat_path=Path(args.output_mat),
        output_pt_dir=Path(args.output_pt_dir),
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        repo_root=Path(args.repo_root),
        default_dtype=args.default_dtype,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
