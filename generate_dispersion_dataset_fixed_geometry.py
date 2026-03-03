"""
Deterministic Python generation flow for fixed-geometry equivalence checks.

This mirrors the high-level flow of generate_dispersion_dataset_Han_Alex.m
but replaces random geometry generation with a fixed, provided geometry.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import scipy.io as sio
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
    }


def run_fixed_geometry_generation(
    geometry_path: Path,
    output_pt_dir: Path,
    repo_root: Path,
) -> Path:
    _add_python_library_path(repo_root)

    from wavevectors import get_IBZ_wavevectors  # type: ignore
    from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt  # type: ignore
    from design_conversion import design_to_explicit, apply_steel_rubber_paradigm  # type: ignore

    mat = sio.loadmat(str(geometry_path))
    if "FIXED_DESIGN" not in mat:
        raise KeyError("FIXED_DESIGN not found in geometry .mat file")
    design = np.asarray(mat["FIXED_DESIGN"], dtype=np.float16)
    if design.shape != (32, 32, 3):
        raise ValueError(f"Expected FIXED_DESIGN shape (32, 32, 3), got {design.shape}")

    const = _base_constants()
    # Match MATLAB flow: apply steel-rubber paradigm before solving.
    design = apply_steel_rubber_paradigm(design, const)
    design = np.asarray(design, dtype=np.float16)
    const["design"] = design
    const["wavevectors"] = np.asarray(get_IBZ_wavevectors(const["N_wv"], const["a"], "none"), dtype=np.float16)

    wv, fr, ev, _mesh, K, M, T = dispersion_with_matrix_save_opt(const, const["wavevectors"])

    explicit_props = design_to_explicit(
        np.asarray(design, dtype=np.float32),
        const["design_scale"],
        const["E_min"],
        const["E_max"],
        const["rho_min"],
        const["rho_max"],
        const["poisson_min"],
        const["poisson_max"],
    )

    output_pt_dir.mkdir(parents=True, exist_ok=True)

    geometries_full = np.expand_dims(np.asarray(design[:, :, 0], dtype=np.float16), axis=0)  # (1, N_pix, N_pix)
    wavevectors_full = np.expand_dims(np.asarray(wv, dtype=np.float16), axis=0)  # (1, N_wv, 2)
    eigenvalue_data_full = np.expand_dims(np.real(np.asarray(fr, dtype=np.float16)), axis=0)  # (1, N_wv, N_eig)
    eigenvector_data_full = np.expand_dims(
        np.asarray(ev, dtype=np.complex64).transpose(1, 2, 0), axis=0
    )  # (1, N_wv, N_eig, N_dof)

    torch.save(torch.from_numpy(geometries_full), output_pt_dir / "geometries_full.pt")
    torch.save(torch.from_numpy(wavevectors_full), output_pt_dir / "wavevectors_full.pt")
    torch.save(torch.from_numpy(eigenvalue_data_full), output_pt_dir / "eigenvalue_data_full.pt")
    torch.save(torch.from_numpy(eigenvector_data_full), output_pt_dir / "eigenvector_data_full.pt")

    # Keep constitutive fields available in PT-native format for downstream checks.
    torch.save(torch.from_numpy(np.expand_dims(np.asarray(explicit_props["E"], dtype=np.float32), axis=0)), output_pt_dir / "elastic_modulus_full.pt")
    torch.save(torch.from_numpy(np.expand_dims(np.asarray(explicit_props["rho"], dtype=np.float32), axis=0)), output_pt_dir / "density_full.pt")
    torch.save(torch.from_numpy(np.expand_dims(np.asarray(explicit_props["nu"], dtype=np.float32), axis=0)), output_pt_dir / "poisson_full.pt")

    manifest = {
        "n_struct": 1,
        "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "geometry_source": str(geometry_path),
        "output_pt_dir": str(output_pt_dir),
        "symmetry_type": "p4mm",
        "saved_files": [
            "geometries_full.pt",
            "wavevectors_full.pt",
            "eigenvalue_data_full.pt",
            "eigenvector_data_full.pt",
            "elastic_modulus_full.pt",
            "density_full.pt",
            "poisson_full.pt",
        ],
    }
    (output_pt_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_pt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-geometry PT dataset in Python.")
    parser.add_argument("--geometry-mat", required=True, help="Path to .mat file containing FIXED_DESIGN.")
    parser.add_argument("--output-pt-dir", required=True, help="Output directory for PT files.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parent),
        help="Repository root path.",
    )
    args = parser.parse_args()

    out = run_fixed_geometry_generation(
        geometry_path=Path(args.geometry_mat),
        output_pt_dir=Path(args.output_pt_dir),
        repo_root=Path(args.repo_root),
    )
    print(f"PY_FIXED_OUTPUT_PT_DIR={out}")


if __name__ == "__main__":
    main()
