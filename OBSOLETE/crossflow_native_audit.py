from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch


REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = REPO_ROOT / "crossflow_outputs"
OUT_DIR = REPO_ROOT / "crossflow_native_exports"
MATLAB_NATIVE_DIR = OUT_DIR / "matlab_native"
PYTHON_NATIVE_DIR = OUT_DIR / "python_native"
PLOTS_DIR = OUT_DIR / "PLOTS"


def _strip_meta_keys(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if not k.startswith("__")}


def _safe_array(x: Any) -> np.ndarray:
    return np.asarray(x)


def _squeeze(x: Any) -> np.ndarray:
    return np.squeeze(_safe_array(x))


def _numeric_compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {
            "shape_match": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    abs_diff = np.abs(a - b)
    rel_diff = abs_diff / np.maximum(np.abs(a), 1e-14)
    return {
        "shape_match": True,
        "shape": list(a.shape),
        "max_abs": float(np.max(abs_diff)),
        "mean_abs": float(np.mean(abs_diff)),
        "max_rel": float(np.max(rel_diff)),
        "mean_rel": float(np.mean(rel_diff)),
        "allclose_1e-8_1e-10": bool(np.allclose(a, b, rtol=1e-8, atol=1e-10)),
    }


def _eigenvector_mode_map(evec_data: np.ndarray, n_pix: int = 32, k_idx: int = 0, b_idx: int = 0) -> np.ndarray:
    vec = np.asarray(evec_data[:, k_idx, b_idx]).reshape(-1)
    n_dof = 2 * n_pix * n_pix
    if vec.size != n_dof:
        return np.zeros((n_pix, n_pix), dtype=float)
    ux = vec[0::2].reshape(n_pix, n_pix)
    uy = vec[1::2].reshape(n_pix, n_pix)
    mag = np.sqrt(np.abs(ux) ** 2 + np.abs(uy) ** 2)
    return mag.astype(float)


def _extract_design3(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[:, :, :, 0]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unexpected design shape: {arr.shape}")


def _save_pair_plot(mat_arr: np.ndarray, py_arr: np.ndarray, title: str, out_path: Path, cmap: str = "viridis") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    vmin = float(np.min([np.nanmin(mat_arr), np.nanmin(py_arr)]))
    vmax = float(np.max([np.nanmax(mat_arr), np.nanmax(py_arr)]))
    diff = py_arr - mat_arr

    im0 = axes[0].imshow(mat_arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("MATLAB flow")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(py_arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title("Python flow")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, origin="lower", cmap="coolwarm", aspect="auto")
    axes[2].set_title("Python - MATLAB")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MATLAB_NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    PYTHON_NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fixed_input_mat = INPUT_DIR / "fixed_geometry_input.mat"
    matlab_mat = INPUT_DIR / "matlab_fixed_output.mat"
    python_mat = INPUT_DIR / "python_fixed_output.mat"
    if not fixed_input_mat.exists() or not matlab_mat.exists() or not python_mat.exists():
        raise FileNotFoundError("Expected files missing in crossflow_outputs/")

    fixed_in = _strip_meta_keys(sio.loadmat(fixed_input_mat, squeeze_me=False, struct_as_record=False))
    mat_data = _strip_meta_keys(sio.loadmat(matlab_mat, squeeze_me=False, struct_as_record=False))
    py_data = _strip_meta_keys(sio.loadmat(python_mat, squeeze_me=False, struct_as_record=False))

    # Save MATLAB-native copy (.mat)
    sio.savemat(MATLAB_NATIVE_DIR / "fixed_geometry_input.mat", fixed_in)
    sio.savemat(MATLAB_NATIVE_DIR / "matlab_flow_output.mat", mat_data)

    # Save Python-native copy (.pt)
    fixed_design = _squeeze(fixed_in["FIXED_DESIGN"]).astype(np.float64)
    py_native = {}
    for k, v in py_data.items():
        try:
            arr = np.asarray(v)
            if arr.dtype == object:
                py_native[k] = v
            else:
                py_native[k] = torch.from_numpy(arr.copy())
        except Exception:
            py_native[k] = v
    torch.save({"FIXED_DESIGN": torch.from_numpy(fixed_design.copy())}, PYTHON_NATIVE_DIR / "fixed_geometry_input.pt")
    torch.save(py_native, PYTHON_NATIVE_DIR / "python_flow_output.pt")

    # Input equality checks
    mat_design = _extract_design3(np.asarray(mat_data["designs"]))
    py_design = _extract_design3(np.asarray(py_data["designs"]))
    input_checks = {
        "fixed_input_vs_mat_design": _numeric_compare(fixed_design, mat_design),
        "fixed_input_vs_python_design": _numeric_compare(fixed_design, py_design),
        "mat_design_vs_python_design": _numeric_compare(mat_design, py_design),
    }

    # Quantity-by-quantity comparisons
    comparable = ["WAVEVECTOR_DATA", "EIGENVALUE_DATA", "EIGENVECTOR_DATA", "designs"]
    comparisons: dict[str, Any] = {}
    for key in comparable:
        if key not in mat_data or key not in py_data:
            comparisons[key] = {"present_in_both": False}
            continue
        a = _squeeze(mat_data[key])
        b = _squeeze(py_data[key])
        comparisons[key] = {"present_in_both": True, **_numeric_compare(a, b)}

    # Plot pairs (same style)
    for ch_idx in range(3):
        _save_pair_plot(
            mat_design[:, :, ch_idx],
            py_design[:, :, ch_idx],
            f"Input Design Channel {ch_idx} (fixed geometry)",
            PLOTS_DIR / f"input_design_channel_{ch_idx}.png",
        )

    mat_eig = _squeeze(mat_data["EIGENVALUE_DATA"])
    py_eig = _squeeze(py_data["EIGENVALUE_DATA"])
    _save_pair_plot(
        mat_eig,
        py_eig,
        "Output: EIGENVALUE_DATA",
        PLOTS_DIR / "output_eigenvalue_data.png",
    )

    mat_evec = _squeeze(mat_data["EIGENVECTOR_DATA"])
    py_evec = _squeeze(py_data["EIGENVECTOR_DATA"])
    mat_mode_map = _eigenvector_mode_map(mat_evec, n_pix=32, k_idx=0, b_idx=0)
    py_mode_map = _eigenvector_mode_map(py_evec, n_pix=32, k_idx=0, b_idx=0)
    _save_pair_plot(
        mat_mode_map,
        py_mode_map,
        "Output: Eigenvector mode magnitude (k=0, band=0)",
        PLOTS_DIR / "output_eigenvector_mode_map_k0_b0.png",
    )

    report = {
        "folders": {
            "matlab_native_folder": str(MATLAB_NATIVE_DIR),
            "python_native_folder": str(PYTHON_NATIVE_DIR),
            "plots_folder": str(PLOTS_DIR),
        },
        "saved_quantities": {
            "matlab_native_mat_files": [
                "fixed_geometry_input.mat",
                "matlab_flow_output.mat",
            ],
            "python_native_pt_files": [
                "fixed_geometry_input.pt",
                "python_flow_output.pt",
            ],
            "python_flow_pt_payload_keys": sorted(list(py_native.keys())),
            "matlab_flow_mat_keys": sorted(list(mat_data.keys())),
        },
        "input_equality_checks": input_checks,
        "quantity_comparisons": comparisons,
        "plot_files": sorted([p.name for p in PLOTS_DIR.glob("*.png")]),
    }

    (OUT_DIR / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
