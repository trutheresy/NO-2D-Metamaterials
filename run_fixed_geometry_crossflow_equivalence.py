"""
Orchestrate fixed-geometry MATLAB/Python generation and strict comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.io as sio


def create_fixed_design(repo_root: Path, n_pix: int = 32, design_number: int = 1) -> np.ndarray:
    """Generate one deterministic p4mm geometry via library random-design pipeline."""
    sys.path.insert(0, str(repo_root / "2d-dispersion-py"))
    from design_parameters import DesignParameters  # type: ignore
    from get_design2 import get_design2  # type: ignore

    dp = DesignParameters(design_number)
    dp.property_coupling = "coupled"
    dp.design_style = "kernel"
    dp.design_options = {
        "kernel": "periodic",
        "sigma_f": 1.0,
        "sigma_l": 1.0,
        "symmetry_type": "p4mm",
        "N_value": np.inf,
    }
    dp.N_pix = [n_pix, n_pix]
    dp = dp.prepare()
    design = np.asarray(get_design2(dp), dtype=np.float64)
    return np.clip(design, 0.0, 1.0)


def _run_command(command: list[str], cwd: Path) -> None:
    proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    print(proc.stdout.strip())


def run_matlab_flow(repo_root: Path, geometry_mat: Path, matlab_output: Path) -> None:
    script_path = repo_root / "generate_dispersion_dataset_Han_Alex_fixed_geometry.m"
    cmd = (
        f"fixed_geometry_path='{geometry_mat.as_posix()}'; "
        f"matlab_output_path='{matlab_output.as_posix()}'; "
        f"run('{script_path.as_posix()}');"
    )
    _run_command(["matlab", "-batch", cmd], cwd=repo_root)


def run_python_flow(repo_root: Path, geometry_mat: Path, python_output: Path) -> None:
    script = repo_root / "generate_dispersion_dataset_Han_Alex_fixed_geometry.py"
    _run_command(
        [
            sys.executable,
            str(script),
            "--geometry-mat",
            str(geometry_mat),
            "--output-mat",
            str(python_output),
            "--repo-root",
            str(repo_root),
        ],
        cwd=repo_root,
    )


def _extract_array(payload: Dict, key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"{key} not found in dataset payload")
    arr = np.asarray(payload[key])
    return np.squeeze(arr)


def _align_vector(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    ref_norm = np.linalg.norm(ref)
    tgt_norm = np.linalg.norm(target)
    if ref_norm == 0 or tgt_norm == 0:
        return target
    alpha = np.vdot(ref, target) / np.vdot(target, target)
    return target * alpha


def compare_outputs(matlab_mat: Path, python_mat: Path) -> Dict:
    m_data = sio.loadmat(str(matlab_mat), squeeze_me=False, struct_as_record=False)
    p_data = sio.loadmat(str(python_mat), squeeze_me=False, struct_as_record=False)

    m_eval = _extract_array(m_data, "EIGENVALUE_DATA")
    p_eval = _extract_array(p_data, "EIGENVALUE_DATA")
    if m_eval.shape != p_eval.shape:
        raise ValueError(f"EIGENVALUE_DATA shape mismatch: matlab={m_eval.shape}, python={p_eval.shape}")

    eval_abs = np.abs(m_eval - p_eval)
    eval_rel = eval_abs / np.maximum(np.abs(m_eval), 1e-14)
    eval_metrics = {
        "shape": list(m_eval.shape),
        "max_abs": float(np.max(eval_abs)),
        "mean_abs": float(np.mean(eval_abs)),
        "max_rel": float(np.max(eval_rel)),
        "mean_rel": float(np.mean(eval_rel)),
        "allclose": bool(np.allclose(m_eval, p_eval, rtol=1e-8, atol=1e-10)),
    }

    m_evec = _extract_array(m_data, "EIGENVECTOR_DATA")
    p_evec = _extract_array(p_data, "EIGENVECTOR_DATA")
    if m_evec.shape != p_evec.shape:
        raise ValueError(f"EIGENVECTOR_DATA shape mismatch: matlab={m_evec.shape}, python={p_evec.shape}")

    if m_evec.ndim != 3:
        raise ValueError(f"Expected EIGENVECTOR_DATA to be 3D after squeeze, got ndim={m_evec.ndim}")

    n_k = m_evec.shape[1]
    n_eig = m_evec.shape[2]
    per_mode_abs = []
    per_mode_rel = []
    per_mode_corr = []

    for k_idx in range(n_k):
        for eig_idx in range(n_eig):
            ref = m_evec[:, k_idx, eig_idx].astype(np.complex128, copy=False)
            tgt = p_evec[:, k_idx, eig_idx].astype(np.complex128, copy=False)
            tgt_aligned = _align_vector(ref, tgt)
            diff = np.linalg.norm(ref - tgt_aligned)
            denom = max(np.linalg.norm(ref), 1e-14)
            rel = diff / denom
            corr = np.abs(np.vdot(ref, tgt)) / max(np.linalg.norm(ref) * np.linalg.norm(tgt), 1e-14)
            per_mode_abs.append(float(diff))
            per_mode_rel.append(float(rel))
            per_mode_corr.append(float(corr))

    evec_metrics = {
        "shape": list(m_evec.shape),
        "max_abs_norm_diff": float(np.max(per_mode_abs)),
        "mean_abs_norm_diff": float(np.mean(per_mode_abs)),
        "max_rel_norm_diff": float(np.max(per_mode_rel)),
        "mean_rel_norm_diff": float(np.mean(per_mode_rel)),
        "min_correlation": float(np.min(per_mode_corr)),
        "mean_correlation": float(np.mean(per_mode_corr)),
        "strict_pass": bool((np.max(per_mode_rel) <= 1e-7) and (np.min(per_mode_corr) >= 1.0 - 1e-8)),
    }

    return {
        "eigenvalue_metrics": eval_metrics,
        "eigenvector_metrics": evec_metrics,
        "overall_pass": bool(eval_metrics["allclose"] and evec_metrics["strict_pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-geometry MATLAB/Python equivalence flow.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "crossflow_outputs"),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry_mat = out_dir / "fixed_geometry_input.mat"
    matlab_output = out_dir / "matlab_fixed_output.mat"
    python_output = out_dir / "python_fixed_output.mat"
    report_path = out_dir / "equivalence_report.json"

    fixed = create_fixed_design(repo_root=repo_root, n_pix=32, design_number=1)
    sio.savemat(str(geometry_mat), {"FIXED_DESIGN": fixed})
    print(f"FIXED_GEOMETRY={geometry_mat}")

    run_matlab_flow(repo_root, geometry_mat, matlab_output)
    run_python_flow(repo_root, geometry_mat, python_output)

    report = compare_outputs(matlab_output, python_output)
    report["paths"] = {
        "geometry_mat": str(geometry_mat),
        "matlab_output": str(matlab_output),
        "python_output": str(python_output),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT_JSON={report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
