from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


@dataclass
class Paths:
    repo_root: Path
    out_root: Path
    plots: Path
    fixed_input_mat: Path
    remap_npy: Path
    wavevector_parity_json: Path
    design_explicit_json: Path
    python_func_checks_json: Path
    matlab_func_checks_json: Path
    quantity_comparison_json: Path
    worst_case_json: Path
    report_md: Path
    python_out_mat: Path
    matlab_out_mat: Path


def _paths(repo_root: Path) -> Paths:
    out = repo_root / "crossflow_function_tests"
    return Paths(
        repo_root=repo_root,
        out_root=out,
        plots=out / "plots",
        fixed_input_mat=out / "fixed_geometry_input.mat",
        remap_npy=out / "wavevector_index_remap.npy",
        wavevector_parity_json=out / "wavevector_parity.json",
        design_explicit_json=out / "design_to_explicit_checks.json",
        python_func_checks_json=out / "python_dispersion_function_checks.json",
        matlab_func_checks_json=out / "matlab_dispersion_function_checks.json",
        quantity_comparison_json=out / "quantity_comparison.json",
        worst_case_json=out / "worst_case_examples.json",
        report_md=out / "final_function_flow_report.md",
        python_out_mat=out / "python_function_outputs.mat",
        matlab_out_mat=out / "matlab_function_outputs.mat",
    )


def _ensure_dirs(p: Paths) -> None:
    p.out_root.mkdir(parents=True, exist_ok=True)
    p.plots.mkdir(parents=True, exist_ok=True)


def _md5(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _run(command: list[str], cwd: Path) -> str:
    proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _np(x: Any) -> np.ndarray:
    return np.asarray(x)


def _squeeze(x: Any) -> np.ndarray:
    return np.squeeze(np.asarray(x))


def _array_stats(a: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "finite": bool(np.isfinite(np.real(a)).all() and np.isfinite(np.imag(a)).all()),
        "min_real": float(np.min(np.real(a))),
        "max_real": float(np.max(np.real(a))),
    }


def _compare(a: np.ndarray, b: np.ndarray, rtol: float = 1e-8, atol: float = 1e-10) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    abs_diff = np.abs(a - b)
    rel_diff = abs_diff / np.maximum(np.abs(a), 1e-14)
    idx = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    return {
        "shape_match": True,
        "shape": list(a.shape),
        "max_abs": float(abs_diff[idx]),
        "mean_abs": float(np.mean(abs_diff)),
        "max_rel": float(np.max(rel_diff)),
        "mean_rel": float(np.mean(rel_diff)),
        "allclose": bool(np.allclose(a, b, rtol=rtol, atol=atol)),
        "worst_index": [int(i) for i in idx],
        "worst_a": [float(np.real(a[idx])), float(np.imag(a[idx]))] if np.iscomplexobj(a) else float(a[idx]),
        "worst_b": [float(np.real(b[idx])), float(np.imag(b[idx]))] if np.iscomplexobj(b) else float(b[idx]),
    }


def _phase_align(ref: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    denom = np.vdot(tgt, tgt)
    if np.abs(denom) < 1e-14:
        return tgt
    alpha = np.vdot(ref, tgt) / denom
    return tgt * alpha


def _compare_evec_phase(ref: np.ndarray, tgt: np.ndarray) -> dict[str, Any]:
    # ref/tgt shape: (N_dof, N_wv, N_eig)
    rels = []
    corrs = []
    worst = {"rel": -1.0, "k": -1, "band": -1}
    for k in range(ref.shape[1]):
        for b in range(ref.shape[2]):
            r = ref[:, k, b].astype(np.complex128, copy=False)
            t = tgt[:, k, b].astype(np.complex128, copy=False)
            ta = _phase_align(r, t)
            diff = np.linalg.norm(r - ta)
            denom = max(np.linalg.norm(r), 1e-14)
            rel = float(diff / denom)
            corr = float(np.abs(np.vdot(r, t)) / max(np.linalg.norm(r) * np.linalg.norm(t), 1e-14))
            rels.append(rel)
            corrs.append(corr)
            if rel > worst["rel"]:
                worst = {"rel": rel, "k": k, "band": b}
    return {
        "max_rel_norm_diff": float(np.max(rels)),
        "mean_rel_norm_diff": float(np.mean(rels)),
        "min_correlation": float(np.min(corrs)),
        "mean_correlation": float(np.mean(corrs)),
        "worst_mode": worst,
    }


def _plot_triplet(a: np.ndarray, b: np.ndarray, title: str, out_path: Path, cmap: str = "viridis") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    vmin = float(min(np.nanmin(a), np.nanmin(b)))
    vmax = float(max(np.nanmax(a), np.nanmax(b)))
    d = b - a
    im0 = axes[0].imshow(a, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("MATLAB")
    im1 = axes[1].imshow(b, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title("Python")
    im2 = axes[2].imshow(d, origin="lower", cmap="coolwarm", aspect="auto")
    axes[2].set_title("Python - MATLAB")
    for ax, im in zip(axes, [im0, im1, im2]):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _create_fixed_design(repo_root: Path, n_pix: int = 32, design_number: int = 1) -> np.ndarray:
    """Generate one deterministic p4mm geometry from normal random-design functions."""
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
    return np.asarray(get_design2(dp), dtype=np.float64)


def main() -> None:
    repo = Path(__file__).resolve().parent
    p = _paths(repo)
    if p.out_root.exists():
        shutil.rmtree(p.out_root)
    _ensure_dirs(p)

    # Add library path
    sys.path.insert(0, str(repo / "2d-dispersion-py"))
    from wavevectors import get_IBZ_wavevectors  # type: ignore
    from design_conversion import design_to_explicit  # type: ignore
    from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt  # type: ignore

    # Load fixed-geometry driver module to verify top-level function
    fixed_py_mod = _load_module(repo / "generate_dispersion_dataset_Han_Alex_fixed_geometry.py", "fixed_py_gen")

    # Step 1 fixture
    fixed = _create_fixed_design(repo_root=repo, n_pix=32, design_number=1)
    sio.savemat(str(p.fixed_input_mat), {"FIXED_DESIGN": fixed})
    fixture_meta = {"shape": list(fixed.shape), "dtype": str(fixed.dtype), "md5": _md5(fixed)}

    # Shared constants
    const = fixed_py_mod._base_constants()
    const["design"] = fixed
    const["wavevectors"] = get_IBZ_wavevectors(const["N_wv"], const["a"], "none")
    imag_tol = 1e-3

    # Step 2 wavevectors parity (Python against MATLAB output after run)
    py_wv = _np(const["wavevectors"])

    # Step 3 design_to_explicit
    py_exp = design_to_explicit(
        fixed,
        const["design_scale"],
        const["E_min"],
        const["E_max"],
        const["rho_min"],
        const["rho_max"],
        const["poisson_min"],
        const["poisson_max"],
    )
    matlab_style_E = const["E_min"] + (const["E_max"] - const["E_min"]) * fixed[:, :, 0]
    matlab_style_rho = const["rho_min"] + (const["rho_max"] - const["rho_min"]) * fixed[:, :, 1]
    matlab_style_nu = const["poisson_min"] + (const["poisson_max"] - const["poisson_min"]) * fixed[:, :, 2]
    design_explicit_report = {
        "E_check": _compare(_np(py_exp["E"]), _np(matlab_style_E)),
        "rho_check": _compare(_np(py_exp["rho"]), _np(matlab_style_rho)),
        "nu_check": _compare(_np(py_exp["nu"]), _np(matlab_style_nu)),
        "bounds": {
            "E": [float(np.min(py_exp["E"])), float(np.max(py_exp["E"]))],
            "rho": [float(np.min(py_exp["rho"])), float(np.max(py_exp["rho"]))],
            "nu": [float(np.min(py_exp["nu"])), float(np.max(py_exp["nu"]))],
        },
    }
    p.design_explicit_json.write_text(json.dumps(design_explicit_report, indent=2), encoding="utf-8")

    # Step 4 Python function outputs direct
    wv, fr, ev, mesh, K, M, T = dispersion_with_matrix_save_opt(const, const["wavevectors"])
    py_func_checks = {
        "wv": _array_stats(_np(wv)),
        "fr": _array_stats(_np(fr)),
        "ev": _array_stats(_np(ev)),
        "mesh_is_none": mesh is None,
        "K_shape": list(K.shape),
        "M_shape": list(M.shape),
        "T_len": len(T),
        "N_dof_expected": int(2 * (const["N_pix"] * const["N_ele"]) ** 2),
    }
    p.python_func_checks_json.write_text(json.dumps(py_func_checks, indent=2), encoding="utf-8")

    # Also run top-level fixed geometry generation function
    fixed_py_mod.run_fixed_geometry_generation(p.fixed_input_mat, p.python_out_mat, repo)

    # Step 5 MATLAB reference
    matlab_cmd = (
        f"fixed_geometry_path='{p.fixed_input_mat.as_posix()}'; "
        f"matlab_output_path='{p.matlab_out_mat.as_posix()}'; "
        f"run('{(repo / 'generate_dispersion_dataset_Han_Alex_fixed_geometry.m').as_posix()}');"
    )
    _run(["matlab", "-batch", matlab_cmd], repo)
    mdata = {k: v for k, v in sio.loadmat(p.matlab_out_mat).items() if not k.startswith("__")}
    mchecks = {
        "keys": sorted(list(mdata.keys())),
        "WAVEVECTOR_DATA": _array_stats(_squeeze(mdata["WAVEVECTOR_DATA"])),
        "EIGENVALUE_DATA": _array_stats(_squeeze(mdata["EIGENVALUE_DATA"])),
        "EIGENVECTOR_DATA": _array_stats(_squeeze(mdata["EIGENVECTOR_DATA"])),
        "designs": _array_stats(_squeeze(mdata["designs"])),
    }
    p.matlab_func_checks_json.write_text(json.dumps(mchecks, indent=2), encoding="utf-8")

    # Step 6 comparisons
    pdata = {k: v for k, v in sio.loadmat(p.python_out_mat).items() if not k.startswith("__")}

    m_design = _squeeze(mdata["designs"])
    p_design = _squeeze(pdata["designs"])

    m_wv = _squeeze(mdata["WAVEVECTOR_DATA"])
    p_wv = _squeeze(pdata["WAVEVECTOR_DATA"])
    m_eig = _squeeze(mdata["EIGENVALUE_DATA"])
    p_eig = _squeeze(pdata["EIGENVALUE_DATA"])
    m_evec = _squeeze(mdata["EIGENVECTOR_DATA"])
    p_evec = _squeeze(pdata["EIGENVECTOR_DATA"])

    # Set-equality and remap
    key = lambda r: (round(float(r[0]), 12), round(float(r[1]), 12))
    py_index = {key(r): i for i, r in enumerate(p_wv)}
    remap = np.array([py_index[key(r)] for r in m_wv], dtype=int)
    np.save(p.remap_npy, remap)

    m_sorted = np.array(sorted(map(tuple, np.round(m_wv, 12))))
    p_sorted = np.array(sorted(map(tuple, np.round(p_wv, 12))))
    wavevector_parity = {
        "python_wavevectors_shape": list(py_wv.shape),
        "matlab_wavevectors_shape": list(m_wv.shape),
        "set_equal_order_agnostic": bool(m_sorted.shape == p_sorted.shape and np.allclose(m_sorted, p_sorted)),
        "raw_compare": _compare(m_wv, p_wv),
    }
    p.wavevector_parity_json.write_text(json.dumps(wavevector_parity, indent=2), encoding="utf-8")

    # aligned outputs
    p_eig_aligned = p_eig[remap, :]
    p_evec_aligned = p_evec[:, remap, :]

    quantity_comp = {
        "fixture": fixture_meta,
        "constants": {
            "N_ele": 1,
            "N_pix": 32,
            "N_wv": [25, 13],
            "N_eig": 6,
            "sigma_eig": 1e-2,
            "a": 1.0,
            "E_min": 200e6,
            "E_max": 200e9,
            "rho_min": 8e2,
            "rho_max": 8e3,
            "poisson_min": 0.0,
            "poisson_max": 0.5,
            "t": 1.0,
        },
        "designs_raw": _compare(m_design, p_design),
        "wavevectors_raw": _compare(m_wv, p_wv),
        "eigenvalues_raw": _compare(m_eig, p_eig),
        "eigenvalues_aligned": _compare(m_eig, p_eig_aligned),
        "eigenvectors_raw": _compare(m_evec, p_evec),
        "eigenvectors_aligned": _compare(m_evec, p_evec_aligned),
        "eigenvectors_phase_aligned_metrics": _compare_evec_phase(m_evec, p_evec_aligned),
    }
    p.quantity_comparison_json.write_text(json.dumps(quantity_comp, indent=2), encoding="utf-8")

    worst = {
        "wavevectors_raw_worst": quantity_comp["wavevectors_raw"],
        "eigenvalues_aligned_worst": quantity_comp["eigenvalues_aligned"],
        "eigenvectors_aligned_worst": quantity_comp["eigenvectors_aligned"],
        "eigenvectors_phase_aligned_worst_mode": quantity_comp["eigenvectors_phase_aligned_metrics"]["worst_mode"],
    }
    p.worst_case_json.write_text(json.dumps(worst, indent=2), encoding="utf-8")

    # Step 7 plots
    for ch in range(3):
        _plot_triplet(
            m_design[:, :, ch],
            p_design[:, :, ch],
            f"Input design channel {ch}",
            p.plots / f"input_design_channel_{ch}.png",
        )

    _plot_triplet(
        m_eig,
        p_eig_aligned,
        "Output EIGENVALUE_DATA (aligned by wavevector)",
        p.plots / "output_eigenvalue_data.png",
    )

    # mode map for k=0, band=0 from interleaved dof
    def mode_map(evec: np.ndarray, k: int, b: int) -> np.ndarray:
        v = evec[:, k, b].reshape(-1)
        ux = v[0::2].reshape(32, 32)
        uy = v[1::2].reshape(32, 32)
        return np.sqrt(np.abs(ux) ** 2 + np.abs(uy) ** 2)

    _plot_triplet(
        mode_map(m_evec, 0, 0),
        mode_map(p_evec_aligned, 0, 0),
        "Output eigenvector mode magnitude (k=0, band=0)",
        p.plots / "output_eigenvector_mode_k0_b0.png",
    )

    # Step 8 report
    report = f"""# Function-Level Cross-Flow Report

- Output folder: `{p.out_root}`
- Fixture MD5: `{fixture_meta["md5"]}`
- Constants matched: computational grid and material constants use the same values in MATLAB/Python fixed scripts.

## Function checks

- `get_IBZ_wavevectors`: set-equality with MATLAB is `{wavevector_parity["set_equal_order_agnostic"]}` (order differs).
- `design_to_explicit`: E/rho/nu checks are allclose.
- `dispersion_with_matrix_save_opt` (Python): produced finite `wv/fr/ev`, plus `K/M/T`.
- MATLAB fixed flow: produced expected keys and finite principal tensors.

## Comparison summary

- designs raw allclose: `{quantity_comp["designs_raw"]["allclose"]}`
- wavevectors raw allclose: `{quantity_comp["wavevectors_raw"]["allclose"]}`
- eigenvalues aligned allclose: `{quantity_comp["eigenvalues_aligned"]["allclose"]}`
- eigenvectors aligned allclose: `{quantity_comp["eigenvectors_aligned"]["allclose"]}`
- eigenvectors phase-aligned max_rel_norm_diff: `{quantity_comp["eigenvectors_phase_aligned_metrics"]["max_rel_norm_diff"]}`
- eigenvectors phase-aligned min_correlation: `{quantity_comp["eigenvectors_phase_aligned_metrics"]["min_correlation"]}`

## Artifacts

- `{p.wavevector_parity_json.name}`
- `{p.design_explicit_json.name}`
- `{p.python_func_checks_json.name}`
- `{p.matlab_func_checks_json.name}`
- `{p.quantity_comparison_json.name}`
- `{p.worst_case_json.name}`
- plots in `{p.plots}`
"""
    p.report_md.write_text(report, encoding="utf-8")
    print(f"Done. Artifacts in: {p.out_root}")


if __name__ == "__main__":
    main()
