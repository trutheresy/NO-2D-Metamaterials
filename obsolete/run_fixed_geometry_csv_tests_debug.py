"""
DEBUG ONLY: delete after debugging is complete.

Orchestrates fixed-geometry CSV-driven debug tests and writes consolidated reports.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio


REPO = Path(__file__).resolve().parent


def _run(command: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(command, cwd=str(cwd or REPO), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(command)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return (p.stdout or "") + "\n" + (p.stderr or "")


def _create_fixed_geometry_input(out_dir: Path) -> Path:
    sys.path.insert(0, str(REPO / "2d-dispersion-py"))
    from design_parameters import DesignParameters  # type: ignore
    from get_design2 import get_design2  # type: ignore

    dp = DesignParameters(1)
    dp.property_coupling = "coupled"
    dp.design_style = "kernel"
    dp.design_options = {
        "kernel": "periodic",
        "sigma_f": 1.0,
        "sigma_l": 1.0,
        "symmetry_type": "p4mm",
        "N_value": np.inf,
    }
    dp.N_pix = [32, 32]
    dp = dp.prepare()
    design = np.asarray(get_design2(dp), dtype=np.float64)
    design = np.clip(design, 0.0, 1.0)
    geom_path = out_dir / "fixed_geometry_input_debug.mat"
    sio.savemat(str(geom_path), {"FIXED_DESIGN": design})
    return geom_path


def _create_matlab_geometry_from_fixed(fixed_geom_mat: Path, out_mat: Path) -> None:
    d = sio.loadmat(str(fixed_geom_mat))
    fixed = np.asarray(d["FIXED_DESIGN"], dtype=np.float64)
    sio.savemat(str(out_mat), {"FIXED_DESIGN": fixed})


def _build_csv_row_test_map(csv_path: Path, out_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    mapped = []
    for r in rows:
        name = r.get("Mathematical Manipulation", "")
        if "Save " in name or "Create output" in name:
            test_type = "save_artifact_test"
        elif "wavevector" in name.lower() or "Transpose" in name:
            test_type = "ordering_test"
        elif "Initialize" in name or "Stack" in name or "Convert" in name:
            test_type = "shape_type_test"
        elif "Solve generalized EVP" in name or "Frequency conversion" in name:
            test_type = "parity_test"
        else:
            test_type = "range_physical_test"
        mapped.append(
            {
                "row_name": name,
                "test_type": test_type,
                "expected_equivalence": r.get("equivalence_status", ""),
                "deviation_risk": r.get("deviation_risk", ""),
            }
        )
    out_path.write_text(json.dumps({"row_test_map": mapped}, indent=2), encoding="utf-8")


def _build_logic_fix_list(row_report_path: Path, out_path: Path) -> None:
    rows = json.loads(row_report_path.read_text(encoding="utf-8")).get("rows", [])
    fixes = []
    for r in rows:
        if r.get("deviation_source") != "logic":
            continue
        row_name = r.get("row_name", "")
        if "wavevector" in row_name.lower():
            action = "Use deterministic remap contract when comparing MATLAB/Python wavevector-indexed arrays."
        elif "eigenvalue" in row_name.lower() or "EVP" in row_name:
            action = "Align eigensolver branch/settings for parity runs and compare in float64 before PT export."
        elif "Save " in row_name:
            action = "Define explicit cross-language schema contract for non-equivalent Python-only artifacts."
        else:
            action = "Review corresponding function implementation for logic-parity mismatch."
        fixes.append({"row_name": row_name, "proposed_fix": action})
    out_path.write_text(json.dumps({"logic_fix_list": fixes}, indent=2), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "debug_fixed_geometry_outputs" / timestamp
    py_stage = out_dir / "python_stage"
    mat_stage = out_dir / "matlab_stage"
    comp_stage = out_dir / "comparisons"
    row_stage = out_dir / "row_reports"
    for p in [out_dir, py_stage, mat_stage, comp_stage, row_stage]:
        p.mkdir(parents=True, exist_ok=True)

    csv_path = REPO / "dataset_generation_manipulations.csv"
    csv_map_path = row_stage / "csv_row_test_map.json"
    _build_csv_row_test_map(csv_path, csv_map_path)

    geometry_mat = _create_fixed_geometry_input(out_dir)
    py_out_mat = py_stage / "python_fixed_output_debug.mat"
    py_out_pt = py_stage / "python_fixed_output_debug_pt"
    py_debug = py_stage / "intermediates"
    mat_out_mat = mat_stage / "matlab_fixed_output_debug.mat"
    mat_debug = mat_stage / "intermediates"
    compare_summary = comp_stage / "comparison_summary_debug.json"
    intermediate_report = comp_stage / "intermediate_comparison_debug.json"
    full_saved_outputs_report = comp_stage / "full_saved_outputs_comparison_debug.json"
    row_report = row_stage / "row_report_debug.json"
    fix_list = row_stage / "logic_fix_list.json"

    # Stage 0: float16-first python debug generation
    _run(
        [
            sys.executable,
            str(REPO / "generate_dispersion_dataset_Han_Alex_fixed_geometry_debug.py"),
            "--geometry-mat",
            str(geometry_mat),
            "--output-mat",
            str(py_out_mat),
            "--output-pt-dir",
            str(py_out_pt),
            "--debug-dir",
            str(py_debug),
            "--repo-root",
            str(REPO),
            "--default-dtype",
            "float16",
        ]
    )

    # MATLAB debug generation
    # Build MATLAB geometry input directly from fixed design (pre-mapping), to avoid remapping NaN issues.
    geometries_for_matlab = mat_stage / "geometries_from_fixed_debug.mat"
    _create_matlab_geometry_from_fixed(geometry_mat, geometries_for_matlab)
    matlab_cmd = (
        f"geometry_mat_path='{geometries_for_matlab.as_posix()}'; "
        f"output_mat_path='{mat_out_mat.as_posix()}'; "
        f"debug_save_dir='{mat_debug.as_posix()}'; "
        f"run('{(REPO / 'generate_dispersion_from_prescribed_geometries_debug.m').as_posix()}');"
    )
    _run(["matlab", "-batch", matlab_cmd], cwd=REPO)

    # Comparison + row attribution
    _run(
        [
            sys.executable,
            str(REPO / "compare_eigenvalue_full_pt_vs_matlab_debug.py"),
            "--pt-dir",
            str(py_out_pt),
            "--mat-out",
            str(mat_out_mat),
            "--csv-path",
            str(csv_path),
            "--report-json",
            str(compare_summary),
            "--row-report-json",
            str(row_report),
            "--python-default-dtype",
            "float16",
        ]
    )
    _run(
        [
            sys.executable,
            str(REPO / "compare_intermediates_fixed_geometry_debug.py"),
            "--python-debug-dir",
            str(py_debug),
            "--matlab-stage-mat",
            str(mat_debug / "matlab_stage_struct_1.mat"),
            "--report-json",
            str(intermediate_report),
        ]
    )
    _run(
        [
            sys.executable,
            str(REPO / "compare_saved_outputs_full_fixed_debug.py"),
            "--pt-dir",
            str(py_out_pt),
            "--mat-out",
            str(mat_out_mat),
            "--report-json",
            str(full_saved_outputs_report),
        ]
    )

    _build_logic_fix_list(row_report, fix_list)

    final = {
        "output_root": str(out_dir),
        "csv_row_test_map": str(csv_map_path),
        "comparison_summary": str(compare_summary),
        "intermediate_comparison": str(intermediate_report),
        "full_saved_outputs_comparison": str(full_saved_outputs_report),
        "row_report": str(row_report),
        "logic_fix_list": str(fix_list),
    }
    (out_dir / "debug_suite_manifest.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
