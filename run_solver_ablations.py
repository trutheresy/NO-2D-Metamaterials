from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent
VALIDATION_DIR = REPO / "validation_eigenvalue_full_vs_matlab"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> str:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    p = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.stdout + "\n" + p.stderr


def parse_pt_dir(gen_output: str) -> Path:
    m = re.search(r"PyTorch dataset bundle saved to:\s*(.+)", gen_output)
    if not m:
        raise RuntimeError("Could not parse PT output path from generator output.")
    return Path(m.group(1).strip())


def run_one(tag: str, env_flags: dict[str, str]) -> dict:
    print(f"\n=== Running ablation: {tag} ===")
    gen_out = run_cmd(["python", "generate_dispersion_dataset_Han_Alex.py"], env=env_flags)
    pt_dir = parse_pt_dir(gen_out)

    geom_mat = VALIDATION_DIR / f"geometries_{tag}.mat"
    mat_out = VALIDATION_DIR / f"matlab_prescribed_{tag}.mat"
    rep_out = VALIDATION_DIR / f"solver_toggle_{tag}.json"

    run_cmd(
        [
            "python",
            "pt_tensor_to_mat.py",
            "--pt-path",
            str(pt_dir / "geometries_full.pt"),
            "--mat-out",
            str(geom_mat),
            "--var-name",
            "geometries_full",
        ]
    )

    matlab_cmd = (
        f"geometry_mat_path='{geom_mat.as_posix()}'; "
        f"output_mat_path='{mat_out.as_posix()}'; "
        f"run('{(REPO / 'generate_dispersion_from_prescribed_geometries.m').as_posix()}');"
    )
    run_cmd(["matlab", "-batch", matlab_cmd])

    run_cmd(
        [
            "python",
            "compare_eigenvalue_full_pt_vs_matlab.py",
            "--pt-dir",
            str(pt_dir),
            "--mat-out",
            str(mat_out),
            "--report-json",
            str(rep_out),
        ]
    )

    summary = json.loads(rep_out.read_text(encoding="utf-8"))["summary"]
    return {
        "tag": tag,
        "env_flags": env_flags,
        "pt_dir": str(pt_dir),
        "report": str(rep_out),
        "summary": summary,
    }


def main() -> None:
    runs = [
        ("baseline_no_flags", {}),
        ("use_complex128", {"PARITY_USE_COMPLEX128": "1"}),
        ("force_sparse_eigs", {"PARITY_FORCE_SPARSE_EIGS": "1"}),
        ("disable_neg_clamp", {"PARITY_DISABLE_NEG_CLAMP": "1"}),
        ("fr_float64", {"PARITY_FR_FLOAT64": "1"}),
        (
            "parity_bundle",
            {
                "PARITY_USE_COMPLEX128": "1",
                "PARITY_FORCE_SPARSE_EIGS": "1",
                "PARITY_DISABLE_NEG_CLAMP": "1",
                "PARITY_FR_FLOAT64": "1",
                "PARITY_WAVEVECTOR_FLOAT64": "1",
            },
        ),
    ]

    results = []
    for tag, env_flags in runs:
        results.append(run_one(tag, env_flags))

    out = VALIDATION_DIR / "solver_ablation_summary.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved solver ablation summary: {out}")
    for r in results:
        s = r["summary"]
        print(
            f"{r['tag']}: max={s['aligned_max_abs_over_structures']:.6f}, "
            f"mean={s['aligned_mean_abs_over_structures']:.6f}"
        )


if __name__ == "__main__":
    main()
