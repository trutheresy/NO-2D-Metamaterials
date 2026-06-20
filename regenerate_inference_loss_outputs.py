"""Regenerate per-sample loss arrays and downstream plots for C_/B_ inference folders."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
LOSSES = ["mae", "mse", "nmae", "nmse", "nrms"]
PRED_NAME = "predictions_I3O5_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat.pt"

DATASETS = [
    {
        "tag": "c_test",
        "pt_dir": ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt",
        "geometries": ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt/geometries_full.pt",
        "inf_dir": ROOT
        / "INFERENCE/C_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-023704",
        "boundary_scatter": False,
    },
    {
        "tag": "b_test",
        "pt_dir": ROOT / "DATASETS/b_test/binarized_2026-03-08_16-34-27_pt",
        "geometries": ROOT / "DATASETS/b_test/binarized_2026-03-08_16-34-27_pt/geometries_full.pt",
        "inf_dir": ROOT
        / "INFERENCE/B_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-024713",
        "boundary_scatter": True,
    },
]


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def loss_folder(inf_dir: Path, loss: str) -> Path:
    return inf_dir / f"{loss.upper()}_sample_case_plots"


def loss_npy(inf_dir: Path, loss: str, tag: str) -> Path:
    return loss_folder(inf_dir, loss) / f"per_sample_loss_{loss}_{tag}.npy"


def main() -> None:
    zero_report: list[str] = []

    for cfg in DATASETS:
        tag = cfg["tag"]
        inf_dir = cfg["inf_dir"]
        pt_dir = cfg["pt_dir"]
        pred = inf_dir / PRED_NAME
        if not pred.is_file():
            raise FileNotFoundError(pred)

        print(f"\n========== {tag} ==========", flush=True)
        for loss in LOSSES:
            out_dir = loss_folder(inf_dir, loss)
            out_dir.mkdir(parents=True, exist_ok=True)
            for png in out_dir.glob("*.png"):
                png.unlink()
            run(
                [
                    PYTHON,
                    "per_sample_loss.py",
                    "--dataset-pt-dir",
                    str(pt_dir),
                    "--inference",
                    str(pred),
                    "--losses",
                    loss,
                    "--tag",
                    tag,
                    "--output-dir",
                    str(out_dir),
                ]
            )

        run(
            [
                PYTHON,
                "plot_loss_histograms.py",
                "--dataset-pt-dir",
                str(pt_dir),
                "--inference",
                str(pred),
                "--losses",
                *LOSSES,
                "--tag",
                tag,
                "--output-dir",
                str(inf_dir),
            ]
        )

        for loss in LOSSES:
            plot_dir = loss_folder(inf_dir, loss)
            for png in plot_dir.glob("*.png"):
                png.unlink()
            run(
                [
                    PYTHON,
                    "plot_sample_cases.py",
                    "--dataset-pt-dir",
                    str(pt_dir),
                    "--predictions",
                    str(pred),
                    "--loss-array",
                    loss,
                    str(loss_npy(inf_dir, loss, tag)),
                    "--output-dir",
                    str(loss_folder(inf_dir, loss)),
                    "--tag",
                    tag,
                    "--no-show-eigfreq",
                    "--no-title",
                ]
            )

        for sub in ("NMAE_high_loss_samples", "NMSE_high_loss_samples"):
            d = inf_dir / sub
            if d.is_dir():
                for png in d.glob("*.png"):
                    png.unlink()
        run(
            [
                PYTHON,
                "plot_high_loss_samples.py",
                "--dataset-pt-dir",
                str(pt_dir),
                "--predictions",
                str(pred),
                "--loss-array",
                "nmae",
                str(loss_npy(inf_dir, "nmae", tag)),
                "--loss-array",
                "nmse",
                str(loss_npy(inf_dir, "nmse", tag)),
                "--output-dir",
                str(inf_dir),
                "--tag",
                tag,
                "--no-show-eigfreq",
            ]
        )

        if cfg["boundary_scatter"]:
            run(
                [
                    PYTHON,
                    "scatter_loss_vs_boundary.py",
                    "--dataset-pt-dir",
                    str(pt_dir),
                    "--inference",
                    str(pred),
                    "--geometries",
                    str(cfg["geometries"]),
                    "--losses",
                    "mae",
                    "mse",
                    "nmae",
                    "nmse",
                    "nrms",
                    "--tag",
                    tag,
                    "--output-dir",
                    str(inf_dir / "boundary_length_vs_loss"),
                ]
            )
            run(
                [
                    PYTHON,
                    "scatter_loss_vs_boundary_by_band.py",
                    "--dataset-pt-dir",
                    str(pt_dir),
                    "--inference",
                    str(pred),
                    "--geometries",
                    str(cfg["geometries"]),
                    "--losses",
                    "mae",
                    "mse",
                    "nmae",
                    "nmse",
                    "nrms",
                    "--tag",
                    tag,
                    "--output-dir",
                    str(inf_dir / "boundary_length_vs_loss_by_band"),
                ]
            )

        for loss in LOSSES:
            arr = np.load(loss_npy(inf_dir, loss, tag))
            vals = arr[:, 4]
            n_zero = int(np.sum(vals == 0.0))
            n_neg = int(np.sum(vals < 0.0))
            n_nan = int(np.sum(~np.isfinite(vals)))
            zero_report.append(
                f"{tag} {loss.upper():4s}: zeros={n_zero}  negatives={n_neg}  non-finite={n_nan}  "
                f"min={vals.min():.6e}  max={vals.max():.6e}  mean={vals.mean():.6e}"
            )

    print("\n========== ZERO-LOSS REPORT ==========", flush=True)
    for line in zero_report:
        print(line, flush=True)


if __name__ == "__main__":
    main()
