"""Regenerate loss .npy arrays, percentile case plots, and histograms for C_/B_."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
LOSSES = ["mae", "mse", "nmae", "nmse", "nrms"]
NMAE_EPS = 1e-5
NMSE_EPS = 1e-5
PRED = "predictions_I3O5_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat.pt"

DATASETS = [
    (
        "c_test",
        ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt",
        ROOT / "INFERENCE/C_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-023704",
    ),
    (
        "b_test",
        ROOT / "DATASETS/b_test/binarized_2026-03-08_16-34-27_pt",
        ROOT / "INFERENCE/B_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-024713",
    ),
]


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    for tag, pt_dir, inf_dir in DATASETS:
        pred = inf_dir / PRED
        print(f"\n========== {tag} ==========", flush=True)

        for loss in LOSSES:
            out_dir = inf_dir / f"{loss.upper()}_sample_case_plots"
            out_dir.mkdir(parents=True, exist_ok=True)
            run([
                PYTHON, "per_sample_loss.py",
                "--dataset-pt-dir", str(pt_dir),
                "--inference", str(pred),
                "--losses", loss,
                "--tag", tag,
                "--output-dir", str(out_dir),
                "--nmae-eps", str(NMAE_EPS),
                "--nmse-eps", str(NMSE_EPS),
            ])

        for p in inf_dir.glob("loss_histogram*.png"):
            p.unlink()

        run([
            PYTHON, "plot_loss_histograms.py",
            "--dataset-pt-dir", str(pt_dir),
            "--inference", str(pred),
            "--losses", *LOSSES,
            "--tag", tag,
            "--output-dir", str(inf_dir),
            "--nmae-eps", str(NMAE_EPS),
            "--nmse-eps", str(NMSE_EPS),
        ])

        for loss in LOSSES:
            plot_dir = inf_dir / f"{loss.upper()}_sample_case_plots"
            for png in plot_dir.glob("*.png"):
                png.unlink()
            npy = plot_dir / f"per_sample_loss_{loss}_{tag}.npy"
            run([
                PYTHON, "plot_sample_cases.py",
                "--dataset-pt-dir", str(pt_dir),
                "--predictions", str(pred),
                "--loss-array", loss, str(npy),
                "--output-dir", str(plot_dir),
                "--tag", tag,
                "--no-show-eigfreq",
                "--no-title",
            ])

        print(f"{tag}: done", flush=True)


if __name__ == "__main__":
    main()
