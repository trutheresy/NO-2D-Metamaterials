"""Delete stale case/high-loss PNGs and regenerate from existing .npy arrays."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
PRED_NAME = "predictions_I3O5_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat.pt"
LOSSES = ["mae", "mse", "nmae", "nmse", "nrms"]

DATASETS = [
    ("c_test", ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt",
     ROOT / "INFERENCE/C_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-023704"),
    ("b_test", ROOT / "DATASETS/b_test/binarized_2026-03-08_16-34-27_pt",
     ROOT / "INFERENCE/B_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-024713"),
]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    for tag, pt_dir, inf_dir in DATASETS:
        pred = inf_dir / PRED_NAME
        for loss in LOSSES:
            out_dir = inf_dir / f"{loss.upper()}_sample_case_plots"
            for png in out_dir.glob("*.png"):
                png.unlink()
            run([
                PYTHON, "plot_sample_cases.py",
                "--dataset-pt-dir", str(pt_dir),
                "--predictions", str(pred),
                "--loss-array", loss, str(out_dir / f"per_sample_loss_{loss}_{tag}.npy"),
                "--output-dir", str(out_dir),
                "--tag", tag, "--no-show-eigfreq", "--no-title",
            ])
        for sub in ("NMAE_high_loss_samples", "NMSE_high_loss_samples"):
            d = inf_dir / sub
            if d.is_dir():
                for png in d.glob("*.png"):
                    png.unlink()
        run([
            PYTHON, "plot_high_loss_samples.py",
            "--dataset-pt-dir", str(pt_dir),
            "--predictions", str(pred),
            "--loss-array", "nmae", str(inf_dir / "NMAE_sample_case_plots" / f"per_sample_loss_nmae_{tag}.npy"),
            "--loss-array", "nmse", str(inf_dir / "NMSE_sample_case_plots" / f"per_sample_loss_nmse_{tag}.npy"),
            "--output-dir", str(inf_dir),
            "--tag", tag, "--no-show-eigfreq",
        ])
        print(f"{tag}: plots refreshed", flush=True)


if __name__ == "__main__":
    main()
