"""Regenerate per-sample loss arrays and downstream plots under the PLOTS/INFERENCE layout.

Outputs follow the shared layout (see output_layout.py):

    PLOTS/<model>/<dataset>/<script-output-folder>/      # plots (+ the .npy / .csv that ride with them)
    INFERENCE/<model>/<dataset>/<predictions/...>         # dense prediction tensors and data files

The dense prediction tensor for each dataset is read from
``INFERENCE/<model>/<dataset>/<PRED_NAME>``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from output_layout import resolve_output_dir

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
LOSSES = ["mae", "mse", "nmae", "nmse", "nrms"]

# Default model (the 260401 run that established the layout). Override with --model.
MODEL = "NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat"
PRED_NAME = "predictions_I3O5_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat.pt"

DATASETS = [
    {
        "tag": "c_test",
        "pt_dir": ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt",
        "geometries": ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt/geometries_full.pt",
        "boundary_scatter": False,
    },
    {
        "tag": "b_test",
        "pt_dir": ROOT / "DATASETS/b_test/binarized_2026-03-08_16-34-27_pt",
        "geometries": ROOT / "DATASETS/b_test/binarized_2026-03-08_16-34-27_pt/geometries_full.pt",
        "boundary_scatter": True,
    },
]


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def sample_case_subdir(loss: str) -> str:
    return f"{loss.upper()}_sample_case_plots"


def plots_dir(model: str, tag: str, subdir: str) -> Path:
    """PLOTS/<model>/<tag>/<subdir>, created if missing."""
    return resolve_output_dir("plots", model, tag, subdir)


def inference_dir(model: str, tag: str) -> Path:
    """INFERENCE/<model>/<tag>, created if missing."""
    return resolve_output_dir("inference", model, tag)


def loss_npy(model: str, tag: str, loss: str) -> Path:
    return plots_dir(model, tag, sample_case_subdir(loss)) / f"per_sample_loss_{loss}_{tag}.npy"


def find_predictions(inf_dir: Path, pred_name: str | None) -> Path:
    """Resolve the dense prediction tensor in an INFERENCE/<model>/<tag> folder."""
    if pred_name:
        p = inf_dir / pred_name
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    cands = sorted(inf_dir.glob("predictions_*.pt"))
    if not cands:
        raise FileNotFoundError(f"No predictions_*.pt in {inf_dir}")
    if len(cands) > 1:
        print(f"Warning: multiple prediction tensors in {inf_dir}; using {cands[0].name}")
    return cands[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL, help="Model folder name under PLOTS/ and INFERENCE/.")
    p.add_argument("--pred-name", default=None,
                   help="Prediction filename in INFERENCE/<model>/<tag>/ (default: auto-detect predictions_*.pt).")
    p.add_argument("--datasets", nargs="+", default=[d["tag"] for d in DATASETS],
                   help="Which dataset tags to process (default: all configured).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model
    selected = set(args.datasets)
    zero_report: list[str] = []

    for cfg in DATASETS:
        tag = cfg["tag"]
        if tag not in selected:
            continue
        pt_dir = cfg["pt_dir"]
        pred = find_predictions(inference_dir(model, tag), args.pred_name)

        print(f"\n========== {tag} ==========", flush=True)

        # 1) per-sample loss arrays (.npy) -> PLOTS/<model>/<tag>/<LOSS>_sample_case_plots/
        for loss in LOSSES:
            out_dir = plots_dir(model, tag, sample_case_subdir(loss))
            for png in out_dir.glob("*.png"):
                png.unlink()
            run([
                PYTHON, "per_sample_loss.py",
                "--dataset-pt-dir", str(pt_dir),
                "--inference", str(pred),
                "--losses", loss,
                "--tag", tag,
                "--model-name", model,
                "--dataset", tag,
                "--category", "plots",
                "--output-subdir", sample_case_subdir(loss),
            ])

        # 2) loss histograms -> PLOTS/<model>/<tag>/histograms/
        run([
            PYTHON, "plot_loss_histograms.py",
            "--dataset-pt-dir", str(pt_dir),
            "--inference", str(pred),
            "--losses", *LOSSES,
            "--tag", tag,
            "--model-name", model,
            "--dataset", tag,
            "--output-subdir", "histograms",
        ])

        # 3) per-percentile sample-case plots -> PLOTS/<model>/<tag>/<LOSS>_sample_case_plots/
        for loss in LOSSES:
            run([
                PYTHON, "plot_sample_cases.py",
                "--dataset-pt-dir", str(pt_dir),
                "--predictions", str(pred),
                "--loss-array", loss, str(loss_npy(model, tag, loss)),
                "--tag", tag,
                "--model-name", model,
                "--dataset", tag,
                "--output-subdir", sample_case_subdir(loss),
                "--no-show-eigfreq",
                "--no-title",
            ])

        # 4) high-loss analysis -> PLOTS/<model>/<tag>/high_loss_analysis/<LOSS>_high_loss_samples/
        hla = plots_dir(model, tag, "high_loss_analysis")
        for sub in ("NMAE_high_loss_samples", "NMSE_high_loss_samples"):
            d = hla / sub
            if d.is_dir():
                for png in d.glob("*.png"):
                    png.unlink()
        run([
            PYTHON, "plot_high_loss_samples.py",
            "--dataset-pt-dir", str(pt_dir),
            "--predictions", str(pred),
            "--loss-array", "nmae", str(loss_npy(model, tag, "nmae")),
            "--loss-array", "nmse", str(loss_npy(model, tag, "nmse")),
            "--tag", tag,
            "--model-name", model,
            "--dataset", tag,
            "--output-subdir", "high_loss_analysis",
            "--no-show-eigfreq",
        ])

        # 5) boundary-length scatters (binary geometries only) -> PLOTS/<model>/<tag>/...
        if cfg["boundary_scatter"]:
            run([
                PYTHON, "scatter_loss_vs_boundary.py",
                "--dataset-pt-dir", str(pt_dir),
                "--inference", str(pred),
                "--geometries", str(cfg["geometries"]),
                "--losses", "mae", "mse", "nmae", "nmse", "nrms",
                "--tag", tag,
                "--model-name", model,
                "--dataset", tag,
                "--output-subdir", "boundary_length_vs_loss",
            ])
            run([
                PYTHON, "scatter_loss_vs_boundary_by_band.py",
                "--dataset-pt-dir", str(pt_dir),
                "--inference", str(pred),
                "--geometries", str(cfg["geometries"]),
                "--losses", "mae", "mse", "nmae", "nmse", "nrms",
                "--tag", tag,
                "--model-name", model,
                "--dataset", tag,
                "--output-subdir", "boundary_length_vs_loss_by_band",
            ])

        for loss in LOSSES:
            arr = np.load(loss_npy(model, tag, loss))
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
