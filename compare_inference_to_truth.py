"""
Compare saved model-inference outputs against the ground-truth targets and write
per-channel / overall loss statistics to a CSV.

This consumes the dense prediction tensor produced by ``run_model_inference.py``
(shape ``(n_geometries * n_waveforms * n_bands, out_channels, 32, 32)``, indexed as
``idx = geom*(n_waveforms*n_bands) + wave*n_bands + band``) and compares it to the
ground-truth target assembled like full-validation training
(``FullIndexTensorPairDataset`` / ``per_sample_loss.py``):

    - channel 0  : eigenfrequency, from ``eigenfrequency_<encoding>_full.pt`` (flat index)
    - channels 1..4 : displacement targets, from ``displacements_dataset.pt`` tensors 0–3

All ``n_geom * n_waveforms * n_bands`` samples are scored (the full test grid).

Outputs a CSV with:
    - rows   : metric (MAE, MSE) x statistic (average, best, p25, median, p75, worst)
    - columns: one per output channel, plus an ``overall_sample`` column

The "statistic" is computed over the per-sample loss distribution for that column:
each sample contributes one scalar per channel (mean of the per-pixel error over the
32x32 field) and one ``overall_sample`` scalar (mean over all channels x pixels).
``average`` is the mean, ``best``/``worst`` are the min/max sample, and p25/median/p75
are the quartile boundaries of the distribution.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from output_layout import resolve_script_output_dir
from per_sample_loss import load_displacements_dataset


EIGEN_CH0_FILES = {
    "uniform": "eigenfrequency_uniform_full.pt",
    "fft": "eigenfrequency_fft_full.pt",
}

# Output-channel semantics, matching run_model_inference.py / the training scripts.
CASE_CHANNEL_LABELS = {
    1: ["eigenfrequency"],
    4: ["disp_x_real", "disp_x_imag", "disp_y_real", "disp_y_imag"],
    5: ["eigenfrequency", "disp_x_real", "disp_x_imag", "disp_y_real", "disp_y_imag"],
}

# Statistic rows, in output order. (label -> kind)
STATISTICS = [
    ("average", "mean"),
    ("best", "min"),
    ("p25", "p25"),
    ("median", "p50"),
    ("p75", "p75"),
    ("worst", "max"),
]


def channel_labels(out_channels: int) -> list[str]:
    if out_channels in CASE_CHANNEL_LABELS:
        return [f"ch{i}_{name}" for i, name in enumerate(CASE_CHANNEL_LABELS[out_channels])]
    return [f"ch{i}" for i in range(out_channels)]


def find_predictions_file(inference_dir: Path) -> Path:
    cands = sorted(inference_dir.glob("predictions_*.pt"))
    if not cands:
        cands = sorted(inference_dir.glob("*.pt"))
        cands = [c for c in cands if c.name != "selected_geometry_indices.pt"]
    if not cands:
        raise FileNotFoundError(f"No predictions_*.pt found in {inference_dir}")
    if len(cands) > 1:
        print(f"Warning: multiple prediction tensors in {inference_dir}; using {cands[0].name}")
    return cands[0]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def compute_per_sample_losses(
    predictions: torch.Tensor,
    eigen_flat: torch.Tensor,
    displacements,
    out_channels: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mae, mse) arrays of shape (M, out_channels), one row per dense sample.

    Flat index ``j`` matches ``run_model_inference.py``:
      ``j = geom*(n_waveforms*n_bands) + wave*n_bands + band``.

    Truth ch0 comes from ``eigen_flat[j]``; truth ch1–4 from
    ``displacements_dataset.pt`` tensors 0–3 at the same flat index.
    Errors are averaged over the 32x32 field to produce one scalar per channel.
    """
    m = int(predictions.shape[0])
    mae = np.empty((m, out_channels), dtype=np.float64)
    mse = np.empty((m, out_channels), dtype=np.float64)

    for start in tqdm(range(0, m, batch_size), desc="Scoring", unit="batch"):
        end = min(start + batch_size, m)

        pred_b = predictions[start:end].to(device, dtype=torch.float32)
        truth_ch0 = eigen_flat[start:end].to(device, dtype=torch.float32).unsqueeze(1)
        if out_channels == 1:
            truth_b = truth_ch0
        else:
            rest = torch.stack(
                [displacements.tensors[k][start:end] for k in range(out_channels - 1)],
                dim=1,
            ).to(device, dtype=torch.float32)
            truth_b = torch.cat([truth_ch0, rest], dim=1)

        err = pred_b - truth_b
        mae_b = err.abs().mean(dim=(2, 3))
        mse_b = err.square().mean(dim=(2, 3))

        mae[start:end] = mae_b.double().cpu().numpy()
        mse[start:end] = mse_b.double().cpu().numpy()

    return mae, mse


def summarize_column(values: np.ndarray) -> dict[str, float]:
    """Compute the statistic rows for a single (M,) per-sample loss vector."""
    return {
        "average": float(values.mean()),
        "best": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "worst": float(values.max()),
    }


def build_rows(
    mae: np.ndarray,
    mse: np.ndarray,
    col_labels: list[str],
) -> list[dict[str, object]]:
    """Build CSV row dicts: one per (metric, statistic), columns = channels + overall."""
    overall_mae = mae.mean(axis=1)
    overall_mse = mse.mean(axis=1)

    summaries: dict[str, dict[str, dict[str, float]]] = {"MAE": {}, "MSE": {}}
    for c, label in enumerate(col_labels):
        summaries["MAE"][label] = summarize_column(mae[:, c])
        summaries["MSE"][label] = summarize_column(mse[:, c])
    summaries["MAE"]["overall_sample"] = summarize_column(overall_mae)
    summaries["MSE"]["overall_sample"] = summarize_column(overall_mse)

    all_cols = col_labels + ["overall_sample"]
    rows: list[dict[str, object]] = []
    for metric in ("MAE", "MSE"):
        for stat_label, _ in STATISTICS:
            row: dict[str, object] = {"metric": metric, "statistic": stat_label}
            for col in all_cols:
                row[col] = summaries[metric][col][stat_label]
            rows.append(row)
    return rows


def write_csv(out_csv: Path, rows: list[dict[str, object]], col_labels: list[str]) -> None:
    fieldnames = ["metric", "statistic", *col_labels, "overall_sample"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v) for k, v in row.items()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inference-dir", help="Folder containing predictions_*.pt (output of run_model_inference.py).")
    p.add_argument("--predictions", help="Explicit path to the prediction tensor (.pt). Overrides --inference-dir.")
    p.add_argument(
        "--dataset-pt-dir",
        required=True,
        help="Dataset *_pt folder with eigenfrequency_*_full.pt and displacements_dataset.pt.",
    )
    p.add_argument("--eigen-encoding", choices=tuple(EIGEN_CH0_FILES), default="uniform", help="Channel-0 eigenfrequency encoding to compare against (default: uniform).")
    p.add_argument("--out-csv", default="", help="Explicit CSV output path (overrides the model/dataset layout below).")
    p.add_argument("--model-name", default="", help="Model name for the INFERENCE/<model>/<dataset>/<subdir> layout.")
    p.add_argument("--dataset", default="", help="Dataset folder for the layout (default: dataset *_pt parent name).")
    p.add_argument("--output-subdir", default="", help="Optional script output folder name under INFERENCE/<model>/<dataset> (default: none).")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="cpu", choices=("auto", "cuda", "cpu"),
                   help="Compute device (default: cpu). Use 'cuda' or 'auto' to opt into GPU.")
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.predictions:
        pred_path = Path(args.predictions)
        inference_dir = pred_path.parent
    elif args.inference_dir:
        inference_dir = Path(args.inference_dir)
        pred_path = find_predictions_file(inference_dir)
    else:
        raise SystemExit("Provide --predictions or --inference-dir.")
    print(f"Predictions: {pred_path}")

    pt_dir = Path(args.dataset_pt_dir)
    eigen_path = pt_dir / EIGEN_CH0_FILES[args.eigen_encoding]
    if not eigen_path.exists():
        raise FileNotFoundError(f"Missing required file: {eigen_path}")

    print("Loading tensors (memory-mapped where possible)...")
    predictions = torch.load(pred_path, map_location="cpu", mmap=True, weights_only=True)
    eigen_ch0 = torch.load(eigen_path, map_location="cpu", mmap=True, weights_only=True)
    displacements = load_displacements_dataset(pt_dir)

    out_channels = int(predictions.shape[1])
    n_geom = int(eigen_ch0.shape[0])
    n_waveforms = int(eigen_ch0.shape[1])
    n_bands = int(eigen_ch0.shape[2])
    field_h = int(eigen_ch0.shape[3])
    field_w = int(eigen_ch0.shape[4])
    expected_dense = n_geom * n_waveforms * n_bands

    print(f"  predictions     : {tuple(predictions.shape)} dtype={predictions.dtype}")
    print(f"  eigen_ch0       : {tuple(eigen_ch0.shape)} dtype={eigen_ch0.dtype} ({args.eigen_encoding})")
    print(f"  displacements   : 4 x {tuple(displacements.tensors[0].shape)}")
    print(
        f"  index_source    : indices_full (dense flat, M={expected_dense})"
    )
    print(f"  out_channels={out_channels}  n_geom={n_geom}  n_waveforms={n_waveforms}  n_bands={n_bands}")

    if predictions.shape[0] != expected_dense:
        raise ValueError(
            f"Prediction sample count {predictions.shape[0]} != n_geom*n_waveforms*n_bands "
            f"({expected_dense}). The prediction tensor must be a dense full-dataset run "
            f"(no --geometries subset) for index alignment to hold."
        )
    if tuple(predictions.shape[2:]) != (field_h, field_w):
        raise ValueError(
            f"Prediction field size {tuple(predictions.shape[2:])} != dataset {field_h}x{field_w}."
        )
    for k, disp in enumerate(displacements.tensors):
        if tuple(disp.shape) != (expected_dense, field_h, field_w):
            raise ValueError(
                f"displacements_dataset tensor {k} shape {tuple(disp.shape)} != "
                f"expected ({expected_dense}, {field_h}, {field_w})."
            )

    eigen_flat = eigen_ch0.reshape(expected_dense, field_h, field_w)
    col_labels = channel_labels(out_channels)

    mae, mse = compute_per_sample_losses(
        predictions=predictions,
        eigen_flat=eigen_flat,
        displacements=displacements,
        out_channels=out_channels,
        device=device,
        batch_size=args.batch_size,
    )

    rows = build_rows(mae, mse, col_labels)

    dataset_tag = args.dataset or pt_dir.parent.name  # e.g. c_test / b_test
    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_dir = resolve_script_output_dir(
            explicit=None,
            category="inference",
            model_name=args.model_name or None,
            dataset=dataset_tag,
            subdir=args.output_subdir,
            fallback=inference_dir,
        )
        out_csv = out_dir / f"loss_comparison_{dataset_tag}.csv"
    write_csv(out_csv, rows, col_labels)

    # Console summary.
    all_cols = col_labels + ["overall_sample"]
    print("\nLoss comparison (rows = metric/statistic, columns = channels + overall):")
    header = "metric  statistic   " + "  ".join(f"{c:>16}" for c in all_cols)
    print(header)
    for row in rows:
        cells = "  ".join(f"{float(row[c]):16.6e}" for c in all_cols)
        print(f"{row['metric']:<6}  {row['statistic']:<9}  {cells}")
    print(f"\nWrote: {out_csv}")
    print(f"Scored {mae.shape[0]} full-grid samples across {out_channels} channels.")


if __name__ == "__main__":
    main()
