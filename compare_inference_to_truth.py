"""
Compare saved model-inference outputs against the ground-truth targets and write
per-channel / overall loss statistics to a CSV.

This consumes the dense prediction tensor produced by ``run_model_inference.py``
(shape ``(n_geometries * n_waveforms * n_bands, out_channels, 32, 32)``, indexed as
``idx = geom*(n_waveforms*n_bands) + wave*n_bands + band``) and compares it to the
ground-truth target assembled exactly like the trainer / ``evaluate_from_disk.py``:

    - channel 0  : eigenfrequency, from ``eigenfrequency_<encoding>_full.pt[d, w, b]``
    - channels 1.. : displacement targets, from ``outputs.pt[j, 1:out_channels]``

where ``(d, w, b) = reduced_indices.pt[j]`` selects the valid (geometry, wavevector,
band) combinations. Only those reduced combinations are scored (the dense prediction
tensor is gathered at the matching flat index for each ``j``).

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
    reduced_indices: torch.Tensor,
    outputs: torch.Tensor,
    eigen_ch0: torch.Tensor,
    out_channels: int,
    n_waveforms: int,
    n_bands: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mae, mse) arrays of shape (M, out_channels), one row per reduced sample.

    For each reduced sample j with triplet (d, w, b):
      - prediction  = predictions[d*n_waveforms*n_bands + w*n_bands + b]
      - truth ch0   = eigen_ch0[d, w, b]
      - truth ch1.. = outputs[j, 1:out_channels]
    Errors are averaged over the 32x32 field to produce one scalar per channel.
    """
    m = reduced_indices.shape[0]
    mae = np.empty((m, out_channels), dtype=np.float64)
    mse = np.empty((m, out_channels), dtype=np.float64)

    d_all = reduced_indices[:, 0].to(torch.long)
    w_all = reduced_indices[:, 1].to(torch.long)
    b_all = reduced_indices[:, 2].to(torch.long)
    dense_all = d_all * (n_waveforms * n_bands) + w_all * n_bands + b_all

    for start in tqdm(range(0, m, batch_size), desc="Scoring", unit="batch"):
        end = min(start + batch_size, m)
        dense_idx = dense_all[start:end]
        d = d_all[start:end]
        w = w_all[start:end]
        b = b_all[start:end]

        pred_b = predictions[dense_idx].to(device, dtype=torch.float32)

        truth_ch0 = eigen_ch0[d, w, b].to(device, dtype=torch.float32).unsqueeze(1)
        if out_channels == 1:
            truth_b = truth_ch0
        else:
            rest = outputs[start:end, 1:out_channels].to(device, dtype=torch.float32)
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
    p.add_argument("--dataset-pt-dir", required=True, help="Dataset *_pt folder with reduced_indices.pt, outputs.pt, eigenfrequency_*_full.pt.")
    p.add_argument("--eigen-encoding", choices=tuple(EIGEN_CH0_FILES), default="uniform", help="Channel-0 eigenfrequency encoding to compare against (default: uniform).")
    p.add_argument("--out-csv", default="", help="CSV output path. Default: <inference-dir>/loss_comparison_<dataset>.csv")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
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
    ri_path = pt_dir / "reduced_indices.pt"
    out_path = pt_dir / "outputs.pt"
    eigen_path = pt_dir / EIGEN_CH0_FILES[args.eigen_encoding]
    for pth in (ri_path, out_path, eigen_path):
        if not pth.exists():
            raise FileNotFoundError(f"Missing required file: {pth}")

    print("Loading tensors (memory-mapped where possible)...")
    predictions = torch.load(pred_path, map_location="cpu", mmap=True, weights_only=True)
    outputs = torch.load(out_path, map_location="cpu", mmap=True, weights_only=True)
    eigen_ch0 = torch.load(eigen_path, map_location="cpu", mmap=True, weights_only=True)
    reduced_list = torch.load(ri_path, map_location="cpu", weights_only=False)
    reduced_indices = torch.as_tensor(reduced_list, dtype=torch.long)

    out_channels = int(predictions.shape[1])
    n_geom, n_waveforms, n_bands = int(eigen_ch0.shape[0]), int(eigen_ch0.shape[1]), int(eigen_ch0.shape[2])
    expected_dense = n_geom * n_waveforms * n_bands

    print(f"  predictions     : {tuple(predictions.shape)} dtype={predictions.dtype}")
    print(f"  outputs         : {tuple(outputs.shape)} dtype={outputs.dtype}")
    print(f"  eigen_ch0       : {tuple(eigen_ch0.shape)} dtype={eigen_ch0.dtype} ({args.eigen_encoding})")
    print(f"  reduced_indices : {tuple(reduced_indices.shape)} (M reduced samples)")
    print(f"  out_channels={out_channels}  n_geom={n_geom}  n_waveforms={n_waveforms}  n_bands={n_bands}")

    if predictions.shape[0] != expected_dense:
        raise ValueError(
            f"Prediction sample count {predictions.shape[0]} != n_geom*n_waveforms*n_bands "
            f"({expected_dense}). The prediction tensor must be a dense full-dataset run "
            f"(no --geometries subset) for index alignment to hold."
        )
    if outputs.shape[0] != reduced_indices.shape[0]:
        raise ValueError(
            f"outputs.pt rows ({outputs.shape[0]}) != reduced_indices length "
            f"({reduced_indices.shape[0]})."
        )
    if int(reduced_indices[:, 0].max()) >= n_geom:
        raise ValueError("reduced_indices geometry index exceeds eigen_ch0 geometry dim.")

    col_labels = channel_labels(out_channels)

    mae, mse = compute_per_sample_losses(
        predictions=predictions,
        reduced_indices=reduced_indices,
        outputs=outputs,
        eigen_ch0=eigen_ch0,
        out_channels=out_channels,
        n_waveforms=n_waveforms,
        n_bands=n_bands,
        device=device,
        batch_size=args.batch_size,
    )

    rows = build_rows(mae, mse, col_labels)

    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        dataset_tag = pt_dir.parent.name  # e.g. c_test / b_test
        out_csv = inference_dir / f"loss_comparison_{dataset_tag}.csv"
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
    print(f"Scored {mae.shape[0]} reduced samples across {out_channels} channels.")


if __name__ == "__main__":
    main()
