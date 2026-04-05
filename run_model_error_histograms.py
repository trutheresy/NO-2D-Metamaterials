from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


CHANNEL_NAMES = [
    "eigenfrequency_fft",
    "disp_x_real",
    "disp_x_imag",
    "disp_y_real",
    "disp_y_imag",
]


@dataclass
class ChannelStats:
    count: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    max_abs: float = 0.0

    def update(self, abs_err: torch.Tensor) -> None:
        v = abs_err.reshape(-1).to(torch.float64)
        self.count += int(v.numel())
        self.sum_abs += float(v.sum().item())
        self.sum_sq += float((v * v).sum().item())
        self.max_abs = max(self.max_abs, float(v.max().item()))

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "mae": float("nan"), "rmse": float("nan"), "max_abs_err": float("nan")}
        mae = self.sum_abs / self.count
        rmse = math.sqrt(self.sum_sq / self.count)
        return {"count": self.count, "mae": mae, "rmse": rmse, "max_abs_err": self.max_abs}


class FourierNeuralOperator(nn.Module):
    """Match train_from_disk.py architecture."""

    def __init__(self, modes_height: int, modes_width: int, hidden_channels: int, n_layers: int):
        super().__init__()
        from neuralop.models import FNO2d

        self.fno = FNO2d(
            in_channels=3,
            out_channels=5,
            n_modes_height=modes_height,
            n_modes_width=modes_width,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run model on datasets and save per-channel error histogram reports.")
    p.add_argument(
        "--model-path",
        default=(
            "D:/Research/NO-2D-Metamaterials/MODELS/training_runs/"
            "NO_I3O5_BCF16_L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_260311_20260311_175808_32860/"
            "NO_I3O5_BCF16_L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_260311_best.pth"
        ),
    )
    p.add_argument("--datasets-root", default="D:/Research/NO-2D-Metamaterials/DATASETS")
    p.add_argument("--datasets", nargs="+", default=["b_test", "c_test", "b_train_01", "c_train_01"])
    p.add_argument("--batch-size", type=int, default=260)
    p.add_argument("--max-samples", type=int, default=50000, help="0 means use all samples in each dataset.")
    p.add_argument("--sample-values-per-channel", type=int, default=300000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--modes-height", type=int, default=32)
    p.add_argument("--modes-width", type=int, default=32)
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--timeout-sec", type=int, default=3600)
    # internal single-dataset mode for robust subprocess execution
    p.add_argument("--one-dataset", default="")
    p.add_argument("--one-pt-dir", default="")
    return p.parse_args()


def latest_pt_dir(dataset_dir: Path) -> Path:
    cands = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not cands:
        raise FileNotFoundError(f"No *_pt folder under {dataset_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = FourierNeuralOperator(
        modes_height=args.modes_height,
        modes_width=args.modes_width,
        hidden_channels=args.hidden_channels,
        n_layers=args.layers,
    ).to(device)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        # Some checkpoints include framework metadata entries not part of nn.Module parameters.
        state_dict = {k: v for k, v in state_dict.items() if k != "_metadata"}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def save_channel_histograms(samples: dict[int, np.ndarray], out_dir: Path) -> list[str]:
    out_files = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for ch in range(5):
        vals = samples.get(ch, np.array([], dtype=np.float32))
        ax = axes[ch]
        if vals.size > 0:
            bins = max(60, min(240, int(round(math.sqrt(vals.size)))))
            ax.hist(vals, bins=bins, alpha=0.85)
            ax.set_yscale("log")
        ax.set_title(f"Abs Error Histogram - ch{ch} ({CHANNEL_NAMES[ch]})")
        ax.set_xlabel("absolute error")
        ax.set_ylabel("count")
        out_png = out_dir / f"model_error_hist_ch{ch}_{CHANNEL_NAMES[ch]}.png"
        # also save individual
        fig_i, ax_i = plt.subplots(figsize=(9, 5))
        if vals.size > 0:
            bins_i = max(60, min(240, int(round(math.sqrt(vals.size)))))
            ax_i.hist(vals, bins=bins_i, alpha=0.85)
            ax_i.set_yscale("log")
        ax_i.set_title(f"Abs Error Histogram - ch{ch} ({CHANNEL_NAMES[ch]})")
        ax_i.set_xlabel("absolute error")
        ax_i.set_ylabel("count")
        fig_i.tight_layout()
        fig_i.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig_i)
        out_files.append(str(out_png))
    axes[5].axis("off")
    fig.tight_layout()
    combo = out_dir / "model_error_hist_all_channels.png"
    fig.savefig(combo, dpi=160, bbox_inches="tight")
    plt.close(fig)
    out_files.append(str(combo))
    return out_files


def evaluate_one_dataset(args: argparse.Namespace, dataset_name: str, pt_dir: Path) -> dict[str, Any]:
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    model = load_model(args, device)

    x = torch.load(pt_dir / "inputs.pt", map_location="cpu", mmap=True, weights_only=True)
    y = torch.load(pt_dir / "outputs.pt", map_location="cpu", mmap=True, weights_only=True)
    n = int(min(x.shape[0], y.shape[0]))
    if args.max_samples > 0:
        n = min(n, int(args.max_samples))
    batch_size = max(1, int(args.batch_size))

    stats = {ch: ChannelStats() for ch in range(5)}
    collected: dict[int, list[np.ndarray]] = {ch: [] for ch in range(5)}
    per_batch_target = max(200, args.sample_values_per_channel // max(1, (n // batch_size) + 1))

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = x[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            yb = y[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            pred = model(xb)
            abs_err = (pred - yb).abs()
            for ch in range(5):
                ch_err = abs_err[:, ch, :, :]
                stats[ch].update(ch_err)
                flat = ch_err.reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)
                if flat.size > per_batch_target:
                    idx = rng.choice(flat.size, size=per_batch_target, replace=False)
                    flat = flat[idx]
                collected[ch].append(flat)
            del xb, yb, pred, abs_err

    samples: dict[int, np.ndarray] = {}
    for ch in range(5):
        if collected[ch]:
            merged = np.concatenate(collected[ch], axis=0)
            if merged.size > args.sample_values_per_channel:
                idx = rng.choice(merged.size, size=args.sample_values_per_channel, replace=False)
                merged = merged[idx]
            samples[ch] = merged
        else:
            samples[ch] = np.array([], dtype=np.float32)

    hist_files = save_channel_histograms(samples, pt_dir)

    channel_metrics = {}
    for ch in range(5):
        channel_metrics[f"ch{ch}_{CHANNEL_NAMES[ch]}"] = stats[ch].as_dict()

    report = {
        "dataset": dataset_name,
        "pt_dir": str(pt_dir),
        "model_path": str(Path(args.model_path)),
        "device": str(device),
        "batch_size": batch_size,
        "samples_evaluated": n,
        "sample_values_per_channel": int(args.sample_values_per_channel),
        "channels": channel_metrics,
        "hist_files": hist_files,
    }
    (pt_dir / "model_error_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Model Error Report - {dataset_name}",
        "",
        f"- model: `{args.model_path}`",
        f"- dataset folder: `{pt_dir}`",
        f"- device: `{device}`",
        f"- samples evaluated: `{n}`",
        f"- batch size: `{batch_size}`",
        "",
        "| channel | mae | rmse | max_abs_err | count |",
        "|---|---:|---:|---:|---:|",
    ]
    for ch in range(5):
        m = channel_metrics[f"ch{ch}_{CHANNEL_NAMES[ch]}"]
        lines.append(
            f"| ch{ch} `{CHANNEL_NAMES[ch]}` | {m['mae']:.6e} | {m['rmse']:.6e} | {m['max_abs_err']:.6e} | {m['count']} |"
        )
    lines += ["", "## Histogram Files"]
    for f in hist_files:
        lines.append(f"- `{f}`")
    (pt_dir / "model_error_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_all_datasets(args: argparse.Namespace) -> None:
    root = Path(args.datasets_root)
    script_path = Path(__file__).resolve()
    status: list[dict[str, Any]] = []
    for i, ds_name in enumerate(args.datasets, start=1):
        ds_dir = root / ds_name
        pt_dir = latest_pt_dir(ds_dir)
        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            "--model-path",
            args.model_path,
            "--datasets-root",
            str(root),
            "--one-dataset",
            ds_name,
            "--one-pt-dir",
            str(pt_dir),
            "--batch-size",
            str(args.batch_size),
            "--max-samples",
            str(args.max_samples),
            "--sample-values-per-channel",
            str(args.sample_values_per_channel),
            "--seed",
            str(args.seed + i),
            "--device",
            args.device,
            "--modes-height",
            str(args.modes_height),
            "--modes-width",
            str(args.modes_width),
            "--hidden-channels",
            str(args.hidden_channels),
            "--layers",
            str(args.layers),
        ]
        print(f"[run] {i}/{len(args.datasets)} {ds_name}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_sec)
        ok = (pt_dir / "model_error_report.md").exists() and (pt_dir / "model_error_report.json").exists()
        status.append(
            {
                "dataset": ds_name,
                "pt_dir": str(pt_dir),
                "ok": ok,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-1000:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
            }
        )
        print(f"[{'ok' if ok else 'fail'}] {ds_name}", flush=True)
    (root / "model_error_hist_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[done] status file: {root / 'model_error_hist_status.json'}")


def main() -> None:
    args = parse_args()
    if args.one_dataset:
        report = evaluate_one_dataset(args, args.one_dataset, Path(args.one_pt_dir))
        print(f"[done] {args.one_dataset} samples={report['samples_evaluated']}", flush=True)
        # Hard exit avoids occasional torch shutdown crash on this workstation.
        os._exit(0)
    run_all_datasets(args)


if __name__ == "__main__":
    main()
