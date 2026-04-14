from __future__ import annotations

import argparse
import bisect
import csv
import faulthandler
from functools import partial
import hashlib
import json
import logging
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from diagnostic_panels import save_random_test_diagnostic_panels


TRAIN_PREFIXES = ("c_train", "b_train")
TEST_PREFIXES = ("c_test", "b_test")
_MAIN_FAULT_FH: Any | None = None
_WORKER_FAULT_FH: Any | None = None

EIGEN_CH0_FILES = {
    "uniform": "eigenfrequency_uniform_full.pt",
    "fft": "eigenfrequency_fft_full.pt",
}

OUT_CHANNELS = 5


@dataclass
class ShardInfo:
    name: str
    pt_dir: Path
    inputs_path: Path
    outputs_path: Path
    reduced_indices_path: Path
    eigen_ch0_path: Path
    n: int


class ShardedTensorPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Disk-backed dataset: inputs.pt; output channel 0 from eigenfrequency_*_full.pt at
    (design, wavevector, band) from reduced_indices.pt; channels 1–4 from outputs.pt.
    """

    def __init__(self, shards: list[ShardInfo], eigen_ch0_encoding: str):
        if eigen_ch0_encoding not in EIGEN_CH0_FILES:
            raise ValueError(f"Unknown eigen_ch0_encoding: {eigen_ch0_encoding!r}")
        if not shards:
            raise ValueError("No shards were provided.")
        self.shards = shards
        self.eigen_ch0_encoding = eigen_ch0_encoding
        self._lengths = [s.n for s in shards]
        self._offsets = np.cumsum([0, *self._lengths]).tolist()
        self._total = self._offsets[-1]

        self._loaded_shard_idx: int | None = None
        self._loaded_inputs: torch.Tensor | None = None
        self._loaded_outputs: torch.Tensor | None = None
        self._loaded_eigen_ch0: torch.Tensor | None = None
        self._loaded_ridx: Any = None

    def __len__(self) -> int:
        return self._total

    @property
    def offsets(self) -> list[int]:
        """Global index offsets for each shard start; length = n_shards + 1."""
        return self._offsets

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index out of range: {idx}")
        shard_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[shard_idx]
        return shard_idx, local_idx

    def _load_shard(self, shard_idx: int) -> None:
        if self._loaded_shard_idx == shard_idx:
            return
        shard = self.shards[shard_idx]
        self._loaded_inputs = torch.load(shard.inputs_path, map_location="cpu", mmap=True, weights_only=True)
        self._loaded_outputs = torch.load(shard.outputs_path, map_location="cpu", mmap=True, weights_only=True)
        self._loaded_eigen_ch0 = torch.load(shard.eigen_ch0_path, map_location="cpu", mmap=True, weights_only=True)
        self._loaded_ridx = torch.load(shard.reduced_indices_path, map_location="cpu", weights_only=False)
        self._loaded_shard_idx = shard_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx, local_idx = self._resolve(idx)
        self._load_shard(shard_idx)
        assert self._loaded_inputs is not None and self._loaded_outputs is not None
        assert self._loaded_eigen_ch0 is not None and self._loaded_ridx is not None
        x = self._loaded_inputs[local_idx]
        triplet = self._loaded_ridx[local_idx]
        d, w, b = int(triplet[0]), int(triplet[1]), int(triplet[2])
        y0 = self._loaded_eigen_ch0[d, w, b]
        y_rest = self._loaded_outputs[local_idx, 1:5]
        y = torch.cat([y0.unsqueeze(0), y_rest], dim=0)
        return x, y


class ShardAwareBatchSampler(Sampler[list[int]]):
    """
    Shuffle globally at the shard/batch level while keeping each batch shard-local.

    This avoids heavy cross-shard random seeks that occur with default global index
    shuffling, preserving I/O locality and improving GPU feed throughput.
    """

    def __init__(
        self,
        shard_offsets: list[int],
        batch_size: int,
        drop_last: bool,
        seed: int = 0,
    ) -> None:
        if len(shard_offsets) < 2:
            raise ValueError("shard_offsets must include at least one shard start and end.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self._offsets = shard_offsets
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self._num_samples = int(shard_offsets[-1])
        self._num_batches = self._estimate_num_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _estimate_num_batches(self) -> int:
        n_batches = 0
        for s in range(len(self._offsets) - 1):
            n = self._offsets[s + 1] - self._offsets[s]
            if self.drop_last:
                n_batches += n // self.batch_size
            else:
                n_batches += (n + self.batch_size - 1) // self.batch_size
        return n_batches

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)

        shard_batches: list[list[list[int]]] = []
        for s in range(len(self._offsets) - 1):
            start = self._offsets[s]
            end = self._offsets[s + 1]
            n = end - start
            if n <= 0:
                shard_batches.append([])
                continue
            perm_local = torch.randperm(n, generator=gen).tolist()
            global_idx = [start + i for i in perm_local]
            batches: list[list[int]] = []
            for i in range(0, n, self.batch_size):
                b = global_idx[i : i + self.batch_size]
                if len(b) < self.batch_size and self.drop_last:
                    continue
                batches.append(b)
            shard_batches.append(batches)

        # Interleave shards in randomized rounds so no single shard dominates too long.
        ptr = [0] * len(shard_batches)
        active = [i for i, b in enumerate(shard_batches) if b]
        while active:
            order = torch.randperm(len(active), generator=gen).tolist()
            next_active: list[int] = []
            for k in order:
                s = active[k]
                p = ptr[s]
                if p < len(shard_batches[s]):
                    yield shard_batches[s][p]
                    p += 1
                    ptr[s] = p
                if p < len(shard_batches[s]):
                    next_active.append(s)
            active = next_active


class FourierNeuralOperator(nn.Module):
    """Match NO_trainer_4.ipynb model style via neuralop FNO2d."""

    def __init__(self, modes_height: int, modes_width: int, hidden_channels: int, n_layers: int):
        super().__init__()
        try:
            from neuralop.models import FNO2d
        except Exception as e:
            raise RuntimeError(
                "Failed to import neuralop FNO2d. "
                "This environment likely has a broken neuralop/wandb dependency chain. "
                "Fix env packages and retry."
            ) from e
        self.model = FNO2d(
            in_channels=3,
            out_channels=OUT_CHANNELS,
            n_modes_height=modes_height,
            n_modes_width=modes_width,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # Preserve unwrapped FNO2d checkpoint keys by delegating state_dict I/O to the inner model.
    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        return self.model.load_state_dict(state_dict, strict=strict)


def parse_args() -> argparse.Namespace:
    # DataLoader defaults (batch 520, workers 2, prefetch 3, pin_memory) match the stable Windows
    # I3O5 L1 disk run (~12.4 ks/epoch after resume vs ~25 ks/epoch with workers=4 on this machine).
    p = argparse.ArgumentParser(description="Disk-backed training pipeline with local file logging.")
    p.add_argument("--output-root", default="D:/Research/NO-2D-Metamaterials/DATASETS")
    p.add_argument("--save-dir", default="D:/Research/NO-2D-Metamaterials/MODELS/training_runs")
    p.add_argument(
        "--output-run-dir",
        default="",
        help="Optional run output directory. New runs write directly here; resumed runs copy --resume-run-dir here and continue in the copied directory.",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument(
        "--resume-run-dir",
        default="",
        help="Existing run directory to resume and append logs/checkpoints in-place.",
    )
    p.add_argument(
        "--extend-epochs",
        type=int,
        default=0,
        help="When resuming, train this many additional epochs beyond the last completed epoch.",
    )
    p.add_argument("--batch-size", type=int, default=520)
    p.add_argument(
        "--train-shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle training indices each epoch (recommended). Disable with --no-train-shuffle if needed.",
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=3)
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin DataLoader CPU memory for host->GPU transfer.",
    )
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--modes-height", type=int, default=32)
    p.add_argument("--modes-width", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--loss", choices=("mse", "l1", "smoothl1"), default="l1")
    p.add_argument("--scheduler", choices=("steplr", "cosine", "none"), default="steplr")
    p.add_argument("--step-size", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--t-max", type=int, default=0, help="CosineAnnealingLR T_max. 0 means epochs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback when CUDA is unavailable. Default is to require GPU.",
    )
    p.add_argument("--max-train-samples", type=int, default=0, help="0 means use all.")
    p.add_argument("--max-test-samples", type=int, default=0, help="0 means use all.")
    p.add_argument(
        "--eigen-ch0-encoding",
        choices=tuple(EIGEN_CH0_FILES.keys()),
        default="uniform",
        help="Output ch0 from eigenfrequency_uniform_full.pt (uniform) or eigenfrequency_fft_full.pt (fft). "
        "Channels 1–4 always come from outputs.pt.",
    )
    p.add_argument(
        "--progress-mode",
        choices=("tqdm", "plain"),
        default="tqdm",
        help="Progress output mode: tqdm bar or plain periodic logs.",
    )
    p.add_argument(
        "--log-every-batches",
        type=int,
        default=100,
        help="Heartbeat/postfix interval in train batches.",
    )
    p.add_argument(
        "--diagnostic-panels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each epoch's validation, save random test-set diagnostic PNGs (see diagnostic_panels).",
    )
    p.add_argument(
        "--diagnostic-samples",
        type=int,
        default=10,
        help="Number of random test samples to render when --diagnostic-panels is on.",
    )
    return p.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    raise RuntimeError(
        "CUDA is unavailable but GPU is required for training. "
        "If you intentionally want CPU fallback, pass --allow-cpu. "
        f"Current CUDA_VISIBLE_DEVICES={cuda_visible}"
    )


def latest_pt_dir(dataset_dir: Path) -> Path:
    cands = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not cands:
        raise FileNotFoundError(f"No *_pt folder under {dataset_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def discover_shards(output_root: Path, prefixes: tuple[str, ...], eigen_ch0_encoding: str) -> list[ShardInfo]:
    if eigen_ch0_encoding not in EIGEN_CH0_FILES:
        raise ValueError(f"Unknown eigen_ch0_encoding: {eigen_ch0_encoding!r}")
    eigen_fname = EIGEN_CH0_FILES[eigen_ch0_encoding]
    shards: list[ShardInfo] = []
    ds_dirs = sorted([p for p in output_root.iterdir() if p.is_dir() and p.name.startswith(prefixes)], key=lambda p: p.name)
    validated_contract = False
    for d in ds_dirs:
        pt = latest_pt_dir(d)
        in_path = pt / "inputs.pt"
        out_path = pt / "outputs.pt"
        ridx_path = pt / "reduced_indices.pt"
        eigen_path = pt / eigen_fname
        if not in_path.exists() or not out_path.exists():
            raise FileNotFoundError(f"Missing inputs/outputs in {pt}")
        if not ridx_path.exists():
            raise FileNotFoundError(f"Missing reduced_indices.pt in {pt}")
        if not eigen_path.exists():
            raise FileNotFoundError(
                f"Missing {eigen_fname} in {pt} (required for --eigen-ch0-encoding={eigen_ch0_encoding})"
            )

        ridx = torch.load(ridx_path, map_location="cpu", weights_only=False)
        n = len(ridx)

        if not validated_contract:
            x = torch.load(in_path, map_location="cpu", mmap=True, weights_only=True)
            y = torch.load(out_path, map_location="cpu", mmap=True, weights_only=True)
            eig = torch.load(eigen_path, map_location="cpu", mmap=True, weights_only=True)
            if x.ndim != 4 or y.ndim != 4:
                raise ValueError(f"Invalid tensor dims in {pt}: inputs={tuple(x.shape)}, outputs={tuple(y.shape)}")
            if tuple(x.shape[1:]) != (3, 32, 32):
                raise ValueError(f"Invalid inputs shape in {pt}: {tuple(x.shape)}")
            if tuple(y.shape[1:]) != (5, 32, 32):
                raise ValueError(f"Invalid outputs shape in {pt}: {tuple(y.shape)}")
            if eig.ndim != 5 or tuple(eig.shape[-2:]) != (32, 32):
                raise ValueError(
                    f"Invalid {eigen_fname} in {pt}: expected 5D with trailing (32,32), got shape={tuple(eig.shape)}"
                )
            if int(y.shape[0]) != n:
                raise ValueError(
                    f"Sample count mismatch in {pt}: outputs N={int(y.shape[0])} vs len(reduced_indices)={n}"
                )
            arr = np.asarray(ridx, dtype=np.int64)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"reduced_indices must be shape [N,3] when treated as array, got {arr.shape}")
            dmax, wmax, bmax = int(arr[:, 0].max()), int(arr[:, 1].max()), int(arr[:, 2].max())
            dmin, wmin, bmin = int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 2].min())
            if dmin < 0 or wmin < 0 or bmin < 0:
                raise ValueError(f"Negative (design,wavevector,band) index in reduced_indices under {pt}")
            if dmax >= eig.shape[0] or wmax >= eig.shape[1] or bmax >= eig.shape[2]:
                raise ValueError(
                    f"reduced_indices out of range for {eigen_fname} in {pt}: "
                    f"max indices ({dmax},{wmax},{bmax}) vs tensor shape {tuple(eig.shape[:3])}"
                )
            validated_contract = True

        shards.append(
            ShardInfo(
                name=d.name,
                pt_dir=pt,
                inputs_path=in_path,
                outputs_path=out_path,
                reduced_indices_path=ridx_path,
                eigen_ch0_path=eigen_path,
                n=n,
            )
        )
    if not shards:
        raise FileNotFoundError(f"No dataset shards found with prefixes={prefixes} under {output_root}")
    return shards


def dataset_version_hash(shards: list[ShardInfo], eigen_ch0_encoding: str) -> str:
    h = hashlib.sha256()
    h.update(eigen_ch0_encoding.encode("utf-8"))
    for s in shards:
        h.update(str(s.pt_dir).encode("utf-8"))
        h.update(str(s.inputs_path.stat().st_size).encode("utf-8"))
        h.update(str(s.outputs_path.stat().st_size).encode("utf-8"))
        h.update(str(s.eigen_ch0_path.stat().st_size).encode("utf-8"))
        h.update(str(int(s.inputs_path.stat().st_mtime)).encode("utf-8"))
        h.update(str(int(s.outputs_path.stat().st_mtime)).encode("utf-8"))
        h.update(str(int(s.eigen_ch0_path.stat().st_mtime)).encode("utf-8"))
    return h.hexdigest()[:12]


def build_run_name(args: argparse.Namespace) -> str:
    ds = datetime.now().strftime("%y%m%d")
    ch0_tag = "ch0u" if args.eigen_ch0_encoding == "uniform" else "ch0fft"
    loss_tag = {"mse": "L2", "l1": "L1", "smoothl1": "SL1"}[args.loss]
    return (
        f"NO_I3O5_BCF16_{loss_tag}_HC{args.hidden_channels}_"
        f"LR{args.learning_rate:.0e}_WD{args.weight_decay:.0e}_"
        f"SS{args.step_size}_G{args.gamma:.0e}_{ch0_tag}_{ds}"
    )


def git_info() -> dict[str, str]:
    out: dict[str, str] = {"git_commit": "unknown", "git_dirty": "unknown"}
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        out["git_commit"] = c.stdout.strip()
    except Exception:
        pass
    try:
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        out["git_dirty"] = "true" if d.stdout.strip() else "false"
    except Exception:
        pass
    return out


def maybe_cap_shards(shards: list[ShardInfo], cap: int) -> list[ShardInfo]:
    if cap <= 0:
        return shards
    out: list[ShardInfo] = []
    remaining = cap
    for s in shards:
        if remaining <= 0:
            break
        keep = min(s.n, remaining)
        out.append(
            ShardInfo(
                s.name,
                s.pt_dir,
                s.inputs_path,
                s.outputs_path,
                s.reduced_indices_path,
                s.eigen_ch0_path,
                keep,
            )
        )
        remaining -= keep
    return out


def amp_context(device: torch.device, amp_mode: str):
    if device.type != "cuda" or amp_mode == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    if amp_mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def build_criterion(loss_name: str) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "l1":
        return nn.L1Loss()
    return nn.SmoothL1Loss(beta=1e-5)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    t_max = args.t_max if args.t_max > 0 else args.epochs
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)


def per_channel_loss_mean(pred: torch.Tensor, yb: torch.Tensor, loss_name: str) -> torch.Tensor:
    """
    Per-channel mean of the given loss, shape [C]: mean over (N, H, W) per channel.
    Matches the per-channel breakdown implied by nn.MSELoss / L1Loss / SmoothL1Loss with default reduction.
    """
    with torch.no_grad():
        pf = pred.detach().float()
        yf = yb.float()
        if loss_name == "mse":
            return (pf - yf).square().mean(dim=(0, 2, 3)).cpu().to(torch.float64)
        if loss_name == "l1":
            return (pf - yf).abs().mean(dim=(0, 2, 3)).cpu().to(torch.float64)
        if loss_name == "smoothl1":
            return F.smooth_l1_loss(pf, yf, reduction="none", beta=1e-5).mean(dim=(0, 2, 3)).cpu().to(torch.float64)
        raise ValueError(f"Unknown loss_name: {loss_name!r}")


def metrics_csv_fieldnames(out_channels: int, *, dual_compare: bool) -> list[str]:
    cols = [
        "epoch",
        "train_loss",
        *[f"train_loss_ch{i}" for i in range(out_channels)],
        "val_loss",
        "lr",
        "epoch_time_sec",
        "data_time_sec",
        "train_samples_per_sec",
        "val_samples_per_sec",
        *[f"val_loss_ch{i}" for i in range(out_channels)],
    ]
    if dual_compare:
        cols += [
            "train_compare_loss",
            "val_compare_loss",
            *[f"train_compare_loss_ch{i}" for i in range(out_channels)],
            *[f"val_compare_loss_ch{i}" for i in range(out_channels)],
        ]
    return cols


def _to_float_or_nan(v: str | None) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _migrate_metrics_to_dual_header(metrics_csv: Path, out_channels: int) -> None:
    """
    Upgrade single-loss metrics.csv to dual-loss header in-place by seeding compare columns
    from the active loss columns. This preserves CSV append compatibility for resumed runs
    that switch active loss and need compare-loss tracking.
    """
    dual_header = metrics_csv_fieldnames(out_channels, dual_compare=True)
    single_header = metrics_csv_fieldnames(out_channels, dual_compare=False)
    with metrics_csv.open("r", newline="", encoding="utf-8") as rf:
        rows = list(csv.DictReader(rf))
        existing_header = rows[0].keys() if rows else single_header
    if list(existing_header) != single_header:
        return

    tmp_path = metrics_csv.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=dual_header)
        writer.writeheader()
        for row in rows:
            # For historical rows we only have one loss family in CSV; copy it into compare
            # columns as the best available apples-to-apples baseline.
            merged = {k: row.get(k, "") for k in single_header}
            merged["train_compare_loss"] = row.get("train_loss", "")
            merged["val_compare_loss"] = row.get("val_loss", "")
            for i in range(out_channels):
                merged[f"train_compare_loss_ch{i}"] = row.get(f"train_loss_ch{i}", "")
                merged[f"val_compare_loss_ch{i}"] = row.get(f"val_loss_ch{i}", "")
            writer.writerow(merged)
    tmp_path.replace(metrics_csv)


def _best_from_metrics_csv(metrics_csv: Path, metric_col: str) -> tuple[int, float]:
    if not metrics_csv.is_file():
        return -1, float("inf")
    best_epoch = -1
    best_val = float("inf")
    with metrics_csv.open("r", newline="", encoding="utf-8") as rf:
        for row in csv.DictReader(rf):
            val = _to_float_or_nan(row.get(metric_col))
            epoch = int(_to_float_or_nan(row.get("epoch")))
            if np.isfinite(val) and val < best_val:
                best_val = float(val)
                best_epoch = epoch
    return best_epoch, best_val


def _sync_best_weights_from_epoch(run_dir: Path, run_name: str, best_epoch: int, best_path: Path) -> None:
    if best_epoch < 0:
        return
    src = run_dir / f"{run_name}_E{best_epoch}.pth"
    if src.is_file():
        shutil.copyfile(src, best_path)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_mode: str,
    criterion: nn.Module,
    criterion_compare: nn.Module | None,
    active_loss_name: str,
    compare_loss_name: str | None,
) -> tuple[float, float | None, list[float], list[float] | None, float]:
    model.eval()
    running_loss = 0.0
    running_loss_compare = 0.0 if criterion_compare is not None else None
    running_per_ch = torch.zeros(OUT_CHANNELS, dtype=torch.float64)
    running_per_ch_cmp = (
        torch.zeros(OUT_CHANNELS, dtype=torch.float64) if compare_loss_name is not None else None
    )
    n_samples = 0
    start = time.time()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device, dtype=torch.float32, non_blocking=True)
            with amp_context(device, amp_mode):
                pred = model(xb)
                loss = criterion(pred, yb)
                loss_compare = criterion_compare(pred, yb) if criterion_compare is not None else None
            bs = xb.shape[0]
            running_loss += float(loss.item()) * bs
            if running_loss_compare is not None and loss_compare is not None:
                running_loss_compare += float(loss_compare.item()) * bs
            running_per_ch += per_channel_loss_mean(pred, yb, active_loss_name) * bs
            if running_per_ch_cmp is not None and compare_loss_name is not None:
                running_per_ch_cmp += per_channel_loss_mean(pred, yb, compare_loss_name) * bs
            n_samples += bs
    elapsed = max(time.time() - start, 1e-9)
    mean_loss = running_loss / max(n_samples, 1)
    mean_loss_compare = (running_loss_compare / max(n_samples, 1)) if running_loss_compare is not None else None
    mean_ch = (running_per_ch / max(n_samples, 1)).tolist()
    mean_ch_cmp = (
        (running_per_ch_cmp / max(n_samples, 1)).tolist() if running_per_ch_cmp is not None else None
    )
    sps = n_samples / elapsed
    return mean_loss, mean_loss_compare, mean_ch, mean_ch_cmp, sps


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train_from_disk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def env_info() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def emit_progress(message: str, use_tqdm: bool) -> None:
    if use_tqdm:
        tqdm.write(message, file=sys.stdout)
    else:
        print(message)


def enable_main_fault_handler(run_dir: Path) -> str:
    global _MAIN_FAULT_FH
    fault_path = run_dir / "main_process_fault.log"
    _MAIN_FAULT_FH = fault_path.open("a", encoding="utf-8")
    faulthandler.enable(file=_MAIN_FAULT_FH, all_threads=True)
    return str(fault_path)


def dataloader_worker_init(_worker_id: int, run_dir_str: str) -> None:
    global _WORKER_FAULT_FH
    run_dir = Path(run_dir_str)
    fault_path = run_dir / f"worker_{os.getpid()}_fault.log"
    try:
        _WORKER_FAULT_FH = fault_path.open("a", encoding="utf-8")
        faulthandler.enable(file=_WORKER_FAULT_FH, all_threads=True)
        _WORKER_FAULT_FH.write(f"{datetime.now(timezone.utc).isoformat()} | worker_start pid={os.getpid()}\n")
        _WORKER_FAULT_FH.flush()
    except Exception:
        # Best-effort fallback so worker init failures are still visible.
        with (run_dir / f"worker_{os.getpid()}_init_error.log").open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} | worker init failed\n")
            f.write(traceback.format_exc())
            f.write("\n")


def save_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_val: float,
    best_epoch: int,
) -> None:
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict(),
    }
    torch.save(payload, path)


def _latest_epoch_from_ckpts(run_dir: Path) -> int:
    max_epoch = -1
    for ckpt in run_dir.glob("*_E*.pth"):
        stem = ckpt.stem
        if "_E" not in stem:
            continue
        try:
            n = int(stem.rsplit("_E", 1)[-1])
        except ValueError:
            continue
        max_epoch = max(max_epoch, n)
    return max_epoch


def _try_load_model_state_with_compat_fallback(
    model: nn.Module,
    state_dict: Any,
    run_dir: Path,
) -> tuple[bool, Path | None]:
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("_metadata", None)
    try:
        model.load_state_dict(state_dict)
        return True, None
    except RuntimeError:
        compat_ckpts = sorted(run_dir.glob("*best_fno2d_compat*.pth"))
        if not compat_ckpts:
            raise
        compat_path = compat_ckpts[-1]
        compat_blob = torch.load(compat_path, map_location="cpu", weights_only=False)
        compat_state = compat_blob["model_state_dict"] if isinstance(compat_blob, dict) and "model_state_dict" in compat_blob else compat_blob
        if isinstance(compat_state, dict) and "_metadata" in compat_state:
            compat_state = dict(compat_state)
            compat_state.pop("_metadata", None)
        model.load_state_dict(compat_state)
        return False, compat_path


def _infer_previous_loss_from_run(run_dir: Path) -> str | None:
    train_log = run_dir / "train.log"
    if train_log.is_file():
        loss_pat = re.compile(r"\bloss=([a-zA-Z0-9_]+)")
        try:
            last_match: str | None = None
            with train_log.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    m = loss_pat.search(line)
                    if m:
                        last_match = m.group(1).lower()
            if last_match in {"mse", "l1", "smoothl1"}:
                return last_match
            if last_match == "l2":
                return "mse"
        except Exception:
            pass
    rc_path = run_dir / "resolved_config.json"
    if rc_path.is_file():
        try:
            rc = json.loads(rc_path.read_text(encoding="utf-8"))
            loss = str(rc.get("args", {}).get("loss", "")).lower()
            if loss in {"mse", "l1", "smoothl1"}:
                return loss
        except Exception:
            pass
    return None


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    output_run_dir = Path(args.output_run_dir).resolve() if args.output_run_dir.strip() else None
    resume_mode = bool(args.resume_run_dir.strip())
    if resume_mode:
        src_run_dir = Path(args.resume_run_dir).resolve()
        if not src_run_dir.is_dir():
            raise FileNotFoundError(f"--resume-run-dir does not exist: {src_run_dir}")
        if output_run_dir is not None:
            if output_run_dir.exists():
                raise FileExistsError(f"--output-run-dir already exists: {output_run_dir}")
            shutil.copytree(src_run_dir, output_run_dir)
            run_dir = output_run_dir
        else:
            run_dir = src_run_dir
        run_metadata_path = run_dir / "run_metadata.json"
        if not run_metadata_path.is_file():
            raise FileNotFoundError(f"Missing run_metadata.json in resume dir: {run_dir}")
        prev_meta = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if output_run_dir is not None:
            run_name = run_dir.name
            run_id = run_name
        else:
            run_name = str(prev_meta.get("run_name", run_dir.name))
            run_id = str(prev_meta.get("run_id", run_dir.name))
        if args.extend_epochs <= 0:
            raise ValueError("--extend-epochs must be > 0 when --resume-run-dir is set.")
    else:
        if output_run_dir is not None:
            run_dir = output_run_dir
            if run_dir.exists():
                raise FileExistsError(f"--output-run-dir already exists: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=False)
            run_name = run_dir.name
            run_id = run_name
        else:
            run_name = build_run_name(args)
            run_id = run_name
            run_dir = save_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

    main_fault_path = enable_main_fault_handler(run_dir)

    logger = setup_logger(run_dir / "train.log")
    start_ts = datetime.now(timezone.utc)
    run_metadata_path = run_dir / "run_metadata.json"
    if resume_mode:
        metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "running"
        metadata["resumed_at_utc"] = start_ts.isoformat()
        metadata["resume_extend_epochs"] = int(args.extend_epochs)
    else:
        metadata = {
            "run_name": run_name,
            "run_id": run_id,
            "status": "running",
            "started_at_utc": start_ts.isoformat(),
            "git": git_info(),
            "environment": env_info(),
        }
    write_json(run_metadata_path, metadata)
    crash_state: dict[str, Any] = {"phase": "startup", "epoch": 0, "batch": 0}

    try:
        output_root = Path(args.output_root)
        train_shards = discover_shards(output_root, TRAIN_PREFIXES, args.eigen_ch0_encoding)
        test_shards = discover_shards(output_root, TEST_PREFIXES, args.eigen_ch0_encoding)
        train_shards = maybe_cap_shards(train_shards, args.max_train_samples)
        test_shards = maybe_cap_shards(test_shards, args.max_test_samples)

        train_ds = ShardedTensorPairDataset(train_shards, args.eigen_ch0_encoding)
        test_ds = ShardedTensorPairDataset(test_shards, args.eigen_ch0_encoding)

        train_loader_kwargs: dict[str, Any] = {
            "num_workers": args.num_workers,
            "pin_memory": bool(args.pin_memory),
            "drop_last": False,
        }
        test_loader_kwargs: dict[str, Any] = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": bool(args.pin_memory),
            "drop_last": False,
        }
        if args.num_workers > 0:
            train_loader_kwargs["persistent_workers"] = True
            train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
            train_loader_kwargs["worker_init_fn"] = partial(dataloader_worker_init, run_dir_str=str(run_dir))
            train_loader_kwargs["timeout"] = 180
            test_loader_kwargs["persistent_workers"] = True
            test_loader_kwargs["prefetch_factor"] = args.prefetch_factor
            test_loader_kwargs["worker_init_fn"] = partial(dataloader_worker_init, run_dir_str=str(run_dir))
            test_loader_kwargs["timeout"] = 180

        train_batch_sampler: ShardAwareBatchSampler | None = None
        if args.train_shuffle:
            train_batch_sampler = ShardAwareBatchSampler(
                shard_offsets=train_ds.offsets,
                batch_size=args.batch_size,
                drop_last=False,
                seed=args.seed,
            )
            train_loader_kwargs["batch_sampler"] = train_batch_sampler
        else:
            train_loader_kwargs["batch_size"] = args.batch_size
            train_loader_kwargs["shuffle"] = False
        train_loader = DataLoader(train_ds, **train_loader_kwargs)
        test_loader = DataLoader(test_ds, **test_loader_kwargs)

        device = resolve_device(args.allow_cpu)
        model = FourierNeuralOperator(
            modes_height=args.modes_height,
            modes_width=args.modes_width,
            hidden_channels=args.hidden_channels,
            n_layers=args.layers,
        ).to(device)
        inferred_prev_loss = _infer_previous_loss_from_run(run_dir) if resume_mode else None
        criterion = build_criterion(args.loss)
        compare_loss_name: str | None = inferred_prev_loss
        criterion_compare = build_criterion(compare_loss_name) if compare_loss_name and compare_loss_name != args.loss else None
        if criterion_compare is not None:
            logger.info("Dual-loss logging enabled | active_loss=%s compare_loss=%s", args.loss, compare_loss_name)
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))
        start_epoch = 1
        total_epochs = args.epochs
        best_val = float("inf")
        best_epoch = -1

        state_latest_path = run_dir / "training_state_latest.pt"
        if resume_mode:
            if state_latest_path.is_file():
                state_blob = torch.load(state_latest_path, map_location="cpu", weights_only=False)
                state_dict = state_blob["model_state_dict"]
                resumed_full_state, compat_path = _try_load_model_state_with_compat_fallback(model, state_dict, run_dir)
                if resumed_full_state:
                    optimizer.load_state_dict(state_blob["optimizer_state_dict"])
                    if scheduler is not None and state_blob.get("scheduler_state_dict") is not None:
                        scheduler.load_state_dict(state_blob["scheduler_state_dict"])
                    scaler.load_state_dict(state_blob.get("scaler_state_dict", {}))
                    start_epoch = int(state_blob["epoch"]) + 1
                    best_val = float(state_blob.get("best_val_loss", best_val))
                    best_epoch = int(state_blob.get("best_epoch", best_epoch))
                    logger.info("Resumed full state from %s at epoch=%d", state_latest_path, start_epoch - 1)
                else:
                    fallback_epoch = _latest_epoch_from_ckpts(run_dir)
                    if fallback_epoch >= 0:
                        start_epoch = fallback_epoch + 1
                    else:
                        start_epoch = int(state_blob.get("epoch", 0)) + 1
                    logger.warning(
                        "State dict mismatch while loading %s; used compat checkpoint %s and switched to warm-start at epoch=%d.",
                        state_latest_path,
                        compat_path,
                        start_epoch - 1,
                    )
            else:
                # Legacy runs may only have model-only checkpoints; warm-start from latest epoch.
                ckpts = sorted(run_dir.glob(f"{run_name}_E*.pth"))
                if not ckpts:
                    raise FileNotFoundError(
                        f"No resumable state found in {run_dir}. Need training_state_latest.pt or {run_name}_E*.pth"
                    )
                latest_ckpt = max(
                    ckpts,
                    key=lambda p: int(p.stem.split("_E")[-1]) if "_E" in p.stem else -1,
                )
                epoch_n = int(latest_ckpt.stem.split("_E")[-1])
                legacy_blob = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
                if isinstance(legacy_blob, dict) and "model_state_dict" in legacy_blob:
                    state_dict = legacy_blob["model_state_dict"]
                else:
                    state_dict = legacy_blob
                resumed_full_state, compat_path = _try_load_model_state_with_compat_fallback(model, state_dict, run_dir)
                start_epoch = epoch_n + 1
                if resumed_full_state:
                    logger.warning(
                        "Resumed model-only checkpoint %s (epoch=%d). Optimizer/scheduler state unavailable; "
                        "this is a warm-start, not exact equivalent continuation.",
                        latest_ckpt,
                        epoch_n,
                    )
                else:
                    logger.warning(
                        "State dict mismatch while loading %s; used compat checkpoint %s at epoch=%d. "
                        "Optimizer/scheduler state unavailable; this is a warm-start.",
                        latest_ckpt,
                        compat_path,
                        epoch_n,
                    )
            total_epochs = start_epoch + int(args.extend_epochs) - 1
            metadata["resume_from_epoch"] = start_epoch - 1
            metadata["resume_target_total_epochs"] = total_epochs

        data_ver = dataset_version_hash(train_shards + test_shards, args.eigen_ch0_encoding)
        params: dict[str, Any] = {
            "model_name": "FNO2d",
            "in_channels": 3,
            "out_channels": OUT_CHANNELS,
            "eigen_ch0_encoding": args.eigen_ch0_encoding,
            "hidden_channels": args.hidden_channels,
            "layers": args.layers,
            "modes_height": args.modes_height,
            "modes_width": args.modes_width,
            "optimizer": "adamw",
            "loss": args.loss,
            "scheduler": args.scheduler,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "t_max": args.t_max if args.t_max > 0 else total_epochs,
            "epochs": total_epochs,
            "start_epoch": start_epoch,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "pin_memory": bool(args.pin_memory),
            "seed": args.seed,
            "amp": args.amp,
            "device": str(device),
            "train_shards_count": len(train_shards),
            "test_shards_count": len(test_shards),
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "dataset_version": data_ver,
            "train_sampler": "shard_aware" if args.train_shuffle else "sequential",
        }

        config_payload = {
            "args": vars(args),
            "params": params,
            "resume_mode": resume_mode,
            "train_shards": [{"name": s.name, "pt_dir": str(s.pt_dir), "n": s.n} for s in train_shards],
            "test_shards": [{"name": s.name, "pt_dir": str(s.pt_dir), "n": s.n} for s in test_shards],
        }
        write_json(run_dir / "resolved_config.json", config_payload)
        best_path = run_dir / f"{run_name}_best.pth"
        metrics_csv = run_dir / "metrics.csv"
        metrics_jsonl = run_dir / "metrics.jsonl"

        logger.info(
            "Run started | run_name=%s run_id=%s device=%s train_samples=%d test_samples=%d epoch_range=%d..%d",
            run_name,
            run_id,
            device,
            len(train_ds),
            len(test_ds),
            start_epoch,
            total_epochs,
        )
        logger.info("Fault diagnostics | main_fault_log=%s", main_fault_path)
        crash_state["phase"] = "train_loop_setup"

        dual_compare = criterion_compare is not None
        best_metric_col = "val_compare_loss" if dual_compare else "val_loss"
        expected_metrics_header = metrics_csv_fieldnames(OUT_CHANNELS, dual_compare=dual_compare)
        csv_mode = "a" if (resume_mode and metrics_csv.exists()) else "w"
        if csv_mode == "a" and metrics_csv.exists():
            if dual_compare:
                _migrate_metrics_to_dual_header(metrics_csv, OUT_CHANNELS)
            with metrics_csv.open("r", newline="", encoding="utf-8") as rf:
                existing_header = next(csv.reader(rf))
            if existing_header != expected_metrics_header:
                raise RuntimeError(
                    "metrics.csv header does not match this trainer (expected per-channel train columns). "
                    "Use a new run directory or delete/rename metrics.csv and metrics.jsonl before resuming."
                )
            # Always derive best checkpoint score from CSV across all completed epochs.
            csv_best_epoch, csv_best_val = _best_from_metrics_csv(metrics_csv, best_metric_col)
            if csv_best_epoch >= 0:
                best_epoch = csv_best_epoch
                best_val = csv_best_val
                _sync_best_weights_from_epoch(run_dir, run_name, best_epoch, best_path)

        with metrics_csv.open(csv_mode, newline="", encoding="utf-8") as csv_f, metrics_jsonl.open("a", encoding="utf-8") as jsonl_f:
            writer = csv.writer(csv_f)
            if csv_mode == "w":
                writer.writerow(expected_metrics_header)

            for epoch in range(start_epoch, total_epochs + 1):
                crash_state["phase"] = "train_epoch"
                crash_state["epoch"] = epoch
                if train_batch_sampler is not None:
                    train_batch_sampler.set_epoch(epoch - 1)
                model.train()
                t_epoch0 = time.time()
                running = 0.0
                running_compare = 0.0 if criterion_compare is not None else None
                running_train_per_ch = torch.zeros(OUT_CHANNELS, dtype=torch.float64)
                running_train_per_ch_cmp = (
                    torch.zeros(OUT_CHANNELS, dtype=torch.float64) if criterion_compare is not None else None
                )
                n_seen = 0
                data_time = 0.0
                last = time.time()
                total_batches = len(train_loader)
                use_tqdm = args.progress_mode == "tqdm"

                if use_tqdm:
                    with tqdm(
                        total=total_batches,
                        desc=f"Train E{epoch}/{total_epochs}",
                        unit="batch",
                        file=sys.stdout,
                        dynamic_ncols=False,
                        ascii=True,
                        mininterval=2.0,
                        miniters=max(1, args.log_every_batches),
                        smoothing=0.0,
                        position=0,
                        leave=True,
                    ) as train_bar:
                        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
                            crash_state["batch"] = batch_idx
                            data_time += max(time.time() - last, 0.0)
                            xb = xb.to(device, dtype=torch.float32, non_blocking=True)
                            yb = yb.to(device, dtype=torch.float32, non_blocking=True)

                            optimizer.zero_grad(set_to_none=True)
                            with amp_context(device, args.amp):
                                pred = model(xb)
                                loss = criterion(pred, yb)
                                loss_compare = criterion_compare(pred, yb) if criterion_compare is not None else None

                            if scaler.is_enabled():
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()

                            bs = xb.shape[0]
                            running += float(loss.item()) * bs
                            if running_compare is not None and loss_compare is not None:
                                running_compare += float(loss_compare.item()) * bs
                            running_train_per_ch += per_channel_loss_mean(pred, yb, args.loss) * bs
                            if running_train_per_ch_cmp is not None and compare_loss_name is not None:
                                running_train_per_ch_cmp += per_channel_loss_mean(pred, yb, compare_loss_name) * bs
                            n_seen += bs
                            last = time.time()
                            train_bar.update(1)

                            if args.log_every_batches > 0 and (
                                batch_idx % args.log_every_batches == 0 or batch_idx == total_batches
                            ):
                                train_bar.set_postfix(batch_loss=f"{float(loss.item()):.4e}")
                                emit_progress(
                                    f"E{epoch}/{total_epochs} B{batch_idx}/{total_batches} "
                                    f"batch_loss={float(loss.item()):.4e}",
                                    use_tqdm=use_tqdm,
                                )
                else:
                    for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
                        crash_state["batch"] = batch_idx
                        data_time += max(time.time() - last, 0.0)
                        xb = xb.to(device, dtype=torch.float32, non_blocking=True)
                        yb = yb.to(device, dtype=torch.float32, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        with amp_context(device, args.amp):
                            pred = model(xb)
                            loss = criterion(pred, yb)
                            loss_compare = criterion_compare(pred, yb) if criterion_compare is not None else None

                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                        bs = xb.shape[0]
                        running += float(loss.item()) * bs
                        if running_compare is not None and loss_compare is not None:
                            running_compare += float(loss_compare.item()) * bs
                        running_train_per_ch += per_channel_loss_mean(pred, yb, args.loss) * bs
                        if running_train_per_ch_cmp is not None and compare_loss_name is not None:
                            running_train_per_ch_cmp += per_channel_loss_mean(pred, yb, compare_loss_name) * bs
                        n_seen += bs
                        last = time.time()

                        if args.log_every_batches > 0 and (
                            batch_idx % args.log_every_batches == 0 or batch_idx == total_batches
                        ):
                            emit_progress(
                                f"E{epoch}/{total_epochs} B{batch_idx}/{total_batches} "
                                f"batch_loss={float(loss.item()):.4e}",
                                use_tqdm=use_tqdm,
                            )

                if scheduler is not None:
                    scheduler.step()
                epoch_time = max(time.time() - t_epoch0, 1e-9)
                train_loss = running / max(n_seen, 1)
                train_loss_compare = (running_compare / max(n_seen, 1)) if running_compare is not None else None
                train_ch = (running_train_per_ch / max(n_seen, 1)).tolist()
                train_ch_cmp = (
                    (running_train_per_ch_cmp / max(n_seen, 1)).tolist()
                    if running_train_per_ch_cmp is not None
                    else None
                )
                train_sps = n_seen / epoch_time
                crash_state["phase"] = "eval_epoch"
                val_loss, val_loss_compare, val_ch, val_ch_cmp, val_sps = evaluate(
                    model,
                    test_loader,
                    device,
                    args.amp,
                    criterion,
                    criterion_compare,
                    args.loss,
                    compare_loss_name,
                )
                lr_now = float(optimizer.param_groups[0]["lr"])

                if args.diagnostic_panels:
                    diag_dir = run_dir / "diagnostics" / f"epoch_{epoch:03d}"
                    try:
                        paths = save_random_test_diagnostic_panels(
                            model,
                            test_ds,
                            device,
                            diag_dir,
                            epoch=epoch,
                            n_samples=int(args.diagnostic_samples),
                            amp_mode=args.amp,
                            seed=int(args.seed) + epoch * 1_000_003,
                        )
                        logger.info(
                            "Saved %d diagnostic panel(s) under %s",
                            len(paths),
                            diag_dir,
                        )
                    except Exception:
                        logger.exception("Diagnostic panel export failed (training continues)")

                row = [
                    epoch,
                    train_loss,
                    *train_ch,
                    val_loss,
                    lr_now,
                    epoch_time,
                    data_time,
                    train_sps,
                    val_sps,
                    *val_ch,
                ]
                row_compare_metric: float | None = None
                if train_ch_cmp is not None and val_ch_cmp is not None:
                    row_compare_metric = float(val_loss_compare) if val_loss_compare is not None else None
                    row.extend(
                        [
                            float(train_loss_compare) if train_loss_compare is not None else float("nan"),
                            row_compare_metric if row_compare_metric is not None else float("nan"),
                            *train_ch_cmp,
                            *val_ch_cmp,
                        ]
                    )
                writer.writerow(row)
                csv_f.flush()

                epoch_metrics: dict[str, Any] = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr_now,
                    "epoch_time_sec": epoch_time,
                    "data_time_sec": data_time,
                    "train_samples_per_sec": train_sps,
                    "val_samples_per_sec": val_sps,
                }
                for i in range(OUT_CHANNELS):
                    epoch_metrics[f"train_loss_ch{i}"] = train_ch[i]
                    epoch_metrics[f"val_loss_ch{i}"] = val_ch[i]
                if train_ch_cmp is not None and val_ch_cmp is not None:
                    epoch_metrics["train_compare_loss"] = float(train_loss_compare) if train_loss_compare is not None else None
                    epoch_metrics["val_compare_loss"] = float(val_loss_compare) if val_loss_compare is not None else None
                    for i in range(OUT_CHANNELS):
                        epoch_metrics[f"train_compare_loss_ch{i}"] = train_ch_cmp[i]
                        epoch_metrics[f"val_compare_loss_ch{i}"] = val_ch_cmp[i]
                jsonl_f.write(json.dumps(epoch_metrics) + "\n")
                jsonl_f.flush()

                epoch_ckpt = run_dir / f"{run_name}_E{epoch}.pth"
                torch.save(model.state_dict(), epoch_ckpt)
                tracked_val = row_compare_metric if best_metric_col == "val_compare_loss" else val_loss
                if tracked_val < best_val:
                    best_val = tracked_val
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_path)
                save_training_state(
                    state_latest_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_val=best_val,
                    best_epoch=best_epoch,
                )

                emit_progress(
                    f"epoch={epoch}/{total_epochs} train_loss={train_loss:.6e} "
                    f"val_loss={val_loss:.6e} lr={lr_now:.3e} train_sps={train_sps:.1f}",
                    use_tqdm=use_tqdm,
                )
                if train_loss_compare is not None and val_loss_compare is not None and compare_loss_name is not None:
                    emit_progress(
                        f"epoch={epoch}/{total_epochs} compare_loss({compare_loss_name}) "
                        f"train={train_loss_compare:.6e} val={val_loss_compare:.6e}",
                        use_tqdm=use_tqdm,
                    )
                logger.info(
                    "epoch=%d/%d loss=%s train_ch_%s=%s val_ch_%s=%s train_loss=%.6e val_loss=%.6e lr=%.3e train_sps=%.1f",
                    epoch,
                    total_epochs,
                    args.loss,
                    args.loss,
                    " ".join(f"{x:.3e}" for x in train_ch),
                    args.loss,
                    " ".join(f"{x:.3e}" for x in val_ch),
                    train_loss,
                    val_loss,
                    lr_now,
                    train_sps,
                )
                if (
                    train_loss_compare is not None
                    and val_loss_compare is not None
                    and compare_loss_name is not None
                    and train_ch_cmp is not None
                    and val_ch_cmp is not None
                ):
                    logger.info(
                        "epoch=%d/%d compare_loss=%s train_ch_%s=%s val_ch_%s=%s train_loss=%.6e val_loss=%.6e",
                        epoch,
                        total_epochs,
                        compare_loss_name,
                        compare_loss_name,
                        " ".join(f"{x:.3e}" for x in train_ch_cmp),
                        compare_loss_name,
                        " ".join(f"{x:.3e}" for x in val_ch_cmp),
                        train_loss_compare,
                        val_loss_compare,
                    )
                emit_progress(
                    f"Epoch {epoch}/{total_epochs} done | train_loss={train_loss:.4e} "
                    f"val_loss={val_loss:.4e} lr={lr_now:.2e}",
                    use_tqdm=use_tqdm,
                )

        final_path = run_dir / f"{run_name}_final.pth"
        torch.save(model.state_dict(), final_path)
        crash_state["phase"] = "completed"

        end_ts = datetime.now(timezone.utc)
        summary = {
            "status": "completed",
            "run_name": run_name,
            "run_id": run_id,
            "started_at_utc": start_ts.isoformat(),
            "ended_at_utc": end_ts.isoformat(),
            "duration_sec": (end_ts - start_ts).total_seconds(),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "checkpoints": {
                "best": str(best_path),
                "final": str(final_path),
                "epoch_pattern": str(run_dir / f"{run_name}_E{{epoch}}.pth"),
            },
            "artifacts": {
                "train_log": str(run_dir / "train.log"),
                "resolved_config": str(run_dir / "resolved_config.json"),
                "run_metadata": str(run_dir / "run_metadata.json"),
                "metrics_csv": str(metrics_csv),
                "metrics_jsonl": str(metrics_jsonl),
                "diagnostics": str(run_dir / "diagnostics"),
            },
        }
        write_json(run_dir / "summary.json", summary)

        metadata["status"] = "completed"
        metadata["ended_at_utc"] = end_ts.isoformat()
        metadata["duration_sec"] = (end_ts - start_ts).total_seconds()
        metadata["best_val_loss"] = best_val
        metadata["best_epoch"] = best_epoch
        write_json(run_dir / "run_metadata.json", metadata)

        logger.info("Run complete | run_name=%s run_id=%s best_val_loss=%.6e", run_name, run_id, best_val)
        logger.info("Artifacts saved to: %s", run_dir)
        emit_progress(
            f"Run complete. run_name={run_name} run_id={run_id} best_val_loss={best_val:.6e}",
            use_tqdm=(args.progress_mode == "tqdm"),
        )
        emit_progress(f"Artifacts saved to: {run_dir}", use_tqdm=(args.progress_mode == "tqdm"))

    except Exception as exc:
        end_ts = datetime.now(timezone.utc)
        worker_pids = [p.pid for p in mp.active_children()]
        diagnostic_hint = (
            "DataLoader worker failed. Check worker_*_fault.log files in run directory; "
            "if empty, rerun with --num-workers 0 to isolate worker-related failures."
            if "DataLoader worker" in str(exc)
            else ""
        )
        err = {
            "status": "failed",
            "run_name": run_name,
            "run_id": run_id,
            "started_at_utc": start_ts.isoformat(),
            "ended_at_utc": end_ts.isoformat(),
            "duration_sec": (end_ts - start_ts).total_seconds(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "phase": crash_state["phase"],
            "epoch": crash_state["epoch"],
            "batch": crash_state["batch"],
            "active_child_pids": worker_pids,
            "fault_logs": {
                "main": str(run_dir / "main_process_fault.log"),
                "workers_glob": str(run_dir / "worker_*_fault.log"),
            },
            "diagnostic_hint": diagnostic_hint,
        }
        write_json(run_dir / "summary.json", err)
        metadata["status"] = "failed"
        metadata["ended_at_utc"] = end_ts.isoformat()
        metadata["duration_sec"] = (end_ts - start_ts).total_seconds()
        metadata["error_type"] = type(exc).__name__
        metadata["error_message"] = str(exc)
        write_json(run_dir / "run_metadata.json", metadata)
        logger.exception("Training failed")
        raise


if __name__ == "__main__":
    main()
