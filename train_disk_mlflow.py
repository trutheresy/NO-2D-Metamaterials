from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


TRAIN_PREFIXES = ("c_train", "b_train")
TEST_PREFIXES = ("c_test", "b_test")

EIGEN_CH0_FILES = {
    "uniform": "eigenfrequency_uniform_full.pt",
    "fft": "eigenfrequency_fft_full.pt",
}


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
    p = argparse.ArgumentParser(description="Disk-backed training pipeline with MLflow logging.")
    p.add_argument("--output-root", default="D:/Research/NO-2D-Metamaterials/MODELS")
    p.add_argument("--experiment-name", default="NO_I3O5_BCF16_disk")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--save-dir", default="D:/Research/NO-2D-Metamaterials/training_runs")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=260)
    p.add_argument("--train-shuffle", action="store_true", help="Enable global shuffle across shards (can be heavy).")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--prefetch-factor", type=int, default=3)
    p.add_argument("--hidden-channels", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--modes-height", type=int, default=32)
    p.add_argument("--modes-width", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--loss", choices=("mse", "l1", "smoothl1"), default="mse")
    p.add_argument("--optimizer", choices=("adamw", "adam", "sgd"), default="adamw")
    p.add_argument("--scheduler", choices=("steplr", "cosine", "none"), default="steplr")
    p.add_argument("--step-size", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--t-max", type=int, default=0, help="CosineAnnealingLR T_max. 0 means epochs.")
    p.add_argument("--momentum", type=float, default=0.9, help="Used when optimizer=sgd.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="fp16")
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
    return (
        f"NO_I3O5_BCF16_L2_HC{args.hidden_channels}_"
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
    return nn.SmoothL1Loss()


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    t_max = args.t_max if args.t_max > 0 else args.epochs
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_mode: str,
    criterion: nn.Module,
) -> tuple[float, list[float], float]:
    model.eval()
    running_loss = 0.0
    running_per_ch = torch.zeros(5, dtype=torch.float64)
    n_samples = 0
    start = time.time()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device, dtype=torch.float32, non_blocking=True)
            with amp_context(device, amp_mode):
                pred = model(xb)
                loss = criterion(pred, yb)
            bs = xb.shape[0]
            running_loss += float(loss.item()) * bs
            per_ch = ((pred - yb) ** 2).mean(dim=(0, 2, 3)).detach().cpu().to(torch.float64)
            running_per_ch += per_ch * bs
            n_samples += bs
    elapsed = max(time.time() - start, 1e-9)
    mean_loss = running_loss / max(n_samples, 1)
    mean_ch = (running_per_ch / max(n_samples, 1)).tolist()
    sps = n_samples / elapsed
    return mean_loss, mean_ch, sps


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_root = Path(args.output_root)
    train_shards = discover_shards(output_root, TRAIN_PREFIXES, args.eigen_ch0_encoding)
    test_shards = discover_shards(output_root, TEST_PREFIXES, args.eigen_ch0_encoding)
    train_shards = maybe_cap_shards(train_shards, args.max_train_samples)
    test_shards = maybe_cap_shards(test_shards, args.max_test_samples)

    train_ds = ShardedTensorPairDataset(train_shards, args.eigen_ch0_encoding)
    test_ds = ShardedTensorPairDataset(test_shards, args.eigen_ch0_encoding)

    train_loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": bool(args.train_shuffle),
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    test_loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        test_loader_kwargs["persistent_workers"] = True
        test_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    test_loader = DataLoader(test_ds, **test_loader_kwargs)

    device = resolve_device(args.allow_cpu)
    model = FourierNeuralOperator(
        modes_height=args.modes_height,
        modes_width=args.modes_width,
        hidden_channels=args.hidden_channels,
        n_layers=args.layers,
    ).to(device)
    criterion = build_criterion(args.loss)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))

    run_name = build_run_name(args)
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    git_meta = git_info()
    data_ver = dataset_version_hash(train_shards + test_shards, args.eigen_ch0_encoding)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        run_dir = save_root / f"{run_name}_{run_id[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)

        params: dict[str, Any] = {
            "model_name": "FNO2d",
            "in_channels": 3,
            "out_channels": 5,
            "hidden_channels": args.hidden_channels,
            "layers": args.layers,
            "modes_height": args.modes_height,
            "modes_width": args.modes_width,
            "optimizer": args.optimizer,
            "loss": args.loss,
            "scheduler": args.scheduler,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "t_max": args.t_max if args.t_max > 0 else args.epochs,
            "momentum": args.momentum,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "seed": args.seed,
            "amp": args.amp,
            "device": str(device),
            "train_shards_count": len(train_shards),
            "test_shards_count": len(test_shards),
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "dataset_version": data_ver,
            "eigen_ch0_encoding": args.eigen_ch0_encoding,
        }
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "project": "NO-2D-Metamaterials",
                "task": "surrogate_training",
                "contract": "in3_out5",
                "dataset_version": data_ver,
                "eigen_ch0_encoding": args.eigen_ch0_encoding,
                **git_meta,
            }
        )

        config_payload = {
            "args": vars(args),
            "params": params,
            "train_shards": [{"name": s.name, "pt_dir": str(s.pt_dir), "n": s.n} for s in train_shards],
            "test_shards": [{"name": s.name, "pt_dir": str(s.pt_dir), "n": s.n} for s in test_shards],
        }
        config_path = run_dir / "resolved_config.json"
        config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(config_path))

        best_val = float("inf")
        best_path = run_dir / f"{run_name}_best.pth"
        losses_csv = run_dir / "losses.csv"

        with losses_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "lr",
                    "epoch_time_sec",
                    "train_samples_per_sec",
                    "val_samples_per_sec",
                    "val_loss_ch0",
                    "val_loss_ch1",
                    "val_loss_ch2",
                    "val_loss_ch3",
                    "val_loss_ch4",
                ]
            )

            epoch_bar = tqdm(
                range(1, args.epochs + 1),
                total=args.epochs,
                desc="Epochs",
                unit="epoch",
                file=sys.stdout,
                dynamic_ncols=True,
            )
            for epoch in epoch_bar:
                model.train()
                start_epoch = time.time()
                running = 0.0
                n_seen = 0
                data_time = 0.0
                last = time.time()

                train_iter = tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f"Train E{epoch}/{args.epochs}",
                    unit="batch",
                    leave=False,
                    file=sys.stdout,
                    dynamic_ncols=True,
                )
                for xb, yb in train_iter:
                    data_time += max(time.time() - last, 0.0)
                    xb = xb.to(device, dtype=torch.float32, non_blocking=True)
                    yb = yb.to(device, dtype=torch.float32, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    with amp_context(device, args.amp):
                        pred = model(xb)
                        loss = criterion(pred, yb)

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    bs = xb.shape[0]
                    running += float(loss.item()) * bs
                    n_seen += bs
                    last = time.time()
                    train_iter.set_postfix(batch_loss=f"{float(loss.item()):.4e}")

                if scheduler is not None:
                    scheduler.step()
                epoch_time = max(time.time() - start_epoch, 1e-9)
                train_loss = running / max(n_seen, 1)
                train_sps = n_seen / epoch_time
                val_loss, val_ch, val_sps = evaluate(model, test_loader, device, args.amp, criterion)
                lr_now = float(optimizer.param_groups[0]["lr"])

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": lr_now,
                        "epoch_time_sec": epoch_time,
                        "data_time_sec": data_time,
                        "train_samples_per_sec": train_sps,
                        "val_samples_per_sec": val_sps,
                        "val_loss_ch0": val_ch[0],
                        "val_loss_ch1": val_ch[1],
                        "val_loss_ch2": val_ch[2],
                        "val_loss_ch3": val_ch[3],
                        "val_loss_ch4": val_ch[4],
                    },
                    step=epoch,
                )

                writer.writerow(
                    [
                        epoch,
                        train_loss,
                        val_loss,
                        lr_now,
                        epoch_time,
                        train_sps,
                        val_sps,
                        val_ch[0],
                        val_ch[1],
                        val_ch[2],
                        val_ch[3],
                        val_ch[4],
                    ]
                )
                f.flush()

                epoch_ckpt = run_dir / f"{run_name}_E{epoch}.pth"
                torch.save(model.state_dict(), epoch_ckpt)
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), best_path)

                print(
                    f"epoch={epoch}/{args.epochs} train_loss={train_loss:.6e} "
                    f"val_loss={val_loss:.6e} lr={lr_now:.3e} train_sps={train_sps:.1f}"
                )
                epoch_bar.set_postfix(train_loss=f"{train_loss:.4e}", val_loss=f"{val_loss:.4e}", lr=f"{lr_now:.2e}")

        final_path = run_dir / f"{run_name}_final.pth"
        torch.save(model.state_dict(), final_path)

        mlflow.log_artifact(str(losses_csv))
        mlflow.log_artifact(str(best_path))
        mlflow.log_artifact(str(final_path))

        mlflow.log_metric("best_val_loss", best_val)
        print(f"Run complete. run_name={run_name} run_id={run_id} best_val_loss={best_val:.6e}")
        print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
