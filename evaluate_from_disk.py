from __future__ import annotations

import argparse
import bisect
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


TEST_PREFIXES = ("c_test", "b_test")
EIGEN_CH0_FILES = {
    "uniform": "eigenfrequency_uniform_full.pt",
    "fft": "eigenfrequency_fft_full.pt",
}


class ShardInfo:
    def __init__(
        self,
        *,
        name: str,
        pt_dir: Path,
        inputs_path: Path,
        outputs_path: Path,
        reduced_indices_path: Path,
        eigen_ch0_path: Path,
        n: int,
    ) -> None:
        self.name = name
        self.pt_dir = pt_dir
        self.inputs_path = inputs_path
        self.outputs_path = outputs_path
        self.reduced_indices_path = reduced_indices_path
        self.eigen_ch0_path = eigen_ch0_path
        self.n = n


class ShardedTensorPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Disk-backed dataset:
    - inputs.pt as model input (3x32x32)
    - output channel 0 from eigenfrequency_*_full.pt via reduced_indices
    - output channels 1..N-1 from outputs.pt
    """

    def __init__(self, shards: list[ShardInfo], out_channels: int):
        if not shards:
            raise ValueError("No shards were provided.")
        if out_channels < 1 or out_channels > 5:
            raise ValueError(f"out_channels must be in [1, 5], got {out_channels}")
        self.shards = shards
        self.out_channels = out_channels
        self._lengths = [s.n for s in shards]
        self._offsets = np.cumsum([0, *self._lengths]).tolist()
        self._total = self._offsets[-1]

        self._loaded_shard_idx: int | None = None
        self._loaded_inputs: torch.Tensor | None = None
        self._loaded_outputs: torch.Tensor | None = None
        self._loaded_eigen_ch0: torch.Tensor | None = None
        self._loaded_ridx: object | None = None

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

        if self.out_channels == 1:
            y = y0.unsqueeze(0)
        else:
            y_rest = self._loaded_outputs[local_idx, 1 : self.out_channels]
            y = torch.cat([y0.unsqueeze(0), y_rest], dim=0)
        return x, y


class FourierNeuralOperator(torch.nn.Module):
    def __init__(
        self,
        modes_height: int,
        modes_width: int,
        hidden_channels: int,
        n_layers: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        try:
            from neuralop.models import FNO2d
        except Exception as e:
            raise RuntimeError(
                "Failed to import neuralop FNO2d. Fix neuralop/wandb dependencies and retry."
            ) from e
        self.model = FNO2d(
            in_channels=3,
            out_channels=out_channels,
            n_modes_height=modes_height,
            n_modes_width=modes_width,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        return self.model.load_state_dict(state_dict, strict=strict)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate disk-backed NO model on c_test and b_test shards.")
    p.add_argument("--model-path", required=True, help="Path to model checkpoint (.pth).")
    p.add_argument("--output-root", default="D:/Research/NO-2D-Metamaterials/DATASETS")
    p.add_argument("--eigen-ch0-encoding", choices=tuple(EIGEN_CH0_FILES.keys()), default="uniform")
    p.add_argument("--losses", nargs="+", required=True, help="List of losses: mse, l1, smoothl1, l2")
    p.add_argument("--batch-size", type=int, default=520)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=3)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--modes-height", type=int, default=32)
    p.add_argument("--modes-width", type=int, default=32)
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--out-channels", type=int, default=5, help="1 for eigen-only, 4 for displacement-only, 5 for full.")
    p.add_argument("--save-json", default="", help="Optional output json path.")
    p.add_argument(
        "--write-to-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append evaluation results to log/jsonl/csv artifacts in the model folder "
            "(train.log / metrics.jsonl / evaluation_metrics.csv) when present."
        ),
    )
    return p.parse_args()


def resolve_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is unavailable. Pass --allow-cpu for CPU evaluation.")


def amp_context(device: torch.device, amp_mode: str):
    if device.type != "cuda" or amp_mode == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    if amp_mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def latest_pt_dir(dataset_dir: Path) -> Path:
    cands = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not cands:
        raise FileNotFoundError(f"No *_pt folder under {dataset_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def discover_test_shards(output_root: Path, eigen_ch0_encoding: str) -> list[ShardInfo]:
    if eigen_ch0_encoding not in EIGEN_CH0_FILES:
        raise ValueError(f"Unknown eigen_ch0_encoding: {eigen_ch0_encoding!r}")
    eigen_fname = EIGEN_CH0_FILES[eigen_ch0_encoding]
    shards: list[ShardInfo] = []
    ds_dirs = sorted([p for p in output_root.iterdir() if p.is_dir() and p.name.startswith(TEST_PREFIXES)], key=lambda p: p.name)
    for d in ds_dirs:
        pt = latest_pt_dir(d)
        in_path = pt / "inputs.pt"
        out_path = pt / "outputs.pt"
        ridx_path = pt / "reduced_indices.pt"
        eigen_path = pt / eigen_fname
        if not in_path.exists() or not out_path.exists() or not ridx_path.exists() or not eigen_path.exists():
            raise FileNotFoundError(f"Missing one or more required files in {pt}")
        ridx = torch.load(ridx_path, map_location="cpu", weights_only=False)
        shards.append(
            ShardInfo(
                name=d.name,
                pt_dir=pt,
                inputs_path=in_path,
                outputs_path=out_path,
                reduced_indices_path=ridx_path,
                eigen_ch0_path=eigen_path,
                n=len(ridx),
            )
        )
    if not shards:
        raise FileNotFoundError(f"No test shards found with prefixes={TEST_PREFIXES} under {output_root}")
    return shards


def normalize_loss_name(name: str) -> str:
    n = name.strip().lower()
    if n == "l2":
        return "mse"
    if n not in {"mse", "l1", "smoothl1"}:
        raise ValueError(f"Unsupported loss: {name}. Supported: mse, l1, smoothl1, l2")
    return n


def loss_tensor(loss_name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if loss_name == "mse":
        return (pred - target).pow(2)
    if loss_name == "l1":
        return (pred - target).abs()
    return F.smooth_l1_loss(pred, target, reduction="none", beta=1e-5)


def load_model_state(model: torch.nn.Module, model_path: Path) -> None:
    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    if isinstance(state, dict) and "_metadata" in state:
        state = dict(state)
        state.pop("_metadata", None)
    model.load_state_dict(state)


def maybe_write_to_model_logs(model_path: Path, payload: dict[str, object]) -> None:
    run_dir = model_path.resolve().parent
    ts = datetime.now(timezone.utc).isoformat()

    train_log = run_dir / "train.log"
    if train_log.is_file():
        with train_log.open("a", encoding="utf-8") as f:
            f.write(
                "\n"
                f"{ts} | INFO | evaluation_from_disk | model={model_path.name} "
                f"num_test_samples={payload['num_test_samples']}\n"
            )
            for loss_name, vals in payload["losses"].items():  # type: ignore[index]
                f.write(
                    f"{ts} | INFO | evaluation_from_disk | loss={loss_name} "
                    f"avg_pixel_loss={vals['avg_pixel_loss']:.6e} "
                    f"avg_sample_loss={vals['avg_sample_loss']:.6e}\n"
                )

    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.is_file():
        event = {
            "event": "evaluation_from_disk",
            "timestamp_utc": ts,
            "model_path": str(model_path.resolve()),
            "num_test_samples": payload["num_test_samples"],
            "losses": payload["losses"],
        }
        with metrics_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.is_file():
        eval_csv = run_dir / "evaluation_metrics.csv"
        write_header = not eval_csv.exists()
        with eval_csv.open("a", newline="", encoding="utf-8") as f:
            import csv

            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "timestamp_utc",
                        "model_path",
                        "num_test_samples",
                        "loss",
                        "avg_pixel_loss",
                        "avg_sample_loss",
                    ]
                )
            for loss_name, vals in payload["losses"].items():  # type: ignore[index]
                w.writerow(
                    [
                        ts,
                        str(model_path.resolve()),
                        payload["num_test_samples"],
                        loss_name,
                        vals["avg_pixel_loss"],
                        vals["avg_sample_loss"],
                    ]
                )


def main() -> None:
    args = parse_args()
    losses = [normalize_loss_name(x) for x in args.losses]
    device = resolve_device(args.allow_cpu)

    output_root = Path(args.output_root)
    shards = discover_test_shards(output_root, args.eigen_ch0_encoding)
    dataset = ShardedTensorPairDataset(shards, out_channels=args.out_channels)

    loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["timeout"] = 180
    loader = DataLoader(dataset, **loader_kwargs)

    model = FourierNeuralOperator(
        modes_height=args.modes_height,
        modes_width=args.modes_width,
        hidden_channels=args.hidden_channels,
        n_layers=args.layers,
        out_channels=args.out_channels,
    ).to(device)
    load_model_state(model, Path(args.model_path))
    model.eval()

    stats: dict[str, dict[str, float]] = {
        loss_name: {"pixel_sum": 0.0, "sample_sum": 0.0} for loss_name in losses
    }
    total_pixels = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Evaluating", unit="batch", ascii=True):
            xb = xb.to(device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device, dtype=torch.float32, non_blocking=True)
            with amp_context(device, args.amp):
                pred = model(xb)

            bs = int(xb.shape[0])
            elems_per_sample = int(np.prod(list(yb.shape[1:])))
            total_samples += bs
            total_pixels += bs * elems_per_sample

            for loss_name in losses:
                loss_map = loss_tensor(loss_name, pred, yb).float()
                stats[loss_name]["pixel_sum"] += float(loss_map.sum().item())
                per_sample = loss_map.view(bs, -1).mean(dim=1)
                stats[loss_name]["sample_sum"] += float(per_sample.sum().item())

    results: dict[str, dict[str, float]] = {}
    for loss_name in losses:
        results[loss_name] = {
            "avg_pixel_loss": stats[loss_name]["pixel_sum"] / max(total_pixels, 1),
            "avg_sample_loss": stats[loss_name]["sample_sum"] / max(total_samples, 1),
        }

    payload = {
        "model_path": str(Path(args.model_path).resolve()),
        "output_root": str(output_root.resolve()),
        "test_prefixes": list(TEST_PREFIXES),
        "num_test_samples": total_samples,
        "losses": results,
    }

    print(json.dumps(payload, indent=2))

    if args.write_to_logs:
        maybe_write_to_model_logs(Path(args.model_path), payload)
        print(f"Wrote evaluation results to available logs under: {Path(args.model_path).resolve().parent}")

    if args.save_json.strip():
        out_path = Path(args.save_json)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
