from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import re
import sys
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
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--prefetch-factor", type=int, default=2)
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
        "--epoch",
        type=int,
        default=0,
        help="Override epoch number when the checkpoint filename has no _E{n} (e.g. *_best.pth). 0 means auto.",
    )
    p.add_argument(
        "--write-to-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append evaluation results to train.log / metrics.jsonl / evaluation_metrics.csv when present, "
            "and backfill metrics.csv rows for the inferred epoch (see --epoch)."
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


def per_channel_loss_mean(pred: torch.Tensor, yb: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Match train_from_disk.py: per-channel mean over (N, H, W)."""
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


_EPOCH_IN_CKPT = re.compile(r"_E(\d+)$", re.IGNORECASE)
_EPOCH_ALT = re.compile(r"(?:^|[_-])epoch[_-]?(\d+)$", re.IGNORECASE)


def parse_epoch_from_checkpoint(model_path: Path, override: int) -> int:
    if override > 0:
        return int(override)
    stem = model_path.stem
    m = _EPOCH_IN_CKPT.search(stem)
    if m:
        return int(m.group(1))
    m2 = _EPOCH_ALT.search(stem)
    if m2:
        return int(m2.group(1))
    raise ValueError(
        f"Could not infer epoch from checkpoint name {model_path.name!r}. "
        f"Use a path like *_E31.pth or pass --epoch 31."
    )


def infer_out_channels_from_header(header: list[str]) -> int:
    ch_idx = -1
    for h in header:
        if h.startswith("val_loss_ch"):
            try:
                ch_idx = max(ch_idx, int(h[len("val_loss_ch") :]))
            except ValueError:
                continue
    if ch_idx < 0:
        raise ValueError("Could not infer channel count from metrics.csv header (expected val_loss_ch* columns).")
    return ch_idx + 1


def header_has_dual_compare(header: list[str]) -> bool:
    return "val_compare_loss" in header


def read_trained_loss_from_run_dir(run_dir: Path) -> str | None:
    rc_path = run_dir / "resolved_config.json"
    if not rc_path.is_file():
        return None
    try:
        rc = json.loads(rc_path.read_text(encoding="utf-8"))
        loss = str(rc.get("args", {}).get("loss", "")).lower().strip()
        if loss == "l2":
            loss = "mse"
        if loss in {"mse", "l1", "smoothl1"}:
            return loss
    except Exception:
        return None
    return None


def migrate_row_dict_to_dual(row: dict[str, str], out_channels: int) -> dict[str, str]:
    single_header = metrics_csv_fieldnames(out_channels, dual_compare=False)
    out = {k: row.get(k, "") for k in single_header}
    out["train_compare_loss"] = row.get("train_loss", "")
    out["val_compare_loss"] = row.get("val_loss", "")
    for i in range(out_channels):
        out[f"train_compare_loss_ch{i}"] = row.get(f"train_loss_ch{i}", "")
        out[f"val_compare_loss_ch{i}"] = row.get(f"val_loss_ch{i}", "")
    return out


def ensure_dual_schema(rows: list[dict[str, str]], out_channels: int) -> tuple[list[str], list[dict[str, str]]]:
    dual_header = metrics_csv_fieldnames(out_channels, dual_compare=True)
    new_rows = [migrate_row_dict_to_dual(r, out_channels) for r in rows]
    return dual_header, new_rows


def _epoch_cell_to_int(cell: str | None) -> int | None:
    if cell is None or cell.strip() == "":
        return None
    try:
        return int(float(cell))
    except ValueError:
        return None


def load_model_state(model: torch.nn.Module, model_path: Path) -> None:
    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    if isinstance(state, dict) and "_metadata" in state:
        state = dict(state)
        state.pop("_metadata", None)
    model.load_state_dict(state)


def _append_train_log_messages(run_dir: Path, messages: list[str]) -> None:
    """Match train_from_disk.py FileHandler format: %(asctime)s | %(levelname)s | %(message)s"""
    train_log = run_dir / "train.log"
    if not train_log.is_file():
        for m in messages:
            print(m, file=sys.stderr)
        return
    lg = logging.getLogger("evaluate_from_disk.trainlog")
    lg.handlers.clear()
    lg.propagate = False
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(train_log, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    for m in messages:
        lg.info(m)
    fh.flush()
    fh.close()
    lg.removeHandler(fh)


def _route_losses_for_csv_backfill(
    requested: list[str],
    trained_loss: str | None,
) -> tuple[str | None, str | None, list[str]]:
    """
    Returns (val_loss_name, compare_loss_name, skipped_for_compare_csv).
    val_loss_name: requested loss matching trained loss (updates val_* columns), if any.
    compare_loss_name: first requested loss != trained (updates val_compare_*); CSV has one compare slot.
    """
    skipped: list[str] = []
    val_name: str | None = None
    cmp_name: str | None = None

    if trained_loss is not None:
        for ln in requested:
            if ln == trained_loss:
                val_name = ln
                break
        for ln in requested:
            if ln == trained_loss:
                continue
            if cmp_name is None:
                cmp_name = ln
            else:
                skipped.append(ln)
        return val_name, cmp_name, skipped

    if len(requested) == 1:
        return requested[0], None, []
    val_name = requested[0]
    cmp_name = requested[1]
    skipped = requested[2:]
    return val_name, cmp_name, skipped


def _atomic_write_metrics_csv(metrics_csv: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp = metrics_csv.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as wf:
        w = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)
    tmp.replace(metrics_csv)


def backfill_metrics_csv(
    metrics_csv: Path,
    *,
    target_epoch: int,
    out_channels: int,
    trained_loss: str,
    eval_by_loss: dict[str, dict[str, object]],
    requested_losses: list[str],
) -> tuple[int, bool]:
    """
    Update metrics.csv in place for rows matching target_epoch.
    Returns (rows_updated_count, did_migrate_to_dual).
    """
    if trained_loss not in {"mse", "l1", "smoothl1"}:
        raise ValueError(f"Invalid trained_loss for metrics.csv routing: {trained_loss!r}")

    with metrics_csv.open("r", newline="", encoding="utf-8") as rf:
        reader = csv.DictReader(rf)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"{metrics_csv} has no header.")
        header = list(fieldnames)
        rows = list(reader)

    header_n = infer_out_channels_from_header(header)
    if header_n != out_channels:
        raise ValueError(
            f"metrics.csv channel count ({header_n}) does not match --out-channels ({out_channels})."
        )

    had_dual_before = header_has_dual_compare(header)
    val_loss_name, compare_loss_name, _skipped_route = _route_losses_for_csv_backfill(
        requested_losses, trained_loss
    )
    did_migrate = False
    if compare_loss_name is not None and not had_dual_before:
        header, rows = ensure_dual_schema(rows, out_channels)
        did_migrate = True

    dual = header_has_dual_compare(header)
    expected = metrics_csv_fieldnames(out_channels, dual_compare=dual)
    if list(header) != expected:
        raise RuntimeError(
            f"metrics.csv header does not match the trainer schema after any migration. "
            f"Expected {len(expected)} columns; inspect {metrics_csv} manually."
        )

    updated = 0
    found_any = False
    for r in rows:
        ep = _epoch_cell_to_int(r.get("epoch"))
        if ep != target_epoch:
            continue
        found_any = True
        if val_loss_name is not None and val_loss_name in eval_by_loss:
            v = eval_by_loss[val_loss_name]
            r["val_loss"] = f"{float(v['avg_pixel_loss']):.17g}"
            for i in range(out_channels):
                r[f"val_loss_ch{i}"] = f"{float(v['val_ch'][i]):.17g}"
        if compare_loss_name is not None and compare_loss_name in eval_by_loss:
            v = eval_by_loss[compare_loss_name]
            r["val_compare_loss"] = f"{float(v['avg_pixel_loss']):.17g}"
            for i in range(out_channels):
                r[f"val_compare_loss_ch{i}"] = f"{float(v['val_ch'][i]):.17g}"
        updated += 1

    if not found_any:
        raise ValueError(
            f"No metrics.csv row with epoch={target_epoch}. "
            f"Training must have logged that epoch before backfill."
        )

    _atomic_write_metrics_csv(metrics_csv, expected, rows)
    return updated, did_migrate


def maybe_write_to_model_logs(model_path: Path, payload: dict[str, object]) -> None:
    run_dir = model_path.resolve().parent
    ts_iso = datetime.now(timezone.utc).isoformat()
    epoch = int(payload["epoch"])  # type: ignore[arg-type]
    eval_by_loss = payload["eval_by_loss"]  # type: ignore[assignment]
    requested = list(payload["requested_losses"])  # type: ignore[assignment]
    trained_loss = payload.get("trained_loss")  # type: ignore[assignment]
    out_ch = int(payload["out_channels"])  # type: ignore[arg-type]

    val_n, cmp_n, skipped = _route_losses_for_csv_backfill(requested, trained_loss if isinstance(trained_loss, str) else None)

    log_lines: list[str] = [
        f"evaluation_from_disk | epoch={epoch} checkpoint={model_path.name} "
        f"num_test_samples={payload['num_test_samples']} trained_loss={trained_loss!r}"
    ]
    for loss_name, vals in payload["losses"].items():  # type: ignore[index]
        log_lines.append(
            f"evaluation_from_disk | epoch={epoch} loss={loss_name} "
            f"avg_pixel_loss={vals['avg_pixel_loss']:.6e} avg_sample_loss={vals['avg_sample_loss']:.6e}"
        )
    if skipped:
        log_lines.append(
            f"evaluation_from_disk | epoch={epoch} note=compare_csv_slot_single skipped_losses={skipped}"
        )
    _append_train_log_messages(run_dir, log_lines)

    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.is_file():
        event: dict[str, object] = {
            "epoch": epoch,
            "source": "evaluation_from_disk",
            "timestamp_utc": ts_iso,
            "checkpoint": str(model_path.resolve()),
            "num_test_samples": payload["num_test_samples"],
            "trained_loss": trained_loss,
            "requested_losses": requested,
            "val_loss_backfill_loss": val_n,
            "val_compare_backfill_loss": cmp_n,
            "skipped_csv_compare": skipped,
        }
        for loss_name, vals in payload["losses"].items():  # type: ignore[index]
            eb = eval_by_loss[loss_name]
            event[f"eval_{loss_name}_avg_pixel"] = vals["avg_pixel_loss"]
            event[f"eval_{loss_name}_avg_sample"] = vals["avg_sample_loss"]
            event[f"eval_{loss_name}_val_ch"] = eb["val_ch"]
        with metrics_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.is_file():
        if not isinstance(trained_loss, str) or trained_loss not in {"mse", "l1", "smoothl1"}:
            _append_train_log_messages(
                run_dir,
                [
                    "evaluation_from_disk | warning=skip_metrics_csv_backfill "
                    "reason=missing_or_invalid_resolved_config_loss",
                ],
            )
        else:
            n_up, migrated = backfill_metrics_csv(
                metrics_csv,
                target_epoch=epoch,
                out_channels=out_ch,
                trained_loss=trained_loss,
                eval_by_loss=eval_by_loss,
                requested_losses=requested,
            )
            _append_train_log_messages(
                run_dir,
                [
                    f"evaluation_from_disk | epoch={epoch} metrics_csv rows_updated={n_up} "
                    f"migrated_to_dual_header={migrated}"
                ],
            )

    eval_csv = run_dir / "evaluation_metrics.csv"
    write_header = not eval_csv.exists()
    with eval_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "timestamp_utc",
                    "epoch",
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
                    ts_iso,
                    epoch,
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
    stats_per_ch: dict[str, torch.Tensor] = {
        loss_name: torch.zeros(args.out_channels, dtype=torch.float64) for loss_name in losses
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
                stats_per_ch[loss_name] += per_channel_loss_mean(pred, yb, loss_name) * bs

    results: dict[str, dict[str, float]] = {}
    eval_by_loss: dict[str, dict[str, object]] = {}
    for loss_name in losses:
        avg_px = stats[loss_name]["pixel_sum"] / max(total_pixels, 1)
        avg_sm = stats[loss_name]["sample_sum"] / max(total_samples, 1)
        val_ch = [float(x) for x in (stats_per_ch[loss_name] / max(total_samples, 1)).tolist()]
        results[loss_name] = {
            "avg_pixel_loss": avg_px,
            "avg_sample_loss": avg_sm,
        }
        eval_by_loss[loss_name] = {
            "avg_pixel_loss": avg_px,
            "avg_sample_loss": avg_sm,
            "val_ch": val_ch,
        }

    model_path = Path(args.model_path)
    run_dir = model_path.resolve().parent
    epoch: int | None = None
    trained_loss: str | None = None
    if args.write_to_logs:
        epoch = parse_epoch_from_checkpoint(model_path, args.epoch)
        trained_loss = read_trained_loss_from_run_dir(run_dir)

    payload: dict[str, object] = {
        "model_path": str(model_path.resolve()),
        "output_root": str(output_root.resolve()),
        "test_prefixes": list(TEST_PREFIXES),
        "num_test_samples": total_samples,
        "out_channels": args.out_channels,
        "losses": results,
    }
    if epoch is not None:
        payload["epoch"] = epoch
    if trained_loss is not None:
        payload["trained_loss"] = trained_loss
    if args.write_to_logs:
        payload["requested_losses"] = losses
        payload["eval_by_loss"] = eval_by_loss

    print(json.dumps(payload, indent=2))

    if args.write_to_logs:
        maybe_write_to_model_logs(model_path, payload)
        print(f"Wrote evaluation results to available logs under: {Path(args.model_path).resolve().parent}")

    if args.save_json.strip():
        out_path = Path(args.save_json)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
