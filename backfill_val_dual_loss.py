#!/usr/bin/env python3
"""Backfill val MAE (l1) and val MSE for every epoch checkpoint into metrics.csv."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch

from evaluate_from_disk import (
    FourierNeuralOperator,
    _append_train_log_messages,
    backfill_metrics_csv,
    build_test_dataset,
    ensure_dual_schema,
    header_has_dual_compare,
    infer_out_channels_from_header,
    load_model_state,
    metrics_csv_fieldnames,
    populate_reference_l1_mse_from_row,
    read_trained_loss_from_run_dir,
    resolve_device,
    run_evaluation,
    _atomic_write_metrics_csv,
)

_EPOCH_ACTIVE_LOSS = re.compile(r"epoch=(\d+)/\d+\s+loss=(mse|l1)", re.IGNORECASE)
_EPOCH_CKPT = re.compile(r"_E(\d+)\.pth$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill val l1+mse into metrics.csv from epoch checkpoints.")
    p.add_argument("--run-dir", required=True, help="Training run directory (contains metrics.csv and *_E*.pth).")
    p.add_argument("--output-root", default="D:/Research/NO-2D-Metamaterials/DATASETS")
    p.add_argument(
        "--val-full-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate on indices_full.pt (100%% of test data). Use --no-val-full-test for reduced_indices.",
    )
    p.add_argument(
        "--mse-start-epoch",
        type=int,
        default=0,
        help="First epoch trained with MSE as active loss (0 = auto from train.log).",
    )
    p.add_argument("--epochs", default="", help="Optional epoch range, e.g. 1-12 or 9,10,11.")
    p.add_argument("--batch-size", type=int, default=520)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=3)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def _parse_epoch_filter(spec: str) -> set[int] | None:
    spec = spec.strip()
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _infer_mse_start_epoch(train_log: Path) -> int | None:
    if not train_log.is_file():
        return None
    first: int | None = None
    with train_log.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = _EPOCH_ACTIVE_LOSS.search(line)
            if m and m.group(2).lower() == "mse":
                ep = int(m.group(1))
                first = ep if first is None else min(first, ep)
    return first


def _trained_loss_for_epoch(epoch: int, mse_start: int | None) -> str:
    if mse_start is not None and epoch >= mse_start:
        return "mse"
    return "l1"


def _read_run_config(run_dir: Path) -> dict:
    rc_path = run_dir / "resolved_config.json"
    if not rc_path.is_file():
        raise FileNotFoundError(f"Missing {rc_path}")
    return json.loads(rc_path.read_text(encoding="utf-8"))


def _ensure_dual_metrics_csv(metrics_csv: Path, out_channels: int, *, active_loss: str | None) -> None:
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Missing {metrics_csv}")
    with metrics_csv.open("r", newline="", encoding="utf-8") as rf:
        reader = csv.DictReader(rf)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"{metrics_csv} has no header.")
        header = list(fieldnames)
        rows = list(reader)
    expected = metrics_csv_fieldnames(out_channels, dual_compare=True)
    if header == expected:
        return
    if not header_has_dual_compare(header):
        header, rows = ensure_dual_schema(rows, out_channels, active_loss=active_loss)
    else:
        for row in rows:
            populate_reference_l1_mse_from_row(row, out_channels, active_loss=active_loss)
        header = expected
    _atomic_write_metrics_csv(metrics_csv, header, rows)


def _discover_epoch_checkpoints(run_dir: Path) -> dict[int, Path]:
    run_name = run_dir.name
    candidates: dict[int, list[Path]] = {}
    for path in sorted(run_dir.glob("*.pth")):
        m = _EPOCH_CKPT.search(path.name)
        if m:
            candidates.setdefault(int(m.group(1)), []).append(path)
    by_epoch: dict[int, Path] = {}
    for ep, paths in candidates.items():
        if len(paths) == 1:
            by_epoch[ep] = paths[0]
            continue
        preferred = [p for p in paths if p.name.startswith(f"{run_name}_E")]
        by_epoch[ep] = preferred[0] if preferred else paths[0]
    return by_epoch


def _backup_metrics_csv(metrics_csv: Path) -> Path | None:
    backup = metrics_csv.with_name("metrics.csv.pre_fullval.bak")
    if backup.is_file():
        return None
    if metrics_csv.is_file():
        backup.write_bytes(metrics_csv.read_bytes())
        return backup
    return None


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    rc = _read_run_config(run_dir)
    rc_args = rc.get("args", {})
    out_channels = int(rc.get("params", {}).get("out_channels", 5))
    eigen_ch0 = str(rc_args.get("eigen_ch0_encoding", "uniform"))

    mse_start = args.mse_start_epoch if args.mse_start_epoch > 0 else _infer_mse_start_epoch(run_dir / "train.log")
    epoch_filter = _parse_epoch_filter(args.epochs)

    metrics_csv = run_dir / "metrics.csv"
    config_active_loss = read_trained_loss_from_run_dir(run_dir)
    _ensure_dual_metrics_csv(metrics_csv, out_channels, active_loss=config_active_loss)
    backup_path = _backup_metrics_csv(metrics_csv)
    if backup_path is not None:
        print(f"Backed up {metrics_csv.name} -> {backup_path.name}", flush=True)

    ckpts = _discover_epoch_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"No *_E*.pth checkpoints in {run_dir}")

    epochs = sorted(ep for ep in ckpts if epoch_filter is None or ep in epoch_filter)
    if not epochs:
        raise ValueError("No checkpoints match --epochs filter.")

    output_root = Path(args.output_root)
    dataset = build_test_dataset(
        output_root,
        eigen_ch0,
        out_channels,
        val_full_test=bool(args.val_full_test),
    )
    device = resolve_device(args.allow_cpu)

    model = FourierNeuralOperator(
        modes_height=int(rc_args.get("modes_height", 32)),
        modes_width=int(rc_args.get("modes_width", 32)),
        hidden_channels=int(rc_args.get("hidden_channels", 128)),
        n_layers=int(rc_args.get("layers", 4)),
        out_channels=out_channels,
    ).to(device)

    losses = ["l1", "mse"]
    val_mode = "indices_full" if args.val_full_test else "reduced_indices"
    print(
        f"Backfilling {len(epochs)} epochs in {run_dir.name} | mse_start={mse_start} | "
        f"val_index_source={val_mode} | test_samples={len(dataset)}",
        flush=True,
    )

    for epoch in epochs:
        ckpt = ckpts[epoch]
        if config_active_loss == "smoothl1":
            trained_loss = "smoothl1"
        else:
            trained_loss = _trained_loss_for_epoch(epoch, mse_start)
        load_model_state(model, ckpt, run_dir=run_dir)
        ev = run_evaluation(
            dataset=dataset,
            model=model,
            model_path=ckpt,
            device=device,
            output_root=output_root,
            losses=losses,
            out_channels=out_channels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=bool(args.pin_memory),
            amp=args.amp,
            unified_loss=True,
        )
        n_up, migrated = backfill_metrics_csv(
            metrics_csv,
            target_epoch=epoch,
            out_channels=out_channels,
            trained_loss=trained_loss,
            eval_by_loss=ev.eval_by_loss,
            requested_losses=losses,
        )
        l1_val = ev.losses["l1"]["avg_pixel_loss"]
        mse_val = ev.losses["mse"]["avg_pixel_loss"]
        msg = (
            f"backfill_val_dual_loss | epoch={epoch} trained_loss={trained_loss} "
            f"val_l1={l1_val:.6e} val_mse={mse_val:.6e} rows_updated={n_up} migrated={migrated}"
        )
        print(msg, flush=True)
        _append_train_log_messages(run_dir, [msg])

    print(f"Done. Updated {metrics_csv}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
