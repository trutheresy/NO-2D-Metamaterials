#!/usr/bin/env python3
"""Decide whether to extend Huber training by another 2 epochs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check train plateau / val overfit for Huber run.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("MODELS/training_runs/NO_I3O5_BCF16_SL1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260626"),
    )
    p.add_argument(
        "--window",
        type=int,
        default=4,
        help="Number of most recent epochs to inspect (default: 4).",
    )
    p.add_argument(
        "--train-plateau-rel",
        type=float,
        default=0.01,
        help="Train Huber rel. drop below this over window => plateaued (default: 1%%).",
    )
    p.add_argument(
        "--val-overfit-rel",
        type=float,
        default=0.025,
        help="Last val Huber this much above window min => overfit (default: 2.5%%).",
    )
    p.add_argument(
        "--val-trend-rel",
        type=float,
        default=0.02,
        help="Monotonic val rise above this over window => overfit (default: 2%%).",
    )
    return p.parse_args()


def load_rows(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", newline="", encoding="utf-8") as rf:
        rows = list(csv.DictReader(rf))
    rows.sort(key=lambda r: int(r["epoch"]))
    return rows


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.is_file():
        print(f"ERROR: missing {metrics_csv}", file=sys.stderr)
        return 2

    rows = load_rows(metrics_csv)
    if len(rows) < 2:
        print("STOP: fewer than 2 epochs logged; not extending further.")
        return 1

    window = min(args.window, len(rows))
    tail = rows[-window:]
    epochs = [int(r["epoch"]) for r in tail]
    train = [float(r["train_loss"]) for r in tail]
    val = [float(r["val_loss"]) for r in tail]

    train_drop = (train[0] - train[-1]) / train[0] if train[0] > 0 else 0.0
    plateaued = train_drop < args.train_plateau_rel

    min_val = min(val)
    last_val = val[-1]
    overfit_vs_best = min_val > 0 and (last_val - min_val) / min_val > args.val_overfit_rel

    monotonic_up = all(val[i] <= val[i + 1] for i in range(len(val) - 1))
    val_rise = (val[-1] - val[0]) / val[0] if val[0] > 0 else 0.0
    overfit_trend = monotonic_up and len(val) >= 3 and val_rise > args.val_trend_rel
    clear_overfit = overfit_vs_best or overfit_trend

    print(f"Window epochs {epochs[0]}..{epochs[-1]} (n={window})")
    print(f"  train Huber: {[f'{x:.3e}' for x in train]}")
    print(f"  val   Huber: {[f'{x:.3e}' for x in val]}")
    print(f"  train rel. drop over window: {train_drop:.2%} (plateau if < {args.train_plateau_rel:.2%})")
    print(f"  val min in window: {min_val:.3e}; last: {last_val:.3e}")
    print(f"  plateaued={plateaued}; clear_overfit={clear_overfit} (vs_best={overfit_vs_best}, trend={overfit_trend})")

    if plateaued:
        print("STOP: train Huber appears to be plateauing.")
        return 1
    if clear_overfit:
        print("STOP: val Huber shows clear overfit behavior.")
        return 1

    print("CONTINUE: train still improving and val does not show clear overfit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
