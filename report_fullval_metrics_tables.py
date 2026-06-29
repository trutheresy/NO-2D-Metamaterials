#!/usr/bin/env python3
"""Print MAE/MSE training-progress tables from metrics.csv (full-val backfill)."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "MODELS" / "training_runs"

DEFAULT_RUNS = [
    {
        # Production L1→MSE run (Apr-08 session onward); folder suffix is 260401 from Apr-01 L1 start.
        "label": "0408",
        "dir": RUNS / "NO_I3O5_BCF16_L1&L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401",
        "mse_start_epoch": 11,
        "start_lr": "2e-3",
    },
    {
        "label": "0619",
        "dir": RUNS / "NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619",
        "mse_start_epoch": 9,
        "start_lr": "2e-3",
    },
    {
        "label": "0622",
        "dir": RUNS / "NO_I3O5_BCF16_L1_HC128_LR4e-03_WD0e+00_SS1_G8e-01_ch0u_260622",
        "mse_start_epoch": None,
        "start_lr": "4e-3",
    },
]

_EPOCH_ACTIVE_LOSS = re.compile(r"epoch=(\d+)/\d+\s+loss=(mse|l1)", re.IGNORECASE)


def _fmt_num(x: float) -> str:
    return f"{x:.3e}"


def _trained_loss(epoch: int, mse_start: int | None) -> str:
    if mse_start is not None and epoch >= mse_start:
        return "mse"
    return "l1"


def _infer_mse_start(run_dir: Path) -> int | None:
    train_log = run_dir / "train.log"
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


def load_metrics(run_dir: Path) -> dict[int, dict[str, str]]:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as rf:
        rows = list(csv.DictReader(rf))
    return {int(row["epoch"]): row for row in rows}


def extract_mae_mse(row: dict[str, str], trained: str) -> tuple[float | None, float, float | None, float]:
    if row.get("val_l1_loss") and row.get("val_mse_loss"):
        train_mae = float(row["train_l1_loss"]) if row.get("train_l1_loss") else None
        val_mae = float(row["val_l1_loss"])
        train_mse = float(row["train_mse_loss"]) if row.get("train_mse_loss") else None
        val_mse = float(row["val_mse_loss"])
        return train_mae, val_mae, train_mse, val_mse

    train_loss = float(row["train_loss"])
    val_loss = float(row["val_loss"])
    train_cmp_s = row.get("train_compare_loss") or ""
    val_cmp_s = row.get("val_compare_loss") or ""
    train_compare = float(train_cmp_s) if train_cmp_s else None
    val_compare = float(val_cmp_s) if val_cmp_s else None

    if trained == "l1":
        train_mae: float | None = train_loss
        val_mae = val_loss
        # Migrated rows duplicate MAE into train_compare; dual-logged runs store MSE there.
        if train_compare is None or train_compare >= train_loss * 0.5:
            train_mse: float | None = None
        else:
            train_mse = train_compare
        val_mse = val_compare if val_compare is not None else float("nan")
        return train_mae, val_mae, train_mse, val_mse

    train_mae = train_compare if train_compare is not None else float("nan")
    val_mae = val_compare if val_compare is not None else float("nan")
    return train_mae, val_mae, train_loss, val_loss


def _fmt_lr(x: float | None, mark_best: bool) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    s = f"{x:.2e}"
    return s + "*" if mark_best else s


def _fmt_train(x: float | None, mark_best: bool) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    s = f"*{_fmt_num(x)}*"
    return s + "*" if mark_best else s


def _fmt_val(x: float, mark_best: bool) -> str:
    if math.isnan(x):
        return "-"
    s = f"**{_fmt_num(x)}**"
    return s + "*" if mark_best else s


def build_table(runs: list[dict], metric: str, max_epoch: int | None, *, use_lr: bool = False) -> list[str]:
    metric_key = metric  # "mae" or "mse"
    loaded: dict[str, dict[int, dict[str, str]]] = {}
    mse_starts: dict[str, int | None] = {}
    all_epochs: set[int] = set()

    for spec in runs:
        rows = load_metrics(spec["dir"])
        loaded[spec["label"]] = rows
        mse_start = spec.get("mse_start_epoch")
        if mse_start is None:
            mse_start = _infer_mse_start(spec["dir"])
        mse_starts[spec["label"]] = mse_start
        all_epochs.update(rows.keys())

    epoch_limit = max(all_epochs) if max_epoch is None else max_epoch

    # Collect raw values per column for min tracking.
    col_keys: list[tuple[str, str]] = []  # (label, "lr"|"train"|"val")
    for spec in runs:
        col_keys.append((spec["label"], "lr" if use_lr else "train"))
        col_keys.append((spec["label"], "val"))

    grid: dict[tuple[int, str, str], float | None] = {}
    for ep in range(1, epoch_limit + 1):
        for spec in runs:
            label = spec["label"]
            row = loaded[label].get(ep)
            if row is None:
                grid[(ep, label, "lr" if use_lr else "train")] = None
                grid[(ep, label, "val")] = None
                continue
            trained = _trained_loss(ep, mse_starts[label])
            train_mae, val_mae, train_mse, val_mse = extract_mae_mse(row, trained)
            if use_lr:
                grid[(ep, label, "lr")] = float(row["lr"])
            elif metric_key == "mae":
                grid[(ep, label, "train")] = train_mae
            else:
                grid[(ep, label, "train")] = train_mse
            if metric_key == "mae":
                grid[(ep, label, "val")] = val_mae
            else:
                grid[(ep, label, "val")] = val_mse

    left_kind = "lr" if use_lr else "train"
    best: dict[tuple[str, str], float] = {}
    for label, kind in col_keys:
        vals = [
            v
            for ep in range(1, epoch_limit + 1)
            if (v := grid.get((ep, label, kind))) is not None and not math.isnan(v)
        ]
        best[(label, kind)] = min(vals) if vals else float("inf")

    lines: list[str] = []
    header = ["Ep"]
    for spec in runs:
        lr = spec["start_lr"]
        lbl = spec["label"]
        left_hdr = f"{lbl} {lr} LR" if use_lr else f"{lbl} {lr} train"
        header.append(left_hdr)
        header.append(f"{lbl} {lr} val")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for ep in range(1, epoch_limit + 1):
        cells = [str(ep)]
        for spec in runs:
            label = spec["label"]
            left_v = grid.get((ep, label, left_kind))
            val_v = grid.get((ep, label, "val"))
            left_best = (
                left_v is not None
                and not math.isnan(left_v)
                and abs(left_v - best[(label, left_kind)]) <= abs(left_v) * 1e-6
            )
            val_best = (
                val_v is not None
                and not math.isnan(val_v)
                and abs(val_v - best[(label, "val")]) <= abs(val_v) * 1e-6
            )
            if use_lr:
                cells.append(_fmt_lr(left_v, left_best))
            else:
                cells.append(_fmt_train(left_v, left_best))
            cells.append(_fmt_val(val_v if val_v is not None else float("nan"), val_best))
        lines.append("| " + " | ".join(cells) + " |")

    return lines


def main() -> int:
    p = argparse.ArgumentParser(description="Report full-val MAE/MSE tables for training runs.")
    p.add_argument("--max-epoch", type=int, default=0, help="Limit epochs shown (0 = max across runs).")
    p.add_argument(
        "--layout",
        choices=("train", "lr"),
        default="train",
        help="Left column per run: train loss (default) or LR.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to this file as well as stdout.",
    )
    args = p.parse_args()
    max_epoch = args.max_epoch if args.max_epoch > 0 else None
    use_lr = args.layout == "lr"
    default_out = RUNS / (
        "fullval_metrics_report_lr.txt" if use_lr else "fullval_metrics_report.txt"
    )
    output = args.output if args.output is not None else default_out

    out_lines: list[str] = []
    out_lines.append("Full validation set (`indices_full`, ~3.9M samples). Val losses backfilled from checkpoints.")
    out_lines.append(
        "**0408** = `L1&L2_..._260401` (Apr-08 MSE from ep 11; ep 1-10 L1). "
        "Not the separate Apr-01 L1-only `L1_..._260401` folder."
    )
    if use_lr:
        out_lines.append("**Val losses bold.** Lowest value per column marked with * (LR column = minimum LR reached).")
    else:
        out_lines.append("*Train losses italicized;* **val losses bold.** Lowest value per column marked with *.")
    out_lines.append("")
    out_lines.append("### Table 1 — MAE (L1)")
    out_lines.append("")
    out_lines.extend(build_table(DEFAULT_RUNS, "mae", max_epoch, use_lr=use_lr))
    out_lines.append("")
    out_lines.append("### Table 2 — MSE (L2)")
    out_lines.append("")
    out_lines.extend(build_table(DEFAULT_RUNS, "mse", max_epoch, use_lr=use_lr))

    text = "\n".join(out_lines)
    print(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"\nWrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
