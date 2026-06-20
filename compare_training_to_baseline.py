"""Compare per-epoch train/val losses after training completes (reads metrics.csv only)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE_RUN = (
    ROOT / "MODELS/training_runs/NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401"
)


def metrics_csv_for_run(run_dir: Path) -> Path:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"No metrics.csv in {run_dir}. Run compare only after train_from_disk.py finishes."
        )
    if path.stat().st_size == 0:
        raise FileNotFoundError(
            f"metrics.csv in {run_dir} is empty (training incomplete or killed before epoch 1)."
        )
    return path


def load_metrics(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "epoch" not in reader.fieldnames:
            raise ValueError(f"Invalid metrics header in {path}")
        for row in reader:
            ep = int(row["epoch"])
            rows[ep] = {
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
            }
            for i in range(5):
                k = f"val_loss_ch{i}"
                if k in row and row[k]:
                    rows[ep][k] = float(row[k])
    if not rows:
        raise ValueError(f"No epoch rows in {path}")
    return rows


def fmt_delta(new: float | None, base: float | None) -> str:
    if new is None or base is None:
        return "n/a"
    d = new - base
    pct = 100.0 * d / base if base else float("nan")
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.3e} ({sign}{pct:.2f}%)"


def print_table(baseline: dict[int, dict], new_run: dict[int, dict], max_epoch: int) -> None:
    print(
        f"{'Ep':>3} | {'base_train':>11} {'new_train':>11} {'d_train':>18} | "
        f"{'base_val':>11} {'new_val':>11} {'d_val':>18}"
    )
    print("-" * 95)
    for ep in range(1, max_epoch + 1):
        b = baseline.get(ep)
        n = new_run.get(ep)
        if b is None and n is None:
            continue
        bt = b["train_loss"] if b else None
        bv = b["val_loss"] if b else None
        nt = n["train_loss"] if n else None
        nv = n["val_loss"] if n else None
        bt_s = f"{bt:.3e}" if bt is not None else "-"
        bv_s = f"{bv:.3e}" if bv is not None else "-"
        nt_s = f"{nt:.3e}" if nt is not None else "-"
        nv_s = f"{nv:.3e}" if nv is not None else "-"
        print(
            f"{ep:3d} | {bt_s:>11} {nt_s:>11} {fmt_delta(nt, bt):>18} | "
            f"{bv_s:>11} {nv_s:>11} {fmt_delta(nv, bv):>18}"
        )


def build_report(
    baseline: dict[int, dict],
    new_run: dict[int, dict],
    baseline_path: Path,
    new_path: Path,
    baseline_run: Path,
    new_run_dir: Path,
) -> dict:
    overlap = sorted(set(baseline) & set(new_run))
    val_deltas = [
        new_run[ep]["val_loss"] - baseline[ep]["val_loss"]
        for ep in overlap
        if baseline[ep].get("val_loss") is not None and new_run[ep].get("val_loss") is not None
    ]
    rows = []
    for ep in sorted(set(baseline) | set(new_run)):
        b = baseline.get(ep, {})
        n = new_run.get(ep, {})
        bv, nv = b.get("val_loss"), n.get("val_loss")
        rows.append(
            {
                "epoch": ep,
                "baseline_train_loss": b.get("train_loss"),
                "baseline_val_loss": bv,
                "new_train_loss": n.get("train_loss"),
                "new_val_loss": nv,
                "delta_val_loss": (nv - bv) if nv is not None and bv is not None else None,
            }
        )
    return {
        "baseline_run": str(baseline_run),
        "new_run": str(new_run_dir),
        "baseline_metrics": str(baseline_path),
        "new_metrics": str(new_path),
        "baseline_epochs": len(baseline),
        "new_epochs": len(new_run),
        "overlapping_epochs": len(overlap),
        "mean_delta_val_loss": (sum(val_deltas) / len(val_deltas)) if val_deltas else None,
        "new_better_epochs": sum(1 for d in val_deltas if d < 0),
        "new_worse_epochs": sum(1 for d in val_deltas if d > 0),
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Post-training comparison of per-epoch losses from metrics.csv files."
    )
    p.add_argument(
        "--baseline-run",
        type=Path,
        default=DEFAULT_BASELINE_RUN,
        help="Baseline training run directory (must contain metrics.csv).",
    )
    p.add_argument(
        "--new-run",
        type=Path,
        required=True,
        help="New training run directory (must contain metrics.csv).",
    )
    p.add_argument(
        "--baseline-metrics",
        type=Path,
        default=None,
        help="Optional explicit baseline metrics.csv (overrides --baseline-run).",
    )
    p.add_argument(
        "--new-metrics",
        type=Path,
        default=None,
        help="Optional explicit new metrics.csv (overrides --new-run).",
    )
    p.add_argument(
        "--max-epoch",
        type=int,
        default=0,
        help="Limit table to this epoch (0 = max available in either run).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (default: <new-run>/training_baseline_comparison.json).",
    )
    args = p.parse_args()

    baseline_path = args.baseline_metrics or metrics_csv_for_run(args.baseline_run)
    new_path = args.new_metrics or metrics_csv_for_run(args.new_run)
    out_path = args.output or (args.new_run / "training_baseline_comparison.json")

    baseline = load_metrics(baseline_path)
    new_run = load_metrics(new_path)
    max_epoch = args.max_epoch or max(max(baseline), max(new_run))

    print(f"Baseline: {baseline_path} ({len(baseline)} epochs)")
    print(f"New run:  {new_path} ({len(new_run)} epochs)\n")
    print_table(baseline, new_run, max_epoch)

    report = build_report(baseline, new_run, baseline_path, new_path, args.baseline_run, args.new_run)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")

    if report["mean_delta_val_loss"] is not None:
        d = report["mean_delta_val_loss"]
        better = report["new_better_epochs"]
        worse = report["new_worse_epochs"]
        print(
            f"Summary: mean delta val_loss={d:+.3e} over {report['overlapping_epochs']} epochs "
            f"({better} better, {worse} worse for new run)."
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
