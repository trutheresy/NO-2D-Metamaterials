"""Poll metrics.csv during training; print per-epoch loss vs 04-08 L1&L2 baseline."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE = ROOT / (
    "MODELS/training_runs/NO_I3O5_BCF16_L1&L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401"
)
DEFAULT_NEW_RUN = ROOT / (
    "MODELS/training_runs/NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619"
)
MSE_FIRST_BASELINE_EPOCH = 11  # 04-08 session started MSE at epoch 11


def load_csv(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    if not path.is_file() or path.stat().st_size == 0:
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["epoch"])
            out: dict[str, float] = {
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
            }
            for key in ("train_compare_loss", "val_compare_loss"):
                if key in row and row[key]:
                    out[key] = float(row[key])
            rows[ep] = out
    return rows


def load_l1_compare_from_log(train_log: Path) -> dict[int, dict[str, float]]:
    """Parse compare_loss=l1 lines (metrics.csv compare cols are wrong for MSE epochs)."""
    pat = re.compile(
        r"epoch=(\d+)/\d+ compare_loss=l1 train_loss=([\d.eE+-]+) val_loss=([\d.eE+-]+)"
    )
    out: dict[int, dict[str, float]] = {}
    if not train_log.is_file():
        return out
    for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            ep = int(m.group(1))
            out[ep] = {"train_compare_loss": float(m.group(2)), "val_compare_loss": float(m.group(3))}
    return out


def fmt_delta(new: float, base: float) -> str:
    d = new - base
    pct = 100.0 * d / base if base else float("nan")
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.3e} ({sign}{pct:.2f}%)"


def baseline_ref_epoch(new_epoch: int, active_loss: str) -> tuple[int, str]:
    """Return (baseline epoch, metric description) for comparison."""
    if active_loss == "mse":
        mse_idx = new_epoch - 8  # new ep 9 -> MSE epoch 1
        base_ep = MSE_FIRST_BASELINE_EPOCH + mse_idx - 1
        return base_ep, "val MSE"
    return new_epoch, "val L1"


def report_epoch(
    ep: int,
    new: dict[str, float],
    baseline: dict[int, dict[str, float]],
    baseline_log_l1: dict[int, dict[str, float]],
    active_loss: str,
) -> None:
    train = new["train_loss"]
    val = new["val_loss"]
    base_ep, metric = baseline_ref_epoch(ep, active_loss)
    b = baseline.get(base_ep)
    if b is None:
        print(f"\n=== Epoch {ep} ({active_loss.upper()}) ===")
        print(f"  train_loss={train:.6e}  val_loss={val:.6e}")
        print(f"  (no baseline epoch {base_ep})")
        return

    b_train = b["train_loss"]
    b_val = b["val_loss"]
    print(f"\n=== Epoch {ep} ({active_loss.upper()}) ===")
    print(f"  train_loss={train:.6e}  val_loss={val:.6e}")
    print(f"  vs 04-08 baseline ep {base_ep} ({metric}):")
    print(f"    baseline train={b_train:.6e}  val={b_val:.6e}")
    print(f"    delta train {fmt_delta(train, b_train)}  val {fmt_delta(val, b_val)}")

    if active_loss == "mse":
        l1_train = new.get("train_compare_loss")
        l1_val = new.get("val_compare_loss")
        bl = baseline.get(ep) or baseline_log_l1.get(ep)
        if l1_train is not None and l1_val is not None and bl is not None:
            bt = bl.get("train_compare_loss", bl.get("train_loss"))
            bv = bl.get("val_compare_loss", bl.get("val_loss"))
            print(f"  vs 04-08 baseline ep {ep} (val L1 at same epoch index):")
            print(f"    new L1-compare train={l1_train:.6e}  val={l1_val:.6e}")
            print(f"    baseline L1 train={bt:.6e}  val={bv:.6e}")
            print(f"    delta train {fmt_delta(l1_train, bt)}  val {fmt_delta(l1_val, bv)}")


def infer_loss_for_epoch(ep: int, row: dict[str, float]) -> str:
    if ep >= 9 or "val_compare_loss" in row:
        return "mse"
    return "l1"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--new-run", type=Path, default=DEFAULT_NEW_RUN)
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--poll-sec", type=float, default=30.0)
    p.add_argument("--start-epoch", type=int, default=5)
    p.add_argument("--end-epoch", type=int, default=12)
    p.add_argument("--once", action="store_true", help="Report all available epochs in range and exit.")
    args = p.parse_args()

    metrics = args.new_run / "metrics.csv"
    baseline_csv = args.baseline / "metrics.csv"
    baseline_log = args.baseline / "train.log"
    baseline = load_csv(baseline_csv)
    baseline_log_l1 = load_l1_compare_from_log(baseline_log)
    reported: set[int] = set()

    print(f"Watching {metrics}")
    print(f"Baseline: {args.baseline}")
    print(f"Epoch range: {args.start_epoch}..{args.end_epoch}")

    while True:
        new_rows = load_csv(metrics)
        for ep in range(args.start_epoch, args.end_epoch + 1):
            if ep in new_rows and ep not in reported:
                loss = infer_loss_for_epoch(ep, new_rows[ep])
                report_epoch(ep, new_rows[ep], baseline, baseline_log_l1, loss)
                reported.add(ep)
                sys.stdout.flush()

        if args.once or len(reported) >= args.end_epoch - args.start_epoch + 1:
            if reported:
                print(f"\nReported epochs: {sorted(reported)}")
            break
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
