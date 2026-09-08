"""Generate MAE/MSE epoch comparison tables from comparison_registry.json."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = REPO_ROOT / "MODELS" / "training_runs" / "comparison_registry.json"
TRAINING_RUNS = REPO_ROOT / "MODELS" / "training_runs"

LABEL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&+._-]*_\d{4}$")

MetricRow = tuple[float | None, float | None, float | None, float | None]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to comparison_registry.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default="",
        help="Optional path to write markdown report (stdout always printed)",
    )
    p.add_argument(
        "--val-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Omit train columns; show validation metrics only (default: true).",
    )
    return p.parse_args()


def load_registry(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs: dict[str, dict[str, str]] = data.get("runs", {})
    if not runs:
        raise ValueError(f"No runs in registry: {path}")
    if "aliases" in data or any("aliases" in v for v in runs.values()):
        raise ValueError("Registry must not contain aliases (one label per run).")
    for label in runs:
        if not LABEL_RE.match(label):
            raise ValueError(
                f"Invalid label {label!r}; expected LOSS_MMDD (e.g. L1_0401, SL1_0626)."
            )
    if len(runs) != len(set(runs)):
        raise ValueError("Duplicate labels in registry.")
    run_dirs = [v["run_dir"] for v in runs.values()]
    if len(run_dirs) != len(set(run_dirs)):
        raise ValueError("Duplicate run_dir entries in registry.")
    return runs


def fv(row: dict[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    v = row.get(key, "")
    if v in ("", None):
        return None
    return float(v)


def is_l1_l2_phased(run_dir_name: str) -> bool:
    return "L1&L2" in run_dir_name


def extract_mae_mse(run_dir_name: str, row: dict[str, str]) -> MetricRow:
    """Return train_mae, val_mae, train_mse, val_mse."""
    if is_l1_l2_phased(run_dir_name):
        ep = int(row["epoch"])
        if ep <= 10:
            return fv(row, "train_loss"), fv(row, "val_loss"), None, fv(row, "val_compare_loss")
        return None, fv(row, "val_compare_loss"), fv(row, "train_loss"), fv(row, "val_loss")

    tr_mae = fv(row, "train_l1_loss")
    va_mae = fv(row, "val_l1_loss")
    tr_mse = fv(row, "train_mse_loss")
    va_mse = fv(row, "val_mse_loss")
    if tr_mae is not None or va_mae is not None:
        return tr_mae, va_mae, tr_mse, va_mse

    # Legacy L1-only runs: val_loss / train_loss are MAE.
    return fv(row, "train_loss"), fv(row, "val_loss"), None, None


def load_run_metrics(run_dir: Path, label: str) -> list[MetricRow]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{label}: missing {metrics_path}")
    rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8")))
    return [extract_mae_mse(run_dir.name, r) for r in rows]


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4e}"


def mark_best(cells: list[str], values: list[float | None]) -> list[str]:
    """Bold+asterisk the single best (lowest) value per column; earliest epoch on ties."""
    nums = [(i, v) for i, v in enumerate(values) if v is not None and not math.isnan(v)]
    if not nums:
        return cells
    best_i, best_v = min(nums, key=lambda iv: (iv[1], iv[0]))
    tol = max(abs(best_v) * 1e-6, 1e-15)
    out = list(cells)
    if values[best_i] is not None and abs(values[best_i] - best_v) <= tol:
        out[best_i] = f"**{cells[best_i]}***"
    return out


def build_manifest(runs: dict[str, dict[str, str]]) -> list[str]:
    lines = ["### Comparison manifest", ""]
    sources = {v.get("val_index_source", "unknown") for v in runs.values()}
    for label, entry in runs.items():
        run_dir = TRAINING_RUNS / entry["run_dir"]
        n_ep = len(load_run_metrics(run_dir, label))
        vis = entry.get("val_index_source", "unknown")
        lines.append(
            f"- **{label}** -> `{entry['run_dir']}` "
            f"({n_ep} epochs, val: {vis})"
        )
    if len(sources) > 1:
        lines.append("")
        lines.append(
            "WARNING: Mixed val_index_source across runs; compare MAE/MSE across columns with care."
        )
    lines.append("")
    return lines


def render_table(
    title: str,
    metric: str,
    labels: list[str],
    parsed: dict[str, list[MetricRow]],
    val_only: bool,
) -> list[str]:
    va_idx = 1 if metric == "mae" else 3
    max_ep = max(len(parsed[lb]) for lb in labels)

    if val_only:
        col_names = [f"{lb} val" for lb in labels]
    else:
        col_names = []
        for lb in labels:
            col_names.extend([f"{lb} train", f"{lb} val"])

    lines = [f"### {title}", ""]
    hdr = "| ep | " + " | ".join(col_names) + " |"
    sep = "|---|" + "|".join(["---:"] * len(col_names)) + "|"
    lines.extend([hdr, sep])

    if val_only:
        col_vals: list[list[float | None]] = [[] for _ in labels]
        row_cells: list[list[str]] = []
        for i in range(max_ep):
            cells: list[str] = []
            for j, lb in enumerate(labels):
                tup = parsed[lb][i] if i < len(parsed[lb]) else (None, None, None, None)
                v = tup[va_idx]
                cells.append(fmt(v))
                col_vals[j].append(v)
            row_cells.append(cells)
        for j in range(len(labels)):
            marked = mark_best([row_cells[i][j] for i in range(max_ep)], col_vals[j])
            for i in range(max_ep):
                row_cells[i][j] = marked[i]
        for i in range(max_ep):
            lines.append(f"| {i + 1} | " + " | ".join(row_cells[i]) + " |")
    else:
        mi = 0 if metric == "mae" else 2
        for i in range(max_ep):
            cells_tr: list[str] = []
            cells_va: list[str] = []
            vals_tr: list[float | None] = []
            vals_va: list[float | None] = []
            for lb in labels:
                tup = parsed[lb][i] if i < len(parsed[lb]) else (None, None, None, None)
                tr, va = tup[mi], tup[va_idx]
                cells_tr.append(fmt(tr))
                cells_va.append(fmt(va))
                vals_tr.append(tr)
                vals_va.append(va)
            cells_tr = mark_best(cells_tr, vals_tr)
            cells_va = mark_best(cells_va, vals_va)
            parts: list[str] = []
            for ct, cv in zip(cells_tr, cells_va):
                parts.extend([ct, cv])
            lines.append(f"| {i + 1} | " + " | ".join(parts) + " |")
    lines.append("")
    return lines


def best_summary(labels: list[str], parsed: dict[str, list[MetricRow]]) -> list[str]:
    lines = ["### Best val per label", ""]
    for lb in labels:
        rows = parsed[lb]
        vmae = min((r[1] for r in rows if r[1] is not None), default=None)
        vmse = min((r[3] for r in rows if r[3] is not None), default=None)
        mae_ep = next((i + 1 for i, r in enumerate(rows) if r[1] == vmae), None)
        mse_ep = next((i + 1 for i, r in enumerate(rows) if r[3] == vmse), None)
        mae_s = f"ep{mae_ep}={vmae:.4e}" if vmae is not None else "n/a"
        mse_s = f"ep{mse_ep}={vmse:.4e}" if vmse is not None else "n/a"
        lines.append(f"- **{lb}**: best val MAE {mae_s}; best val MSE {mse_s}")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    runs = load_registry(args.registry.resolve())
    labels = list(runs.keys())

    for label, entry in runs.items():
        run_dir = TRAINING_RUNS / entry["run_dir"]
        if not run_dir.is_dir():
            raise FileNotFoundError(f"{label}: run_dir does not exist: {run_dir}")

    parsed = {lb: load_run_metrics(TRAINING_RUNS / runs[lb]["run_dir"], lb) for lb in labels}

    report_lines: list[str] = []
    report_lines.extend(build_manifest(runs))
    report_lines.extend(render_table("MAE (L1)", "mae", labels, parsed, args.val_only))
    report_lines.extend(render_table("MSE (L2)", "mse", labels, parsed, args.val_only))
    report_lines.extend(best_summary(labels, parsed))

    text = "\n".join(report_lines)
    print(text)
    if args.output:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
