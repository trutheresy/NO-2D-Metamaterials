from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRAIN_PREFIXES = ("c_train", "b_train")
TEST_PREFIXES = ("c_test", "b_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run validation-train divergence diagnostics using per-dataset subprocess stats.")
    p.add_argument("--output-root", default="D:/Research/NO-2D-Metamaterials/DATASETS")
    p.add_argument("--analysis-root", default="D:/Research/NO-2D-Metamaterials/PLOTS/validation_divergence_debug")
    p.add_argument("--sample-images", type=int, default=2048)
    p.add_argument("--raw-values-per-key", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout-sec", type=int, default=45)
    p.add_argument("--audit-batch-size", type=int, default=260)
    p.add_argument("--audit-num-workers", type=int, default=0)
    p.add_argument("--audit-prefetch-factor", type=int, default=2)
    p.add_argument(
        "--audit-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p.parse_args()


def latest_pt_dir(dataset_dir: Path) -> Path:
    cands = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not cands:
        raise FileNotFoundError(f"No *_pt folder under {dataset_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def discover_dataset_shards(output_root: Path) -> list[dict[str, str]]:
    shards: list[dict[str, str]] = []
    ds_dirs = sorted([p for p in output_root.iterdir() if p.is_dir()])
    for d in ds_dirs:
        split = None
        if d.name.startswith(TRAIN_PREFIXES):
            split = "train"
        elif d.name.startswith(TEST_PREFIXES):
            split = "test"
        if split is None:
            continue
        pt = latest_pt_dir(d)
        shards.append({"dataset_name": d.name, "split": split, "pt_dir": str(pt)})
    return shards


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _vals(rows: list[dict[str, Any]], col: str, split: str | None = None) -> np.ndarray:
    data = []
    for r in rows:
        if split is not None and r["split"] != split:
            continue
        v = r.get(col, np.nan)
        if isinstance(v, (int, float)) and np.isfinite(v):
            data.append(float(v))
    return np.asarray(data, dtype=np.float64)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_s = np.sort(a)
    b_s = np.sort(b)
    grid = np.sort(np.unique(np.concatenate([a_s, b_s])))
    cdf_a = np.searchsorted(a_s, grid, side="right") / a_s.size
    cdf_b = np.searchsorted(b_s, grid, side="right") / b_s.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def compare_train_test(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "geometry_mean",
        "geometry_std",
        "geometry_min",
        "geometry_max",
        "eigen_mean",
        "eigen_std",
        "eigen_min",
        "eigen_max",
        "disp_all_mean",
        "disp_all_std",
        "disp_all_min",
        "disp_all_max",
        "disp_ch0_std",
        "disp_ch1_std",
        "disp_ch2_std",
        "disp_ch3_std",
    ]
    out = []
    for m in metrics:
        tr = _vals(rows, m, split="train")
        te = _vals(rows, m, split="test")
        tr_mean = float(np.mean(tr)) if tr.size > 0 else float("nan")
        te_mean = float(np.mean(te)) if te.size > 0 else float("nan")
        tr_std = float(np.std(tr)) if tr.size > 0 else float("nan")
        te_std = float(np.std(te)) if te.size > 0 else float("nan")
        denom = max(abs(tr_mean), abs(te_mean), 1e-12)
        out.append(
            {
                "metric": m,
                "train_count": int(tr.size),
                "test_count": int(te.size),
                "train_group_mean": tr_mean,
                "test_group_mean": te_mean,
                "abs_mean_gap": float(abs(tr_mean - te_mean)),
                "rel_mean_gap": float(abs(tr_mean - te_mean) / denom),
                "train_group_std": tr_std,
                "test_group_std": te_std,
                "abs_std_gap": float(abs(tr_std - te_std)),
                "ks_distance": ks_distance(tr, te),
            }
        )
    return out


def plot_dataset_metric_hists(rows: list[dict[str, Any]], output_dir: Path) -> None:
    modalities = ("geometry", "eigen", "disp_all")
    stats = ("mean", "std", "min", "max")
    for mod in modalities:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        for i, st in enumerate(stats):
            col = f"{mod}_{st}"
            train_vals = _vals(rows, col, split="train")
            test_vals = _vals(rows, col, split="test")
            ax = axes[i]
            bins = max(15, min(50, int(round(math.sqrt(max(train_vals.size + test_vals.size, 1)) * 2))))
            if train_vals.size > 0:
                ax.hist(train_vals, bins=bins, alpha=0.6, label="train")
            if test_vals.size > 0:
                ax.hist(test_vals, bins=bins, alpha=0.6, label="test")
            ax.set_title(f"{mod} {st}")
            ax.set_xlabel(f"{mod}_{st}")
            ax.set_ylabel("Dataset count")
            if train_vals.size > 0 and test_vals.size > 0:
                ax.legend()
        fig.suptitle(f"Per-dataset {mod} sampled statistics", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"hist_{mod}_dataset_metrics.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i in range(4):
        col = f"disp_ch{i}_std"
        train_vals = _vals(rows, col, split="train")
        test_vals = _vals(rows, col, split="test")
        ax = axes[i]
        bins = max(15, min(50, int(round(math.sqrt(max(train_vals.size + test_vals.size, 1)) * 2))))
        if train_vals.size > 0:
            ax.hist(train_vals, bins=bins, alpha=0.6, label="train")
        if test_vals.size > 0:
            ax.hist(test_vals, bins=bins, alpha=0.6, label="test")
        ax.set_title(f"disp_ch{i}_std")
        ax.set_xlabel("Std")
        ax.set_ylabel("Dataset count")
        if train_vals.size > 0 and test_vals.size > 0:
            ax.legend()
    fig.suptitle("Per-dataset displacement channel std (sampled)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "hist_displacement_channel_std.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_hist(values: np.ndarray, path: Path, title: str, xlabel: str, bins: int = 200) -> None:
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=bins, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_raw_value_histograms(samples: dict[str, np.ndarray], output_dir: Path) -> None:
    _save_hist(
        samples.get("eigen_raw", np.array([], dtype=np.float32)),
        output_dir / "hist_eigenfrequency_raw_values.png",
        "Eigenfrequency FFT raw values (sampled)",
        "Value",
    )
    _save_hist(
        samples.get("disp_all_raw", np.array([], dtype=np.float32)),
        output_dir / "hist_displacement_all_raw_values.png",
        "Displacement raw values (all channels, sampled)",
        "Value",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i in range(4):
        vals = samples.get(f"disp_ch{i}_raw", np.array([], dtype=np.float32))
        ax = axes[i]
        if vals.size > 0:
            ax.hist(vals, bins=200, alpha=0.85)
        ax.set_title(f"Displacement channel {i} raw values (sampled)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_displacement_raw_values_per_channel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def audit_validation_loading(
    output_root: Path,
    rows: list[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> dict[str, Any]:
    train_names = sorted([r["dataset_name"] for r in rows if r["split"] == "train"])
    test_names = sorted([r["dataset_name"] for r in rows if r["split"] == "test"])
    overlap = sorted(set(train_names).intersection(test_names))

    train_samples = int(sum(int(r.get("n_total", 0)) for r in rows if r["split"] == "train"))
    test_samples = int(sum(int(r.get("n_total", 0)) for r in rows if r["split"] == "test"))
    train_batches = int(math.ceil(train_samples / max(batch_size, 1)))
    test_batches = int(math.ceil(test_samples / max(batch_size, 1)))

    trainer_code = (Path(__file__).parent / "train_from_disk.py").read_text(encoding="utf-8")
    static_checks = {
        "test_loader_built_from_test_ds": bool(re.search(r"test_loader\s*=\s*DataLoader\(test_ds,\s*\*\*test_loader_kwargs\)", trainer_code)),
        "train_loader_built_from_train_ds": bool(
            re.search(r"train_loader\s*=\s*DataLoader\(train_ds,\s*\*\*train_loader_kwargs\)", trainer_code)
        ),
        "eval_calls_evaluate_with_test_loader": bool(
            re.search(r"evaluate\(model,\s*test_loader,\s*device,\s*args\.amp,\s*criterion\)", trainer_code)
        ),
        "eval_not_called_with_train_loader": not bool(
            re.search(r"evaluate\(model,\s*train_loader,\s*device,\s*args\.amp,\s*criterion\)", trainer_code)
        ),
        "discover_shards_train_prefixes_present": "discover_shards(output_root, TRAIN_PREFIXES," in trainer_code,
        "discover_shards_test_prefixes_present": "discover_shards(output_root, TEST_PREFIXES," in trainer_code,
    }

    return {
        "output_root": str(output_root),
        "train_shards_count": len(train_names),
        "test_shards_count": len(test_names),
        "train_samples": train_samples,
        "test_samples": test_samples,
        "train_loader_batches_estimate": train_batches,
        "test_loader_batches_estimate": test_batches,
        "train_shard_sample": train_names[:5],
        "test_shard_sample": test_names[:5],
        "train_test_name_overlap_count": len(overlap),
        "train_test_name_overlap": overlap[:10],
        "audit_loader_params": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
            "pin_memory": bool(pin_memory),
        },
        "static_checks": static_checks,
    }


def scale_ratio_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["eigen_std", "disp_all_std", "disp_ch0_std", "disp_ch1_std", "disp_ch2_std", "disp_ch3_std"]
    means = {m: float(np.mean(_vals(rows, m))) for m in metrics}
    out = [
        {"ratio_name": "std_eigen_over_std_disp_all", "value": means["eigen_std"] / max(means["disp_all_std"], 1e-12)},
        {"ratio_name": "std_eigen_over_std_disp_ch0", "value": means["eigen_std"] / max(means["disp_ch0_std"], 1e-12)},
        {"ratio_name": "std_eigen_over_std_disp_ch1", "value": means["eigen_std"] / max(means["disp_ch1_std"], 1e-12)},
        {"ratio_name": "std_eigen_over_std_disp_ch2", "value": means["eigen_std"] / max(means["disp_ch2_std"], 1e-12)},
        {"ratio_name": "std_eigen_over_std_disp_ch3", "value": means["eigen_std"] / max(means["disp_ch3_std"], 1e-12)},
    ]
    return out


def write_findings(
    output_dir: Path,
    dataset_rows: list[dict[str, Any]],
    compare_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    train_count = sum(1 for r in dataset_rows if r["split"] == "train")
    test_count = sum(1 for r in dataset_rows if r["split"] == "test")
    ks_values = [r["ks_distance"] for r in compare_rows if np.isfinite(r["ks_distance"])]
    rel_gaps = [r["rel_mean_gap"] for r in compare_rows if np.isfinite(r["rel_mean_gap"])]
    max_gap_row = max(compare_rows, key=lambda r: r["rel_mean_gap"] if np.isfinite(r["rel_mean_gap"]) else -1.0)
    audit_ok = all(bool(v) for v in audit.get("static_checks", {}).values()) and audit.get("train_test_name_overlap_count", 1) == 0

    lines = [
        "# Findings: Validation-Train Loss Divergence Debug",
        "",
        "## Dataset coverage",
        f"- Analyzed datasets: train={train_count}, test={test_count}, total={len(dataset_rows)}.",
        f"- Per-dataset statistics are computed from sampled images (`n_sample_images` in `dataset_stats.csv`).",
        "",
        "## Train-vs-test distribution similarity",
        f"- Mean KS distance across tracked metrics: `{float(np.mean(ks_values)) if ks_values else float('nan'):.4f}`.",
        f"- Max relative mean gap metric: `{max_gap_row['metric']}` with `rel_mean_gap={max_gap_row['rel_mean_gap']:.4e}`.",
        f"- Median relative mean gap across metrics: `{float(np.median(rel_gaps)) if rel_gaps else float('nan'):.4e}`.",
        "",
        "## Output scale comparison",
    ]
    for row in scale_rows:
        lines.append(f"- `{row['ratio_name']}` = `{row['value']:.6e}`.")

    lines.extend(
        [
            "",
            "## Validation loading audit",
            f"- Validation loader wiring checks all pass: `{audit_ok}`.",
            f"- Train/test shard overlap count: `{audit.get('train_test_name_overlap_count', 'n/a')}`.",
            f"- Train samples: `{audit.get('train_samples', 'n/a')}`, test samples: `{audit.get('test_samples', 'n/a')}`.",
            "",
            "## Recommendations",
            "- If eigenfrequency/dispersion scale ratios are high, test channel-weighted loss or per-channel normalization.",
            "- If train/test gaps are small yet val loss plateaus, test regularization/lr schedule and per-channel loss weighting.",
            "- Keep this validation split audit as a guardrail when editing shard discovery and eval code.",
        ]
    )
    (output_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    run_dir = Path(args.analysis_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = run_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    shards = discover_dataset_shards(output_root)
    print(f"[info] output_root={output_root} shard_count={len(shards)}")

    helper = repo_root / "collect_single_dataset_stats.py"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    statuses = []
    rows = []
    all_samples: dict[str, list[np.ndarray]] = {
        "eigen_raw": [],
        "disp_all_raw": [],
        "disp_ch0_raw": [],
        "disp_ch1_raw": [],
        "disp_ch2_raw": [],
        "disp_ch3_raw": [],
    }
    for i, s in enumerate(shards, start=1):
        dataset_name = s["dataset_name"]
        split = s["split"]
        pt_dir = s["pt_dir"]
        row_json = rows_dir / f"{dataset_name}.json"
        row_npz = rows_dir / f"{dataset_name}_samples.npz"
        cmd = [
            sys.executable,
            "-u",
            str(helper),
            "--dataset-name",
            dataset_name,
            "--split",
            split,
            "--pt-dir",
            pt_dir,
            "--out-json",
            str(row_json),
            "--out-sample-npz",
            str(row_npz),
            "--sample-images",
            str(args.sample_images),
            "--raw-values-per-key",
            str(args.raw_values_per_key),
            "--seed",
            str(args.seed + i),
        ]
        print(f"[collect] {i}/{len(shards)} {dataset_name}")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout_sec)
        ok = row_json.exists() and row_npz.exists()
        statuses.append(
            {
                "dataset_name": dataset_name,
                "returncode": proc.returncode,
                "ok": ok,
                "stdout_tail": proc.stdout[-300:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-300:] if proc.stderr else "",
            }
        )
        if not ok:
            continue
        rows.append(json.loads(row_json.read_text(encoding="utf-8")))
        sample_npz = np.load(row_npz)
        for k in all_samples.keys():
            if k in sample_npz:
                all_samples[k].append(sample_npz[k])

    (run_dir / "collection_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    if not rows:
        raise RuntimeError(f"No dataset rows were collected. See {run_dir / 'collection_status.json'}")

    rows = sorted(rows, key=lambda r: (str(r["split"]), str(r["dataset_name"])))
    write_csv(run_dir / "dataset_stats.csv", rows)
    print(f"[done] wrote {run_dir / 'dataset_stats.csv'}")

    plot_dataset_metric_hists(rows, run_dir)
    print("[done] wrote per-dataset metric histograms")

    merged_samples = {k: np.concatenate(v, axis=0) if v else np.array([], dtype=np.float32) for k, v in all_samples.items()}
    plot_raw_value_histograms(merged_samples, run_dir)
    print("[done] wrote raw value histograms")

    compare_rows = compare_train_test(rows)
    write_csv(run_dir / "train_test_similarity.csv", compare_rows)
    print(f"[done] wrote {run_dir / 'train_test_similarity.csv'}")

    scale_rows = scale_ratio_summary(rows)
    write_csv(run_dir / "scale_ratio_summary.csv", scale_rows)
    print(f"[done] wrote {run_dir / 'scale_ratio_summary.csv'}")

    audit = audit_validation_loading(
        output_root=output_root,
        rows=rows,
        batch_size=args.audit_batch_size,
        num_workers=args.audit_num_workers,
        prefetch_factor=args.audit_prefetch_factor,
        pin_memory=bool(args.audit_pin_memory),
    )
    (run_dir / "validation_loader_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[done] wrote {run_dir / 'validation_loader_audit.json'}")

    write_findings(run_dir, rows, compare_rows, scale_rows, audit)
    print(f"[done] wrote {run_dir / 'findings.md'}")


if __name__ == "__main__":
    main()
