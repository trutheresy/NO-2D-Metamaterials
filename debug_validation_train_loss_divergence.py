from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


TRAIN_PREFIXES = ("c_train", "b_train")
TEST_PREFIXES = ("c_test", "b_test")


@dataclass
class ShardInfo:
    name: str
    pt_dir: Path
    inputs_path: Path
    outputs_path: Path
    n: int


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    min: float = math.inf
    max: float = -math.inf
    non_finite: int = 0

    def update(self, t: torch.Tensor) -> None:
        flat = t.reshape(-1).to(torch.float32)
        if flat.numel() == 0:
            return
        vals = flat.to(torch.float64)
        self.count += int(vals.numel())
        self.sum += float(vals.sum().item())
        self.sumsq += float((vals * vals).sum().item())
        self.min = min(self.min, float(vals.min().item()))
        self.max = max(self.max, float(vals.max().item()))

    def as_dict(self, prefix: str) -> dict[str, float | int]:
        if self.count <= 0:
            return {
                f"{prefix}_count": 0,
                f"{prefix}_mean": float("nan"),
                f"{prefix}_std": float("nan"),
                f"{prefix}_min": float("nan"),
                f"{prefix}_max": float("nan"),
                f"{prefix}_non_finite": self.non_finite,
            }
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        std = math.sqrt(var)
        return {
            f"{prefix}_count": self.count,
            f"{prefix}_mean": mean,
            f"{prefix}_std": std,
            f"{prefix}_min": self.min,
            f"{prefix}_max": self.max,
            f"{prefix}_non_finite": self.non_finite,
        }


class ChunkValueSampler:
    def __init__(self, sample_per_chunk: int, max_values_per_key: int, seed: int = 0):
        self.sample_per_chunk = int(sample_per_chunk)
        self.max_values_per_key = int(max_values_per_key)
        self.rng = np.random.default_rng(seed)
        self._values: dict[str, list[np.ndarray]] = defaultdict(list)

    def add_tensor(self, key: str, t: torch.Tensor) -> None:
        arr = t.detach().cpu().numpy().reshape(-1)
        if arr.size == 0:
            return
        if arr.size > self.sample_per_chunk:
            idx = self.rng.choice(arr.size, size=self.sample_per_chunk, replace=False)
            arr = arr[idx]
        self._values[key].append(arr.astype(np.float32, copy=False))

    def finalize(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, chunks in self._values.items():
            merged = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.float32)
            if merged.size > self.max_values_per_key:
                idx = self.rng.choice(merged.size, size=self.max_values_per_key, replace=False)
                merged = merged[idx]
            out[key] = merged
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug train/validation loss divergence with dataset statistics and audits.")
    p.add_argument("--output-root", default="D:/Research/NO-2D-Metamaterials/MODELS")
    p.add_argument("--analysis-root", default="D:/Research/NO-2D-Metamaterials/PLOTS/validation_divergence_debug")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--sample-per-chunk", type=int, default=8000)
    p.add_argument("--max-samples-per-key", type=int, default=500000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--audit-batch-size", type=int, default=260)
    p.add_argument("--audit-num-workers", type=int, default=0)
    p.add_argument("--audit-prefetch-factor", type=int, default=2)
    p.add_argument(
        "--audit-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p.parse_args()


def safe_torch_load(path: Path, prefer_mmap: bool = True, weights_only: bool = True) -> Any:
    return torch.load(path, map_location="cpu", mmap=bool(prefer_mmap), weights_only=weights_only)


def looks_like_dataset_root(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    names = [p.name for p in root.iterdir() if p.is_dir()]
    return any(n.startswith(TRAIN_PREFIXES) for n in names) and any(n.startswith(TEST_PREFIXES) for n in names)


def resolve_output_root(user_root: Path) -> Path:
    candidates = [
        user_root,
        Path("D:/Research/NO-2D-Metamaterials/DATASETS"),
        Path("D:/Research/NO-2D-Metamaterials/MODELS"),
    ]
    for c in candidates:
        if looks_like_dataset_root(c):
            return c
    raise FileNotFoundError(
        "No valid dataset root found. Checked: " + ", ".join(str(c) for c in candidates)
    )


def _split_from_name(name: str) -> str:
    if name.startswith(TRAIN_PREFIXES):
        return "train"
    if name.startswith(TEST_PREFIXES):
        return "test"
    return "unknown"


def latest_pt_dir(dataset_dir: Path) -> Path:
    cands = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not cands:
        raise FileNotFoundError(f"No *_pt folder under {dataset_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def discover_shards_for_analysis(output_root: Path, prefixes: tuple[str, ...]) -> list[ShardInfo]:
    shards: list[ShardInfo] = []
    ds_dirs = sorted([p for p in output_root.iterdir() if p.is_dir() and p.name.startswith(prefixes)], key=lambda p: p.name)
    for d in ds_dirs:
        pt = latest_pt_dir(d)
        in_path = pt / "inputs.pt"
        out_path = pt / "outputs.pt"
        ridx_path = pt / "reduced_indices.pt"
        if not in_path.exists() or not out_path.exists():
            raise FileNotFoundError(f"Missing inputs/outputs in {pt}")
        if not ridx_path.exists():
            raise FileNotFoundError(f"Missing reduced_indices.pt in {pt}")
        # Defer sample-count determination to per-shard tensor load to avoid extra startup failure points.
        shards.append(ShardInfo(name=d.name, pt_dir=pt, inputs_path=in_path, outputs_path=out_path, n=-1))
    if not shards:
        raise FileNotFoundError(f"No dataset shards found with prefixes={prefixes} under {output_root}")
    return shards


def compute_shard_stats(
    shard: ShardInfo,
    chunk_size: int,
    sampler: ChunkValueSampler,
    global_stats: dict[str, RunningStats],
) -> dict[str, Any]:
    x = safe_torch_load(shard.inputs_path, prefer_mmap=True, weights_only=True)
    y = safe_torch_load(shard.outputs_path, prefer_mmap=True, weights_only=True)

    n_hint = int(shard.n)
    if n_hint > 0:
        n = min(n_hint, int(x.shape[0]), int(y.shape[0]))
    else:
        n = min(int(x.shape[0]), int(y.shape[0]))
    local = {
        "geometry": RunningStats(),
        "eigen": RunningStats(),
        "disp_all": RunningStats(),
        "disp_ch0": RunningStats(),
        "disp_ch1": RunningStats(),
        "disp_ch2": RunningStats(),
        "disp_ch3": RunningStats(),
    }

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xb = x[start:end]
        yb = y[start:end]
        geometry = xb[:, 0, :, :]
        eigen = yb[:, 0, :, :]
        disp = yb[:, 1:5, :, :]

        local["geometry"].update(geometry)
        local["eigen"].update(eigen)
        local["disp_all"].update(disp)
        for i in range(4):
            local[f"disp_ch{i}"].update(disp[:, i, :, :])

        global_stats["geometry"].update(geometry)
        global_stats["eigen"].update(eigen)
        global_stats["disp_all"].update(disp)
        for i in range(4):
            global_stats[f"disp_ch{i}"].update(disp[:, i, :, :])

        sampler.add_tensor("eigen_raw", eigen)
        sampler.add_tensor("disp_all_raw", disp)
        for i in range(4):
            sampler.add_tensor(f"disp_ch{i}_raw", disp[:, i, :, :])

    row: dict[str, Any] = {
        "dataset_name": shard.name,
        "split": _split_from_name(shard.name),
        "pt_dir": str(shard.pt_dir),
        "n_from_reduced_indices": int(shard.n) if shard.n > 0 else None,
        "n_inputs": int(x.shape[0]),
        "n_outputs": int(y.shape[0]),
        "n_used": int(n),
    }
    for key in ("geometry", "eigen", "disp_all", "disp_ch0", "disp_ch1", "disp_ch2", "disp_ch3"):
        row.update(local[key].as_dict(key))
    return row


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
        v = r[col]
        if isinstance(v, (int, float)) and np.isfinite(v):
            data.append(float(v))
    return np.asarray(data, dtype=np.float64)


def _save_hist(values: np.ndarray, path: Path, title: str, xlabel: str, bins: int = 40, logy: bool = False) -> None:
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=bins, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
        fig.suptitle(f"Per-dataset {mod} statistics", fontsize=14)
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
    fig.suptitle("Per-dataset displacement channel std", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "hist_displacement_channel_std.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_raw_value_histograms(samples: dict[str, np.ndarray], output_dir: Path) -> None:
    _save_hist(
        samples.get("eigen_raw", np.array([], dtype=np.float32)),
        output_dir / "hist_eigenfrequency_raw_values.png",
        "Eigenfrequency FFT raw values",
        "Value",
        bins=200,
        logy=True,
    )
    _save_hist(
        samples.get("disp_all_raw", np.array([], dtype=np.float32)),
        output_dir / "hist_displacement_all_raw_values.png",
        "Displacement raw values (all channels combined)",
        "Value",
        bins=200,
        logy=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i in range(4):
        key = f"disp_ch{i}_raw"
        vals = samples.get(key, np.array([], dtype=np.float32))
        ax = axes[i]
        if vals.size > 0:
            ax.hist(vals, bins=200, alpha=0.85)
        ax.set_title(f"Displacement channel {i} raw values")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_displacement_raw_values_per_channel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


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
    out: list[dict[str, Any]] = []
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


def audit_validation_loading(
    output_root: Path,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> dict[str, Any]:
    train_shards = discover_shards_for_analysis(output_root, TRAIN_PREFIXES)
    test_shards = discover_shards_for_analysis(output_root, TEST_PREFIXES)
    train_samples = int(sum(int(safe_torch_load(s.inputs_path, prefer_mmap=True, weights_only=True).shape[0]) for s in train_shards))
    test_samples = int(sum(int(safe_torch_load(s.inputs_path, prefer_mmap=True, weights_only=True).shape[0]) for s in test_shards))
    train_batches = int(math.ceil(train_samples / max(batch_size, 1)))
    test_batches = int(math.ceil(test_samples / max(batch_size, 1)))

    train_names = [s.name for s in train_shards]
    test_names = [s.name for s in test_shards]
    overlap = sorted(set(train_names).intersection(test_names))

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
        "train_shards_count": len(train_shards),
        "test_shards_count": len(test_shards),
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


def compute_scale_ratio_summary(global_stats: dict[str, RunningStats]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eigen_std = global_stats["eigen"].as_dict("eigen")["eigen_std"]
    disp_all_std = global_stats["disp_all"].as_dict("disp_all")["disp_all_std"]
    rows.append(
        {
            "ratio_name": "std_eigen_over_std_disp_all",
            "value": float(eigen_std / max(float(disp_all_std), 1e-12)),
        }
    )
    for i in range(4):
        disp_std = global_stats[f"disp_ch{i}"].as_dict(f"disp_ch{i}")[f"disp_ch{i}_std"]
        rows.append(
            {
                "ratio_name": f"std_eigen_over_std_disp_ch{i}",
                "value": float(eigen_std / max(float(disp_std), 1e-12)),
            }
        )
    return rows


def latest_metrics_channel_summary(repo_root: Path) -> dict[str, Any]:
    runs_root = repo_root / "MODELS" / "training_runs"
    if not runs_root.exists():
        return {"available": False, "reason": f"Missing {runs_root}"}
    run_dirs = [d for d in runs_root.iterdir() if d.is_dir() and (d / "metrics.csv").exists()]
    if not run_dirs:
        return {"available": False, "reason": "No training run with metrics.csv found"}
    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    rows: list[dict[str, Any]] = []
    with (latest / "metrics.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return {"available": False, "reason": f"{latest / 'metrics.csv'} has no rows"}

    first = rows[0]
    last = rows[-1]
    channel_keys = ["val_loss_ch0", "val_loss_ch1", "val_loss_ch2", "val_loss_ch3", "val_loss_ch4"]
    ch = []
    for k in channel_keys:
        a = float(first[k])
        b = float(last[k])
        ratio = b / max(a, 1e-12)
        ch.append({"channel": k, "first": a, "last": b, "last_over_first": ratio})
    return {
        "available": True,
        "run_dir": str(latest),
        "epochs_observed": len(rows),
        "channels": ch,
    }


def write_findings(
    output_dir: Path,
    dataset_rows: list[dict[str, Any]],
    compare_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    latest_channel_summary: dict[str, Any],
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
        f"- Output root used for analysis: `{audit.get('output_root', 'unknown')}`.",
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
            "## Additional checks",
        ]
    )
    if latest_channel_summary.get("available", False):
        lines.append(f"- Latest run inspected: `{latest_channel_summary['run_dir']}`.")
        for c in latest_channel_summary["channels"]:
            lines.append(
                f"- `{c['channel']}` first={c['first']:.6e}, last={c['last']:.6e}, last/first={c['last_over_first']:.6e}."
            )
    else:
        lines.append(f"- Per-channel val-loss trend check unavailable: {latest_channel_summary.get('reason', 'unknown')}.")

    lines.extend(
        [
            "",
            "## Recommendations",
            "- If eigenfrequency/dispersion scale ratios are high, test channel-weighted loss or per-channel normalization.",
            "- If train/test gaps are small yet val loss plateaus, test regularization/lr schedule and sample-level leakage checks.",
            "- Keep current validation split audit in CI-style sanity checks when changing shard discovery/loading logic.",
        ]
    )
    (output_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    print("[info] parsed args", flush=True)
    repo_root = Path(__file__).resolve().parent
    out_root = Path(args.analysis_root)
    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] run_dir={run_dir}", flush=True)

    output_root = resolve_output_root(Path(args.output_root))
    print(f"[info] resolved output_root={output_root}", flush=True)
    train_shards = discover_shards_for_analysis(output_root, TRAIN_PREFIXES)
    print(f"[info] discovered train shards={len(train_shards)}", flush=True)
    test_shards = discover_shards_for_analysis(output_root, TEST_PREFIXES)
    print(f"[info] discovered test shards={len(test_shards)}", flush=True)
    all_shards = train_shards + test_shards
    print(f"[info] output_root={output_root} train_shards={len(train_shards)} test_shards={len(test_shards)}")

    sampler = ChunkValueSampler(
        sample_per_chunk=args.sample_per_chunk,
        max_values_per_key=args.max_samples_per_key,
        seed=args.seed,
    )
    global_stats = {
        "geometry": RunningStats(),
        "eigen": RunningStats(),
        "disp_all": RunningStats(),
        "disp_ch0": RunningStats(),
        "disp_ch1": RunningStats(),
        "disp_ch2": RunningStats(),
        "disp_ch3": RunningStats(),
    }

    rows: list[dict[str, Any]] = []
    for i, shard in enumerate(all_shards, start=1):
        print(f"[stats] {i}/{len(all_shards)} {shard.name}")
        row = compute_shard_stats(
            shard=shard,
            chunk_size=args.chunk_size,
            sampler=sampler,
            global_stats=global_stats,
        )
        rows.append(row)

    rows = sorted(rows, key=lambda r: (str(r["split"]), str(r["dataset_name"])))
    dataset_stats_csv = run_dir / "dataset_stats.csv"
    write_csv(dataset_stats_csv, rows)
    print(f"[done] wrote {dataset_stats_csv}")

    plot_dataset_metric_hists(rows, run_dir)
    print(f"[done] wrote per-dataset metric histograms in {run_dir}")

    raw_samples = sampler.finalize()
    plot_raw_value_histograms(raw_samples, run_dir)
    print(f"[done] wrote raw-value histograms in {run_dir}")

    compare_rows = compare_train_test(rows)
    compare_csv = run_dir / "train_test_similarity.csv"
    write_csv(compare_csv, compare_rows)
    print(f"[done] wrote {compare_csv}")

    scale_rows = compute_scale_ratio_summary(global_stats)
    scale_csv = run_dir / "scale_ratio_summary.csv"
    write_csv(scale_csv, scale_rows)
    print(f"[done] wrote {scale_csv}")

    audit = audit_validation_loading(
        output_root=output_root,
        batch_size=args.audit_batch_size,
        num_workers=args.audit_num_workers,
        prefetch_factor=args.audit_prefetch_factor,
        pin_memory=bool(args.audit_pin_memory),
    )
    audit_json = run_dir / "validation_loader_audit.json"
    audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[done] wrote {audit_json}")

    latest_channel_summary = latest_metrics_channel_summary(repo_root)
    channel_json = run_dir / "latest_run_channel_trends.json"
    channel_json.write_text(json.dumps(latest_channel_summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {channel_json}")

    write_findings(run_dir, rows, compare_rows, scale_rows, audit, latest_channel_summary)
    print(f"[done] wrote {run_dir / 'findings.md'}")


if __name__ == "__main__":
    main()
