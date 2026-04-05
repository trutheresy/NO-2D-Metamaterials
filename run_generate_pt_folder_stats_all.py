from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset


TRAIN_PREFIXES = ("c_train", "b_train")
TEST_PREFIXES = ("c_test", "b_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-folder PT histograms and printout.md for all dataset folders.")
    p.add_argument("--datasets-root", default="D:/Research/NO-2D-Metamaterials/DATASETS")
    p.add_argument("--sample-max", type=int, default=200_000, help="Max sampled values per .pt file")
    p.add_argument("--timeout-sec", type=int, default=240, help="Per-folder subprocess timeout")
    p.add_argument("--seed", type=int, default=0)
    # Internal single-folder mode to keep torch runs short-lived/stable on this machine.
    p.add_argument("--one-pt-dir", default="", help=argparse.SUPPRESS)
    p.add_argument("--dataset-name", default="", help=argparse.SUPPRESS)
    return p.parse_args()


def latest_pt_dir(dataset_dir: Path) -> Path:
    cands = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not cands:
        raise FileNotFoundError(f"No *_pt folder under {dataset_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def discover_dataset_dirs(root: Path) -> list[Path]:
    ds = []
    for p in sorted([d for d in root.iterdir() if d.is_dir()], key=lambda x: x.name.lower()):
        if p.name.startswith(TRAIN_PREFIXES) or p.name.startswith(TEST_PREFIXES):
            ds.append(p)
    return ds


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", mmap=False, weights_only=False)


def _sample_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=k, replace=False))


def numeric_values_from_obj(obj: Any, sample_max: int, rng: np.random.Generator) -> tuple[np.ndarray, str, str]:
    if torch.is_tensor(obj):
        t = obj
        n = int(t.numel())
        shape_desc = "x".join(str(s) for s in t.shape)
        if n == 0:
            return np.array([], dtype=np.float64), f"tensor[{t.dtype}]", shape_desc
        idx = _sample_indices(n, sample_max, rng)
        vals = t.reshape(-1)[torch.from_numpy(idx)].detach().cpu()
        if torch.is_complex(vals):
            vals = vals.real
        return vals.to(torch.float64).numpy(), f"tensor[{t.dtype}]", shape_desc

    if isinstance(obj, np.ndarray):
        arr = obj.reshape(-1)
        n = int(arr.size)
        shape_desc = "x".join(str(s) for s in obj.shape)
        if n == 0:
            return np.array([], dtype=np.float64), f"ndarray[{obj.dtype}]", shape_desc
        idx = _sample_indices(n, sample_max, rng)
        vals = arr[idx]
        if np.iscomplexobj(vals):
            vals = vals.real
        return vals.astype(np.float64, copy=False), f"ndarray[{obj.dtype}]", shape_desc

    if isinstance(obj, TensorDataset):
        chunks: list[np.ndarray] = []
        shape_parts: list[str] = []
        for i, t in enumerate(obj.tensors):
            shape_parts.append(f"t{i}:{'x'.join(str(s) for s in t.shape)}")
            vals, _, _ = numeric_values_from_obj(t, max(10_000, sample_max // max(len(obj.tensors), 1)), rng)
            if vals.size > 0:
                chunks.append(vals)
        if not chunks:
            return np.array([], dtype=np.float64), "TensorDataset", "no numeric tensors"
        vals = np.concatenate(chunks, axis=0)
        if vals.size > sample_max:
            vals = vals[_sample_indices(vals.size, sample_max, rng)]
        return vals.astype(np.float64, copy=False), "TensorDataset", "; ".join(shape_parts)

    if isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
            if np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
                flat = arr.reshape(-1)
                if flat.size == 0:
                    return np.array([], dtype=np.float64), f"{type(obj).__name__}[{arr.dtype}]", str(arr.shape)
                vals = flat[_sample_indices(flat.size, sample_max, rng)]
                if np.iscomplexobj(vals):
                    vals = vals.real
                return vals.astype(np.float64, copy=False), f"{type(obj).__name__}[{arr.dtype}]", str(arr.shape)
        except Exception:
            pass
        return np.array([], dtype=np.float64), type(obj).__name__, "non-numeric-list"

    if isinstance(obj, dict):
        chunks = []
        for v in obj.values():
            vals, _, _ = numeric_values_from_obj(v, max(10_000, sample_max // 8), rng)
            if vals.size > 0:
                chunks.append(vals)
        if not chunks:
            return np.array([], dtype=np.float64), "dict", f"keys={len(obj)} no numeric leaves"
        vals = np.concatenate(chunks, axis=0)
        if vals.size > sample_max:
            vals = vals[_sample_indices(vals.size, sample_max, rng)]
        return vals.astype(np.float64, copy=False), "dict", f"keys={len(obj)}"

    return np.array([], dtype=np.float64), type(obj).__name__, "unsupported"


def stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "non_finite": 0}
    finite = np.isfinite(values)
    non_finite = int((~finite).sum())
    vals = values[finite]
    if vals.size == 0:
        return {
            "count": int(values.size),
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "non_finite": non_finite,
        }
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "non_finite": non_finite,
    }


def save_hist(values: np.ndarray, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    finite = values[np.isfinite(values)] if values.size > 0 else np.array([], dtype=np.float64)
    if finite.size > 0:
        bins = max(40, min(220, int(round(math.sqrt(finite.size)))))
        ax.hist(finite, bins=bins, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_printout(pt_dir: Path, sample_max: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pt_files = sorted([p for p in pt_dir.iterdir() if p.is_file() and p.suffix == ".pt"], key=lambda p: p.name.lower())
    rows: list[dict[str, Any]] = []
    for i, p in enumerate(pt_files, start=1):
        obj = safe_torch_load(p)
        vals, tdesc, sdesc = numeric_values_from_obj(obj, sample_max, rng)
        st = stats(vals)
        out_png = pt_dir / f"hist_{p.stem}.png"
        save_hist(vals, out_png, f"{p.name} sampled value histogram")
        rows.append(
            {
                "file": p.name,
                "type": tdesc,
                "shape_or_structure": sdesc,
                "count": st["count"],
                "mean": st["mean"],
                "std": st["std"],
                "min": st["min"],
                "max": st["max"],
                "non_finite": st["non_finite"],
                "hist_png": out_png.name,
            }
        )
        print(f"[{i}/{len(pt_files)}] {p.name} -> {out_png.name}", flush=True)

    md_path = pt_dir / "printout.md"
    lines = [
        f"# PT Stats for `{pt_dir.name}`",
        "",
        f"- Folder: `{pt_dir}`",
        f"- Files processed: `{len(rows)}`",
        f"- Sampling cap per file: `{sample_max}` values",
        "- Note: stats/histograms are sampled when files exceed the cap.",
        "",
        "| file | type | shape_or_structure | count | mean | std | min | max | non_finite | hist_png |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["file"]),
                    str(r["type"]),
                    str(r["shape_or_structure"]),
                    str(r["count"]),
                    f"{float(r['mean']):.6e}" if np.isfinite(r["mean"]) else "nan",
                    f"{float(r['std']):.6e}" if np.isfinite(r["std"]) else "nan",
                    f"{float(r['min']):.6e}" if np.isfinite(r["min"]) else "nan",
                    f"{float(r['max']):.6e}" if np.isfinite(r["max"]) else "nan",
                    str(r["non_finite"]),
                    str(r["hist_png"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {md_path}", flush=True)


def run_single_mode(pt_dir: Path, sample_max: int, seed: int) -> None:
    build_printout(pt_dir, sample_max, seed)
    # Intentionally hard-exit to avoid torch shutdown crash path on this workstation.
    os._exit(0)


def run_all_mode(args: argparse.Namespace) -> None:
    root = Path(args.datasets_root)
    ds_dirs = discover_dataset_dirs(root)
    if not ds_dirs:
        raise FileNotFoundError(f"No dataset dirs found under {root}")

    status = []
    print(f"[info] datasets={len(ds_dirs)} root={root}")
    script_path = Path(__file__).resolve()
    for i, d in enumerate(ds_dirs, start=1):
        pt_dir = latest_pt_dir(d)
        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            "--one-pt-dir",
            str(pt_dir),
            "--dataset-name",
            d.name,
            "--sample-max",
            str(args.sample_max),
            "--seed",
            str(args.seed + i),
        ]
        print(f"[run] {i}/{len(ds_dirs)} {d.name} -> {pt_dir.name}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_sec)
        ok = (pt_dir / "printout.md").exists()
        status.append(
            {
                "dataset_dir": d.name,
                "pt_dir": str(pt_dir),
                "ok": ok,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-800:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-800:] if proc.stderr else "",
            }
        )
        print(f"[{'ok' if ok else 'fail'}] {d.name}", flush=True)

    status_path = root / "pt_stats_generation_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    ok_count = sum(1 for s in status if s["ok"])
    print(f"[done] {ok_count}/{len(status)} succeeded")
    print(f"[done] status file: {status_path}")


def main() -> None:
    args = parse_args()
    if args.one_pt_dir:
        run_single_mode(Path(args.one_pt_dir), sample_max=args.sample_max, seed=args.seed)
        return
    run_all_mode(args)


if __name__ == "__main__":
    main()
