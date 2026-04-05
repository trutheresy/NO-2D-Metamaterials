from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect sampled stats for one dataset shard and exit immediately.")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--split", required=True, choices=("train", "test"))
    p.add_argument("--pt-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-sample-npz", required=True)
    p.add_argument("--sample-images", type=int, default=2048)
    p.add_argument("--raw-values-per-key", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def stats_from_array(a: np.ndarray, prefix: str) -> dict[str, float | int]:
    flat = a.reshape(-1).astype(np.float64, copy=False)
    finite = np.isfinite(flat)
    non_finite = int((~finite).sum())
    vals = flat[finite]
    if vals.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_non_finite": non_finite,
        }
    return {
        f"{prefix}_count": int(vals.size),
        f"{prefix}_mean": float(vals.mean()),
        f"{prefix}_std": float(vals.std()),
        f"{prefix}_min": float(vals.min()),
        f"{prefix}_max": float(vals.max()),
        f"{prefix}_non_finite": non_finite,
    }


def sample_flat_values(rng: np.random.Generator, arr: np.ndarray, take: int) -> np.ndarray:
    flat = arr.reshape(-1).astype(np.float32, copy=False)
    if flat.size <= take:
        return flat
    idx = rng.choice(flat.size, size=take, replace=False)
    return flat[idx]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    pt_dir = Path(args.pt_dir)
    x = torch.load(pt_dir / "inputs.pt", map_location="cpu", mmap=True, weights_only=True)
    y = torch.load(pt_dir / "outputs.pt", map_location="cpu", mmap=True, weights_only=True)
    n_total = int(min(x.shape[0], y.shape[0]))

    k = int(max(1, min(args.sample_images, n_total)))
    if k >= n_total:
        idx = np.arange(n_total, dtype=np.int64)
    else:
        idx = np.sort(rng.choice(n_total, size=k, replace=False))
    idx_t = torch.from_numpy(idx)

    xb = x[idx_t].to(torch.float32).numpy()
    yb = y[idx_t].to(torch.float32).numpy()
    geometry = xb[:, 0, :, :]
    eigen = yb[:, 0, :, :]
    disp = yb[:, 1:5, :, :]

    row: dict[str, object] = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "pt_dir": str(pt_dir),
        "n_total": n_total,
        "n_sample_images": int(k),
    }
    row.update(stats_from_array(geometry, "geometry"))
    row.update(stats_from_array(eigen, "eigen"))
    row.update(stats_from_array(disp, "disp_all"))
    for i in range(4):
        row.update(stats_from_array(disp[:, i, :, :], f"disp_ch{i}"))

    take = int(max(1000, args.raw_values_per_key))
    npz_payload = {
        "eigen_raw": sample_flat_values(rng, eigen, take),
        "disp_all_raw": sample_flat_values(rng, disp, take),
    }
    for i in range(4):
        npz_payload[f"disp_ch{i}_raw"] = sample_flat_values(rng, disp[:, i, :, :], take)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(row, indent=2), encoding="utf-8")

    out_npz = Path(args.out_sample_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **npz_payload)

    # Exit immediately to avoid unstable interpreter shutdown path in this environment.
    os._exit(0)


if __name__ == "__main__":
    main()
