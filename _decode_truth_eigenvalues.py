"""Decode TRUE uniform-encoded eigenfrequency patches into scalar eigenvalues.

Applies the same patch-mean decode used for model predictions
(``s = exp(100 * mean(patch))``) to ``eigenfrequency_uniform_full.pt``, and stages
a truth folder (decoded eigenvalues + copied geometries/wavevectors) that
``plot_dispersions_true_vs_pred.py`` can consume via ``--true``.

Usage: python _decode_truth_eigenvalues.py <dataset_pt_dir> <output_dir>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def main(pt_dir: str, out_dir: str) -> None:
    pt = Path(pt_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ef = torch.load(pt / "eigenfrequency_uniform_full.pt", map_location="cpu", mmap=True, weights_only=True)
    n_geom, n_wv, n_bands = ef.shape[:3]
    print(f"encoded truth patches: {tuple(ef.shape)} {ef.dtype}")

    decoded = torch.empty((n_geom, n_wv, n_bands), dtype=torch.float32)
    for g in tqdm(range(n_geom), desc="Decoding truth", unit="geom"):
        pixel_mean = ef[g].to(torch.float32).mean(dim=(-2, -1))  # (n_wv, n_bands)
        decoded[g] = torch.exp(100.0 * pixel_mean)

    torch.save(decoded, out / "eigenvalue_data_full.pt")
    arr = decoded.numpy()
    print(f"decoded truth eigenvalues: {tuple(decoded.shape)}  "
          f"min={arr.min():.6g} median={np.median(arr):.6g} max={arr.max():.6g}")

    for name in ("geometries_full.pt", "wavevectors_full.pt"):
        shutil.copy2(pt / name, out / name)
        print(f"copied {name}")
    print(f"staged truth folder: {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
