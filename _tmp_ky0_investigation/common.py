"""Shared helpers for ky=0 hypothesis investigation (read-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache"
REPORT = Path(__file__).resolve().parent / "REPORT.md"

PRED = (
    "predictions_I3O5_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat.pt"
)

DATASETS = {
    "c_test": {
        "pt_dir": ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt",
        "inf_dir": ROOT
        / "INFERENCE/C_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-023704",
    },
    "b_test": {
        "pt_dir": ROOT / "DATASETS/b_test/binarized_2026-03-08_16-34-27_pt",
        "inf_dir": ROOT
        / "INFERENCE/B_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-024713",
    },
}

KY0_WAVES = list(range(25))
KX0_WAVES = [12 + 25 * j for j in range(13)]
CORNERS = {0, 24, 300, 324}
NMAE_EPS = 1e-5


def load_kxy(pt_dir: Path) -> np.ndarray:
    return (
        torch.load(pt_dir / "wavevectors_full.pt", map_location="cpu", weights_only=False)[0]
        .numpy()
        .astype(np.float64)
    )


def classify_waves(kxy: np.ndarray, tol: float = 1e-3) -> dict[str, np.ndarray]:
    n = kxy.shape[0]
    ky0 = np.where(np.abs(kxy[:, 1]) < tol)[0]
    kx0 = np.where(np.abs(kxy[:, 0]) < tol)[0]
    corners = np.array(sorted(CORNERS), dtype=int)
    on_edge = np.zeros(n, dtype=bool)
    on_edge |= np.abs(kxy[:, 1]) < tol
    on_edge |= np.abs(np.abs(kxy[:, 0]) - np.pi) < tol
    on_edge |= np.abs(kxy[:, 1] - np.pi) < tol
    interior = ~on_edge
    kx0_only = np.setdiff1d(kx0, ky0)  # exclude Gamma duplicate
    return {
        "ky0": ky0,
        "kx0": kx0,
        "kx0_only": kx0_only,
        "corners": corners,
        "edge": np.where(on_edge)[0],
        "interior": np.where(interior)[0],
    }


def flat_indices(n_geom: int, n_wv: int, n_bands: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = n_geom * n_wv * n_bands
    combined = np.arange(total, dtype=np.int64)
    per_geom = n_wv * n_bands
    geom = combined // per_geom
    wave = (combined % per_geom) // n_bands
    band = combined % n_bands
    return geom, wave, band


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
