"""H1/H4: IBZ geometry and training-set wave exposure (read-only)."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import torch

from common import CACHE, DATASETS, classify_waves, load_kxy, save_json

ROOT = DATASETS["c_test"]["pt_dir"].parents[1]


def training_wave_exposure() -> dict:
    """Count how often each wave index appears in reduced_indices across train folds."""
    rng_expected = random.Random(20260309)
    counters: Counter[int] = Counter()
    n_folds = 0
    n_pick = 65  # 325 // 5
    for prefix in ("c_train", "b_train"):
        for pt_dir in sorted(ROOT.glob(f"{prefix}_*/*_pt")):
            red = pt_dir / "reduced_indices.pt"
            if not red.exists():
                continue
            idx = torch.load(red, map_location="cpu", weights_only=False)
            waves = [int(t[1]) for t in idx]
            counters.update(waves)
            n_folds += 1

    kxy = load_kxy(DATASETS["c_test"]["pt_dir"])
    groups = classify_waves(kxy)
    n_train_geoms = 24 * 1000  # 24 batches

    def group_rate(name: str, idxs: np.ndarray) -> dict:
        counts = [counters[int(w)] for w in idxs]
        # each (geom,band) picks 65 waves; 24 folds * 1000 geoms * 6 bands per wave count
        expected_max = n_folds * 1000 * 6  # upper bound appearances if always picked
        return {
            "n_waves": len(idxs),
            "mean_count": float(np.mean(counts)),
            "min_count": int(np.min(counts)),
            "max_count": int(np.max(counts)),
            "mean_frac_of_max": float(np.mean(counts) / max(expected_max, 1)),
        }

    # Theoretical pick probability per wave per (geom,band): 65/325 = 0.2
    ky0_pick_prob = 65 / 325
    ky0_row_expected = ky0_pick_prob  # any given ky0 wave

    return {
        "n_train_folds": n_folds,
        "n_pick_per_geom_band": n_pick,
        "pick_probability": ky0_pick_prob,
        "total_wave_counts": dict(counters),
        "by_group": {k: group_rate(k, v) for k, v in groups.items()},
        "ky0_vs_interior_mean_ratio": (
            group_rate("ky0", groups["ky0"])["mean_count"]
            / max(group_rate("interior", groups["interior"])["mean_count"], 1e-9)
        ),
    }


def ibz_geometry_summary() -> dict:
    kxy = load_kxy(DATASETS["c_test"]["pt_dir"])
    groups = classify_waves(kxy)
    return {
        "n_total": int(kxy.shape[0]),
        "grid_note": "symmetry_type=none rectangle [-pi,pi] x [0,pi], 25x13",
        "group_sizes": {k: int(len(v)) for k, v in groups.items()},
        "ky0_fraction": len(groups["ky0"]) / len(kxy),
        "kx0_fraction": len(groups["kx0"]) / len(kxy),
        "ky0_is_ibz_bottom_edge": True,
        "kx0_is_mostly_interior_line": True,
        "gamma_x_waves": [int(w) for w in groups["ky0"]],
        "design_symmetry": "p4mm",
        "wavevector_symmetry": "none (half-plane ky>=0)",
    }


def main() -> None:
    out = {
        "ibz_geometry": ibz_geometry_summary(),
        "training_exposure": training_wave_exposure(),
    }
    save_json(CACHE / "h1_h4_geometry_training.json", out)
    print("Wrote", CACHE / "h1_h4_geometry_training.json")
    te = out["training_exposure"]
    print(f"Train folds: {te['n_train_folds']}, pick prob: {te['pick_probability']:.3f}")
    print(f"ky0/interior mean count ratio: {te['ky0_vs_interior_mean_ratio']:.3f}")


if __name__ == "__main__":
    main()
