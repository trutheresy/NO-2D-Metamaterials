"""Fast sweep of wavelet embedding params; scores FFT-log cosine similarity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import NO_utilities as NU
from inspect_wavelet_embeddings_ibz import (
    N_KX,
    compute_fft_log_magnitude,
    compute_fft_magnitude,
    fft_log_cosine_similarity_matrix,
    load_wavevectors,
)

ROOT = Path(__file__).resolve().parent
PT_DIR = ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt"
OUT = ROOT / "PLOTS/wavelet_embed_param_sweep"


def embed_with_params(kxy: np.ndarray, size: int, freq_range: float, params: dict) -> np.ndarray:
    emb = NU.embed_2const_wavelet(
        kxy[:, 0],
        kxy[:, 1],
        size=size,
        freq_range=freq_range,
        verbose=False,
        freq_scale=params.get("freq_scale"),
        freq_offset=params.get("freq_offset"),
        sigma_numerator=params.get("sigma_numerator"),
        kx_cycles=params.get("kx_cycles"),
        ky_cycles=params.get("ky_cycles"),
    )
    return emb.astype(np.float32)


def score_params(kxy: np.ndarray, size: int, freq_range: float, params: dict) -> dict:
    emb = embed_with_params(kxy, size, freq_range, params)
    fft_mag = compute_fft_magnitude(emb)
    fft_log = compute_fft_log_magnitude(fft_mag)
    sim = fft_log_cosine_similarity_matrix(fft_log)

    off = sim.copy()
    np.fill_diagonal(off, -np.inf)
    max_off = float(np.max(off))

    ky0_idxs = list(range(N_KX))
    sub = sim[np.ix_(ky0_idxs, ky0_idxs)]
    sub_off = sub.copy()
    np.fill_diagonal(sub_off, -np.inf)
    ky0_max = float(np.max(sub_off))

    pairs_99 = int(np.sum(np.triu(sim, k=1) >= 0.99))
    pairs_95 = int(np.sum(np.triu(sim, k=1) >= 0.95))
    pairs_90 = int(np.sum(np.triu(sim, k=1) >= 0.90))

    cfg = NU.embed_2const_wavelet_params(
        size=size,
        freq_range=freq_range,
        **{k: v for k, v in params.items() if v is not None},
    )

    return {
        **cfg,
        "max_offdiag": max_off,
        "mean_max_offdiag": float(np.max(off, axis=1).mean()),
        "ky0_max_offdiag": ky0_max,
        "pairs_ge_0.99": pairs_99,
        "pairs_ge_0.95": pairs_95,
        "pairs_ge_0.90": pairs_90,
    }


def run_experiments(experiments: list[dict], label: str) -> list[dict]:
    kxy = load_wavevectors(PT_DIR)
    results = []
    print(f"\n=== {label} ===")
    for i, exp in enumerate(experiments):
        size = exp.get("size", 32)
        freq_range = exp.get("freq_range", 1.0)
        params = {k: exp[k] for k in ("freq_scale", "freq_offset", "sigma_numerator", "kx_cycles", "ky_cycles") if k in exp}
        r = score_params(kxy, size, freq_range, params)
        r["experiment"] = exp.get("name", f"exp_{i}")
        r["note"] = exp.get("note", "")
        results.append(r)
        print(
            f"{r['experiment']:28s}  max={r['max_offdiag']:.4f}  ky0={r['ky0_max_offdiag']:.4f}  "
            f"p95={r['pairs_ge_0.95']:3d}  p90={r['pairs_ge_0.90']:4d}  "
            f"fscale={r['freq_scale']} off={r['freq_offset']} sig={r['sigma_numerator']} "
            f"kx={r['kx_cycles']} ky={r['ky_cycles']}"
        )
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    # Round 0: baseline
    all_results += run_experiments(
        [{"name": "baseline", "note": "current production defaults"}],
        "Round 0 — baseline",
    )

    # Round 1: vary freq_scale (biggest lever for spectral peak separation)
    all_results += run_experiments(
        [
            {"name": "fscale_20", "freq_scale": 20},
            {"name": "fscale_24", "freq_scale": 24},
            {"name": "fscale_28", "freq_scale": 28},
            {"name": "fscale_32", "freq_scale": 32},
        ],
        "Round 1 — increase freq_scale",
    )

    # Round 2: vary cycles (reduce modular aliasing in rotation)
    all_results += run_experiments(
        [
            {"name": "cycles_8_11", "kx_cycles": 8, "ky_cycles": 11},
            {"name": "cycles_9_13", "kx_cycles": 9, "ky_cycles": 13},
            {"name": "cycles_11_17", "kx_cycles": 11, "ky_cycles": 17},
            {"name": "cycles_13_19", "kx_cycles": 13, "ky_cycles": 19},
        ],
        "Round 2 — larger coprime kx/ky cycles",
    )

    # Round 3: vary sigma (envelope width)
    all_results += run_experiments(
        [
            {"name": "sigma_0.25", "sigma_numerator": 0.25},
            {"name": "sigma_0.30", "sigma_numerator": 0.30},
            {"name": "sigma_0.50", "sigma_numerator": 0.50},
            {"name": "sigma_0.60", "sigma_numerator": 0.60},
        ],
        "Round 3 — sigma_numerator",
    )

    # Round 4: vary freq_offset
    all_results += run_experiments(
        [
            {"name": "offset_0.5", "freq_offset": 0.5},
            {"name": "offset_1.5", "freq_offset": 1.5},
            {"name": "offset_2.0", "freq_offset": 2.0},
            {"name": "offset_3.0", "freq_offset": 3.0},
        ],
        "Round 4 — freq_offset",
    )

    # Round 5: combine best directions from rounds 1-4
    all_results += run_experiments(
        [
            {"name": "combo_a", "freq_scale": 24, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.30},
            {"name": "combo_b", "freq_scale": 28, "kx_cycles": 11, "ky_cycles": 17, "sigma_numerator": 0.25},
            {"name": "combo_c", "freq_scale": 32, "kx_cycles": 13, "ky_cycles": 19, "sigma_numerator": 0.30},
            {"name": "combo_d", "freq_scale": 28, "freq_offset": 1.5, "kx_cycles": 11, "ky_cycles": 17, "sigma_numerator": 0.25},
            {"name": "combo_e", "freq_scale": 24, "freq_offset": 2.0, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.30},
        ],
        "Round 5 — combinations",
    )

    # Round 6: refine around best combos
    all_results += run_experiments(
        [
            {"name": "refine_1", "freq_scale": 30, "freq_offset": 1.5, "kx_cycles": 11, "ky_cycles": 17, "sigma_numerator": 0.25},
            {"name": "refine_2", "freq_scale": 30, "freq_offset": 2.0, "kx_cycles": 11, "ky_cycles": 17, "sigma_numerator": 0.25},
            {"name": "refine_3", "freq_scale": 32, "freq_offset": 1.5, "kx_cycles": 13, "ky_cycles": 19, "sigma_numerator": 0.25},
            {"name": "refine_4", "freq_scale": 28, "freq_offset": 2.0, "kx_cycles": 11, "ky_cycles": 17, "sigma_numerator": 0.20},
            {"name": "refine_5", "freq_scale": 26, "freq_offset": 1.5, "kx_cycles": 10, "ky_cycles": 16, "sigma_numerator": 0.25},
        ],
        "Round 6 — refinement",
    )

    ranked = sorted(all_results, key=lambda r: (r["pairs_ge_0.95"], r["max_offdiag"], r["pairs_ge_0.90"]), reverse=False)
    best = ranked[0]

    with open(OUT / "sweep_results.json", "w", encoding="utf-8") as f:
        json.dump({"all": all_results, "best": best, "ranked_top10": ranked[:10]}, f, indent=2)

    print("\n=== BEST (lowest collisions) ===")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
