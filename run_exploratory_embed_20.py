"""
20 exploratory wavelet-embedding runs via Latin Hypercube + unlikely combos.

Outputs: PLOTS/wavelet_embedding_inspection/low_fscale_sweep/exploratory_20/<config>/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import NO_utilities as NU
from inspect_wavelet_embeddings_ibz import OUT_PARENT, output_dir_from_cfg, run_inspection

SWEEP_ROOT = OUT_PARENT / "low_fscale_sweep" / "exploratory_20"

# Continuous parameter bounds for LHS
BOUNDS = {
    "freq_scale": (6.0, 52.0),
    "freq_offset": (0.55, 2.8),
    "sigma_numerator": (0.22, 0.58),
    "kx_cycles": (5, 21),  # mapped to odd ints
    "ky_cycles": (5, 23),
}

# Hand-picked unlikely / edge-case combinations (appended after LHS)
UNLIKELY_COMBOS = [
    {
        "name": "wild_low_f_high_cycles",
        "freq_scale": 3.0,
        "freq_offset": 2.2,
        "sigma_numerator": 0.55,
        "kx_cycles": 19,
        "ky_cycles": 21,
    },
    {
        "name": "wild_high_f_tiny_cycles",
        "freq_scale": 54.0,
        "freq_offset": 0.7,
        "sigma_numerator": 0.23,
        "kx_cycles": 3,  # deliberately even/small
        "ky_cycles": 5,
    },
    {
        "name": "wild_offset_only",
        "freq_scale": 20.0,
        "freq_offset": 2.9,
        "sigma_numerator": 0.4,
        "kx_cycles": 11,
        "ky_cycles": 13,
    },
    {
        "name": "wild_wide_cycle_gap",
        "freq_scale": 34.0,
        "freq_offset": 1.01,
        "sigma_numerator": 0.35,
        "kx_cycles": 7,
        "ky_cycles": 23,
    },
]


def to_odd_int(x: float, lo: int, hi: int) -> int:
    n = int(round(x))
    if n % 2 == 0:
        n += 1
    return int(np.clip(n, lo | 1, hi if hi % 2 else hi - 1))


def lhs_samples(n: int, seed: int = 42) -> list[dict]:
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    unit = sampler.random(n)
    lo = np.array([BOUNDS[k][0] for k in ("freq_scale", "freq_offset", "sigma_numerator", "kx_cycles", "ky_cycles")])
    hi = np.array([BOUNDS[k][1] for k in ("freq_scale", "freq_offset", "sigma_numerator", "kx_cycles", "ky_cycles")])
    scaled = qmc.scale(unit, lo, hi)

    samples = []
    for i, row in enumerate(scaled):
        kx = to_odd_int(row[3], 5, 21)
        ky = to_odd_int(row[4], 5, 23)
        samples.append(
            {
                "name": f"lhs_{i:02d}",
                "freq_scale": float(row[0]),
                "freq_offset": float(row[1]),
                "sigma_numerator": float(row[2]),
                "kx_cycles": kx,
                "ky_cycles": ky,
            }
        )
    return samples


def build_experiments() -> list[dict]:
    # 16 LHS + 4 unlikely = 20
    exps = lhs_samples(16)
    exps.extend(UNLIKELY_COMBOS)
    return exps[:20]


def run_experiment(exp: dict) -> dict:
    overrides = {
        "freq_scale": exp["freq_scale"],
        "freq_offset": exp["freq_offset"],
        "sigma_numerator": exp["sigma_numerator"],
        "kx_cycles": exp["kx_cycles"],
        "ky_cycles": exp["ky_cycles"],
    }
    cfg = NU.embed_2const_wavelet_params(**overrides)
    out_dir = SWEEP_ROOT / output_dir_from_cfg(cfg).name
    if exp.get("name"):
        tag = SWEEP_ROOT / f"{exp['name']}__{output_dir_from_cfg(cfg).name}"
        out_dir = tag

    report = run_inspection(out_dir=out_dir, embed_overrides=overrides, quiet=True)
    return {
        "experiment": exp.get("name", "unnamed"),
        **overrides,
        "max_offdiag": report["global_max_offdiag_cos"],
        "avg_similarity": report["avg_similarity"],
        "mean_max_offdiag": report["global_mean_max_offdiag_cos"],
        "pairs_ge_0.95": len(report["pairs_above_threshold"]),
        "pairs_ge_0.90": int(np.sum(np.triu(np.load(out_dir / "fft_log_cosine_similarity.npy"), k=1) >= 0.90)),
        "ky0_max": report["per_ky_row"][0]["max_offdiag_cos"],
        "output_dir": str(out_dir),
    }


def main() -> None:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments()
    results: list[dict] = []

    print(f"Running {len(experiments)} exploratory combinations -> {SWEEP_ROOT}\n")
    for i, exp in enumerate(experiments):
        print(f"[{i + 1}/{len(experiments)}] {exp.get('name', '?')}")
        r = run_experiment(exp)
        results.append(r)
        print(
            f"  fscale={r['freq_scale']:.2f} off={r['freq_offset']:.3f} sig={r['sigma_numerator']:.3f} "
            f"kx={r['kx_cycles']} ky={r['ky_cycles']}  "
            f"max={r['max_offdiag']:.4f} avg={r['avg_similarity']:.4f} p95={r['pairs_ge_0.95']}"
        )

    ranked = sorted(results, key=lambda x: (x["max_offdiag"], x["pairs_ge_0.95"], x["avg_similarity"]))
    manifest = {
        "method": "16x LatinHypercube (5D) + 4 unlikely combos",
        "bounds": BOUNDS,
        "unlikely_combos": UNLIKELY_COMBOS,
        "results": results,
        "ranked_by_max_offdiag": ranked,
        "best": ranked[0],
    }
    manifest_path = SWEEP_ROOT / "exploratory_20_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("TOP 5 (lowest max off-diag similarity)")
    print("=" * 60)
    for r in ranked[:5]:
        print(
            f"  {r['experiment']:24s} max={r['max_offdiag']:.4f} avg={r['avg_similarity']:.4f}  "
            f"fscale={r['freq_scale']:.1f} kx={r['kx_cycles']} ky={r['ky_cycles']}"
        )
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
