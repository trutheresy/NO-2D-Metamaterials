"""
Low freq_scale sweep + odd coprime (kx, ky) cycle sweep at best fscale.

Each experiment writes a full inspection subfolder under:
  PLOTS/wavelet_embedding_inspection/low_fscale_sweep/
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_wavelet_embeddings_ibz import OUT_PARENT, mean_offdiag_similarity, output_dir_from_cfg, run_inspection
import NO_utilities as NU

SWEEP_ROOT = OUT_PARENT / "low_fscale_sweep"
CYCLES_SWEEP_ROOT = SWEEP_ROOT / "cycles_at_best_fscale"

# Fixed odd close cycles during freq_scale-only sweep
FSCALE_SWEEP_CYCLES = (9, 11)

# Odd pairs with |kx - ky| <= 2
ODD_CLOSE_CYCLE_PAIRS = [
    (5, 7),
    (7, 9),
    (7, 11),
    (9, 11),
    (9, 13),
    (11, 13),
    (11, 15),
    (13, 15),
    (13, 17),
    (15, 17),
]

FSCALE_VALUES = [1, 2, 4, 8, 12, 24]


def rank_key(report: dict) -> tuple:
    n95 = len(report.get("pairs_above_threshold", []))
    return (
        report["global_max_offdiag_cos"],
        n95,
        report["global_mean_max_offdiag_cos"],
    )


def run_fscale_sweep() -> tuple[list[dict], dict]:
    print("\n" + "=" * 60)
    print("PHASE 1: freq_scale sweep (fixed kx=9, ky=11)")
    print("=" * 60)
    results: list[dict] = []
    kx_fix, ky_fix = FSCALE_SWEEP_CYCLES

    for fscale in FSCALE_VALUES:
        overrides = {"freq_scale": float(fscale), "kx_cycles": kx_fix, "ky_cycles": ky_fix}
        cfg = NU.embed_2const_wavelet_params(freq_scale=fscale, kx_cycles=kx_fix, ky_cycles=ky_fix)
        out_dir = SWEEP_ROOT / "fscale_only" / output_dir_from_cfg(cfg).name
        report = run_inspection(out_dir=out_dir, embed_overrides=overrides, quiet=True)
        summary = {
            "phase": "fscale_sweep",
            "freq_scale": fscale,
            "kx_cycles": kx_fix,
            "ky_cycles": ky_fix,
            "max_offdiag": report["global_max_offdiag_cos"],
            "avg_similarity": report["avg_similarity"],
            "mean_max_offdiag": report["global_mean_max_offdiag_cos"],
            "pairs_ge_0.95": len(report["pairs_above_threshold"]),
            "output_dir": str(out_dir),
        }
        results.append(summary)
        print(
            f"  fscale={fscale:2d}  max={summary['max_offdiag']:.4f}  "
            f"avg={summary['avg_similarity']:.4f}  "
            f"p95={summary['pairs_ge_0.95']:3d}  -> {out_dir.name}"
        )

    best = min(results, key=lambda r: (r["max_offdiag"], r["pairs_ge_0.95"]))
    print(f"\nBest freq_scale from phase 1: {best['freq_scale']} (max off-diag {best['max_offdiag']:.4f})")
    return results, best


def run_cycles_sweep(best_fscale: float) -> list[dict]:
    print("\n" + "=" * 60)
    print(f"PHASE 2: kx/ky cycles sweep at freq_scale={best_fscale}")
    print("=" * 60)
    results: list[dict] = []

    for kx_c, ky_c in ODD_CLOSE_CYCLE_PAIRS:
        overrides = {
            "freq_scale": float(best_fscale),
            "kx_cycles": kx_c,
            "ky_cycles": ky_c,
        }
        cfg = NU.embed_2const_wavelet_params(freq_scale=best_fscale, kx_cycles=kx_c, ky_cycles=ky_c)
        out_dir = CYCLES_SWEEP_ROOT / output_dir_from_cfg(cfg).name
        report = run_inspection(out_dir=out_dir, embed_overrides=overrides, quiet=True)
        summary = {
            "phase": "cycles_sweep",
            "freq_scale": best_fscale,
            "kx_cycles": kx_c,
            "ky_cycles": ky_c,
            "max_offdiag": report["global_max_offdiag_cos"],
            "avg_similarity": report["avg_similarity"],
            "mean_max_offdiag": report["global_mean_max_offdiag_cos"],
            "pairs_ge_0.95": len(report["pairs_above_threshold"]),
            "ky0_max": report["per_ky_row"][0]["max_offdiag_cos"],
            "output_dir": str(out_dir),
        }
        results.append(summary)
        print(
            f"  kx={kx_c:2d} ky={ky_c:2d}  max={summary['max_offdiag']:.4f}  "
            f"avg={summary['avg_similarity']:.4f}  "
            f"ky0={summary['ky0_max']:.4f}  p95={summary['pairs_ge_0.95']:3d}  -> {out_dir.name}"
        )

    best = min(results, key=lambda r: (r["max_offdiag"], r["pairs_ge_0.95"]))
    print(
        f"\nBest cycles from phase 2: kx={best['kx_cycles']}, ky={best['ky_cycles']} "
        f"(max off-diag {best['max_offdiag']:.4f})"
    )
    return results


def main() -> None:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    fscale_results, best_fscale = run_fscale_sweep()
    cycles_results = run_cycles_sweep(best_fscale["freq_scale"])

    best_cycles = min(cycles_results, key=lambda r: (r["max_offdiag"], r["pairs_ge_0.95"]))
    manifest = {
        "fscale_sweep_fixed_cycles": {"kx": FSCALE_SWEEP_CYCLES[0], "ky": FSCALE_SWEEP_CYCLES[1]},
        "fscale_values": FSCALE_VALUES,
        "fscale_results": fscale_results,
        "best_fscale": best_fscale,
        "odd_close_cycle_pairs": ODD_CLOSE_CYCLE_PAIRS,
        "cycles_results": cycles_results,
        "best_cycles": best_cycles,
    }
    manifest_path = SWEEP_ROOT / "sweep_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Manifest: {manifest_path}")
    print(f"Best fscale: {best_fscale['freq_scale']} -> {best_fscale['output_dir']}")
    print(
        f"Best cycles at that fscale: kx={best_cycles['kx_cycles']}, ky={best_cycles['ky_cycles']} "
        f"-> {best_cycles['output_dir']}"
    )


if __name__ == "__main__":
    main()
