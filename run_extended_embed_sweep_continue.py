"""Continue fscale sweep 36-52 after early stop at 32; then cycles at best fscale."""

from __future__ import annotations

import json
from pathlib import Path

from run_extended_embed_sweep import (
    FSCALE_ONLY_ROOT,
    SWEEP_ROOT,
    load_prior_manifest,
    run_single,
)
from run_low_fscale_experiments import FSCALE_SWEEP_CYCLES, ODD_CLOSE_CYCLE_PAIRS

FSCALE_VALUES = [36, 40, 44, 48, 52, 56]


def main() -> None:
    kx, ky = FSCALE_SWEEP_CYCLES
    results = []
    print("Additional fscale sweep (kx=9, ky=11)")
    for fs in FSCALE_VALUES:
        s = run_single(fs, kx, ky, FSCALE_ONLY_ROOT, "fscale_extended")
        results.append(s)
        print(
            f"  fscale={fs:2d}  max={s['max_offdiag']:.4f}  "
            f"avg={s['avg_similarity']:.4f}  p95={s['pairs_ge_0.95']:3d}"
        )

    best = min(results, key=lambda r: (r["max_offdiag"], r["pairs_ge_0.95"]))
    print(f"\nBest in this batch: fscale={best['freq_scale']} max={best['max_offdiag']:.4f}")

    prior = load_prior_manifest()
    all_fscale = prior.get("fscale_results", []) + results
    overall = min(all_fscale, key=lambda r: (r["max_offdiag"], r["pairs_ge_0.95"]))
    print(
        f"Overall best fscale: {overall['freq_scale']} max={overall['max_offdiag']:.4f} "
        f"avg={overall['avg_similarity']:.4f}"
    )

    # Cycles at overall best fscale
    bf = overall["freq_scale"]
    print(f"\nCycles at fscale={bf}")
    cycle_results = []
    prev_max = float("inf")
    for kx_c, ky_c in ODD_CLOSE_CYCLE_PAIRS:
        s = run_single(bf, kx_c, ky_c, SWEEP_ROOT / "cycles_at_best_fscale", "cycles_extended")
        cycle_results.append(s)
        print(
            f"  kx={kx_c:2d} ky={ky_c:2d}  max={s['max_offdiag']:.4f}  "
            f"avg={s['avg_similarity']:.4f}  ky0={s['ky0_max']:.4f}"
        )
        if s["max_offdiag"] >= prev_max - 1e-4 and len(cycle_results) > 1:
            print("  -> plateau; stopping")
            break
        prev_max = s["max_offdiag"]

    out = SWEEP_ROOT / "extended_sweep_manifest.json"
    ext = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            ext = json.load(f)
    ext["additional_fscale"] = results
    ext["overall_best_fscale"] = overall
    ext["cycles_at_overall_best"] = cycle_results
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ext, f, indent=2)


if __name__ == "__main__":
    main()
