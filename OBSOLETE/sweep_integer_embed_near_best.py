"""Integer grid (+ 2-decimal freq_offset) near exploratory lhs_15 best params."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from inspect_wavelet_embeddings_ibz import load_wavevectors
from sweep_wavelet_embed_params import OUT, score_params

ROOT = Path(__file__).resolve().parent
PT_DIR = ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt"
MANIFEST = OUT / "integer_near_lhs15_manifest.json"

# lhs_15 reference: fscale≈42.57, offset≈2.20, sigma≈0.50, kx=11, ky=23
FREQ_SCALES = [40, 41, 42, 43, 44, 45, 46]
FREQ_OFFSETS = [2.15, 2.18, 2.20, 2.22, 2.25]
SIGMA_NUMERATORS = [0.4, 0.5]
KX_CYCLES = [9, 10, 11, 12, 13]
KY_CYCLES = [21, 22, 23, 24, 25]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kxy = load_wavevectors(PT_DIR)
    results = []
    combos = list(
        itertools.product(FREQ_SCALES, FREQ_OFFSETS, SIGMA_NUMERATORS, KX_CYCLES, KY_CYCLES)
    )
    print(f"Scoring {len(combos)} integer combos near lhs_15 ...")
    for i, (fscale, offset, sigma, kx, ky) in enumerate(combos):
        params = {
            "freq_scale": fscale,
            "freq_offset": offset,
            "sigma_numerator": sigma,
            "kx_cycles": kx,
            "ky_cycles": ky,
        }
        r = score_params(kxy, 32, 1.0, params)
        r["experiment"] = (
            f"f{fscale}_o{offset:.2f}_s{sigma}_kx{kx}_ky{ky}".replace(".", "p")
        )
        results.append(r)
        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(combos)}] {r['experiment']}  "
                f"max={r['max_offdiag']:.4f}  ky0={r['ky0_max_offdiag']:.4f}"
            )

    ranked = sorted(results, key=lambda x: (x["max_offdiag"], x["ky0_max_offdiag"], x["pairs_ge_0.90"]))
    best = ranked[0]
    print("\n=== Top 10 ===")
    for r in ranked[:10]:
        print(
            f"{r['experiment']:32s}  max={r['max_offdiag']:.4f}  ky0={r['ky0_max_offdiag']:.4f}  "
            f"p90={r['pairs_ge_0.90']:3d}  fscale={r['freq_scale']} off={r['freq_offset']:.2f} "
            f"sig={r['sigma_numerator']} kx={r['kx_cycles']} ky={r['ky_cycles']}"
        )
    print("\n=== Best ===")
    print(json.dumps(best, indent=2))

    manifest = {
        "reference": "lhs_15 exploratory best",
        "grid": {
            "freq_scale": FREQ_SCALES,
            "freq_offset": FREQ_OFFSETS,
            "sigma_numerator": SIGMA_NUMERATORS,
            "kx_cycles": KX_CYCLES,
            "ky_cycles": KY_CYCLES,
        },
        "n_combos": len(combos),
        "best": best,
        "ranked_top20": ranked[:20],
        "all_results": results,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {MANIFEST}")


if __name__ == "__main__":
    main()
