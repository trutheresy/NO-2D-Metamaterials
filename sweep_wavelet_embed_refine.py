"""Round 7+ refinement around fscale=32 winner."""

from __future__ import annotations

import json
from pathlib import Path

from sweep_wavelet_embed_params import OUT, run_experiments

def main() -> None:
    results = []
    results += run_experiments(
        [
            {"name": "fscale_34", "freq_scale": 34},
            {"name": "fscale_36", "freq_scale": 36},
            {"name": "fscale_38", "freq_scale": 38},
            {"name": "fscale_40", "freq_scale": 40},
        ],
        "Round 7 — higher freq_scale",
    )
    results += run_experiments(
        [
            {"name": "fs32_c8_11", "freq_scale": 32, "kx_cycles": 8, "ky_cycles": 11},
            {"name": "fs32_c9_13", "freq_scale": 32, "kx_cycles": 9, "ky_cycles": 13},
            {"name": "fs32_c11_17", "freq_scale": 32, "kx_cycles": 11, "ky_cycles": 17},
            {"name": "fs32_c13_19", "freq_scale": 32, "kx_cycles": 13, "ky_cycles": 19},
        ],
        "Round 8 — fscale 32 + larger cycles",
    )
    results += run_experiments(
        [
            {"name": "fs32_sig35", "freq_scale": 32, "sigma_numerator": 0.35},
            {"name": "fs32_sig45", "freq_scale": 32, "sigma_numerator": 0.45},
            {"name": "fs32_sig50", "freq_scale": 32, "sigma_numerator": 0.50},
            {"name": "fs32_off12", "freq_scale": 32, "freq_offset": 1.2},
            {"name": "fs32_off08", "freq_scale": 32, "freq_offset": 0.8},
        ],
        "Round 9 — fscale 32 + sigma/offset tweaks",
    )
    results += run_experiments(
        [
            {"name": "fs36_c9_13_s35", "freq_scale": 36, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.35},
            {"name": "fs36_c11_17_s40", "freq_scale": 36, "kx_cycles": 11, "ky_cycles": 17, "sigma_numerator": 0.40},
            {"name": "fs38_c9_13_s40", "freq_scale": 38, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.40},
            {"name": "fs40_c8_11_s40", "freq_scale": 40, "kx_cycles": 8, "ky_cycles": 11, "sigma_numerator": 0.40},
            {"name": "fs34_c9_13_s40", "freq_scale": 34, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.40},
        ],
        "Round 10 — combined finalists",
    )

    ranked = sorted(results, key=lambda r: (r["max_offdiag"], r["pairs_ge_0.90"], r["pairs_ge_0.95"]))
    best = ranked[0]
    with open(OUT / "refinement_results.json", "w", encoding="utf-8") as f:
        json.dump({"all": results, "best": best, "ranked_top10": ranked[:10]}, f, indent=2)
    print("\n=== REFINEMENT BEST ===")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
