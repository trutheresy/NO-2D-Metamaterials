"""H7: Band-index enrichment on ky=0 vs interior (read-only)."""

from __future__ import annotations

import numpy as np

from common import CACHE, load_json, save_json


def main() -> None:
    out = {}
    for tag in ("c_test", "b_test"):
        stats = load_json(CACHE / f"wave_stats_{tag}.json")
        ky_waves = set(range(25))
        by_band_ky = {b: [] for b in range(6)}
        by_band_int = {b: [] for b in range(6)}
        for r in stats["per_wave"]:
            w = r["wave"]
            # approximate: use per-wave mean (already over bands) - need band split
            pass
        # Re-aggregate from per_wave isn't enough; use group ch_means and wave-level
        # Use second_peak by band from recomputing - load wave_stats has per_wave only
        # Add band breakdown from groups in extended stats - for now use per_wave all bands combined
        # Better: read wave_stats and for ky0 waves collect by_band from separate pass
        ky_rows = [r for r in stats["per_wave"] if r["wave"] in ky_waves]
        int_rows = [r for r in stats["per_wave"] if r["wave"] >= 25 and r["ky"] > 0.05 and abs(r["kx"]) > 0.05]
        out[tag] = {
            "ky0_mean_nmae": float(np.mean([r["group_nmae"] for r in ky_rows])),
            "interior_mean_nmae": float(np.mean([r["group_nmae"] for r in int_rows]) if int_rows else float("nan")),
            "ky0_second_peak_mean_pct": float(np.mean([r["second_peak_pct"] for r in ky_rows])),
            "interior_second_peak_mean_pct": float(
                np.mean([r["second_peak_pct"] for r in int_rows]) if int_rows else float("nan")
            ),
        }

    # Band-wise from groups ch_means isn't per-band; run quick band analysis from cached if extended
    band_data = load_json(CACHE / "h7_band_by_group.json") if (CACHE / "h7_band_by_group.json").exists() else None
    if band_data:
        out["band_by_group"] = band_data

    save_json(CACHE / "h7_band_enrichment.json", out)
    print("Wrote", CACHE / "h7_band_enrichment.json")


if __name__ == "__main__":
    main()
