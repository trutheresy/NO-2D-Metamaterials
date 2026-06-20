"""H5/H12: Mirror-pair asymmetry on ky=0 row (read-only)."""

from __future__ import annotations

import numpy as np

from common import CACHE, load_json, load_kxy, save_json


def main() -> None:
    kxy = load_kxy(CACHE.parents[1] / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt")
    pairs = []
    for tag in ("c_test", "b_test"):
        stats = load_json(CACHE / f"wave_stats_{tag}.json")
        by_w = {r["wave"]: r for r in stats["per_wave"]}
        tag_pairs = []
        for i in range(13):  # mirror about center of 25-point row
            a, b = i, 24 - i
            ra, rb = by_w[a], by_w[b]
            tag_pairs.append(
                {
                    "w_a": a,
                    "w_b": b,
                    "kx_a": float(kxy[a, 0]),
                    "kx_b": float(kxy[b, 0]),
                    "nmae_a": ra["group_nmae"],
                    "nmae_b": rb["group_nmae"],
                    "delta_a_minus_b": float(ra["group_nmae"] - rb["group_nmae"]),
                    "ch0_a": ra["ch0"],
                    "ch0_b": rb["ch0"],
                    "disp_a": ra["disp"],
                    "disp_b": rb["disp"],
                    "second_a": ra["second_peak_pct"],
                    "second_b": rb["second_peak_pct"],
                }
            )
        pairs.append({"tag": tag, "pairs": tag_pairs})

    # Summary stats
    summary = {}
    for block in pairs:
        tag = block["tag"]
        deltas = [p["delta_a_minus_b"] for p in block["pairs"]]
        summary[tag] = {
            "mean_abs_delta_nmae": float(np.mean(np.abs(deltas))),
            "max_abs_delta_nmae": float(np.max(np.abs(deltas))),
            "mean_delta_nmae": float(np.mean(deltas)),
            "pairs_both_above_50pct": sum(
                1 for p in block["pairs"] if p["second_a"] > 50 and p["second_b"] > 50
            ),
        }

    save_json(CACHE / "h5_h12_mirror_pairs.json", {"pairs": pairs, "summary": summary})
    print("Wrote", CACHE / "h5_h12_mirror_pairs.json")
    for tag, s in summary.items():
        print(f"  {tag} mean|delta|={s['mean_abs_delta_nmae']:.4f}  both>50%={s['pairs_both_above_50pct']}")


if __name__ == "__main__":
    main()
