"""H3/H9: Channel decomposition — eigenfrequency vs displacement on ky=0 (read-only)."""

from __future__ import annotations

from common import CACHE, load_json, save_json


def main() -> None:
    out = {}
    for tag in ("c_test", "b_test"):
        stats = load_json(CACHE / f"wave_stats_{tag}.json")
        g = stats["groups"]
        ky = g["ky0"]
        kx = g["kx0_only"] if "kx0_only" in g else g["kx0"]
        interior = g["interior"]

        def ratio(ch0, disp):
            return float(disp / max(ch0, 1e-12))

        out[tag] = {
            "ky0": {
                "ch0_nmae": ky["ch0_nmae_mean"],
                "disp_nmae": ky["disp_nmae_mean"],
                "disp_over_ch0": ratio(ky["ch0_nmae_mean"], ky["disp_nmae_mean"]),
                "second_peak_pct": ky["second_peak_pct"],
            },
            "kx0_only": {
                "ch0_nmae": kx["ch0_nmae_mean"],
                "disp_nmae": kx["disp_nmae_mean"],
                "disp_over_ch0": ratio(kx["ch0_nmae_mean"], kx["disp_nmae_mean"]),
                "second_peak_pct": kx["second_peak_pct"],
            },
            "interior": {
                "ch0_nmae": interior["ch0_nmae_mean"],
                "disp_nmae": interior["disp_nmae_mean"],
                "disp_over_ch0": ratio(interior["ch0_nmae_mean"], interior["disp_nmae_mean"]),
                "second_peak_pct": interior["second_peak_pct"],
            },
            "ky0_minus_interior_disp_excess": float(
                ky["disp_nmae_mean"] - interior["disp_nmae_mean"]
            ),
            "ky0_minus_interior_ch0_excess": float(ky["ch0_nmae_mean"] - interior["ch0_nmae_mean"]),
        }

        # Per-wave: is failure driven by disp while ch0 ok?
        bad_waves = [r for r in stats["per_wave"] if r["second_peak_pct"] > 50]
        if bad_waves:
            out[tag]["waves_above_50pct_second_peak"] = {
                "n": len(bad_waves),
                "mean_ch0": float(sum(r["ch0"] for r in bad_waves) / len(bad_waves)),
                "mean_disp": float(sum(r["disp"] for r in bad_waves) / len(bad_waves)),
                "disp_over_ch0": ratio(
                    sum(r["ch0"] for r in bad_waves) / len(bad_waves),
                    sum(r["disp"] for r in bad_waves) / len(bad_waves),
                ),
            }

    save_json(CACHE / "h3_h9_channel_decomposition.json", out)
    print("Wrote", CACHE / "h3_h9_channel_decomposition.json")
    for tag, d in out.items():
        print(
            f"  {tag} ky0 disp/ch0={d['ky0']['disp_over_ch0']:.2f}  "
            f"interior={d['interior']['disp_over_ch0']:.2f}  "
            f"ky0 2nd%={d['ky0']['second_peak_pct']:.1f}"
        )


if __name__ == "__main__":
    main()
