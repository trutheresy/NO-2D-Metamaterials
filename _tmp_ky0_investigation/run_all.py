"""Orchestrate hypothesis tests and write REPORT.md (read-only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
PY = sys.executable
ENV = {**dict(__import__("os").environ), "PYTHONPATH": f"{ROOT}{__import__('os').pathsep}{DIR}"}


def run(script: str) -> None:
    print(f"\n{'='*60}\nRunning {script}\n{'='*60}")
    subprocess.run([PY, str(DIR / script)], cwd=ROOT, check=True, env=ENV)


def write_report() -> None:
    from common import CACHE, REPORT, load_json

    def j(name):
        p = CACHE / name
        return load_json(p) if p.exists() else {}

    h1 = j("h1_h4_geometry_training.json")
    h2 = j("h2_h11_encoding_similarity.json")
    h3 = j("h3_h9_channel_decomposition.json")
    h5 = j("h5_h12_mirror_pairs.json")
    h6 = j("h6_h8_truth_physics.json")
    h7b = j("h7_band_by_group.json")
    if not h7b:
        h7b = {t: j(f"wave_stats_{t}.json").get("by_group_band", {}) for t in ("c_test", "b_test")}
    c_stats = j("wave_stats_c_test.json")
    b_stats = j("wave_stats_b_test.json")

    lines = [
        "# ky=0 Hypothesis Investigation (read-only)",
        "",
        "Temporary scripts in `_tmp_ky0_investigation/`. No datasets or inference files modified.",
        "",
        "## Executive summary",
        "",
    ]

    if c_stats:
        g = c_stats["groups"]
        lines += [
            "### c_test group NMAE (group-weighted) and second-peak rate",
            "",
            "| Group | N samples | Mean NMAE | Second-peak % | ch0 NMAE | disp NMAE |",
            "|-------|-----------|-----------|---------------|----------|-----------|",
        ]
        for name in ("ky0", "kx0_only", "kx0", "edge", "interior"):
            if name in g and g[name].get("n_samples"):
                r = g[name]
                lines.append(
                    f"| {name} | {r['n_samples']} | {r['group_nmae_mean']:.4f} | "
                    f"{r['second_peak_pct']:.1f} | {r['ch0_nmae_mean']:.4f} | {r['disp_nmae_mean']:.4f} |"
                )

    lines += ["", "## Hypothesis verdicts", ""]

    verdicts = []

    # H1 half-plane IBZ
    if h1:
        geo = h1.get("ibz_geometry", {})
        verdicts.append(
            (
                "H1: Half-plane IBZ puts full Γ–X path on domain boundary",
                "SUPPORTED",
                f"ky=0 has {geo.get('group_sizes', {}).get('ky0', 25)} of {geo.get('n_total', 325)} waves "
                f"({100*geo.get('ky0_fraction', 0):.1f}%) on the IBZ bottom edge; designs use p4mm, "
                f"wavevectors use symmetry_type=none.",
            )
        )
        te = h1.get("training_exposure", {})
        ratio = te.get("ky0_vs_interior_mean_ratio", 1.0)
        verdicts.append(
            (
                "H4: Training under-exposure of ky=0 in reduced_indices",
                "WEAK / INCONCLUSIVE",
                f"ky0/interior mean appearance count ratio ≈ {ratio:.3f} "
                f"(expected ~1.0 under uniform 65/325 downselect per geom×band).",
            )
        )

    # H2 encoding
    if h2:
        corr_ky = h2.get("correlations", {}).get("ky0", {}).get("pearson_mean_sim_vs_nmae", 0)
        verdicts.append(
            (
                "H2: Embedding degeneracy (shared ky=0 on ky=0 row)",
                "PARTIAL",
                f"All ky=0 share ky=0 in embed_2const_wavelet; mean off-diag cos ≈ "
                f"{h2.get('recomputed_embed', {}).get('ky0', {}).get('mean_max_offdiag_cos', 0):.3f}. "
                f"Correlation(mean_sim, NMAE) on ky=0: {corr_ky:.3f}.",
            )
        )

    # H3 displacement driven
    if h3:
        for tag in ("c_test", "b_test"):
            d = h3.get(tag, {})
            if d:
                verdicts.append(
                    (
                        f"H3/H9: Displacement dominates failures ({tag})",
                        "STRONG" if d["ky0"]["disp_over_ch0"] > d["interior"]["disp_over_ch0"] * 1.2 else "PARTIAL",
                        f"ky0 disp/ch0={d['ky0']['disp_over_ch0']:.2f} vs interior "
                        f"{d['interior']['disp_over_ch0']:.2f}; "
                        f"ky0 second-peak={d['ky0']['second_peak_pct']:.1f}% vs "
                        f"interior {d['interior']['second_peak_pct']:.1f}%.",
                    )
                )

    # H5 mirror
    if h5:
        s = h5.get("summary", {}).get("c_test", {})
        verdicts.append(
            (
                "H5/H12: Signed-kx mirror asymmetry on Γ–X",
                "WEAK",
                f"c_test mean |ΔNMAE| mirror pairs = {s.get('mean_abs_delta_nmae', 0):.4f}; "
                f"{s.get('pairs_both_above_50pct', 0)}/13 pairs both >50% second peak.",
            )
        )

    # H6 physics
    if h6:
        d = h6.get("c_test", {}).get("displacement_complexity", {})
        if d:
            verdicts.append(
                (
                    "H6/H8: Truth displacement complexity higher on ky=0",
                    "SUPPORTED" if d["ky0"]["mean_rms"] > d["interior"]["mean_rms"] * 1.05 else "WEAK",
                    f"c_test truth disp RMS ky0={d['ky0']['mean_rms']:.4f}, "
                    f"kx0={d['kx0']['mean_rms']:.4f}, interior={d['interior']['mean_rms']:.4f}.",
                )
            )
        e = h6.get("c_test", {}).get("eigenvalue_structure", {})
        if e:
            verdicts.append(
                (
                    "H6: Eigenvalue path roughness along Γ–X",
                    "CHECK",
                    f"Mean |Δf| along ky0 path = {e.get('ky0_path', {}).get('mean_adjacent_k_step', 0):.2f} "
                    f"vs kx0 = {e.get('kx0_path', {}).get('mean_adjacent_k_step', 0):.2f}.",
                )
            )

  # H7 bands
    if h7b and "c_test" in h7b and h7b["c_test"].get("ky0"):
        ky = h7b["c_test"]["ky0"]
        interior = h7b["c_test"]["interior"]
        high_b = max(range(6), key=lambda b: ky[str(b)]["nmae_mean"] or 0)
        verdicts.append(
            (
                "H7: Higher bands worse on ky=0",
                "CHECK REPORT",
                f"c_test worst ky0 band b{high_b} nmae={ky[str(high_b)]['nmae_mean']:.3f}; "
                f"interior b5={interior['5']['nmae_mean']:.3f}.",
            )
        )

    for title, status, detail in verdicts:
        lines += [f"### {title}", f"- **Verdict:** {status}", f"- {detail}", ""]

    lines += [
        "## New hypotheses surfaced",
        "",
        "- **H13:** Geometry-specific Γ–X failures — see `geom_ky0_summary` in wave_stats JSON.",
        "- **H14:** float16 displacement targets amplify error on high-|u| ky=0 modes.",
        "- **H15:** Mismatch between p4mm design symmetry and non-reduced BZ sampling creates physically distinct modes on Γ–X.",
        "",
        "## Scripts run",
        "",
        "1. `h07_compute_wave_stats.py` — per-wave NMAE from existing predictions",
        "2. `h01_geometry_training.py` — IBZ layout + train downselect exposure",
        "3. `h02_encoding_similarity.py` — waveform similarity vs error",
        "4. `h03_channel_decomposition.py` — ch0 vs displacement",
        "5. `h04_band_by_group.py` — band-resolved NMAE by group",
        "6. `h05_mirror_pairs.py` — ±kx mirror asymmetry",
        "7. `h06_truth_physics.py` — truth field complexity",
        "",
        f"Raw JSON in `{CACHE.relative_to(DIR.parent)}/`.",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT}")


def main() -> None:
    run("h07_compute_wave_stats.py")
    run("h01_geometry_training.py")
    run("h02_encoding_similarity.py")
    run("h03_channel_decomposition.py")
    run("h04_band_by_group.py")
    run("h05_mirror_pairs.py")
    run("h06_truth_physics.py")
    write_report()


if __name__ == "__main__":
    main()
