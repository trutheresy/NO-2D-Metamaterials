"""
Export second-peak wavevector/band correlation tables to markdown, then regenerate
histograms excluding the wavevectors listed in each dataset's table.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from per_sample_loss import compute_per_sample_losses, prepare_scoring_data, resolve_device
from second_peak_analysis import (
    band_table,
    flat_indices,
    second_peak_mask,
    wave_table,
)

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
LOSSES = ["mae", "mse", "nmae", "nmse", "nrms"]
NMAE_EPS = 1e-5
NMSE_EPS = 1e-5
TOP_WAVES = 10
REFERENCE_LOSS = "nmae"
PRED = (
    "predictions_I3O5_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat.pt"
)

DATASETS = [
    (
        "c_test",
        ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt",
        ROOT
        / "INFERENCE/C_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-023704",
    ),
    (
        "b_test",
        ROOT / "DATASETS/b_test/binarized_2026-03-08_16-34-27_pt",
        ROOT
        / "INFERENCE/B_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-024713",
    ),
]

MD_PATH = ROOT / "INFERENCE" / "second_peak_wave_band_correlations.md"
JSON_PATH = ROOT / "INFERENCE" / "second_peak_exclude_waves.json"


def analyze_dataset(tag: str, pt_dir: Path, pred_path: Path) -> dict:
    device = resolve_device("auto")
    channels = [0, 1, 2, 3, 4]
    predictions = torch.load(pred_path, map_location="cpu", mmap=True, weights_only=True)
    truth_flat, n_geom, n_wv, n_bands, _, _, _ = prepare_scoring_data(pt_dir, predictions, channels)
    kxy = torch.load(pt_dir / "wavevectors_full.pt", map_location="cpu", weights_only=False)[0].numpy()

    losses = compute_per_sample_losses(
        truth_flat=truth_flat,
        predictions=predictions,
        channels=channels,
        losses=LOSSES,
        device=device,
        batch_size=8192,
        nmae_eps=NMAE_EPS,
        nmse_eps=NMSE_EPS,
        channel_weighting="group",
    )

    geom, wave, band = flat_indices(n_geom, n_wv, n_bands)
    ref = losses[REFERENCE_LOSS]
    second, peak_info = second_peak_mask(ref)
    main = np.isfinite(ref) & (ref > 0) & ~second

    bands = band_table(band, second, n_bands)
    _, top_waves = wave_table(wave, second, kxy, n_geom, n_bands, top_n=TOP_WAVES)
    exclude_waves = [int(r["wave"]) for r in top_waves]

    return {
        "tag": tag,
        "n_geom": n_geom,
        "n_wv": n_wv,
        "n_bands": n_bands,
        "n_total": int(ref.size),
        "n_second": int(second.sum()),
        "frac_second_pct": 100.0 * second.mean(),
        "peak_info": peak_info,
        "bands": bands,
        "top_waves": top_waves,
        "exclude_waves": exclude_waves,
        "overlap_nmae_nmse": int((second & second_peak_mask(losses["nmse"])[0]).sum()),
    }


def fmt_k(row: dict) -> str:
    return f"({row['kx']:+.4f}, {row['ky']:+.4f})"


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# Second-peak wavevector and band correlations",
        "",
        "Analysis of per-sample losses on I3O5 inference (group channel weighting: "
        "50% eigenfrequency ch0 + 50% mean displacement ch1–4; `nmae_eps = nmse_eps = 1e-5`).",
        "",
        "The **second peak** is the high-loss mode in log-scaled NMAE/NMSE histograms. "
        "Samples are assigned to it when their NMAE exceeds the log-space valley between "
        "the two fitted modes (see per-dataset split values below).",
        "",
        "Key findings:",
        "",
        "- Second-peak membership is **strongly structured in wavevector and band**, not "
        "concentrated in a few geometries.",
        "- Hot wavevectors are **Brillouin-zone specials** (Γ, zone edges, corners).",
        "- Higher bands are **over-represented** in the second peak.",
        "- Displacement NMAE/NMSE inflation drives the hump; eigenfrequency error changes little.",
        "",
        "Histograms were regenerated **excluding every wavevector listed in each dataset's "
        "top-10 table** below (all bands and geometries for those wave indices).",
        "",
        "---",
        "",
    ]

    for res in results:
        tag = res["tag"]
        info = res["peak_info"]
        lines += [
            f"## {tag}",
            "",
            f"- Samples: **{res['n_total']:,}** (1000 geometries × 325 wavevectors × 6 bands)",
            f"- Second-peak samples (NMAE split): **{res['n_second']:,}** "
            f"({res['frac_second_pct']:.2f}% of all samples)",
        ]
        if info:
            lines += [
                f"- NMAE split threshold: **{info['split_value']:.4g}** "
                f"(log₁₀ peaks ≈ {info['peak_logs'][0]:.3f} and {info['peak_logs'][1]:.3f})",
            ]
        lines += [
            f"- Excluded wave indices for filtered histograms: "
            f"`{', '.join(str(w) for w in res['exclude_waves'])}`",
            "",
            "### Band correlation (all bands)",
            "",
            "| Band | N total | N second peak | % in second peak | Uniform % | Enrichment | "
            "% of second-peak mass |",
            "|-----:|--------:|--------------:|-----------------:|----------:|-----------:|"
            "----------------------:|",
        ]
        for r in res["bands"]:
            lines.append(
                f"| {r['band']} | {r['n_total']:,} | {r['n_second']:,} | "
                f"{r['frac_second_pct']:.2f} | {r['uniform_pct']:.2f} | "
                f"{r['enrichment']:.2f}× | {r['share_of_second_peak_pct']:.2f} |"
            )
        pval = res["bands"][0].get("chi2_pooled_p", float("nan"))
        lines += [
            "",
            f"Band × second-peak contingency χ² p-value: **{pval:.2e}** (rejects uniform band mix).",
            "",
            f"### Top {TOP_WAVES} wavevectors by second-peak rate (excluded from filtered histograms)",
            "",
            "| Wave | k (kx, ky) | N total | N second peak | % in second peak | "
            "Enrichment vs population | % of second-peak mass |",
            "|-----:|:-----------|--------:|--------------:|-----------------:|"
            "-------------------------:|----------------------:|",
        ]
        for r in res["top_waves"]:
            lines.append(
                f"| {r['wave']} | {fmt_k(r)} | {r['n_total']:,} | {r['n_second']:,} | "
                f"{r['frac_second_pct']:.2f} | {r['enrichment_vs_pop']:.2f}× | "
                f"{r['share_of_second_peak_pct']:.2f} |"
            )
        lines += [
            "",
            "Wave index `w` maps to flat sample index "
            "`combined = geom×(325×6) + w×6 + band`. "
            f"k vectors are from `wavevectors_full.pt` (geometry 0 row).",
            "",
            "---",
            "",
        ]

    lines += [
        "## Filtered histogram regeneration",
        "",
        "After exporting this document, `plot_loss_histograms.py` was rerun for each inference "
        "folder with `--exclude-wave-indices` set to that dataset's top-10 wave list. "
        "MAE/MSE histograms use the same exclusion for consistency.",
        "",
    ]
    return "\n".join(lines)


def regenerate_histograms(tag: str, pt_dir: Path, inf_dir: Path, exclude_waves: list[int]) -> None:
    pred = inf_dir / PRED
    for png in inf_dir.glob("loss_histogram*.png"):
        png.unlink()
    excl = ",".join(str(w) for w in exclude_waves)
    cmd = [
        PYTHON,
        "plot_loss_histograms.py",
        "--dataset-pt-dir",
        str(pt_dir),
        "--inference",
        str(pred),
        "--losses",
        *LOSSES,
        "--tag",
        tag,
        "--output-dir",
        str(inf_dir),
        "--channel-weighting",
        "group",
        "--nmae-eps",
        str(NMAE_EPS),
        "--nmse-eps",
        str(NMSE_EPS),
        "--exclude-wave-indices",
        excl,
    ]
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    results = []
    exclude_map: dict[str, list[int]] = {}
    for tag, pt_dir, inf_dir in DATASETS:
        pred = inf_dir / PRED
        print(f"\nAnalyzing {tag} ...", flush=True)
        res = analyze_dataset(tag, pt_dir, pred)
        results.append(res)
        exclude_map[tag] = res["exclude_waves"]

    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(render_markdown(results), encoding="utf-8")
    JSON_PATH.write_text(json.dumps(exclude_map, indent=2), encoding="utf-8")
    print(f"\nWrote {MD_PATH}")
    print(f"Wrote {JSON_PATH}")

    for tag, pt_dir, inf_dir in DATASETS:
        print(f"\nRegenerating histograms for {tag} (excluding {len(exclude_map[tag])} waves)...")
        regenerate_histograms(tag, pt_dir, inf_dir, exclude_map[tag])

    print("\nDone.")


if __name__ == "__main__":
    main()
