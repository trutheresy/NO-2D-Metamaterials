"""Find all wavevectors with >50% second-peak rate; update markdown catalog."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from per_sample_loss import compute_per_sample_losses, prepare_scoring_data, resolve_device
from second_peak_analysis import flat_indices, second_peak_mask

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "INFERENCE" / "second_peak_wave_band_correlations.md"
JSON_PATH = ROOT / "INFERENCE" / "second_peak_exclude_waves.json"
PRED = (
    "predictions_I3O5_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat.pt"
)
THRESHOLD_PCT = 50.0
PI = math.pi


def kx_to_pi_frac(kx: float) -> str:
    r = kx / PI
    n = round(r * 12)
    if abs(r - n / 12) > 2e-3:
        return f"{r:.4f}π"
    if n == 0:
        return "0"
    sign = "-" if n < 0 else "+"
    n = abs(n)
    from math import gcd

    g = gcd(n, 12)
    n, d = n // g, 12 // g
    if n == 1:
        return f"{sign}π/{d}"
    if n == d:
        return f"{sign}π"
    return f"{sign}{n}π/{d}"


def ky_to_pi_frac(ky: float) -> str:
    if abs(ky) < 1e-3:
        return "0"
    return kx_to_pi_frac(ky)


def format_k(kx: float, ky: float) -> str:
    return f"({kx_to_pi_frac(kx)}, {ky_to_pi_frac(ky)})"


def classify_k(kx: float, ky: float) -> str:
    tags = []
    tol = 0.02
    if abs(kx) < tol and abs(ky) < tol:
        tags.append("Γ")
    if abs(ky) < tol:
        tags.append("ky=0 (Γ–X)")
    if abs(kx) < tol and abs(ky) > tol:
        tags.append("kx=0")
    if abs(abs(kx) - PI) < tol:
        tags.append("|kx|=π")
    if abs(abs(ky) - PI) < tol:
        tags.append("|ky|=π")
    if abs(abs(kx) - PI) < tol and abs(abs(ky) - PI) < tol:
        tags.append("corner")
    if not tags:
        tags.append("interior/other")
    return ", ".join(tags)


def waves_above_threshold(tag: str, pt_dir: Path, pred_path: Path) -> dict:
    device = resolve_device("auto")
    channels = [0, 1, 2, 3, 4]
    predictions = torch.load(pred_path, map_location="cpu", mmap=True, weights_only=True)
    truth, n_geom, n_wv, n_bands, *_ = prepare_scoring_data(pt_dir, predictions, channels)
    kxy = torch.load(pt_dir / "wavevectors_full.pt", map_location="cpu", weights_only=False)[0].numpy()

    nmae = compute_per_sample_losses(
        truth, predictions, channels, ["nmae"], device, 8192, 1e-5, 1e-5, "group",
    )["nmae"]
    _, wave, _ = flat_indices(n_geom, n_wv, n_bands)
    second, peak_info = second_peak_mask(nmae)
    pop = float(second.mean())

    rows = []
    for w in range(n_wv):
        mask = wave == w
        n = int(mask.sum())
        n_sec = int((second & mask).sum())
        pct = 100.0 * n_sec / n
        if pct <= THRESHOLD_PCT:
            continue
        kx, ky = float(kxy[w, 0]), float(kxy[w, 1])
        rows.append(
            {
                "wave": w,
                "kx": kx,
                "ky": ky,
                "k_pi": format_k(kx, ky),
                "tags": classify_k(kx, ky),
                "n_total": n,
                "n_second": n_sec,
                "frac_second_pct": pct,
                "enrichment": (pct / 100.0) / max(pop, 1e-12),
                "share_of_second_peak_pct": 100.0 * n_sec / max(second.sum(), 1),
            }
        )
    rows.sort(key=lambda r: (-r["frac_second_pct"], r["wave"]))
    return {
        "tag": tag,
        "peak_info": peak_info,
        "pop_pct": 100.0 * pop,
        "waves": rows,
        "wave_indices": [r["wave"] for r in rows],
    }


def render_wave_rows(rows: list[dict]) -> list[str]:
    out = [
        "| Wave | k (×π) | BZ tag | N second / N total | % in 2nd peak | Enrichment | % of 2nd-peak mass |",
        "|-----:|:-------|:-------|-------------------:|--------------:|-----------:|-------------------:|",
    ]
    for r in rows:
        out.append(
            f"| {r['wave']} | {r['k_pi']} | {r['tags']} | "
            f"{r['n_second']:,} / {r['n_total']:,} | {r['frac_second_pct']:.2f} | "
            f"{r['enrichment']:.2f}× | {r['share_of_second_peak_pct']:.2f} |"
        )
    return out


def update_markdown(results: list[dict]) -> None:
    lines = [
        "# Second-peak wavevector and band correlations",
        "",
        "Analysis of per-sample losses on I3O5 inference (group channel weighting: "
        "50% eigenfrequency ch0 + 50% mean displacement ch1–4; `nmae_eps = nmse_eps = 1e-5`).",
        "",
        "The **second peak** is the high-loss mode in log-scaled NMAE histograms. "
        "A sample is in the second peak when its NMAE exceeds the log-space valley between "
        "the two fitted modes.",
        "",
        "## Wavevectors with >50% second-peak membership",
        "",
        "Complete catalog of wave indices where **more than half** of all samples at that "
        "wavevector (1000 geometries × 6 bands = 6000 samples per wave) fall in the dataset's "
        "global NMAE second peak. k components are exact multiples of π/12 from the IBZ grid "
        "(`wavevectors_full.pt`, geometry 0).",
        "",
    ]

    for res in results:
        info = res["peak_info"]
        lines += [
            f"### {res['tag']}",
            "",
            f"- Global second-peak rate: **{res['pop_pct']:.2f}%** of all samples",
        ]
        if info:
            lines.append(
                f"- NMAE split threshold: **{info['split_value']:.4g}** "
                f"(log₁₀ peaks ≈ {info['peak_logs'][0]:.3f}, {info['peak_logs'][1]:.3f})"
            )
        lines.append(f"- Wave indices above 50%: **{len(res['waves'])}** — `{', '.join(str(w) for w in res['wave_indices'])}`")
        lines.append("")
        lines.extend(render_wave_rows(res["waves"]))
        lines += ["", "---", ""]

    lines += [
        "## Band correlations (reference)",
        "",
        "See prior band tables; higher bands remain enriched in the second peak.",
        "",
        "## Histogram filtering note",
        "",
        "Filtered histograms initially excluded only the first top-10 list. "
        "The tables above are the **full >50% catalog** per dataset.",
        "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    excl = {res["tag"]: res["wave_indices"] for res in results}
    JSON_PATH.write_text(json.dumps(excl, indent=2), encoding="utf-8")


def main() -> None:
    datasets = [
        ("c_test", ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt",
         ROOT / "INFERENCE/C_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-023704" / PRED),
        ("b_test", ROOT / "DATASETS/b_test/binarized_2026-03-08_16-34-27_pt",
         ROOT / "INFERENCE/B_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-024713" / PRED),
    ]
    results = [waves_above_threshold(tag, pt, pred) for tag, pt, pred in datasets]
    update_markdown(results)

    for res in results:
        print(f"\n{res['tag']}: {len(res['waves'])} waves > {THRESHOLD_PCT}%")
        for r in res["waves"]:
            print(f"  w{r['wave']:3d}  {r['k_pi']:20s}  {r['frac_second_pct']:6.2f}%  {r['tags']}")
    print(f"\nWrote {MD_PATH}")


if __name__ == "__main__":
    main()
