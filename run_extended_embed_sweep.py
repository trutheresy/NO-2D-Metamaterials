"""
Continue freq_scale and cycles sweeps until performance plateaus.

Extends low_fscale_sweep from fscale=28 upward (+4 steps), then extends odd
close cycle pairs at the best fscale found. Skips runs whose output folder
already contains distinctiveness_report.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import NO_utilities as NU
from inspect_wavelet_embeddings_ibz import OUT_PARENT, output_dir_from_cfg, run_inspection

from run_low_fscale_experiments import FSCALE_SWEEP_CYCLES, ODD_CLOSE_CYCLE_PAIRS

SWEEP_ROOT = OUT_PARENT / "low_fscale_sweep"
FSCALE_ONLY_ROOT = SWEEP_ROOT / "fscale_only"
CYCLES_SWEEP_ROOT = SWEEP_ROOT / "cycles_at_best_fscale"

FSCALE_STEP = 4
FSCALE_CONTINUE_FROM = 28
FSCALE_CAP = 72

# Continue odd close pairs beyond the initial sweep
EXTENDED_CYCLE_PAIRS = [
    (17, 19),
    (19, 21),
    (21, 23),
    (23, 25),
    (25, 27),
    (27, 29),
]


def load_prior_manifest() -> dict:
    path = SWEEP_ROOT / "sweep_manifest.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def summary_from_report(
    report: dict,
    *,
    phase: str,
    freq_scale: float,
    kx_cycles: int,
    ky_cycles: int,
    out_dir: Path,
) -> dict:
    return {
        "phase": phase,
        "freq_scale": freq_scale,
        "kx_cycles": kx_cycles,
        "ky_cycles": ky_cycles,
        "max_offdiag": report["global_max_offdiag_cos"],
        "avg_similarity": report["avg_similarity"],
        "mean_max_offdiag": report["global_mean_max_offdiag_cos"],
        "pairs_ge_0.95": len(report["pairs_above_threshold"]),
        "ky0_max": report["per_ky_row"][0]["max_offdiag_cos"],
        "output_dir": str(out_dir),
    }


def load_existing_summary(out_dir: Path, **meta) -> dict | None:
    report_path = out_dir / "distinctiveness_report.json"
    if not report_path.exists():
        return None
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    if "avg_similarity" not in report:
        import numpy as np
        from inspect_wavelet_embeddings_ibz import mean_offdiag_similarity

        sim = np.load(out_dir / "fft_log_cosine_similarity.npy")
        report["avg_similarity"] = mean_offdiag_similarity(sim)
    return summary_from_report(report, out_dir=out_dir, **meta)


def run_single(
    freq_scale: float,
    kx_cycles: int,
    ky_cycles: int,
    out_parent: Path,
    phase: str,
) -> dict:
    overrides = {
        "freq_scale": float(freq_scale),
        "kx_cycles": kx_cycles,
        "ky_cycles": ky_cycles,
    }
    cfg = NU.embed_2const_wavelet_params(**overrides)
    out_dir = out_parent / output_dir_from_cfg(cfg).name
    meta = {
        "phase": phase,
        "freq_scale": freq_scale,
        "kx_cycles": kx_cycles,
        "ky_cycles": ky_cycles,
    }
    existing = load_existing_summary(out_dir, **meta)
    if existing is not None:
        print(f"  (cached) {out_dir.name}")
        return existing
    report = run_inspection(out_dir=out_dir, embed_overrides=overrides, quiet=True)
    return summary_from_report(report, out_dir=out_dir, **meta)


def run_extended_fscale(prior_fscale_results: list[dict]) -> tuple[list[dict], dict]:
    print("\n" + "=" * 60)
    print(f"EXTENDED fscale sweep: +{FSCALE_STEP} from {FSCALE_CONTINUE_FROM} (fixed kx=9, ky=11)")
    print("=" * 60)
    kx_fix, ky_fix = FSCALE_SWEEP_CYCLES

    all_results = list(prior_fscale_results)
    best = min(all_results, key=lambda r: (r["max_offdiag"], r["pairs_ge_0.95"]))
    best_max = best["max_offdiag"]
    prev_step_max = best_max
    new_results: list[dict] = []

    fscale = FSCALE_CONTINUE_FROM
    while fscale <= FSCALE_CAP:
        s = run_single(fscale, kx_fix, ky_fix, FSCALE_ONLY_ROOT, "fscale_extended")
        new_results.append(s)
        all_results.append(s)
        print(
            f"  fscale={fscale:2d}  max={s['max_offdiag']:.4f}  "
            f"avg={s['avg_similarity']:.4f}  p95={s['pairs_ge_0.95']:3d}"
        )

        if s["max_offdiag"] < best_max - 1e-4:
            best_max = s["max_offdiag"]
            best = s

        if fscale > FSCALE_CONTINUE_FROM and s["max_offdiag"] >= prev_step_max - 1e-4:
            print(f"  -> no improvement over previous step ({prev_step_max:.4f}); stopping fscale sweep")
            break

        prev_step_max = s["max_offdiag"]
        fscale += FSCALE_STEP

    print(f"\nBest fscale overall: {best['freq_scale']} (max off-diag {best['max_offdiag']:.4f})")
    return new_results, best


def all_cycle_pairs() -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for pair in list(ODD_CLOSE_CYCLE_PAIRS) + EXTENDED_CYCLE_PAIRS:
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def run_extended_cycles(
    best_fscale: float,
    prior_cycles: list[dict],
) -> tuple[list[dict], dict]:
    print("\n" + "=" * 60)
    print(f"EXTENDED cycles sweep at freq_scale={best_fscale}")
    print("=" * 60)

    CYCLES_SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    pairs = all_cycle_pairs()
    # Prior results at other fscales don't count toward stop logic
    at_fscale = [r for r in prior_cycles if r["freq_scale"] == best_fscale]
    new_results: list[dict] = []
    best: dict | None = min(at_fscale, key=lambda r: (r["max_offdiag"], r["pairs_ge_0.95"])) if at_fscale else None
    best_max = best["max_offdiag"] if best else float("inf")
    prev_step_max = best_max

    for kx_c, ky_c in pairs:
        s = run_single(best_fscale, kx_c, ky_c, CYCLES_SWEEP_ROOT, "cycles_extended")
        new_results.append(s)
        print(
            f"  kx={kx_c:2d} ky={ky_c:2d}  max={s['max_offdiag']:.4f}  "
            f"avg={s['avg_similarity']:.4f}  ky0={s['ky0_max']:.4f}  p95={s['pairs_ge_0.95']:3d}"
        )

        if s["max_offdiag"] < best_max - 1e-4:
            best_max = s["max_offdiag"]
            best = s

        # Stop only when extending into larger odd pairs beyond initial sweep
        if (kx_c, ky_c) in EXTENDED_CYCLE_PAIRS and len(new_results) > 1:
            if s["max_offdiag"] >= prev_step_max - 1e-4:
                print(f"  -> no improvement over previous step ({prev_step_max:.4f}); stopping cycles sweep")
                break

        prev_step_max = s["max_offdiag"]

    if best is None:
        best = new_results[0]
    print(
        f"\nBest cycles at fscale={best_fscale}: kx={best['kx_cycles']}, ky={best['ky_cycles']} "
        f"(max off-diag {best['max_offdiag']:.4f})"
    )
    return new_results, best


def main() -> None:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    prior = load_prior_manifest()
    prior_fscale = prior.get("fscale_results", [])
    prior_cycles = prior.get("cycles_results", [])

    new_fscale, best_fscale = run_extended_fscale(prior_fscale)

    # Merge all fscale results for overall best
    all_fscale = prior_fscale + [r for r in new_fscale if r not in prior_fscale]
    overall_best_fscale = min(all_fscale, key=lambda r: (r["max_offdiag"], r["pairs_ge_0.95"]))

    # Run extended cycles at overall best fscale (may differ from phase-1 best if higher fscale wins)
    new_cycles, best_cycles = run_extended_cycles(overall_best_fscale["freq_scale"], prior_cycles)

    extended_manifest = {
        "extended_fscale_step": FSCALE_STEP,
        "extended_fscale_from": FSCALE_CONTINUE_FROM,
        "new_fscale_results": new_fscale,
        "overall_best_fscale": overall_best_fscale,
        "extended_cycle_pairs": EXTENDED_CYCLE_PAIRS,
        "new_cycles_results": new_cycles,
        "overall_best_cycles": best_cycles,
        "all_fscale_results": all_fscale,
        "all_cycles_at_best_fscale": prior_cycles + new_cycles,
    }
    manifest_path = SWEEP_ROOT / "extended_sweep_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(extended_manifest, f, indent=2)

    # Update main manifest
    merged = {**prior}
    merged["fscale_results"] = all_fscale
    merged["best_fscale"] = overall_best_fscale
    merged["cycles_results_extended"] = prior_cycles + new_cycles
    merged["best_cycles_overall"] = best_cycles
    with open(SWEEP_ROOT / "sweep_manifest.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Extended manifest: {manifest_path}")
    print(
        f"Best fscale: {overall_best_fscale['freq_scale']}  "
        f"max={overall_best_fscale['max_offdiag']:.4f}  "
        f"avg={overall_best_fscale['avg_similarity']:.4f}"
    )
    print(
        f"Best cycles: kx={best_cycles['kx_cycles']} ky={best_cycles['ky_cycles']}  "
        f"max={best_cycles['max_offdiag']:.4f}  "
        f"avg={best_cycles['avg_similarity']:.4f}"
    )


if __name__ == "__main__":
    main()
