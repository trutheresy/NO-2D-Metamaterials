"""Backfill avg_similarity into sweep_manifest.json from saved similarity matrices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inspect_wavelet_embeddings_ibz import mean_offdiag_similarity

MANIFEST = Path(__file__).resolve().parent / (
    "PLOTS/wavelet_embedding_inspection/low_fscale_sweep/sweep_manifest.json"
)


def backfill_entry(entry: dict) -> dict:
    out_dir = Path(entry["output_dir"])
    sim_path = out_dir / "fft_log_cosine_similarity.npy"
    if sim_path.exists():
        sim = np.load(sim_path)
        entry["avg_similarity"] = mean_offdiag_similarity(sim)
    return entry


def main() -> None:
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    for key in ("fscale_results", "cycles_results", "best_fscale", "best_cycles"):
        val = manifest[key]
        if isinstance(val, list):
            manifest[key] = [backfill_entry(e) for e in val]
        else:
            manifest[key] = backfill_entry(val)

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Updated {MANIFEST}")
    for e in manifest["fscale_results"]:
        print(f"  fscale={e['freq_scale']:2d}  avg={e['avg_similarity']:.4f}")
    print("---")
    for e in manifest["cycles_results"]:
        print(f"  kx={e['kx_cycles']:2d} ky={e['ky_cycles']:2d}  avg={e['avg_similarity']:.4f}")


if __name__ == "__main__":
    main()
