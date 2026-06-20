"""H2/H11: Waveform embedding similarity vs inference error (read-only)."""

from __future__ import annotations

import numpy as np
import torch

import NO_utilities as NU
from common import CACHE, DATASETS, classify_waves, load_json, load_kxy, save_json


def waveform_similarity_matrix(wf: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    sub = wf[idxs].reshape(len(idxs), -1)
    sub = sub - sub.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12
    sub = sub / norms
    return sub @ sub.T


def main() -> None:
    pt_dir = DATASETS["c_test"]["pt_dir"]
    kxy = load_kxy(pt_dir)
    wf = torch.load(pt_dir / "waveforms_full.pt", map_location="cpu", weights_only=False).numpy().astype(
        np.float32
    )
    emb = NU.embed_2const_wavelet(kxy[:, 0], kxy[:, 1], size=32, verbose=False).astype(np.float32)
    groups = classify_waves(kxy)

    stats = load_json(CACHE / "wave_stats_c_test.json")
    wave_nmae = {r["wave"]: r["group_nmae"] for r in stats["per_wave"]}

    results = {"stored_waveforms": {}, "recomputed_embed": {}, "correlations": {}}

    for label, arr in [("stored_waveforms", wf), ("recomputed_embed", emb)]:
        for gname, idxs in [("ky0", groups["ky0"]), ("kx0", groups["kx0"]), ("interior", groups["interior"])]:
            sim = waveform_similarity_matrix(arr, idxs)
            np.fill_diagonal(sim, np.nan)
            row_max = np.nanmax(sim, axis=1)
            results[label][gname] = {
                "mean_max_offdiag_cos": float(np.nanmean(row_max)),
                "max_offdiag_cos": float(np.nanmax(row_max)),
            }

    # Correlation: mean embedding similarity to other waves in group vs wave mean NMAE
    for gname, idxs in [("ky0", groups["ky0"]), ("kx0", groups["kx0"])]:
        sim = waveform_similarity_matrix(emb, idxs)
        np.fill_diagonal(sim, np.nan)
        mean_sim = np.nanmean(sim, axis=1)
        losses = np.array([wave_nmae[int(w)] for w in idxs])
        r = float(np.corrcoef(mean_sim, losses)[0, 1]) if len(idxs) > 2 else float("nan")
        results["correlations"][gname] = {
            "pearson_mean_sim_vs_nmae": r,
            "all_share_ky_zero": bool(gname == "ky0" and np.allclose(kxy[idxs, 1], 0)),
            "all_share_kx_zero": bool(gname == "kx0" and np.allclose(kxy[idxs, 0], 0)),
        }

    # Nearest-neighbor confusion: for each ky0 wave, most similar other ky0 wave
    sim_ky = waveform_similarity_matrix(emb, groups["ky0"])
    np.fill_diagonal(sim_ky, -np.inf)
    nn_idx = np.argmax(sim_ky, axis=1)
    nn_sim = sim_ky[np.arange(len(nn_idx)), nn_idx]
    ky_list = groups["ky0"]
    pairs = []
    for i, j in enumerate(nn_idx):
        w_i, w_j = int(ky_list[i]), int(ky_list[j])
        pairs.append(
            {
                "w_i": w_i,
                "w_j": w_j,
                "cos": float(nn_sim[i]),
                "nmae_i": wave_nmae[w_i],
                "nmae_j": wave_nmae[w_j],
                "delta_nmae": float(wave_nmae[w_i] - wave_nmae[w_j]),
            }
        )
    results["ky0_nearest_neighbor_pairs"] = pairs

    save_json(CACHE / "h2_h11_encoding_similarity.json", results)
    print("Wrote", CACHE / "h2_h11_encoding_similarity.json")
    for g in ("ky0", "kx0"):
        print(f"  {g} corr(sim,nmae)={results['correlations'][g]['pearson_mean_sim_vs_nmae']:.3f}")


if __name__ == "__main__":
    main()
