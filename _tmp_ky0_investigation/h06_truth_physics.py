"""H6/H8: Truth physics proxies along ky=0 vs kx=0 (read-only)."""

from __future__ import annotations

import numpy as np
import torch

from common import CACHE, DATASETS, classify_waves, load_kxy, save_json


def displacement_complexity(pt_dir) -> dict:
    """RMS and spatial std of truth displacement fields by wave group."""
    disp_ds = torch.load(pt_dir / "displacements_dataset.pt", map_location="cpu", weights_only=False)
    # 4 tensors (n_samples, 32, 32)
    n_ch = len(disp_ds.tensors)
    # Load indices to map flat -> wave
    eigen = torch.load(pt_dir / "eigenvalue_data_full.pt", map_location="cpu", weights_only=False)
    n_geom, n_wv, n_bands = eigen.shape
    kxy = load_kxy(pt_dir)
    groups = classify_waves(kxy)

    # sample one band (band 0) all geoms for speed -> or average all
    # Use subset: 100 geoms x all waves x band 0
    n_geom_use = min(200, n_geom)
    geoms = np.linspace(0, n_geom - 1, n_geom_use, dtype=int)

    def collect_for_waves(wave_idxs):
        rms_list = []
        std_list = []
        for w in wave_idxs:
            flat_idx = []
            for g in geoms:
                base = int(g) * n_wv * n_bands + int(w) * n_bands
                flat_idx.extend(range(base, base + n_bands))
            flat_idx = np.array(flat_idx, dtype=np.int64)
            # mean over channels and samples of RMS of field
            rms_ch = []
            std_ch = []
            for c in range(n_ch):
                fields = disp_ds.tensors[c][flat_idx].numpy().astype(np.float64)
                rms_ch.append(float(np.sqrt((fields**2).mean())))
                std_ch.append(float(fields.std()))
            rms_list.append(float(np.mean(rms_ch)))
            std_list.append(float(np.mean(std_ch)))
        return {"mean_rms": float(np.mean(rms_list)), "mean_std": float(np.mean(std_list))}

    return {g: collect_for_waves(idxs) for g, idxs in groups.items() if g in ("ky0", "kx0", "interior")}


def eigenvalue_structure(pt_dir) -> dict:
    eig = torch.load(pt_dir / "eigenvalue_data_full.pt", map_location="cpu", weights_only=False).numpy().astype(
        np.float64
    )
    n_geom, n_wv, n_bands = eig.shape
    kxy = load_kxy(pt_dir)
    groups = classify_waves(kxy)

    def path_stats(idxs):
        # along-path gradient: mean |f(w+1)-f(w)| per geom per band
        idxs = np.array(sorted(idxs), dtype=int)
        if len(idxs) < 2:
            return {}
        diffs = []
        gaps = []
        for g in range(n_geom):
            for b in range(n_bands):
                f = eig[g, idxs, b]
                diffs.append(float(np.mean(np.abs(np.diff(f)))))
                if n_bands > 1:
                    gaps.append(float(np.mean(np.diff(np.sort(f)))))
        return {
            "mean_adjacent_k_step": float(np.mean(diffs)),
            "std_across_geoms_bands": float(np.std(eig[:, idxs, :].reshape(-1))),
            "mean_frequency": float(np.mean(eig[:, idxs, :])),
        }

    return {
        "ky0_path": path_stats(groups["ky0"]),
        "kx0_path": path_stats(groups["kx0"]),
        "interior_sample": path_stats(groups["interior"][:25]),
    }


def main() -> None:
    out = {}
    for tag, cfg in DATASETS.items():
        pt = cfg["pt_dir"]
        out[tag] = {
            "displacement_complexity": displacement_complexity(pt),
            "eigenvalue_structure": eigenvalue_structure(pt),
        }

    save_json(CACHE / "h6_h8_truth_physics.json", out)
    print("Wrote", CACHE / "h6_h8_truth_physics.json")
    for tag in out:
        d = out[tag]["displacement_complexity"]
        print(
            f"  {tag} disp RMS ky0={d['ky0']['mean_rms']:.4f}  "
            f"kx0={d['kx0']['mean_rms']:.4f}  interior={d['interior']['mean_rms']:.4f}"
        )


if __name__ == "__main__":
    main()
