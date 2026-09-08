import os
from pathlib import Path

import scipy.io as sio
import torch


def main():
    repo = Path(r"D:/Research/NO-2D-Metamaterials")
    manifest_path = repo / "OUTPUT" / "fixed_geometry_samples" / "fixed_geometry_manifest.mat"
    out_root = repo / "OUTPUT" / "fixed_geometry_samples" / "pt_subsets_for_plot"
    out_root.mkdir(parents=True, exist_ok=True)
    n_make = 5

    M = sio.loadmat(manifest_path, squeeze_me=False, struct_as_record=False)
    dataset_name = M["dataset_name"]
    source_pt_dir = M["source_pt_dir"]
    geom_idx = M["geometry_idx_0based"].reshape(-1)
    n_total = int(dataset_name.shape[0])
    n_use = min(n_make, n_total)

    for i in range(n_use):
        ds = str(dataset_name[i, 0][0])
        src_pt = Path(str(source_pt_dir[i, 0][0]))
        g = int(geom_idx[i])
        subset_dir = out_root / f"{i+1:02d}_{ds}_g{g:04d}"
        subset_dir.mkdir(parents=True, exist_ok=True)

        geometries = torch.load(src_pt / "geometries_full.pt", map_location="cpu")
        wavevectors = torch.load(src_pt / "wavevectors_full.pt", map_location="cpu")
        eigenvalues = torch.load(src_pt / "eigenvalue_data_full.pt", map_location="cpu")

        torch.save(geometries[g : g + 1], subset_dir / "geometries_full.pt")
        torch.save(wavevectors[g : g + 1], subset_dir / "wavevectors_full.pt")
        torch.save(eigenvalues[g : g + 1], subset_dir / "eigenvalue_data_full.pt")
        print(f"PT_SUBSET={subset_dir}")

    print(f"PT_SUBSET_ROOT={out_root}")
    print(f"PT_SUBSET_COUNT={n_use}")


if __name__ == "__main__":
    main()
