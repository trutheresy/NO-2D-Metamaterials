import os
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
import torch


def _latest_pt_dir(dataset_dir: str) -> str:
    pt_dirs = [os.path.join(dataset_dir, p) for p in os.listdir(dataset_dir) if p.endswith("_pt")]
    if not pt_dirs:
        raise FileNotFoundError(f"No *_pt folder found in {dataset_dir}")
    pt_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pt_dirs[0]


def main():
    root = r"D:/Research/NO-2D-Metamaterials/OUTPUT"
    out_mat = os.path.join(root, "zero_output_combinations_catalog.mat")

    # Geometry indices are 0-based in the Python audit.
    flagged: Dict[str, List[int]] = {
        "c_train_01": [523],
        "c_train_04": [251],
        "c_train_06": [998],
        "c_train_11": [546],
        "c_train_12": [620],
        "c_train_14": [359, 374],
        "c_train_16": [153, 579],
        "c_train_20": [948],
        "c_train_21": [561],
        "c_train_23": [228],
    }

    dataset_name_all: List[str] = []
    pt_dir_all: List[str] = []
    design_number_all: List[int] = []
    geometry_idx_0_all: List[int] = []
    geometry_idx_1_all: List[int] = []
    wavevector_idx_0_all: List[int] = []
    wavevector_idx_1_all: List[int] = []
    band_idx_0_all: List[int] = []
    band_idx_1_all: List[int] = []
    wavevector_xy_all: List[Tuple[float, float]] = []

    rep_dataset_name: List[str] = []
    rep_pt_dir: List[str] = []
    rep_design_number: List[int] = []
    rep_geometry_idx_0: List[int] = []
    rep_geometry_idx_1: List[int] = []
    rep_wavevector_idx_0: List[int] = []
    rep_wavevector_idx_1: List[int] = []
    rep_band_idx_0: List[int] = []
    rep_band_idx_1: List[int] = []
    rep_wavevector_xy: List[Tuple[float, float]] = []

    for dataset, gidx_list in flagged.items():
        dataset_dir = os.path.join(root, dataset)
        pt_dir = _latest_pt_dir(dataset_dir)

        design_params = torch.load(os.path.join(pt_dir, "design_params_full.pt"), map_location="cpu")
        wavevectors_full = torch.load(os.path.join(pt_dir, "wavevectors_full.pt"), map_location="cpu")
        eigenvals = torch.load(os.path.join(pt_dir, "eigenvalue_data_full.pt"), map_location="cpu")

        n_wv = int(eigenvals.shape[1])
        n_band = int(eigenvals.shape[2])

        for g in gidx_list:
            design_number = int(round(float(design_params[g, 0].item())))
            wv_for_geom = wavevectors_full[g].cpu().numpy().astype(np.float64)

            # Representative combo: first wavevector, first band.
            rep_dataset_name.append(dataset)
            rep_pt_dir.append(pt_dir)
            rep_design_number.append(design_number)
            rep_geometry_idx_0.append(g)
            rep_geometry_idx_1.append(g + 1)
            rep_wavevector_idx_0.append(0)
            rep_wavevector_idx_1.append(1)
            rep_band_idx_0.append(0)
            rep_band_idx_1.append(1)
            rep_wavevector_xy.append((float(wv_for_geom[0, 0]), float(wv_for_geom[0, 1])))

            # Full combo list for this geometry (all wavevectors x all bands).
            for w in range(n_wv):
                for b in range(n_band):
                    dataset_name_all.append(dataset)
                    pt_dir_all.append(pt_dir)
                    design_number_all.append(design_number)
                    geometry_idx_0_all.append(g)
                    geometry_idx_1_all.append(g + 1)
                    wavevector_idx_0_all.append(w)
                    wavevector_idx_1_all.append(w + 1)
                    band_idx_0_all.append(b)
                    band_idx_1_all.append(b + 1)
                    wavevector_xy_all.append((float(wv_for_geom[w, 0]), float(wv_for_geom[w, 1])))

    wavevector_xy_all_np = np.asarray(wavevector_xy_all, dtype=np.float64)
    rep_wavevector_xy_np = np.asarray(rep_wavevector_xy, dtype=np.float64)

    mdict = {
        "combo_all_count": np.array([[len(dataset_name_all)]], dtype=np.int32),
        "combo_representative_count": np.array([[len(rep_dataset_name)]], dtype=np.int32),
        "combo_all_dataset_name": np.asarray(dataset_name_all, dtype=object).reshape(-1, 1),
        "combo_all_pt_dir": np.asarray(pt_dir_all, dtype=object).reshape(-1, 1),
        "combo_all_design_number": np.asarray(design_number_all, dtype=np.int64).reshape(-1, 1),
        "combo_all_geometry_idx_0based": np.asarray(geometry_idx_0_all, dtype=np.int32).reshape(-1, 1),
        "combo_all_geometry_idx_1based": np.asarray(geometry_idx_1_all, dtype=np.int32).reshape(-1, 1),
        "combo_all_wavevector_idx_0based": np.asarray(wavevector_idx_0_all, dtype=np.int32).reshape(-1, 1),
        "combo_all_wavevector_idx_1based": np.asarray(wavevector_idx_1_all, dtype=np.int32).reshape(-1, 1),
        "combo_all_band_idx_0based": np.asarray(band_idx_0_all, dtype=np.int32).reshape(-1, 1),
        "combo_all_band_idx_1based": np.asarray(band_idx_1_all, dtype=np.int32).reshape(-1, 1),
        "combo_all_wavevector_xy": wavevector_xy_all_np,
        "combo_rep_dataset_name": np.asarray(rep_dataset_name, dtype=object).reshape(-1, 1),
        "combo_rep_pt_dir": np.asarray(rep_pt_dir, dtype=object).reshape(-1, 1),
        "combo_rep_design_number": np.asarray(rep_design_number, dtype=np.int64).reshape(-1, 1),
        "combo_rep_geometry_idx_0based": np.asarray(rep_geometry_idx_0, dtype=np.int32).reshape(-1, 1),
        "combo_rep_geometry_idx_1based": np.asarray(rep_geometry_idx_1, dtype=np.int32).reshape(-1, 1),
        "combo_rep_wavevector_idx_0based": np.asarray(rep_wavevector_idx_0, dtype=np.int32).reshape(-1, 1),
        "combo_rep_wavevector_idx_1based": np.asarray(rep_wavevector_idx_1, dtype=np.int32).reshape(-1, 1),
        "combo_rep_band_idx_0based": np.asarray(rep_band_idx_0, dtype=np.int32).reshape(-1, 1),
        "combo_rep_band_idx_1based": np.asarray(rep_band_idx_1, dtype=np.int32).reshape(-1, 1),
        "combo_rep_wavevector_xy": rep_wavevector_xy_np,
    }
    sio.savemat(out_mat, mdict, do_compression=True)
    print(f"SAVED_COMBO_MAT={out_mat}")
    print(f"combo_all_count={len(dataset_name_all)}")
    print(f"combo_representative_count={len(rep_dataset_name)}")


if __name__ == "__main__":
    main()
