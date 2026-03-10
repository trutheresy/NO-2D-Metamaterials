import os
import random
from typing import List

import numpy as np
import scipy.io as sio
import torch


def _latest_pt_dir(dataset_dir: str) -> str:
    pt_dirs = [os.path.join(dataset_dir, p) for p in os.listdir(dataset_dir) if p.endswith("_pt")]
    if not pt_dirs:
        raise FileNotFoundError(f"No *_pt directory under: {dataset_dir}")
    pt_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pt_dirs[0]


def _reconstruct_three_channel_from_Epane(geom_e: np.ndarray) -> np.ndarray:
    # Match generation constants used in generate_dispersion_dataset_Han_Alex.py.
    E_min, E_max = 200e6, 200e9
    rho_min, rho_max = 8e2, 8e3
    nu_min, nu_max = 0.0, 0.5

    # Steel-rubber paradigm constants from apply_steel_rubber_paradigm.m.
    E_polymer, E_steel = 100e6, 200e9
    rho_polymer, rho_steel = 1200.0, 8e3
    nu_polymer, nu_steel = 0.45, 0.3

    e_poly_lin = (E_polymer - E_min) / (E_max - E_min)
    e_steel_lin = (E_steel - E_min) / (E_max - E_min)
    rho_poly_lin = (rho_polymer - rho_min) / (rho_max - rho_min)
    rho_steel_lin = (rho_steel - rho_min) / (rho_max - rho_min)
    nu_poly_lin = (nu_polymer - nu_min) / (nu_max - nu_min)
    nu_steel_lin = (nu_steel - nu_min) / (nu_max - nu_min)

    denom = max(e_steel_lin - e_poly_lin, 1e-12)
    latent = (geom_e.astype(np.float64) - e_poly_lin) / denom
    latent = np.clip(latent, 0.0, 1.0)

    pane_e = e_poly_lin + latent * (e_steel_lin - e_poly_lin)
    pane_rho = rho_poly_lin + latent * (rho_steel_lin - rho_poly_lin)
    pane_nu = nu_poly_lin + latent * (nu_steel_lin - nu_poly_lin)
    return np.stack([pane_e, pane_rho, pane_nu], axis=2).astype(np.float64)


def main():
    root = r"D:/Research/NO-2D-Metamaterials/OUTPUT"
    out_root = os.path.join(root, "fixed_geometry_samples")
    os.makedirs(out_root, exist_ok=True)

    rng = random.Random(20260307)
    c_train_names = [f"c_train_{i:02d}" for i in range(1, 25)]

    dataset_name: List[str] = []
    pt_dir_list: List[str] = []
    fixed_geometry_path: List[str] = []
    matlab_output_path: List[str] = []
    geometry_idx_0based: List[int] = []
    geometry_idx_1based: List[int] = []
    design_number: List[int] = []

    for ds in c_train_names:
        dataset_dir = os.path.join(root, ds)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")
        pt_dir = _latest_pt_dir(dataset_dir)
        geoms = torch.load(os.path.join(pt_dir, "geometries_full.pt"), map_location="cpu").numpy()
        design_params = torch.load(os.path.join(pt_dir, "design_params_full.pt"), map_location="cpu").numpy()
        n_geom = int(geoms.shape[0])
        if n_geom < 5:
            raise RuntimeError(f"{ds} has only {n_geom} geometries, need >=5")

        selected = sorted(rng.sample(range(n_geom), 5))
        ds_out = os.path.join(out_root, ds)
        os.makedirs(ds_out, exist_ok=True)

        for g in selected:
            design_no = int(round(float(design_params[g, 0])))
            fixed_design = _reconstruct_three_channel_from_Epane(geoms[g])
            geom_path = os.path.join(ds_out, f"fixed_design_g{g:04d}.mat")
            mat_out = os.path.join(ds_out, f"matlab_fixed_g{g:04d}.mat")
            sio.savemat(
                geom_path,
                {
                    "FIXED_DESIGN": fixed_design,
                    "dataset_name": ds,
                    "geometry_idx_0based": np.array([[g]], dtype=np.int32),
                    "design_number": np.array([[design_no]], dtype=np.int64),
                    "source_pt_dir": pt_dir,
                },
            )

            dataset_name.append(ds)
            pt_dir_list.append(pt_dir)
            fixed_geometry_path.append(geom_path)
            matlab_output_path.append(mat_out)
            geometry_idx_0based.append(g)
            geometry_idx_1based.append(g + 1)
            design_number.append(design_no)

    manifest_path = os.path.join(out_root, "fixed_geometry_manifest.mat")
    sio.savemat(
        manifest_path,
        {
            "dataset_name": np.asarray(dataset_name, dtype=object).reshape(-1, 1),
            "source_pt_dir": np.asarray(pt_dir_list, dtype=object).reshape(-1, 1),
            "fixed_geometry_path": np.asarray(fixed_geometry_path, dtype=object).reshape(-1, 1),
            "matlab_output_path": np.asarray(matlab_output_path, dtype=object).reshape(-1, 1),
            "geometry_idx_0based": np.asarray(geometry_idx_0based, dtype=np.int32).reshape(-1, 1),
            "geometry_idx_1based": np.asarray(geometry_idx_1based, dtype=np.int32).reshape(-1, 1),
            "design_number": np.asarray(design_number, dtype=np.int64).reshape(-1, 1),
            "n_samples_total": np.array([[len(dataset_name)]], dtype=np.int32),
            "samples_per_dataset": np.array([[5]], dtype=np.int32),
            "seed": np.array([[20260307]], dtype=np.int32),
        },
    )
    print(f"MANIFEST_PATH={manifest_path}")
    print(f"N_SAMPLES_TOTAL={len(dataset_name)}")


if __name__ == "__main__":
    main()
