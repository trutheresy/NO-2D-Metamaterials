import glob
import json
import os
import argparse

import torch


def _resolve_pt_dir(dataset_dir: str):
    pt_dirs = glob.glob(os.path.join(dataset_dir, "*_pt"))
    if not pt_dirs:
        return None
    pt_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pt_dirs[0]


def _any_nan_per_row(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    return torch.isnan(flat).any(dim=1)


def _any_nonzero_per_row(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    return flat.ne(0).any(dim=1)


def _load_required(pt_dir: str):
    paths = {
        "geometries_full": os.path.join(pt_dir, "geometries_full.pt"),
        "eigenvalue_data_full": os.path.join(pt_dir, "eigenvalue_data_full.pt"),
        "displacements_dataset": os.path.join(pt_dir, "displacements_dataset.pt"),
        "reduced_indices": os.path.join(pt_dir, "reduced_indices.pt"),
    }
    loaded = {}
    missing = []
    for key, path in paths.items():
        if not os.path.exists(path):
            missing.append(key)
            continue
        loaded[key] = torch.load(path, map_location="cpu")
    return loaded, missing


def _audit_dataset(name: str, root: str):
    entry = {
        "dataset": name,
        "exists": False,
        "pt_subdir": None,
        "issues": [],
        "n_geometries": 0,
        "geometry_flags": {
            "zero_geometry_indices": [],
            "nan_geometry_indices": [],
            "zero_eigenvalue_indices": [],
            "nan_eigenvalue_indices": [],
            "zero_eigenvector_indices": [],
            "nan_eigenvector_indices": [],
        },
        "zero_geometry_crosscheck": [],
    }

    dataset_dir = os.path.join(root, name)
    if not os.path.isdir(dataset_dir):
        entry["issues"].append("missing_dataset_dir")
        return entry
    entry["exists"] = True

    pt_dir = _resolve_pt_dir(dataset_dir)
    if pt_dir is None:
        entry["issues"].append("missing_pt_subdir")
        return entry
    entry["pt_subdir"] = pt_dir

    loaded, missing_required = _load_required(pt_dir)
    if missing_required:
        entry["issues"].append(f"missing_required_files:{','.join(missing_required)}")
        return entry

    geometries = loaded["geometries_full"]
    eigenvalues = loaded["eigenvalue_data_full"]
    displacements = loaded["displacements_dataset"]
    reduced_indices = loaded["reduced_indices"]

    if not isinstance(displacements, torch.utils.data.TensorDataset):
        entry["issues"].append("displacements_dataset_not_tensordataset")
        return entry
    if not isinstance(reduced_indices, list):
        entry["issues"].append("reduced_indices_not_list")
        return entry

    n_geom = int(geometries.shape[0])
    entry["n_geometries"] = n_geom

    geom_flat = geometries.reshape(n_geom, -1)
    geom_zero = ~geom_flat.ne(0).any(dim=1)
    geom_nan = torch.isnan(geom_flat).any(dim=1) if geometries.is_floating_point() else torch.zeros(n_geom, dtype=torch.bool)

    eigv_flat = eigenvalues.reshape(n_geom, -1)
    eigv_zero = ~eigv_flat.ne(0).any(dim=1)
    eigv_nan = torch.isnan(eigv_flat).any(dim=1) if eigenvalues.is_floating_point() else torch.zeros(n_geom, dtype=torch.bool)

    design_ids = torch.tensor([int(t[0]) for t in reduced_indices], dtype=torch.long)
    n_samples = int(design_ids.shape[0])
    if n_samples == 0:
        entry["issues"].append("empty_reduced_indices")
        return entry

    sample_nan_any = torch.zeros(n_samples, dtype=torch.bool)
    sample_nonzero_any = torch.zeros(n_samples, dtype=torch.bool)
    for tensor in displacements.tensors:
        sample_nan_any |= _any_nan_per_row(tensor)
        sample_nonzero_any |= _any_nonzero_per_row(tensor)

    eigvec_nan = torch.zeros(n_geom, dtype=torch.bool)
    eigvec_zero = torch.zeros(n_geom, dtype=torch.bool)
    for d in range(n_geom):
        mask = design_ids == d
        if not torch.any(mask):
            eigvec_nan[d] = True
            eigvec_zero[d] = True
            continue
        eigvec_nan[d] = bool(torch.any(sample_nan_any[mask]))
        eigvec_zero[d] = not bool(torch.any(sample_nonzero_any[mask]))

    entry["geometry_flags"]["zero_geometry_indices"] = torch.where(geom_zero)[0].tolist()
    entry["geometry_flags"]["nan_geometry_indices"] = torch.where(geom_nan)[0].tolist()
    entry["geometry_flags"]["zero_eigenvalue_indices"] = torch.where(eigv_zero)[0].tolist()
    entry["geometry_flags"]["nan_eigenvalue_indices"] = torch.where(eigv_nan)[0].tolist()
    entry["geometry_flags"]["zero_eigenvector_indices"] = torch.where(eigvec_zero)[0].tolist()
    entry["geometry_flags"]["nan_eigenvector_indices"] = torch.where(eigvec_nan)[0].tolist()

    for d in entry["geometry_flags"]["zero_geometry_indices"]:
        entry["zero_geometry_crosscheck"].append(
            {
                "geometry_idx": int(d),
                "eigenvalues_all_zero": bool(eigv_zero[d]),
                "eigenvalues_has_nan": bool(eigv_nan[d]),
                "eigenvectors_all_zero": bool(eigvec_zero[d]),
                "eigenvectors_has_nan": bool(eigvec_nan[d]),
            }
        )

    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    root = r"D:/Research/NO-2D-Metamaterials/OUTPUT"
    targets = args.datasets if args.datasets else ([f"c_train_{i:02d}" for i in range(1, 25)] + ["c_test"])
    report = {"root": root, "targets": []}

    for name in targets:
        print(f"CHECK_DATASET {name}", flush=True)
        entry = _audit_dataset(name, root)
        gf = entry["geometry_flags"]
        print(
            "  SUMMARY "
            f"n_geom={entry['n_geometries']} "
            f"zero_geom={len(gf['zero_geometry_indices'])} "
            f"nan_geom={len(gf['nan_geometry_indices'])} "
            f"zero_eigval={len(gf['zero_eigenvalue_indices'])} "
            f"nan_eigval={len(gf['nan_eigenvalue_indices'])} "
            f"zero_eigvec={len(gf['zero_eigenvector_indices'])} "
            f"nan_eigvec={len(gf['nan_eigenvector_indices'])}",
            flush=True,
        )
        report["targets"].append(entry)

    out = os.path.join(root, "dataset_zero_nan_per_geometry_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("saved", out)
    for t in report["targets"]:
        if (not t["exists"]) or t["issues"]:
            print("DATASET", t["dataset"], "issues", t["issues"])
            continue
        gf = t["geometry_flags"]
        total_flags = (
            len(gf["zero_geometry_indices"])
            + len(gf["nan_geometry_indices"])
            + len(gf["zero_eigenvalue_indices"])
            + len(gf["nan_eigenvalue_indices"])
            + len(gf["zero_eigenvector_indices"])
            + len(gf["nan_eigenvector_indices"])
        )
        print("DATASET", t["dataset"], "geom_flags", total_flags)
        if t["zero_geometry_crosscheck"]:
            print("  zero_geometry_crosscheck_count", len(t["zero_geometry_crosscheck"]))


if __name__ == "__main__":
    main()
