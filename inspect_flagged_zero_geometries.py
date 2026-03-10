import json
import os

import torch


def main():
    root = r"D:/Research/NO-2D-Metamaterials/OUTPUT"
    flagged = {
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

    out = []
    for ds, idxs in flagged.items():
        ddir = os.path.join(root, ds)
        pt_dirs = [os.path.join(ddir, p) for p in os.listdir(ddir) if p.endswith("_pt")] if os.path.isdir(ddir) else []
        if not pt_dirs:
            out.append({"dataset": ds, "error": "missing_pt_dir"})
            continue
        pt_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        pdir = pt_dirs[0]

        eigvals = torch.load(os.path.join(pdir, "eigenvalue_data_full.pt"), map_location="cpu")
        ridx = torch.load(os.path.join(pdir, "reduced_indices.pt"), map_location="cpu")
        disp = torch.load(os.path.join(pdir, "displacements_dataset.pt"), map_location="cpu")

        design_ids = torch.tensor([int(t[0]) for t in ridx], dtype=torch.long)
        w_ids = torch.tensor([int(t[1]) for t in ridx], dtype=torch.long)
        b_ids = torch.tensor([int(t[2]) for t in ridx], dtype=torch.long)

        rec = {"dataset": ds, "pt_dir": pdir, "geometries": []}
        expected_samples = int(eigvals.shape[1] * eigvals.shape[2])  # N_wv * N_eig

        for g in idxs:
            ev_slice = eigvals[g]  # [N_wv, N_eig]
            ev_nonzero = int(torch.count_nonzero(ev_slice).item())
            ev_has_nan = bool(torch.isnan(ev_slice).any().item()) if ev_slice.is_floating_point() else False

            mask = design_ids == g
            sample_idxs = torch.where(mask)[0]

            disp_summary = []
            disp_nonzero_total = 0
            disp_has_nan = False
            for ti, t in enumerate(disp.tensors):
                samp = t[sample_idxs]
                nz = int(torch.count_nonzero(samp).item())
                hn = bool(torch.isnan(samp).any().item()) if samp.is_floating_point() else False
                disp_nonzero_total += nz
                disp_has_nan = disp_has_nan or hn
                disp_summary.append(
                    {
                        "tensor_idx": ti,
                        "shape": tuple(samp.shape),
                        "nonzero_count": nz,
                        "has_nan": hn,
                    }
                )

            unique_w = int(torch.unique(w_ids[mask]).numel()) if sample_idxs.numel() > 0 else 0
            unique_b = int(torch.unique(b_ids[mask]).numel()) if sample_idxs.numel() > 0 else 0

            rec["geometries"].append(
                {
                    "geometry_idx": g,
                    "n_samples": int(sample_idxs.numel()),
                    "expected_samples": expected_samples,
                    "unique_wavevectors": unique_w,
                    "unique_bands": unique_b,
                    "eigenvalues_shape": tuple(ev_slice.shape),
                    "eigenvalues_nonzero_count": ev_nonzero,
                    "eigenvalues_has_nan": ev_has_nan,
                    "eigenvectors_total_nonzero_count": disp_nonzero_total,
                    "eigenvectors_has_nan": disp_has_nan,
                    "eigenvector_tensor_details": disp_summary,
                }
            )

        out.append(rec)

    out_path = os.path.join(root, "flagged_zero_geometry_eig_details.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("saved", out_path)
    for r in out:
        if "error" in r:
            print("DATASET", r["dataset"], "ERROR", r["error"])
            continue
        for g in r["geometries"]:
            print(
                "DATASET",
                r["dataset"],
                "geom",
                g["geometry_idx"],
                "samples",
                g["n_samples"],
                "wv",
                g["unique_wavevectors"],
                "bands",
                g["unique_bands"],
                "eig_nz",
                g["eigenvalues_nonzero_count"],
                "eig_nan",
                g["eigenvalues_has_nan"],
                "eigvec_nz",
                g["eigenvectors_total_nonzero_count"],
                "eigvec_nan",
                g["eigenvectors_has_nan"],
            )


if __name__ == "__main__":
    main()
