import os
import torch


def check_case(root: str, dataset: str, geometry_idx: int):
    ddir = os.path.join(root, dataset)
    pt_dirs = [os.path.join(ddir, p) for p in os.listdir(ddir) if p.endswith("_pt")]
    pt_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    pdir = pt_dirs[0]

    eigvals = torch.load(os.path.join(pdir, "eigenvalue_data_full.pt"), map_location="cpu")
    ridx = torch.load(os.path.join(pdir, "reduced_indices.pt"), map_location="cpu")
    disp = torch.load(os.path.join(pdir, "displacements_dataset.pt"), map_location="cpu")

    design_ids = torch.tensor([int(t[0]) for t in ridx], dtype=torch.long)
    w_ids = torch.tensor([int(t[1]) for t in ridx], dtype=torch.long)
    b_ids = torch.tensor([int(t[2]) for t in ridx], dtype=torch.long)

    mask = design_ids == geometry_idx
    sel = torch.where(mask)[0]
    expected = int(eigvals.shape[1] * eigvals.shape[2])

    print(f"DATASET={dataset} GEOMETRY={geometry_idx}")
    print(f"  samples={int(sel.numel())} expected={expected}")
    print(f"  unique_wavevectors={int(torch.unique(w_ids[mask]).numel())}")
    print(f"  unique_bands={int(torch.unique(b_ids[mask]).numel())}")

    ev = eigvals[geometry_idx]
    print(
        f"  eigenvalues shape={tuple(ev.shape)} nonzero={int(torch.count_nonzero(ev).item())} has_nan={bool(torch.isnan(ev).any().item())}"
    )

    for i, t in enumerate(disp.tensors):
        s = t[sel]
        print(
            f"  eigvec_tensor_{i} shape={tuple(s.shape)} nonzero={int(torch.count_nonzero(s).item())} has_nan={bool(torch.isnan(s).any().item())}"
        )


if __name__ == "__main__":
    ROOT = r"D:/Research/NO-2D-Metamaterials/OUTPUT"
    # One single-zero case and one double-zero case.
    check_case(ROOT, "c_train_01", 523)
    check_case(ROOT, "c_train_14", 359)
