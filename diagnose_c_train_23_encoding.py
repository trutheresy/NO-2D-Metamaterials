from pathlib import Path
import numpy as np
import torch

import NO_utilities
from encode_eigenfrequency_fft_full import discover_dataset_dirs, resolve_pt_dir, EPS


def main():
    repo_root = Path(r"D:/Research/NO-2D-Metamaterials")
    rng = np.random.default_rng(20260307)
    dataset_dirs = discover_dataset_dirs(repo_root)

    target_name = "c_train_23"
    target_pt = None
    target_sel = None
    target_n = None

    # Reproduce exact random sampling sequence from prior run.
    for ds_dir in dataset_dirs:
        name = ds_dir.name
        pt_dir = resolve_pt_dir(ds_dir)
        eig_path = pt_dir / "eigenvalue_data_full.pt"
        eig = torch.load(eig_path, map_location="cpu")
        if isinstance(eig, torch.Tensor):
            n = int(eig.numel())
        else:
            n = int(np.asarray(eig).size)
        sel = rng.choice(n, size=min(5, n), replace=False)
        if name == target_name:
            target_pt = pt_dir
            target_sel = sel
            target_n = n
            break

    if target_pt is None:
        raise RuntimeError(f"{target_name} not found")

    eig = torch.load(target_pt / "eigenvalue_data_full.pt", map_location="cpu")
    enc = torch.load(target_pt / "eigenfrequency_fft_full.pt", map_location="cpu")
    eig_np = eig.detach().cpu().numpy().astype(np.float64, copy=False)
    enc_np = enc.detach().cpu().numpy().astype(np.float64, copy=False)

    flat_eig = eig_np.reshape(-1)
    flat_enc = enc_np.reshape(target_n, enc_np.shape[-2], enc_np.shape[-1])

    # Global distribution diagnostics.
    n_total = flat_eig.size
    n_le_0 = int(np.sum(flat_eig <= 0))
    n_abs_le_1e12 = int(np.sum(np.abs(flat_eig) <= 1e-12))
    n_abs_le_1e9 = int(np.sum(np.abs(flat_eig) <= 1e-9))
    n_abs_le_1e6 = int(np.sum(np.abs(flat_eig) <= 1e-6))
    n_abs_le_1e3 = int(np.sum(np.abs(flat_eig) <= 1e-3))

    pos_vals = flat_eig[flat_eig > 0]
    min_pos = float(np.min(pos_vals)) if pos_vals.size else float("nan")
    max_val = float(np.max(flat_eig))

    print(f"DATASET={target_name}")
    print(f"PT_DIR={target_pt}")
    print(f"N_TOTAL={n_total}")
    print(f"N_LE_0={n_le_0}")
    print(f"N_ABS_LE_1e-12={n_abs_le_1e12}")
    print(f"N_ABS_LE_1e-9={n_abs_le_1e9}")
    print(f"N_ABS_LE_1e-6={n_abs_le_1e6}")
    print(f"N_ABS_LE_1e-3={n_abs_le_1e3}")
    print(f"MIN_POSITIVE={min_pos:.12g}")
    print(f"MAX_VALUE={max_val:.12g}")
    print(f"REPRODUCED_SAMPLE_INDICES={target_sel.tolist()}")

    rels = []
    for idx in target_sel:
        orig = float(flat_eig[idx])
        dec, k_dec, th_dec = NO_utilities.extract_eigenfrequency_from_wavelet(
            flat_enc[idx], size=flat_enc.shape[-1]
        )
        rel = abs(dec - orig) / max(abs(orig), EPS)
        rels.append(rel)
        print(
            f"SAMPLE idx={int(idx)} orig={orig:.12g} dec={float(dec):.12g} "
            f"rel={rel:.6e} k_dec={float(k_dec):.6g} theta_dec={float(th_dec):.6g} "
            f"is_le_0={orig <= 0} is_abs_le_1e-12={abs(orig) <= 1e-12}"
        )

    rels = np.asarray(rels, dtype=np.float64)
    print(f"SAMPLE_MEAN_REL={float(np.mean(rels)):.6e}")
    print(f"SAMPLE_MAX_REL={float(np.max(rels)):.6e}")


if __name__ == "__main__":
    main()
