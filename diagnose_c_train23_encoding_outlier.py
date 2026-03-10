from pathlib import Path
from typing import List

import numpy as np
import torch

import NO_utilities


PREFIXES = ("c_train", "b_train", "c_test", "b_test")
RNG_SEED = 20260307
EPS = 1e-12


def discover_dataset_dirs(repo_root: Path) -> List[Path]:
    candidate_roots = [repo_root / "OUTPUT", repo_root / "data", repo_root]
    found: List[Path] = []
    seen = set()
    for root in candidate_roots:
        if not root.exists() or not root.is_dir():
            continue
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith(PREFIXES):
                continue
            key = str(entry.resolve()).lower()
            if key not in seen:
                seen.add(key)
                found.append(entry)
    return sorted(found, key=lambda p: str(p))


def resolve_pt_dir(dataset_dir: Path) -> Path:
    direct = dataset_dir / "eigenvalue_data_full.pt"
    if direct.exists():
        return dataset_dir
    pt_dirs = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not pt_dirs:
        raise FileNotFoundError(f"No *_pt folder under {dataset_dir}")
    pt_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pt_dirs[0]


def main():
    repo = Path(r"D:/Research/NO-2D-Metamaterials")
    rng = np.random.default_rng(RNG_SEED)
    datasets = discover_dataset_dirs(repo)

    target_name = "c_train_23"
    target_pt = None
    sampled_idx = None
    sampled_n = None

    # Reproduce index sampling behavior used during previous fidelity run.
    for ds_dir in datasets:
        pt_dir = resolve_pt_dir(ds_dir)
        eig = torch.load(pt_dir / "eigenvalue_data_full.pt", map_location="cpu")
        eig_np = eig.detach().cpu().numpy() if isinstance(eig, torch.Tensor) else np.asarray(eig)
        flat_n = int(eig_np.size)
        k = min(5, flat_n)
        idx = rng.choice(flat_n, size=k, replace=False)
        if ds_dir.name == target_name:
            target_pt = pt_dir
            sampled_idx = idx
            sampled_n = flat_n
            break

    if target_pt is None or sampled_idx is None:
        raise RuntimeError(f"Could not locate target dataset: {target_name}")

    eig = torch.load(target_pt / "eigenvalue_data_full.pt", map_location="cpu")
    enc = torch.load(target_pt / "eigenfrequency_fft_full.pt", map_location="cpu")
    eig_np = eig.detach().cpu().numpy().astype(np.float64, copy=False)
    enc_np = enc.detach().cpu().numpy().astype(np.float64, copy=False)

    flat_orig = eig_np.reshape(-1)
    flat_enc = enc_np.reshape(-1, enc_np.shape[-2], enc_np.shape[-1])

    print(f"TARGET_PT_DIR={target_pt}")
    print(f"FLAT_COUNT={sampled_n}")
    print("REPRO_SAMPLED_FLAT_INDICES=" + ",".join(str(int(i)) for i in sampled_idx))
    print("DETAILS_PER_SAMPLED_POINT")

    rels = []
    n_huge = 0
    for j, flat_i in enumerate(sampled_idx.tolist(), start=1):
        orig = float(flat_orig[flat_i])
        dec, k_ext, theta_ext = NO_utilities.extract_eigenfrequency_from_wavelet(flat_enc[flat_i], size=32)
        rel = abs(dec - orig) / max(abs(orig), EPS)
        rels.append(rel)
        if rel > 1e3:
            n_huge += 1
        print(
            f"  sample_{j}: idx={int(flat_i)} orig={orig:.12g} decoded={dec:.12g} "
            f"rel_err={rel:.6e} k_ext={k_ext:.6f} theta_ext={theta_ext:.6f}"
        )

    rels_np = np.asarray(rels, dtype=np.float64)
    print(f"REPRO_MEAN_REL_ERR={float(np.mean(rels_np)):.6e}")
    print(f"REPRO_MAX_REL_ERR={float(np.max(rels_np)):.6e}")
    print(f"REPRO_COUNT_REL_GT_1E3={n_huge}")

    # Distribution checks for zeros / near-zeros.
    absv = np.abs(flat_orig)
    n_zero = int(np.sum(absv == 0))
    n_lt_1e9 = int(np.sum(absv < 1e-9))
    n_lt_1e6 = int(np.sum(absv < 1e-6))
    n_lt_1e3 = int(np.sum(absv < 1e-3))
    print(f"N_ZERO={n_zero}")
    print(f"N_LT_1E-9={n_lt_1e9}")
    print(f"N_LT_1E-6={n_lt_1e6}")
    print(f"N_LT_1E-3={n_lt_1e3}")
    print(f"MIN_ABS={float(np.min(absv)):.6e}")
    print(f"P01_ABS={float(np.percentile(absv,1)):.6e}")
    print(f"P50_ABS={float(np.percentile(absv,50)):.6e}")


if __name__ == "__main__":
    main()
