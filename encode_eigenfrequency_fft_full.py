import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import NO_utilities


PREFIXES = ("c_train", "b_train", "c_test", "b_test")
OUT_FILENAME = "eigenfrequency_fft_full.pt"
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
    direct_file = dataset_dir / "eigenvalue_data_full.pt"
    if direct_file.exists():
        return dataset_dir
    pt_candidates = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not pt_candidates:
        raise FileNotFoundError(f"No pt dir under {dataset_dir}")
    pt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pt_candidates[0]


def encode_full(eigenvalues: np.ndarray, size: int = 32) -> np.ndarray:
    flat = eigenvalues.reshape(-1).astype(np.float64, copy=False)
    unique_vals, inverse_idx = np.unique(flat, return_inverse=True)

    encoded_unique = np.zeros((unique_vals.shape[0], size, size), dtype=np.float16)
    for i, s in enumerate(unique_vals):
        # Clamp non-positive values to tiny positive for log-domain encoder.
        sval = float(s)
        if sval <= 0:
            sval = 1e-6
        img, _, _ = NO_utilities.embed_eigenfrequency_wavelet(sval, size=size)
        encoded_unique[i] = img.astype(np.float16, copy=False)

    encoded = encoded_unique[inverse_idx]
    return encoded.reshape(*eigenvalues.shape, size, size)


def decode_error_check(encoded: np.ndarray, original: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    flat_orig = original.reshape(-1).astype(np.float64, copy=False)
    flat_enc = encoded.reshape(-1, encoded.shape[-2], encoded.shape[-1]).astype(np.float64, copy=False)
    n = flat_orig.shape[0]
    k = min(5, n)
    sel = rng.choice(n, size=k, replace=False)
    rels = []
    for idx in sel:
        orig = float(flat_orig[idx])
        img = flat_enc[idx]
        dec, _, _ = NO_utilities.extract_eigenfrequency_from_wavelet(img, size=img.shape[-1])
        rel = abs(dec - orig) / max(abs(orig), EPS)
        rels.append(rel)
    rels = np.asarray(rels, dtype=np.float64)
    return float(np.mean(rels)), float(np.max(rels))


def main():
    repo_root = Path(r"D:/Research/NO-2D-Metamaterials")
    rng = np.random.default_rng(RNG_SEED)
    dataset_dirs = discover_dataset_dirs(repo_root)
    if not dataset_dirs:
        print("NO_DATASETS_FOUND matching c_train/b_train/c_test/b_test")
        return

    print(f"FOUND_DATASETS={len(dataset_dirs)}")
    results = []
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        try:
            pt_dir = resolve_pt_dir(dataset_dir)
            eig_path = pt_dir / "eigenvalue_data_full.pt"
            if not eig_path.exists():
                raise FileNotFoundError(f"Missing eigenvalue_data_full.pt in {pt_dir}")

            eigenvalues = torch.load(eig_path, map_location="cpu")
            if isinstance(eigenvalues, torch.Tensor):
                eig_np = eigenvalues.detach().cpu().numpy()
            else:
                eig_np = np.asarray(eigenvalues)

            encoded = encode_full(eig_np, size=32)
            out_path = pt_dir / OUT_FILENAME
            torch.save(torch.from_numpy(encoded), out_path)

            mean_rel, max_rel = decode_error_check(encoded, eig_np, rng)
            results.append((dataset_name, str(pt_dir), "ok", mean_rel, max_rel, str(out_path)))
            print(
                f"DATASET={dataset_name} STATUS=ok "
                f"MEAN_REL_ERR={mean_rel:.6e} MAX_REL_ERR={max_rel:.6e} OUT={out_path}"
            )
        except Exception as e:
            results.append((dataset_name, "", f"error:{e}", np.nan, np.nan, ""))
            print(f"DATASET={dataset_name} STATUS=error MSG={e}")

    ok = [r for r in results if r[2] == "ok"]
    print(f"ENCODE_COMPLETE ok={len(ok)} total={len(results)}")


if __name__ == "__main__":
    main()
