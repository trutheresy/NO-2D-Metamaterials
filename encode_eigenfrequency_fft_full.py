"""Bulk-write ``eigenfrequency_fft_full.pt`` under discovered dataset folders.

The batched driver ``run_generate_dispersion_batched.py`` calls the same
:func:`run_generate_dispersion_batched.write_eigenfrequency_fft_full` after each
successful generator batch. Use this script to backfill or refresh all
``c_train*`` / ``b_train*`` / ``*test`` trees under ``OUTPUT`` / ``data`` / repo root.
"""
from pathlib import Path
from typing import List

from run_generate_dispersion_batched import write_eigenfrequency_fft_full

PREFIXES = ("c_train", "b_train", "c_test", "b_test")
RNG_SEED = 0
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


def main():
    repo_root = Path(r"D:/Research/NO-2D-Metamaterials")
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

            r = write_eigenfrequency_fft_full(
                pt_dir,
                wavelet_size=32,
                hist_seed=RNG_SEED,
                decode_check_seed=RNG_SEED,
            )
            if not r.get("ok"):
                raise RuntimeError(r.get("error", "unknown error"))
            out_path = r["dst"]
            results.append(
                (
                    dataset_name,
                    str(pt_dir),
                    "ok",
                    r["mean_rel_err"],
                    r["max_rel_err"],
                    out_path,
                )
            )
            print(
                f"DATASET={dataset_name} STATUS=ok "
                f"MEAN_REL_ERR={r['mean_rel_err']:.6e} MAX_REL_ERR={r['max_rel_err']:.6e} OUT={out_path}"
            )
        except Exception as e:
            results.append((dataset_name, "", f"error:{e}", float("nan"), float("nan"), ""))
            print(f"DATASET={dataset_name} STATUS=error MSG={e}")

    ok = [r for r in results if r[2] == "ok"]
    print(f"ENCODE_COMPLETE ok={len(ok)} total={len(results)}")


if __name__ == "__main__":
    main()
