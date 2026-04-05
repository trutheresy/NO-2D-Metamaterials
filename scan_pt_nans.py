from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch


PREFIXES = ("c_train", "b_train", "c_test", "b_test")


def discover_dataset_dirs(repo_root: Path) -> List[Path]:
    candidate_roots = [repo_root / "OUTPUT", repo_root / "DATASETS", repo_root / "data", repo_root]
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
    if (dataset_dir / "eigenvalue_data_full.pt").exists():
        return dataset_dir

    pt_candidates = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not pt_candidates:
        raise FileNotFoundError(f"No *_pt folder under {dataset_dir}")
    pt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pt_candidates[0]


def iter_tensors(obj: Any, path: str = "root") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, torch.Tensor):
        yield path, obj
        return

    if isinstance(obj, np.ndarray):
        yield path, obj
        return

    if hasattr(obj, "tensors") and isinstance(getattr(obj, "tensors"), tuple):
        tensors = getattr(obj, "tensors")
        for i, t in enumerate(tensors):
            yield from iter_tensors(t, f"{path}.tensors[{i}]")
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_tensors(v, f"{path}[{repr(k)}]")
        return

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_tensors(v, f"{path}[{i}]")
        return


def has_nan(t: Any) -> Tuple[bool, int]:
    if isinstance(t, torch.Tensor):
        if not (torch.is_floating_point(t) or torch.is_complex(t)):
            return False, 0
        mask = torch.isnan(t)
        cnt = int(mask.sum().item())
        return cnt > 0, cnt

    if isinstance(t, np.ndarray):
        if not (np.issubdtype(t.dtype, np.floating) or np.issubdtype(t.dtype, np.complexfloating)):
            return False, 0
        mask = np.isnan(t)
        cnt = int(np.count_nonzero(mask))
        return cnt > 0, cnt

    return False, 0


def scan_pt_file(pt_path: Path) -> Tuple[bool, List[Tuple[str, int]], str]:
    try:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, [], f"LOAD_ERROR: {exc}"

    hits: List[Tuple[str, int]] = []
    for tpath, tensor_like in iter_tensors(obj):
        found, count = has_nan(tensor_like)
        if found:
            hits.append((tpath, count))

    return len(hits) > 0, hits, ""


def scan_folder(folder: Path, recursive: bool = True) -> Tuple[List[Tuple[Path, List[Tuple[str, int]]]], List[Tuple[Path, str]]]:
    pattern = "**/*.pt" if recursive else "*.pt"
    pt_files = sorted(folder.glob(pattern))

    with_nan: List[Tuple[Path, List[Tuple[str, int]]]] = []
    load_errors: List[Tuple[Path, str]] = []

    for pt_path in pt_files:
        found, hits, err = scan_pt_file(pt_path)
        if err:
            load_errors.append((pt_path, err))
            continue
        if found:
            with_nan.append((pt_path, hits))

    return with_nan, load_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan .pt files for NaN values.")
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="Folder to scan for .pt files.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(r"D:/Research/NO-2D-Metamaterials"),
        help="Repository root used by --all-datasets.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Scan discovered dataset folders (c_train/b_train/c_test/b_test).",
    )
    args = parser.parse_args()

    if not args.all_datasets and args.folder is None:
        raise ValueError("Provide a folder, or pass --all-datasets.")

    targets: List[Tuple[str, Path]] = []

    if args.all_datasets:
        datasets = discover_dataset_dirs(args.repo_root)
        for ds in datasets:
            try:
                pt_dir = resolve_pt_dir(ds)
                targets.append((ds.name, pt_dir))
            except Exception as exc:
                print(f"DATASET={ds.name} STATUS=SKIP MSG={exc}")
    else:
        folder = args.folder.resolve()
        targets.append((folder.name, folder))

    print(f"TARGET_FOLDERS={len(targets)}")
    total_with_nan = 0
    total_files = 0
    total_load_errors = 0

    for name, folder in targets:
        with_nan, load_errors = scan_folder(folder, recursive=True)
        n_pt = len(list(folder.glob("**/*.pt")))
        total_files += n_pt
        total_with_nan += len(with_nan)
        total_load_errors += len(load_errors)

        print(
            f"DATASET={name} FOLDER={folder} PT_FILES={n_pt} "
            f"FILES_WITH_NAN={len(with_nan)} LOAD_ERRORS={len(load_errors)}"
        )

        for pt_path, hits in with_nan:
            total_nan = sum(c for _, c in hits)
            print(f"  NAN_FILE={pt_path} NAN_COUNT={total_nan} TENSOR_PATHS={len(hits)}")
            for tpath, count in hits[:10]:
                print(f"    TENSOR={tpath} NAN_COUNT={count}")
            if len(hits) > 10:
                print(f"    ... and {len(hits) - 10} more tensor entries")

        for pt_path, err in load_errors:
            print(f"  LOAD_ERROR_FILE={pt_path} MSG={err}")

    print(
        f"SUMMARY TARGET_FOLDERS={len(targets)} PT_FILES={total_files} "
        f"FILES_WITH_NAN={total_with_nan} LOAD_ERRORS={total_load_errors}"
    )


if __name__ == "__main__":
    main()
