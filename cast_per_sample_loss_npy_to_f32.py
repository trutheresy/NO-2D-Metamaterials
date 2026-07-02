"""Cast legacy per_sample_loss .npy arrays to float32 (all five columns).

Cols 0–3 are integer indices stored as float32 (exact up to ~16M); col 4 is loss.
No rescoring needed — equivalent to computing in float32 for values already derived
from float32 torch math.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def cast_array(path: Path, dry_run: bool = False) -> tuple[str, str]:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"Expected shape (N, 5); got {arr.shape} in {path}")
    old_dtype = str(arr.dtype)
    if arr.dtype == np.float32:
        return old_dtype, old_dtype
    out = arr.astype(np.float32, copy=False)
    if not dry_run:
        np.save(path, out)
    return old_dtype, str(out.dtype)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help=".npy files or directories to scan.")
    p.add_argument("--glob", default="per_sample_loss_*.npy", help="Glob when a path is a directory.")
    p.add_argument("--dry-run", action="store_true", help="Report only; do not overwrite.")
    args = p.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob(args.glob)))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)

    if not files:
        print("No files matched.")
        return

    changed = 0
    for f in files:
        old, new = cast_array(f, dry_run=args.dry_run)
        tag = "skip" if old == new else ("dry-run" if args.dry_run else "cast")
        print(f"{tag:7s}  {old:>8s} -> {new:>8s}  {f}")
        if old != new:
            changed += 1
    print(f"\n{'Would cast' if args.dry_run else 'Cast'} {changed} / {len(files)} file(s).")


if __name__ == "__main__":
    main()
