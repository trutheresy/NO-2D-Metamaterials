from pathlib import Path
import random
from typing import List, Tuple

import torch

from encode_eigenfrequency_fft_full import discover_dataset_dirs, resolve_pt_dir


SEED = 20260309
PREFIXES = ("c_train", "b_train", "c_test", "b_test")


def build_downselected_indices(n_design: int, n_wv: int, n_band: int, rng: random.Random) -> List[Tuple[int, int, int]]:
    n_pick = max(1, n_wv // 5)
    out: List[Tuple[int, int, int]] = []
    wv_all = list(range(n_wv))
    for d in range(n_design):
        for b in range(n_band):
            chosen = rng.sample(wv_all, n_pick)
            chosen.sort()
            for w in chosen:
                out.append((d, w, b))
    return out


def main():
    repo_root = Path(r"D:/Research/NO-2D-Metamaterials")
    dataset_dirs = [d for d in discover_dataset_dirs(repo_root) if d.name.startswith(PREFIXES)]
    rng = random.Random(SEED)

    print(f"FOUND_DATASETS={len(dataset_dirs)}")
    success = 0

    for ds in dataset_dirs:
        try:
            pt_dir = resolve_pt_dir(ds)
            reduced_path = pt_dir / "reduced_indices.pt"
            full_path = pt_dir / "indices_full.pt"

            if not reduced_path.exists():
                raise FileNotFoundError(f"Missing reduced_indices.pt in {pt_dir}")

            # Rename once; keep existing indices_full if already present.
            if not full_path.exists():
                reduced_path.rename(full_path)
            else:
                # If already renamed from a prior run, ensure reduced is removed before recreating.
                reduced_path.unlink(missing_ok=True)

            eigenvalues = torch.load(pt_dir / "eigenvalue_data_full.pt", map_location="cpu", weights_only=False)
            n_design, n_wv, n_band = map(int, eigenvalues.shape)

            reduced_indices = build_downselected_indices(n_design, n_wv, n_band, rng)
            torch.save(reduced_indices, reduced_path)

            print(
                f"DATASET={ds.name} STATUS=ok PT_DIR={pt_dir} "
                f"N_FULL={n_design*n_wv*n_band} N_REDUCED={len(reduced_indices)} "
                f"N_PICK_PER_GEOM_BAND={max(1, n_wv // 5)}"
            )
            success += 1
        except Exception as e:
            print(f"DATASET={ds.name} STATUS=error MSG={e}")

    print(f"DOWNSAMPLE_COMPLETE ok={success} total={len(dataset_dirs)}")


if __name__ == "__main__":
    main()
