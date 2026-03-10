from pathlib import Path
import numpy as np

from encode_eigenfrequency_fft_full import discover_dataset_dirs
from plot_sample_2x4_from_pt import resolve_pt_folder, load_pt_data, plot_one


PREFIXES = ("c_train", "b_train", "c_test", "b_test")
N_UNIQUE_GEOMS = 6


def select_unique_geometry_and_band(reduced_indices, n_select: int):
    pair_to_w = {}
    for tup in reduced_indices:
        d, w, b = map(int, tup)
        key = (d, b)
        if key not in pair_to_w:
            pair_to_w[key] = []
        pair_to_w[key].append(w)

    rng = np.random.default_rng()
    selected = []
    seen_d = set()
    seen_b = set()
    for tup in reduced_indices:
        d, _, b = map(int, tup)
        if d in seen_d or b in seen_b:
            continue
        ws = pair_to_w[(d, b)]
        w = int(rng.choice(ws))
        selected.append((d, w, b))
        seen_d.add(d)
        seen_b.add(b)
        if len(selected) >= n_select:
            break
    if len(selected) < n_select:
        raise RuntimeError(
            f"Could not find {n_select} tuples with unique geometry and unique band. "
            f"Found only {len(selected)}."
        )
    return selected


def delete_old_plots(pt_folder: Path):
    patterns = [
        "uniquegeom_*.png",
        "sample_*.png",
        "g*_wv*_b*.png",
    ]
    deleted = 0
    for pat in patterns:
        for p in pt_folder.glob(pat):
            if p.is_file():
                p.unlink()
                deleted += 1
    return deleted


def main():
    repo_root = Path(r"D:/Research/NO-2D-Metamaterials")
    dataset_dirs = [d for d in discover_dataset_dirs(repo_root) if d.name.startswith(PREFIXES)]
    print(f"FOUND_DATASETS={len(dataset_dirs)}")
    ok = 0

    for ds in dataset_dirs:
        try:
            pt_folder = resolve_pt_folder(ds)
            datafolder_name = pt_folder.name
            geometries, wavevectors, eigenvalues, waveforms, band_fft, eigenfrequency_fft, disp_tensors, reduced_indices = load_pt_data(pt_folder)
            deleted = delete_old_plots(pt_folder)
            selected = select_unique_geometry_and_band(reduced_indices, N_UNIQUE_GEOMS)

            for i, (d, w, b) in enumerate(selected):
                stem = f"g{d:03d}_wv{w:03d}_b{b}"
                plot_one(
                    pt_folder,
                    datafolder_name,
                    geometries,
                    wavevectors,
                    eigenvalues,
                    waveforms,
                    band_fft,
                    eigenfrequency_fft,
                    disp_tensors,
                    reduced_indices,
                    d,
                    w,
                    b,
                    stem,
                )
            print(
                f"DATASET={ds.name} STATUS=ok PT_DIR={pt_folder} "
                f"N_PLOTS={len(selected)} DELETED_OLD={deleted}"
            )
            ok += 1
        except Exception as e:
            print(f"DATASET={ds.name} STATUS=error MSG={e}")

    print(f"PLOT_2X4_ALL_COMPLETE ok={ok} total={len(dataset_dirs)}")


if __name__ == "__main__":
    main()
