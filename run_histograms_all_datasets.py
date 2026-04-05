from pathlib import Path

from dataset_pt_discovery import discover_dataset_dirs
from plot_dataset_histograms import resolve_pt_folder, plot_eigenfrequency_hist, plot_reduced_indices_hists


PREFIXES = ("c_train", "b_train", "c_test", "b_test")


def main():
    repo_root = Path(r"D:/Research/NO-2D-Metamaterials")
    dataset_dirs = [d for d in discover_dataset_dirs(repo_root) if d.name.startswith(PREFIXES)]
    print(f"FOUND_DATASETS={len(dataset_dirs)}")
    ok = 0
    for ds in dataset_dirs:
        try:
            pt_dir = resolve_pt_folder(ds)
            plot_eigenfrequency_hist(pt_dir)
            plot_reduced_indices_hists(pt_dir)
            print(f"DATASET={ds.name} STATUS=ok PT_DIR={pt_dir}")
            ok += 1
        except Exception as e:
            print(f"DATASET={ds.name} STATUS=error MSG={e}")
    print(f"HISTOGRAMS_COMPLETE ok={ok} total={len(dataset_dirs)}")


if __name__ == "__main__":
    main()
