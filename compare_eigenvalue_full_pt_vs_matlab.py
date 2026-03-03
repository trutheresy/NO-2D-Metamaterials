from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    abs_diff = np.abs(a - b)
    rel_diff = abs_diff / np.maximum(np.abs(a), 1e-14)
    idx = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    return {
        "shape_match": True,
        "shape": list(a.shape),
        "max_abs": float(abs_diff[idx]),
        "mean_abs": float(np.mean(abs_diff)),
        "max_rel": float(np.max(rel_diff)),
        "mean_rel": float(np.mean(rel_diff)),
        "max_vs_maxmag_pct": float(abs_diff.max() / max(np.max(np.abs(a)), 1e-14) * 100.0),
        "mean_vs_meanmag_pct": float(abs_diff.mean() / max(np.mean(np.abs(a)), 1e-14) * 100.0),
        "worst_index": [int(i) for i in idx],
        "worst_matlab": float(a[idx]),
        "worst_python": float(b[idx]),
    }


def build_exact_then_nearest_remap(
    matlab_wv: np.ndarray,
    python_wv: np.ndarray,
    decimals: int = 12,
) -> tuple[np.ndarray, dict]:
    """Build remap using exact rounded-key lookup, fallback to nearest."""
    key = lambda row: (round(float(row[0]), decimals), round(float(row[1]), decimals))
    py_map = {key(row): i for i, row in enumerate(python_wv)}

    remap = []
    fallback_count = 0
    nearest_distances = []
    for row in matlab_wv:
        k = key(row)
        if k in py_map:
            remap.append(py_map[k])
            nearest_distances.append(0.0)
            continue
        d2 = np.sum((python_wv - row[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d2))
        remap.append(idx)
        nearest_distances.append(float(np.sqrt(d2[idx])))
        fallback_count += 1

    remap_arr = np.asarray(remap, dtype=np.int64)
    diagnostics = {
        "decimals": int(decimals),
        "fallback_count": int(fallback_count),
        "fallback_fraction": float(fallback_count / max(len(matlab_wv), 1)),
        "unique_mapped_indices": int(len(np.unique(remap_arr))),
        "all_indices_unique": bool(len(np.unique(remap_arr)) == len(remap_arr)),
        "max_nearest_distance": float(np.max(nearest_distances) if nearest_distances else 0.0),
        "mean_nearest_distance": float(np.mean(nearest_distances) if nearest_distances else 0.0),
    }
    return remap_arr, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PT eigenvalue_data_full against MATLAB output")
    parser.add_argument("--pt-dir", required=True, help="PT dataset directory")
    parser.add_argument("--mat-out", required=True, help="MATLAB output .mat path")
    parser.add_argument("--report-json", required=True, help="Output report JSON path")
    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    mat_path = Path(args.mat_out)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # PT data
    pt_wv = torch.load(pt_dir / "wavevectors_full.pt", map_location="cpu").to(torch.float64).numpy()  # (N, W, 2)
    pt_eval = torch.load(pt_dir / "eigenvalue_data_full.pt", map_location="cpu").to(torch.float64).numpy()  # (N, W, B)

    # MATLAB data
    m = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    m_wv = np.asarray(m["WAVEVECTOR_DATA"], dtype=np.float64)        # (W, 2, N)
    m_eval = np.asarray(m["EIGENVALUE_DATA"], dtype=np.float64)      # (W, B, N)

    n_struct = pt_eval.shape[0]
    per_struct = []
    remaps = []
    set_equal_flags = []

    for s in range(n_struct):
        pw = pt_wv[s]                 # (W,2)
        mw = m_wv[:, :, s]            # (W,2)
        pe = pt_eval[s]               # (W,B)
        me = m_eval[:, :, s]          # (W,B)

        remap, remap_diag = build_exact_then_nearest_remap(mw, pw, decimals=12)
        remaps.append(remap.tolist())

        mw_sorted = np.array(sorted(map(tuple, np.round(mw, 12))))
        pw_sorted = np.array(sorted(map(tuple, np.round(pw, 12))))
        set_equal = bool(mw_sorted.shape == pw_sorted.shape and np.allclose(mw_sorted, pw_sorted))
        set_equal_flags.append(set_equal)

        pe_aligned = pe[remap, :]
        per_struct.append(
            {
                "struct_idx": s,
                "wavevector_set_equal_order_agnostic": set_equal,
                "wavevector_remap_diagnostics": remap_diag,
                "raw": compare(me, pe),
                "aligned": compare(me, pe_aligned),
            }
        )

    aligned_max = max(x["aligned"]["max_abs"] for x in per_struct if x["aligned"]["shape_match"])
    aligned_mean = float(np.mean([x["aligned"]["mean_abs"] for x in per_struct if x["aligned"]["shape_match"]]))

    report = {
        "inputs": {
            "pt_dir": str(pt_dir),
            "mat_out": str(mat_path),
            "n_struct": int(n_struct),
        },
        "all_wavevector_sets_equal_order_agnostic": bool(all(set_equal_flags)),
        "per_structure": per_struct,
        "summary": {
            "aligned_max_abs_over_structures": float(aligned_max),
            "aligned_mean_abs_over_structures": aligned_mean,
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
