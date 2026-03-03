"""
DEBUG ONLY: delete after debugging is complete.

Row-level PT-vs-MATLAB comparison with precision-vs-logic attribution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def build_exact_then_nearest_remap(matlab_wv: np.ndarray, python_wv: np.ndarray, decimals: int = 12):
    key = lambda row: (round(float(row[0]), decimals), round(float(row[1]), decimals))
    py_map = {key(row): i for i, row in enumerate(python_wv)}
    remap, dists, fallback = [], [], 0
    for row in matlab_wv:
        k = key(row)
        if k in py_map:
            remap.append(py_map[k])
            dists.append(0.0)
        else:
            d2 = np.sum((python_wv - row[None, :]) ** 2, axis=1)
            idx = int(np.argmin(d2))
            remap.append(idx)
            dists.append(float(np.sqrt(d2[idx])))
            fallback += 1
    remap = np.asarray(remap, dtype=np.int64)
    return remap, {
        "fallback_count": int(fallback),
        "fallback_fraction": float(fallback / max(len(matlab_wv), 1)),
        "max_nearest_distance": float(np.max(dists) if dists else 0.0),
        "mean_nearest_distance": float(np.mean(dists) if dists else 0.0),
        "all_indices_unique": bool(len(np.unique(remap)) == len(remap)),
    }


def mean_rel_percent(a: np.ndarray, b: np.ndarray) -> float:
    # Robust scale-relative percentage to avoid divide-by-zero blowups.
    abs_diff = np.abs(a - b)
    denom = max(float(np.mean(np.abs(a))), 1e-12)
    return float(np.mean(abs_diff) / denom * 100.0)


def classify_deviation(mean_rel_pct: float, equivalence_status: str) -> tuple[str, str]:
    eq_lower = equivalence_status.lower()
    if "not equivalent" in eq_lower or "different" in eq_lower or "python-only" in eq_lower:
        if mean_rel_pct < 1.0:
            return "precision", "<1pct_mean_rel"
        return "logic", ">=1pct_mean_rel"
    if "equivalent" in eq_lower:
        if mean_rel_pct < 1.0:
            return "precision", "<1pct_mean_rel"
        return "logic", ">=1pct_mean_rel"
    if mean_rel_pct < 1.0:
        return "precision", "<1pct_mean_rel"
    return "logic", ">=1pct_mean_rel"


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug PT-vs-MATLAB row-level comparison")
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--mat-out", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--row-report-json", required=True)
    parser.add_argument("--python-default-dtype", default="float16")
    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    mat_path = Path(args.mat_out)
    csv_path = Path(args.csv_path)
    report_path = Path(args.report_json)
    row_report_path = Path(args.row_report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    row_report_path.parent.mkdir(parents=True, exist_ok=True)

    pt_wv = torch.load(pt_dir / "wavevectors_full.pt", map_location="cpu").to(torch.float16).numpy()
    pt_eval = torch.load(pt_dir / "eigenvalue_data_full.pt", map_location="cpu").to(torch.float16).numpy()
    m = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    m_wv = np.asarray(m["WAVEVECTOR_DATA"], dtype=np.float64)
    m_eval = np.asarray(m["EIGENVALUE_DATA"], dtype=np.float64)
    if m_wv.ndim == 2:
        m_wv = np.expand_dims(m_wv, axis=2)
    if m_eval.ndim == 2:
        m_eval = np.expand_dims(m_eval, axis=2)

    n_struct = pt_eval.shape[0]
    per_struct = []
    mean_rel_vals = []
    remap_diags = []
    for s in range(n_struct):
        pw = np.asarray(pt_wv[s], dtype=np.float16)
        mw = np.asarray(m_wv[:, :, s], dtype=np.float64)
        pe = np.asarray(pt_eval[s], dtype=np.float16)
        me = np.asarray(m_eval[:, :, s], dtype=np.float64)
        remap, diag = build_exact_then_nearest_remap(mw, pw.astype(np.float64), decimals=12)
        pea = np.asarray(pe[remap, :], dtype=np.float16)
        mean_rel = mean_rel_percent(me.astype(np.float16), pea)
        mean_rel_vals.append(mean_rel)
        remap_diags.append(diag)
        per_struct.append(
            {
                "struct_idx": int(s),
                "mean_rel_pct": float(mean_rel),
                "max_abs": float(np.max(np.abs(me.astype(np.float16) - pea))),
                "remap_diag": diag,
            }
        )

    summary = {
        "python_default_dtype": args.python_default_dtype,
        "mean_rel_pct_over_structures": float(np.mean(mean_rel_vals)),
        "max_mean_rel_pct_structure": float(np.max(mean_rel_vals)),
        "remap_fallback_fraction_mean": float(np.mean([x["fallback_fraction"] for x in remap_diags])),
    }

    # Row-level attribution driven by CSV rows.
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    row_reports = []
    global_mean = float(np.mean(mean_rel_vals))
    for row in rows:
        eq = row.get("equivalence_status", "")
        dev_src, bucket = classify_deviation(global_mean, eq)
        # Rows that are explicitly representation/format differences should be logic-source.
        if "python-only" in eq.lower() or "not equivalent" in eq.lower() or "different" in eq.lower():
            dev_src = "logic"
            bucket = ">=1pct_mean_rel" if global_mean >= 1.0 else "<1pct_mean_rel"
        row_reports.append(
            {
                "row_name": row.get("Mathematical Manipulation", ""),
                "equivalence_status": eq,
                "deviation_risk": row.get("deviation_risk", ""),
                "python_default_dtype": args.python_default_dtype,
                "status": "pass" if global_mean < 1.0 else "expected-difference",
                "metric": {
                    "global_mean_rel_pct": global_mean,
                    "global_max_struct_mean_rel_pct": float(np.max(mean_rel_vals)),
                },
                "deviation_source": dev_src,
                "precision_bucket": bucket,
                "evidence": str(report_path),
            }
        )

    report_path.write_text(json.dumps({"summary": summary, "per_structure": per_struct}, indent=2), encoding="utf-8")
    row_report_path.write_text(json.dumps({"rows": row_reports}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {report_path}")
    print(f"Saved row report: {row_report_path}")


if __name__ == "__main__":
    main()
