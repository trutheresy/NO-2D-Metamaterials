from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


PREFIXES = ("c_train", "b_train", "c_test", "b_test")

EIGEN_CH0_FILES = {
    "uniform": "eigenfrequency_uniform_full.pt",
    "fft": "eigenfrequency_fft_full.pt",
}

# Stacked 5-channel targets (ch0 from chosen eigen full tensor; ch1–4 displacements).
OUTPUT_STACKED_FILES = {
    "uniform": "outputs_w_uniform.pt",
    "fft": "outputs_w_fft.pt",
}


def discover_dataset_dirs(output_root: Path) -> List[Path]:
    found: List[Path] = []
    for entry in output_root.iterdir():
        if entry.is_dir() and entry.name.startswith(PREFIXES):
            found.append(entry)
    return sorted(found, key=lambda p: p.name)


def latest_pt_dir(dataset_dir: Path) -> Path:
    candidates = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not candidates:
        raise FileNotFoundError(f"No *_pt folder under {dataset_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_pt(path: Path, use_mmap: bool, weights_only: bool):
    if use_mmap:
        # mmap can improve loading behavior for huge tensors, but on some
        # Windows/Torch setups it can cause process-level crashes.
        try:
            return torch.load(path, map_location="cpu", mmap=True, weights_only=weights_only)
        except Exception:
            pass
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def _check_image_tensor(name: str, x: torch.Tensor, expected_last_two: Tuple[int, int], issues: List[str]) -> None:
    if not isinstance(x, torch.Tensor):
        issues.append(f"{name}: expected Tensor, got {type(x).__name__}")
        return
    if x.ndim != 3:
        issues.append(f"{name}: expected ndim=3, got ndim={x.ndim}")
        return
    if tuple(x.shape[-2:]) != expected_last_two:
        issues.append(f"{name}: expected trailing shape {expected_last_two}, got {tuple(x.shape[-2:])}")


def _check_eigenfrequency_full(name: str, x: torch.Tensor, issues: List[str]) -> None:
    if not isinstance(x, torch.Tensor):
        issues.append(f"{name}: expected Tensor, got {type(x).__name__}")
        return
    if x.ndim != 5:
        issues.append(f"{name}: expected ndim=5 [N_design, N_wv, N_band, 32, 32], got ndim={x.ndim}")
        return
    if tuple(x.shape[-2:]) != (32, 32):
        issues.append(f"{name}: expected trailing shape (32, 32), got {tuple(x.shape[-2:])}")


def _as_triplet_arrays(reduced_indices: Sequence[Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(reduced_indices, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"reduced_indices must have shape [N,3], got {arr.shape}")
    return arr[:, 0], arr[:, 1], arr[:, 2]


def build_inputs_outputs_for_pt_dir(
    pt_dir: Path,
    chunk_size: int,
    use_mmap: bool,
    eigen_ch0_encoding: str,
    outputs_only: bool = False,
) -> dict:
    if eigen_ch0_encoding not in EIGEN_CH0_FILES:
        raise ValueError(f"Unknown eigen_ch0_encoding: {eigen_ch0_encoding!r}")
    eigen_fname = EIGEN_CH0_FILES[eigen_ch0_encoding]
    out_stacked_name = OUTPUT_STACKED_FILES[eigen_ch0_encoding]
    issues: List[str] = []

    reduced_indices_path = pt_dir / "reduced_indices.pt"
    geometries_path = pt_dir / "geometries_full.pt"
    waveforms_path = pt_dir / "waveforms_full.pt"
    band_fft_path = pt_dir / "band_fft_full.pt"
    eigen_full_path = pt_dir / eigen_fname
    disp_path = pt_dir / "displacements_dataset.pt"

    for path in (
        reduced_indices_path,
        geometries_path,
        waveforms_path,
        band_fft_path,
        eigen_full_path,
        disp_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    print("    load reduced_indices", flush=True)
    reduced_indices = _load_pt(reduced_indices_path, use_mmap=False, weights_only=False)
    if not isinstance(reduced_indices, list):
        raise TypeError(
            f"Expected list from reduced_indices.pt at {reduced_indices_path}, got {type(reduced_indices).__name__}"
        )
    n = len(reduced_indices)
    if n == 0:
        raise ValueError(f"Empty reduced_indices in {reduced_indices_path}")

    d_np, w_np, b_np = _as_triplet_arrays(reduced_indices)
    del reduced_indices
    print(f"    load geometry/waveform/band/{eigen_fname}", flush=True)
    geometries = _load_pt(geometries_path, use_mmap=use_mmap, weights_only=True)
    waveforms = _load_pt(waveforms_path, use_mmap=use_mmap, weights_only=True)
    band_fft = _load_pt(band_fft_path, use_mmap=use_mmap, weights_only=True)
    eigen_full = _load_pt(eigen_full_path, use_mmap=use_mmap, weights_only=True)

    _check_image_tensor("geometries_full", geometries, (32, 32), issues)
    _check_image_tensor("waveforms_full", waveforms, (32, 32), issues)
    _check_image_tensor("band_fft_full", band_fft, (32, 32), issues)
    _check_eigenfrequency_full(eigen_fname, eigen_full, issues)

    n_design = int(geometries.shape[0])
    n_wv = int(waveforms.shape[0])
    n_band = int(band_fft.shape[0])

    if int(eigen_full.shape[0]) != n_design or int(eigen_full.shape[1]) != n_wv or int(eigen_full.shape[2]) != n_band:
        issues.append(
            f"{eigen_fname} leading dims do not match geometry/waveform/band counts: "
            f"eigen={tuple(eigen_full.shape[:3])}, expected=({n_design}, {n_wv}, {n_band})"
        )

    min_d, max_d = int(d_np.min()), int(d_np.max())
    min_w, max_w = int(w_np.min()), int(w_np.max())
    min_b, max_b = int(b_np.min()), int(b_np.max())
    if min_d < 0 or max_d >= n_design:
        issues.append(f"design index out of range: min={min_d}, max={max_d}, n_design={n_design}")
    if min_w < 0 or max_w >= n_wv:
        issues.append(f"wavevector index out of range: min={min_w}, max={max_w}, n_wv={n_wv}")
    if min_b < 0 or max_b >= n_band:
        issues.append(f"band index out of range: min={min_b}, max={max_b}, n_band={n_band}")

    if issues:
        return {
            "pt_dir": str(pt_dir),
            "status": "mismatch",
            "n": n,
            "issues": issues,
        }

    g_np = geometries.numpy()
    wf_np = waveforms.numpy()
    bf_np = band_fft.numpy()
    ef_np = eigen_full.numpy()

    in_path = pt_dir / "inputs.pt"
    inputs_bytes = int(in_path.stat().st_size) if in_path.exists() else 0
    if not outputs_only:
        tmp_inputs_path = pt_dir / "_inputs_tmp.f16"
        inputs_mm = np.memmap(tmp_inputs_path, mode="w+", dtype=np.float16, shape=(n, 3, 32, 32))
        print("    build inputs", flush=True)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sl = slice(start, end)
            d = d_np[sl]
            w = w_np[sl]
            b = b_np[sl]
            inputs_mm[sl, 0] = g_np[d]
            inputs_mm[sl, 1] = wf_np[w]
            inputs_mm[sl, 2] = bf_np[b]

        inputs_mm.flush()
        print("    save inputs", flush=True)
        torch.save(torch.from_numpy(inputs_mm), in_path)
        del inputs_mm
        if tmp_inputs_path.exists():
            os.remove(tmp_inputs_path)
        gc.collect()
        inputs_bytes = int(in_path.stat().st_size)
    else:
        print("    skip inputs.pt (--outputs-only)", flush=True)

    print("    load displacements", flush=True)
    displacements = _load_pt(disp_path, use_mmap=use_mmap, weights_only=False)
    if not hasattr(displacements, "tensors") or len(displacements.tensors) != 4:
        issues.append(
            "displacements_dataset: expected TensorDataset with 4 tensors "
            f"(got {type(displacements).__name__})"
        )
        raise ValueError("; ".join(issues))

    disp_tensors = displacements.tensors
    for i, t in enumerate(disp_tensors):
        if not isinstance(t, torch.Tensor):
            issues.append(f"displacements tensor {i}: expected Tensor, got {type(t).__name__}")
        elif t.ndim != 3 or tuple(t.shape[1:]) != (32, 32):
            issues.append(f"displacements tensor {i}: expected shape [N,32,32], got {tuple(t.shape)}")
    if issues:
        raise ValueError("; ".join(issues))

    disp_np = [t.numpy() for t in disp_tensors]
    tmp_outputs_path = pt_dir / "_outputs_tmp.f16"
    outputs_mm = np.memmap(tmp_outputs_path, mode="w+", dtype=np.float16, shape=(n, 5, 32, 32))

    # displacements can be either full cardinality [N_design*N_wv*N_band]
    # or already reduced to [n]. We support both and report mode.
    disp_rows = int(disp_tensors[0].shape[0])
    full_rows = n_design * n_wv * n_band
    flat_full = None
    if disp_rows == full_rows:
        disp_mode = "full"
        # Compute once to avoid repeated per-chunk arithmetic.
        flat_full = d_np * (n_wv * n_band) + w_np * n_band + b_np
    elif disp_rows == n:
        disp_mode = "reduced_aligned"
    else:
        return {
            "pt_dir": str(pt_dir),
            "status": "mismatch",
            "n": n,
            "issues": [
                "displacements row count mismatch: "
                f"rows={disp_rows}, expected either full={full_rows} or reduced={n}"
            ],
        }

    print("    build outputs", flush=True)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sl = slice(start, end)
        d = d_np[sl]
        w = w_np[sl]
        b = b_np[sl]

        outputs_mm[sl, 0] = ef_np[d, w, b]
        if disp_mode == "full":
            flat = flat_full[sl]
            outputs_mm[sl, 1] = disp_np[0][flat]
            outputs_mm[sl, 2] = disp_np[1][flat]
            outputs_mm[sl, 3] = disp_np[2][flat]
            outputs_mm[sl, 4] = disp_np[3][flat]
        else:
            outputs_mm[sl, 1] = disp_np[0][sl]
            outputs_mm[sl, 2] = disp_np[1][sl]
            outputs_mm[sl, 3] = disp_np[2][sl]
            outputs_mm[sl, 4] = disp_np[3][sl]

    out_path = pt_dir / out_stacked_name
    outputs_mm.flush()
    print(f"    save {out_stacked_name}", flush=True)
    torch.save(torch.from_numpy(outputs_mm), out_path)
    del outputs_mm
    if tmp_outputs_path.exists():
        os.remove(tmp_outputs_path)
    del disp_np, disp_tensors, displacements, ef_np, bf_np, wf_np, g_np, eigen_full, band_fft, waveforms, geometries
    gc.collect()

    return {
        "pt_dir": str(pt_dir),
        "status": "ok",
        "n": n,
        "eigen_ch0_encoding": eigen_ch0_encoding,
        "eigen_ch0_file": eigen_fname,
        "outputs_stacked_file": out_stacked_name,
        "outputs_only": outputs_only,
        "inputs_shape": [n, 3, 32, 32],
        "outputs_shape": [n, 5, 32, 32],
        "disp_mode": disp_mode,
        "inputs_bytes": inputs_bytes,
        "outputs_bytes": int(out_path.stat().st_size),
        "issues": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build inputs.pt and stacked outputs (outputs_w_uniform.pt or outputs_w_fft.pt) from reduced_indices.pt "
        "for all c_train/b_train/c_test/b_test datasets."
    )
    parser.add_argument(
        "--output-root",
        default=r"D:/Research/NO-2D-Metamaterials/OUTPUT",
        help="Root directory containing c_train_*/b_train_*/c_test/b_test folders.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows per extraction chunk to control RAM usage.",
    )
    parser.add_argument(
        "--report-path",
        default=r"D:/Research/NO-2D-Metamaterials/OUTPUT/inputs_outputs_build_report.json",
        help="Where to save JSON report.",
    )
    parser.add_argument(
        "--use-mmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use memory-mapped torch.load for large tensors (enabled by default).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a dataset if the stacked output for this run (outputs_w_uniform.pt or outputs_w_fft.pt) already exists.",
    )
    parser.add_argument(
        "--outputs-only",
        action="store_true",
        help="Only write the stacked outputs file; do not rebuild inputs.pt (safer alongside training that reads inputs.pt).",
    )
    parser.add_argument(
        "--eigen-ch0-encoding",
        choices=tuple(EIGEN_CH0_FILES.keys()),
        default="uniform",
        help="Which 5D eigenfrequency tensor fills output channel 0: eigenfrequency_uniform_full.pt (uniform) "
        "or eigenfrequency_fft_full.pt (fft).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset directory names (e.g., c_train_01 b_train_01 c_test b_test).",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    dataset_dirs = discover_dataset_dirs(output_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {output_root}")
    if args.datasets:
        wanted = set(args.datasets)
        dataset_dirs = [d for d in dataset_dirs if d.name in wanted]
        if not dataset_dirs:
            raise FileNotFoundError(f"None of requested datasets found: {sorted(wanted)}")

    report = {
        "output_root": str(output_root),
        "n_datasets_found": len(dataset_dirs),
        "chunk_size": int(args.chunk_size),
        "eigen_ch0_encoding": args.eigen_ch0_encoding,
        "eigen_ch0_file": EIGEN_CH0_FILES[args.eigen_ch0_encoding],
        "outputs_stacked_file": OUTPUT_STACKED_FILES[args.eigen_ch0_encoding],
        "outputs_only": bool(args.outputs_only),
        "results": [],
    }

    for dataset_dir in dataset_dirs:
        try:
            pt_dir = latest_pt_dir(dataset_dir)
            stacked_path = pt_dir / OUTPUT_STACKED_FILES[args.eigen_ch0_encoding]
            if args.skip_existing and stacked_path.exists():
                result = {
                    "pt_dir": str(pt_dir),
                    "status": "skipped_existing",
                    "n": 0,
                    "issues": [],
                }
                report["results"].append(result)
                print(
                    f"SKIP {dataset_dir.name} -> {pt_dir.name} (existing {stacked_path.name})",
                    flush=True,
                )
                continue
            print(f"BUILD {dataset_dir.name} -> {pt_dir.name}", flush=True)
            result = build_inputs_outputs_for_pt_dir(
                pt_dir,
                chunk_size=int(args.chunk_size),
                use_mmap=bool(args.use_mmap),
                eigen_ch0_encoding=args.eigen_ch0_encoding,
                outputs_only=bool(args.outputs_only),
            )
        except Exception as e:
            result = {
                "pt_dir": str(dataset_dir),
                "status": "error",
                "n": 0,
                "issues": [str(e)],
            }
        report["results"].append(result)
        print(
            f"  STATUS={result['status']} N={result.get('n', 0)} "
            f"ISSUES={len(result.get('issues', []))}",
            flush=True,
        )

    ok = [r for r in report["results"] if r["status"] == "ok"]
    skipped = [r for r in report["results"] if r["status"] == "skipped_existing"]
    bad = [r for r in report["results"] if r["status"] not in ("ok", "skipped_existing")]
    report["n_ok"] = len(ok)
    report["n_skipped_existing"] = len(skipped)
    report["n_bad"] = len(bad)

    report_path = Path(args.report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"DONE ok={len(ok)} skipped={len(skipped)} bad={len(bad)} report={report_path}")
    if bad:
        print("MISMATCH/ERROR DATASETS:")
        for r in bad:
            print(f"- {r.get('pt_dir')}: {r.get('issues')}")


if __name__ == "__main__":
    main()
