"""
Build non-destructive sinusoidal input sidecars for training.

Writes ONLY new files (never overwrites wavelet products):
  - waveforms_sinusoidal_full.pt
  - band_sinusoidal_full.pt
  - inputs_sinusoidal.pt

Paper encodings (NO_utilities):
  I_k = sin((2/S)(k_x x + k_y y))
  I_b = 0.5 [cos(2 π b x / S) + cos(2 π b y / S)], b = 1..6
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

import NO_utilities as NU

PREFIXES = ("c_train", "b_train", "c_test", "b_test")

WAVEFORMS_SIN_NAME = "waveforms_sinusoidal_full.pt"
BAND_SIN_NAME = "band_sinusoidal_full.pt"
INPUTS_SIN_NAME = "inputs_sinusoidal.pt"

# Forbidden overwrite targets (wavelet / existing products).
FORBIDDEN_WRITE_NAMES = frozenset(
    {
        "inputs.pt",
        "waveforms_full.pt",
        "band_fft_full.pt",
        "outputs.pt",
        "outputs_w_uniform.pt",
        "outputs_w_fft.pt",
        "geometries_full.pt",
        "wavevectors_full.pt",
        "reduced_indices.pt",
        "indices_full.pt",
        "eigenfrequency_uniform_full.pt",
        "eigenfrequency_fft_full.pt",
        "displacements_dataset.pt",
    }
)


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


def _load_pt(path: Path, *, weights_only: bool):
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=weights_only)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=weights_only)


def _as_triplet_arrays(reduced_indices: Sequence[Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(reduced_indices, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"reduced_indices must have shape [N,3], got {arr.shape}")
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _assert_safe_write(path: Path) -> None:
    if path.name in FORBIDDEN_WRITE_NAMES:
        raise RuntimeError(f"Refusing to write forbidden wavelet/existing product: {path}")


def build_encoding_sidecars(pt_dir: Path, *, force: bool, size: int = 32) -> dict:
    """Write waveforms_sinusoidal_full.pt and band_sinusoidal_full.pt if needed."""
    wv_path = pt_dir / "wavevectors_full.pt"
    if not wv_path.exists():
        raise FileNotFoundError(f"Missing {wv_path}")

    wf_out = pt_dir / WAVEFORMS_SIN_NAME
    band_out = pt_dir / BAND_SIN_NAME
    _assert_safe_write(wf_out)
    _assert_safe_write(band_out)

    created = {"waveforms": False, "bands": False}

    if force or not wf_out.exists() or not band_out.exists():
        kxy_all = _load_pt(wv_path, weights_only=False)
        if isinstance(kxy_all, torch.Tensor):
            kxy = kxy_all[0].detach().cpu().numpy().astype(np.float64)
        else:
            raise TypeError(f"Expected Tensor in {wv_path}, got {type(kxy_all).__name__}")
        if kxy.ndim != 2 or kxy.shape[1] != 2:
            raise ValueError(f"Expected wavevectors[0] shape (N_wv, 2), got {kxy.shape}")

        if force or not wf_out.exists():
            emb = NU.embed_wavevector_plane_wave(kxy[:, 0], kxy[:, 1], size=size).astype(np.float32)
            torch.save(torch.from_numpy(emb).to(torch.float16), wf_out)
            created["waveforms"] = True

        if force or not band_out.exists():
            bands = np.arange(1, 7, dtype=np.int32)
            bem = NU.embed_band_sinusoidal(bands, size=size).astype(np.float32)
            torch.save(torch.from_numpy(bem).to(torch.float16), band_out)
            created["bands"] = True

    return {
        "waveforms_path": str(wf_out),
        "band_path": str(band_out),
        "created": created,
    }


def build_inputs_sinusoidal_for_pt_dir(
    pt_dir: Path,
    *,
    chunk_size: int,
    force: bool,
    size: int = 32,
) -> dict:
    reduced_indices_path = pt_dir / "reduced_indices.pt"
    geometries_path = pt_dir / "geometries_full.pt"
    inputs_out = pt_dir / INPUTS_SIN_NAME
    _assert_safe_write(inputs_out)

    for path in (reduced_indices_path, geometries_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    enc = build_encoding_sidecars(pt_dir, force=force, size=size)
    wf_path = Path(enc["waveforms_path"])
    band_path = Path(enc["band_path"])

    if inputs_out.exists() and not force:
        return {
            "pt_dir": str(pt_dir),
            "status": "skipped",
            "reason": f"{INPUTS_SIN_NAME} already exists (use --force to rebuild sinusoidal sidecars only)",
            "inputs_path": str(inputs_out),
            "inputs_bytes": int(inputs_out.stat().st_size),
            "encoding_sidecars": enc,
        }

    reduced_indices = _load_pt(reduced_indices_path, weights_only=False)
    if not isinstance(reduced_indices, list):
        raise TypeError(
            f"Expected list from reduced_indices.pt at {reduced_indices_path}, got {type(reduced_indices).__name__}"
        )
    n = len(reduced_indices)
    if n == 0:
        raise ValueError(f"Empty reduced_indices in {reduced_indices_path}")

    d_np, w_np, b_np = _as_triplet_arrays(reduced_indices)
    del reduced_indices

    geometries = _load_pt(geometries_path, weights_only=True)
    waveforms = _load_pt(wf_path, weights_only=True)
    bands = _load_pt(band_path, weights_only=True)

    if not isinstance(geometries, torch.Tensor) or geometries.ndim != 3:
        raise ValueError(f"Invalid geometries_full in {pt_dir}: {type(geometries)} {getattr(geometries, 'shape', None)}")
    if not isinstance(waveforms, torch.Tensor) or waveforms.ndim != 3:
        raise ValueError(f"Invalid {WAVEFORMS_SIN_NAME} in {pt_dir}")
    if not isinstance(bands, torch.Tensor) or bands.ndim != 3:
        raise ValueError(f"Invalid {BAND_SIN_NAME} in {pt_dir}")

    n_design = int(geometries.shape[0])
    n_wv = int(waveforms.shape[0])
    n_band = int(bands.shape[0])

    if int(d_np.min()) < 0 or int(d_np.max()) >= n_design:
        raise ValueError(f"design index out of range in {pt_dir}")
    if int(w_np.min()) < 0 or int(w_np.max()) >= n_wv:
        raise ValueError(f"wavevector index out of range in {pt_dir}")
    if int(b_np.min()) < 0 or int(b_np.max()) >= n_band:
        raise ValueError(f"band index out of range in {pt_dir}")

    g_np = geometries.numpy()
    wf_np = waveforms.numpy()
    bf_np = bands.numpy()

    tmp_inputs_path = pt_dir / "_inputs_sinusoidal_tmp.f16"
    inputs_mm = np.memmap(tmp_inputs_path, mode="w+", dtype=np.float16, shape=(n, 3, 32, 32))
    print(f"    build {INPUTS_SIN_NAME} (N={n})", flush=True)
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
    print(f"    save {INPUTS_SIN_NAME}", flush=True)
    torch.save(torch.from_numpy(np.asarray(inputs_mm)), inputs_out)
    del inputs_mm
    if tmp_inputs_path.exists():
        os.remove(tmp_inputs_path)
    del bf_np, wf_np, g_np, bands, waveforms, geometries
    gc.collect()

    return {
        "pt_dir": str(pt_dir),
        "status": "ok",
        "n": n,
        "inputs_path": str(inputs_out),
        "inputs_shape": [n, 3, 32, 32],
        "inputs_bytes": int(inputs_out.stat().st_size),
        "n_design": n_design,
        "n_wv": n_wv,
        "n_band": n_band,
        "encoding_sidecars": enc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:/Research/NO-2D-Metamaterials/DATASETS"),
        help="Root containing b_train_*/c_train_*/b_test/c_test",
    )
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild sinusoidal sidecar files only (never wavelet products)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report path (default under output-root)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N dataset dirs (debug)",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    report_path = args.report_path or (output_root / "sinusoidal_inputs_build_report.json")
    ds_dirs = discover_dataset_dirs(output_root)
    if args.limit > 0:
        ds_dirs = ds_dirs[: args.limit]

    if not ds_dirs:
        raise FileNotFoundError(f"No dataset dirs under {output_root} with prefixes {PREFIXES}")

    results = []
    print(f"Building sinusoidal sidecars under {output_root} ({len(ds_dirs)} datasets)", flush=True)
    for d in ds_dirs:
        pt = latest_pt_dir(d)
        print(f"\n== {d.name} :: {pt.name} ==", flush=True)
        try:
            info = build_inputs_sinusoidal_for_pt_dir(
                pt, chunk_size=args.chunk_size, force=args.force
            )
            results.append(info)
            print(f"    status={info['status']}", flush=True)
        except Exception as e:
            results.append({"pt_dir": str(pt), "status": "error", "error": str(e)})
            print(f"    ERROR: {e}", flush=True)

    report = {
        "output_root": str(output_root),
        "force": bool(args.force),
        "n_datasets": len(ds_dirs),
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    n_ok = sum(1 for r in results if r.get("status") in ("ok", "skipped"))
    n_err = sum(1 for r in results if r.get("status") == "error")
    print(f"\nDone. ok/skipped={n_ok} errors={n_err}")
    print(f"Report: {report_path}")
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
