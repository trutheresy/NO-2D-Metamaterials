"""Rename waveforms_full.pt -> waveforms_full_old.pt and regenerate embeddings."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

import NO_utilities as NU
from build_inputs_outputs_from_reduced_indices import discover_dataset_dirs, latest_pt_dir

ROOT = Path(__file__).resolve().parent
PREFIXES = ("c_train", "b_train", "c_test", "b_test")
REPORT_PATH = ROOT / "DATASETS" / "waveforms_regeneration_report.json"


def discover_waveform_pt_dirs(datasets_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for entry in sorted(datasets_root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith(PREFIXES):
            continue
        pt_dir = latest_pt_dir(entry)
        if (pt_dir / "wavevectors_full.pt").exists():
            dirs.append(pt_dir)
    return dirs


def load_kxy(pt_dir: Path) -> np.ndarray:
    kxy = torch.load(pt_dir / "wavevectors_full.pt", map_location="cpu", weights_only=False)[0]
    return kxy.numpy().astype(np.float64)


def patch_inputs_waveform_channel(pt_dir: Path, waveforms: torch.Tensor, chunk_size: int = 4096) -> bool:
    """Update channel 1 of inputs.pt in place; rename old file to inputs_old.pt."""
    inputs_path = pt_dir / "inputs.pt"
    if not inputs_path.exists():
        return False

    reduced = torch.load(pt_dir / "reduced_indices.pt", map_location="cpu", weights_only=False)
    arr = np.asarray(reduced, dtype=np.int64)
    w_np = arr[:, 1]
    n = len(w_np)

    old_path = pt_dir / "inputs_old.pt"
    if not old_path.exists():
        os.replace(inputs_path, old_path)
    else:
        inputs_path.unlink()

    inputs = torch.load(old_path, map_location="cpu", weights_only=True)
    wf_np = waveforms.numpy()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sl = slice(start, end)
        inputs[sl, 1] = torch.from_numpy(wf_np[w_np[sl]])

    torch.save(inputs, inputs_path)
    del inputs
    gc.collect()
    return True


def regenerate_one(
    pt_dir: Path,
    embed_kw: dict,
    *,
    rebuild_inputs: bool,
    dry_run: bool,
) -> dict:
    wf_path = pt_dir / "waveforms_full.pt"
    wf_old = pt_dir / "waveforms_full_old.pt"
    if not wf_path.exists():
        return {"pt_dir": str(pt_dir), "status": "skip", "reason": "no waveforms_full.pt"}

    kxy = load_kxy(pt_dir)
    emb = NU.embed_2const_wavelet(
        kxy[:, 0],
        kxy[:, 1],
        size=32,
        freq_range=1.0,
        verbose=False,
        **embed_kw,
    )
    waveforms = torch.from_numpy(emb.astype(np.float16))

    if dry_run:
        return {
            "pt_dir": str(pt_dir),
            "status": "dry_run",
            "shape": list(waveforms.shape),
        }

    if wf_old.exists():
        wf_old.unlink()
    os.replace(wf_path, wf_old)
    torch.save(waveforms, wf_path)

    inputs_patched = False
    if rebuild_inputs:
        inputs_patched = patch_inputs_waveform_channel(pt_dir, waveforms)

    return {
        "pt_dir": str(pt_dir),
        "status": "ok",
        "waveforms_old": str(wf_old),
        "inputs_patched": inputs_patched,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets-root", type=Path, default=ROOT / "DATASETS")
    p.add_argument("--freq-scale", type=int, required=True)
    p.add_argument("--freq-offset", type=float, required=True)
    p.add_argument("--sigma-numerator", type=float, required=True)
    p.add_argument("--kx-cycles", type=int, required=True)
    p.add_argument("--ky-cycles", type=int, required=True)
    p.add_argument("--no-rebuild-inputs", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    embed_kw = {
        "freq_scale": args.freq_scale,
        "freq_offset": args.freq_offset,
        "sigma_numerator": args.sigma_numerator,
        "kx_cycles": args.kx_cycles,
        "ky_cycles": args.ky_cycles,
    }
    cfg = NU.embed_2const_wavelet_params(**embed_kw)
    pt_dirs = discover_waveform_pt_dirs(args.datasets_root)
    print(f"Found {len(pt_dirs)} dataset pt folders")
    print(f"Embedding params: {cfg}")

    results = []
    for i, pt_dir in enumerate(pt_dirs, 1):
        print(f"[{i}/{len(pt_dirs)}] {pt_dir.parent.name}/{pt_dir.name}", flush=True)
        results.append(
            regenerate_one(
                pt_dir,
                embed_kw,
                rebuild_inputs=not args.no_rebuild_inputs,
                dry_run=args.dry_run,
            )
        )

    report = {
        "embed_params": cfg,
        "n_datasets": len(pt_dirs),
        "rebuild_inputs": not args.no_rebuild_inputs,
        "results": results,
    }
    if not args.dry_run:
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {REPORT_PATH}")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Done: {ok}/{len(pt_dirs)} regenerated")


if __name__ == "__main__":
    main()
