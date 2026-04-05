"""Bulk-write ``eigenfrequency_uniform_full.pt`` under discovered dataset folders.

For each resolved ``*_pt`` folder, fills **whatever is missing**:

- Missing ``eigenfrequency_uniform_full.pt``: read ``eigenvalue_data_full.pt``,
  clamp non-positive to float16 ``1e-6``, :func:`NO_utilities.encode_eigenfrequency_uniform_torch`,
  ``torch.save``.
- Missing ``hist_eigenfrequency_uniform_full_patch_means.png``: build histogram of
  per-patch means (32×32 patches, ≤200k sampled) from the encoded tensor on disk or
  just computed.

If both artifacts already exist, that dataset is skipped.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import NO_utilities as NU

PREFIXES = ("c_train", "b_train", "c_test", "b_test")
OUT_FILENAME = "eigenfrequency_uniform_full.pt"
HIST_FILENAME = "hist_eigenfrequency_uniform_full_patch_means.png"
PATCH_SIZE = 32
SAMPLE_MAX = 200_000
RNG_SEED = 0


def discover_dataset_dirs(repo_root: Path) -> List[Path]:
    candidate_roots = [
        repo_root / "OUTPUT",
        repo_root / "DATASETS",
        repo_root / "data",
        repo_root,
    ]
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
    direct_file = dataset_dir / "eigenvalue_data_full.pt"
    if direct_file.exists():
        return dataset_dir
    pt_candidates = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    if not pt_candidates:
        raise FileNotFoundError(f"No pt dir under {dataset_dir}")
    pt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pt_candidates[0]


def _torch_load(path: Path) -> object:
    load_kw: dict = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    return torch.load(path, **load_kw)


def save_patch_mean_histogram(
    pt_dir: Path,
    encoded: torch.Tensor,
    *,
    patch_size: int,
    sample_max: int,
    seed: int,
) -> Path:
    """Histogram of mean value per H×W patch (≤ ``sample_max`` patches)."""
    if encoded.ndim < 2 or encoded.shape[-2] != patch_size or encoded.shape[-1] != patch_size:
        raise ValueError(
            f"expected ...x{patch_size}x{patch_size} encoded tensor, got shape {tuple(encoded.shape)}"
        )
    patches = encoded.reshape(-1, patch_size, patch_size)
    patch_means = patches.mean(dim=(1, 2)).to(torch.float64)
    n = int(patch_means.numel())
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    if n <= sample_max:
        sampled = patch_means
    else:
        perm = torch.randperm(n, generator=gen)[:sample_max]
        sampled = patch_means[perm]
    vals = sampled.cpu().numpy()
    finite = vals[np.isfinite(vals)] if vals.size > 0 else np.array([], dtype=np.float64)
    out_png = pt_dir / HIST_FILENAME
    fig, ax = plt.subplots(figsize=(9, 5))
    if finite.size > 0:
        bins = max(40, min(220, int(round(math.sqrt(finite.size)))))
        ax.hist(finite, bins=bins, alpha=0.85)
    ax.set_title(
        f"{OUT_FILENAME}: per-patch mean (H×W={patch_size}×{patch_size}), "
        f"sampled patches n={min(n, sample_max)}/{n}"
    )
    ax.set_xlabel("Patch mean value")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {out_png}")
    return out_png


def encode_uniform_full_for_pt_dir(
    pt_dir: Path, *, patch_size: int = PATCH_SIZE, hist_seed: int = RNG_SEED
) -> dict:
    """
    Ensure ``eigenfrequency_uniform_full.pt`` and patch-mean histogram exist.

    Writes only missing artifacts. Returns ``skipped`` True when both already exist.
    """
    dst = pt_dir / OUT_FILENAME
    hist_png = pt_dir / HIST_FILENAME
    need_pt = not dst.is_file()
    need_hist = not hist_png.is_file()

    if not need_pt and not need_hist:
        return {
            "ok": True,
            "skipped": True,
            "wrote_pt": False,
            "wrote_hist": False,
            "dst": str(dst),
            "hist_png": str(hist_png),
        }

    out: dict = {
        "ok": True,
        "skipped": False,
        "wrote_pt": need_pt,
        "wrote_hist": need_hist,
        "dst": str(dst),
    }
    encoded: torch.Tensor | None = None
    eff_patch = patch_size

    if need_pt:
        src = pt_dir / "eigenvalue_data_full.pt"
        if not src.is_file():
            return {"ok": False, "error": f"missing {src}"}
        try:
            blob = _torch_load(src)
        except Exception as e:
            return {"ok": False, "error": f"torch.load {src}: {e}"}
        if not isinstance(blob, torch.Tensor):
            return {"ok": False, "error": f"expected Tensor in {src}, got {type(blob)}"}
        t = blob.detach().cpu()
        n_nonpositive = int((t <= 0).sum().item())
        floor = torch.tensor(1e-6, dtype=torch.float16)
        s_safe = torch.maximum(t.to(torch.float16), floor)
        try:
            encoded = NU.encode_eigenfrequency_uniform_torch(s_safe, size=patch_size)
        except Exception as e:
            return {"ok": False, "error": f"encode_eigenfrequency_uniform_torch: {e}"}
        torch.save(encoded, dst)
        out["shape_in"] = list(s_safe.shape)
        out["shape_out"] = list(encoded.shape)
        out["dtype"] = str(encoded.dtype)
        out["n_nonpositive_clamped"] = n_nonpositive
        eff_patch = int(encoded.shape[-1])
        if encoded.shape[-2] != eff_patch:
            return {
                "ok": False,
                "error": f"encoded tensor last dims not square: {tuple(encoded.shape)}",
            }
    elif need_hist:
        try:
            blob = _torch_load(dst)
        except Exception as e:
            return {"ok": False, "error": f"torch.load {dst}: {e}"}
        if not isinstance(blob, torch.Tensor):
            return {"ok": False, "error": f"expected Tensor in {dst}, got {type(blob)}"}
        encoded = blob.detach().cpu()
        eff_patch = int(encoded.shape[-1])
        if encoded.ndim < 2 or encoded.shape[-2] != eff_patch:
            return {
                "ok": False,
                "error": f"cannot infer patch size from {tuple(encoded.shape)}",
            }

    if need_hist:
        assert encoded is not None
        try:
            hp = save_patch_mean_histogram(
                pt_dir,
                encoded,
                patch_size=eff_patch,
                sample_max=SAMPLE_MAX,
                seed=hist_seed,
            )
            out["hist_png"] = str(hp)
        except Exception as e:
            out["hist_error"] = str(e)
            out["ok"] = False

    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    dataset_dirs = discover_dataset_dirs(repo_root)
    if not dataset_dirs:
        print("NO_DATASETS_FOUND matching c_train/b_train/c_test/b_test")
        return

    print(f"FOUND_DATASETS={len(dataset_dirs)}")
    n_skip = 0
    n_wrote_pt = 0
    n_wrote_hist = 0
    n_err = 0
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        try:
            pt_dir = resolve_pt_dir(dataset_dir)
            if not (pt_dir / "eigenvalue_data_full.pt").is_file() and not (
                pt_dir / OUT_FILENAME
            ).is_file():
                raise FileNotFoundError(
                    f"Missing eigenvalue_data_full.pt and {OUT_FILENAME} in {pt_dir}"
                )

            r = encode_uniform_full_for_pt_dir(
                pt_dir, patch_size=PATCH_SIZE, hist_seed=RNG_SEED
            )
            if not r.get("ok"):
                raise RuntimeError(r.get("error", "unknown error"))
            if r.get("skipped"):
                n_skip += 1
                print(
                    f"DATASET={dataset_name} SKIP: {OUT_FILENAME} and {HIST_FILENAME} "
                    f"already exist under {pt_dir}"
                )
                continue
            if r.get("wrote_pt"):
                n_wrote_pt += 1
            if r.get("wrote_hist"):
                n_wrote_hist += 1
            actions = []
            if r.get("wrote_pt"):
                actions.append("wrote_pt")
            if r.get("wrote_hist"):
                actions.append("wrote_hist")
            msg = f"DATASET={dataset_name} STATUS=ok {' '.join(actions)} dst={r['dst']}"
            if r.get("shape_in") is not None:
                msg += f" shape_in={r['shape_in']} shape_out={r.get('shape_out')}"
            if r.get("n_nonpositive_clamped"):
                msg += f" clamped_nonpositive={r['n_nonpositive_clamped']}"
            if r.get("hist_png"):
                msg += f" hist={r['hist_png']}"
            print(msg)
            if r.get("hist_error"):
                print(f"[WARN] histogram: {r['hist_error']}")
        except Exception as e:
            n_err += 1
            print(f"DATASET={dataset_name} STATUS=error MSG={e}")

    print(
        f"SYNC_COMPLETE wrote_pt={n_wrote_pt} wrote_hist={n_wrote_hist} "
        f"skipped_both_present={n_skip} errors={n_err} total_scanned={len(dataset_dirs)}"
    )


if __name__ == "__main__":
    main()
