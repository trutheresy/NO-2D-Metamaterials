from __future__ import annotations

"""
Batch driver for ``generate_dispersion_dataset_Han_Alex.py``.

After each successful batch this runner writes derived eigenfrequency tensors from
``eigenvalue_data_full.pt`` (unconditional ``torch.save`` overwrite):

- ``eigenfrequency_uniform_full.pt`` + ``hist_eigenfrequency_uniform_full.png``
- ``eigenfrequency_fft_full.pt`` + ``hist_eigenfrequency_fft_full.png``

FFT encoding uses :func:`NO_utilities.embed_eigenfrequency_wavelet` (same logic as the
former standalone numpy ``encode_full`` loop). The bulk DATASETS
scanner ``encode_eigenfrequency_fft_full.py`` calls the same ``write_eigenfrequency_fft_full``.
"""

import argparse
import inspect
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

import NO_utilities as NU
from plot_dataset_histograms import (
    save_eigenfrequency_fft_histogram,
    save_eigenfrequency_uniform_histogram,
)


def _torch_load(path: Path) -> object:
    load_kw: dict = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    return torch.load(path, **load_kw)


def write_eigenfrequency_uniform_full(
    pt_dir: Path, *, patch_size: int = 32, hist_seed: int = 0
) -> dict:
    """
    Build ``eigenfrequency_uniform_full.pt`` from ``eigenvalue_data_full.pt`` in ``pt_dir``.

    Matches ``encode_eigenfrequency_fft_full`` I/O style: unconditional ``torch.save``
    overwrites the output file. Non-positive eigenfrequencies are clamped to float16
    ``1e-6`` before ``log``. Encoding uses :func:`NO_utilities.encode_eigenfrequency_uniform_torch`
    (torch float16 end-to-end). On success, writes ``hist_eigenfrequency_uniform_full.png``.
    """
    src = pt_dir / "eigenvalue_data_full.pt"
    dst = pt_dir / "eigenfrequency_uniform_full.pt"
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
    out: dict = {
        "ok": True,
        "dst": str(dst),
        "shape_in": list(s_safe.shape),
        "shape_out": list(encoded.shape),
        "dtype": str(encoded.dtype),
        "n_nonpositive_clamped": n_nonpositive,
    }
    try:
        hist_path = save_eigenfrequency_uniform_histogram(pt_dir, seed=hist_seed)
        out["hist_png"] = str(hist_path)
    except Exception as e:
        out["hist_error"] = str(e)
    return out


_FFT_DECODE_EPS = 1e-12


def encode_eigenfrequency_fft_tensor(
    eigenvalues: torch.Tensor, *, wavelet_size: int = 32
) -> torch.Tensor:
    """
    Torch-only orchestration: unique scalar eigenvalues → wavelet patches (float16),
    indexed back to the full tensor shape. Non-positive scalars are clamped to ``1e-6``
    before calling :func:`NO_utilities.embed_eigenfrequency_wavelet`.
    """
    t = eigenvalues.detach().cpu().to(torch.float16)
    flat = t.reshape(-1)
    unique_vals, inverse_idx = torch.unique(flat, return_inverse=True)
    patches: list[torch.Tensor] = []
    for i in range(int(unique_vals.numel())):
        sval = float(unique_vals[i].item())
        if sval <= 0:
            sval = 1e-6
        img, _, _ = NU.embed_eigenfrequency_wavelet(sval, size=wavelet_size)
        patches.append(torch.from_numpy(img).to(torch.float16))
    encoded_unique = torch.stack(patches, dim=0)
    encoded = encoded_unique[inverse_idx]
    return encoded.reshape(*t.shape, wavelet_size, wavelet_size)


def fft_decode_error_sample(
    encoded: torch.Tensor, original: torch.Tensor, *, seed: int
) -> tuple[float, float]:
    """Spot-check up to 5 entries with :func:`NO_utilities.extract_eigenfrequency_from_wavelet`."""
    enc = encoded.detach().cpu()
    orig = original.detach().cpu().to(torch.float16)
    flat_orig = orig.reshape(-1)
    flat_enc = enc.reshape(-1, enc.shape[-2], enc.shape[-1]).to(torch.float16)
    n = int(flat_orig.numel())
    k = min(5, n)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen)[:k]
    rels: list[float] = []
    for j in range(k):
        idx = int(perm[j].item())
        o = float(flat_orig[idx].item())
        img = flat_enc[idx].numpy()
        dec, _, _ = NU.extract_eigenfrequency_from_wavelet(img, size=int(img.shape[-1]))
        rel = abs(dec - o) / max(abs(o), _FFT_DECODE_EPS)
        rels.append(rel)
    rel_t = torch.tensor(rels, dtype=torch.float16)
    return float(rel_t.mean().item()), float(rel_t.max().item())


def write_eigenfrequency_fft_full(
    pt_dir: Path,
    *,
    wavelet_size: int = 32,
    hist_seed: int = 0,
    decode_check_seed: int = 0,
) -> dict:
    """
    Build ``eigenfrequency_fft_full.pt`` from ``eigenvalue_data_full.pt`` in ``pt_dir``.

    Unconditional ``torch.save`` overwrite. On success, writes ``hist_eigenfrequency_fft_full.png``
    and records relative decode error from a small random sample (same idea as the old standalone script).
    """
    src = pt_dir / "eigenvalue_data_full.pt"
    dst = pt_dir / "eigenfrequency_fft_full.pt"
    if not src.is_file():
        return {"ok": False, "error": f"missing {src}"}
    try:
        blob = _torch_load(src)
    except Exception as e:
        return {"ok": False, "error": f"torch.load {src}: {e}"}
    if not isinstance(blob, torch.Tensor):
        return {"ok": False, "error": f"expected Tensor in {src}, got {type(blob)}"}
    t = blob.detach().cpu()
    try:
        encoded = encode_eigenfrequency_fft_tensor(t, wavelet_size=wavelet_size)
    except Exception as e:
        return {"ok": False, "error": f"encode_eigenfrequency_fft_tensor: {e}"}
    torch.save(encoded, dst)
    mean_rel, max_rel = fft_decode_error_sample(
        encoded, t, seed=decode_check_seed
    )
    out: dict = {
        "ok": True,
        "dst": str(dst),
        "shape_in": list(t.shape),
        "shape_out": list(encoded.shape),
        "dtype": str(encoded.dtype),
        "mean_rel_err": mean_rel,
        "max_rel_err": max_rel,
    }
    try:
        hist_path = save_eigenfrequency_fft_histogram(pt_dir, seed=hist_seed)
        out["hist_png"] = str(hist_path)
    except Exception as e:
        out["hist_error"] = str(e)
    return out


def _run_one_batch(
    repo_root: Path,
    n_struct: int,
    seed_offset: int,
    binarize: bool,
    log_path: Path,
    parallel_workers: int,
) -> dict:
    print(
        f"[START] n_struct={n_struct} seed_offset={seed_offset} "
        f"binarize={binarize} parallel_workers={parallel_workers} log={log_path}"
    )
    cmd = [
        sys.executable,
        str(repo_root / "generate_dispersion_dataset_Han_Alex.py"),
        "--n-struct",
        str(n_struct),
        "--rng-seed-offset",
        str(seed_offset),
        "--parallel-workers",
        str(parallel_workers),
        "--skip-demo",
    ]
    if binarize:
        cmd.append("--binarize")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_file.write(proc.stdout)

    output_pt_path = ""
    output_pkl_path = ""
    for line in proc.stdout.splitlines():
        if "SUCCESS: PyTorch dataset bundle saved to:" in line:
            output_pt_path = line.split("SUCCESS: PyTorch dataset bundle saved to:", 1)[1].strip()
        elif "SUCCESS: Python pickle file saved to:" in line:
            output_pkl_path = line.split("SUCCESS: Python pickle file saved to:", 1)[1].strip()

    return {
        "exit_code": int(proc.returncode),
        "seed_offset": int(seed_offset),
        "n_struct": int(n_struct),
        "output_pt_path": output_pt_path,
        "output_pkl_path": output_pkl_path,
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generate_dispersion_dataset_Han_Alex.py in seed-shifted batches.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--total-samples", type=int, default=24000, help="Total training samples to generate.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Structures per batch.")
    parser.add_argument("--start-seed-offset", type=int, default=0, help="Seed offset for first training batch.")
    parser.add_argument("--run-validation", action="store_true", help="Generate a validation batch after training batches.")
    parser.add_argument("--validation-size", type=int, default=1000, help="Validation structures to generate.")
    parser.add_argument("--validation-seed-offset", type=int, default=24000, help="Seed offset for validation batch.")
    parser.add_argument("--binarize", action="store_true", help="Generate binarized designs.")
    parser.add_argument("--parallel-workers", type=int, default=16, help="Worker processes per batch run (forwarded to generator).")
    parser.add_argument(
        "--skip-uniform-encoding",
        action="store_true",
        help="Do not write eigenfrequency_uniform_full.pt after each successful batch.",
    )
    parser.add_argument(
        "--uniform-patch-size",
        type=int,
        default=32,
        help="Side length for encode_eigenfrequency_uniform_torch (default 32).",
    )
    parser.add_argument(
        "--uniform-hist-seed",
        type=int,
        default=0,
        help="RNG seed for subsampling values in hist_eigenfrequency_uniform_full.png.",
    )
    parser.add_argument(
        "--skip-fft-encoding",
        action="store_true",
        help="Do not write eigenfrequency_fft_full.pt after each successful batch.",
    )
    parser.add_argument(
        "--fft-wavelet-size",
        type=int,
        default=32,
        help="Patch side length for embed_eigenfrequency_wavelet (default 32).",
    )
    parser.add_argument(
        "--fft-hist-seed",
        type=int,
        default=0,
        help="RNG seed for subsampling values in hist_eigenfrequency_fft_full.png.",
    )
    parser.add_argument(
        "--fft-decode-check-seed",
        type=int,
        default=0,
        help="Seed for random decode spot-check (up to 5 indices).",
    )
    args = parser.parse_args()

    if args.total_samples % args.batch_size != 0:
        raise ValueError("--total-samples must be divisible by --batch-size.")

    repo_root = Path(args.repo_root).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = repo_root / "OUTPUT" / f"batched_generation_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "run_root": str(run_root),
        "repo_root": str(repo_root),
        "batch_size": int(args.batch_size),
        "total_samples": int(args.total_samples),
        "start_seed_offset": int(args.start_seed_offset),
        "binarize": bool(args.binarize),
        "parallel_workers": int(args.parallel_workers),
        "skip_uniform_encoding": bool(args.skip_uniform_encoding),
        "uniform_patch_size": int(args.uniform_patch_size),
        "uniform_hist_seed": int(args.uniform_hist_seed),
        "skip_fft_encoding": bool(args.skip_fft_encoding),
        "fft_wavelet_size": int(args.fft_wavelet_size),
        "fft_hist_seed": int(args.fft_hist_seed),
        "fft_decode_check_seed": int(args.fft_decode_check_seed),
        "train_batches": [],
        "validation_batch": None,
    }

    n_batches = args.total_samples // args.batch_size
    for batch_idx in range(n_batches):
        seed_offset = args.start_seed_offset + batch_idx * args.batch_size
        log_path = run_root / "logs" / f"train_batch_{batch_idx:03d}.log"
        result = _run_one_batch(
            repo_root=repo_root,
            n_struct=args.batch_size,
            seed_offset=seed_offset,
            binarize=args.binarize,
            log_path=log_path,
            parallel_workers=args.parallel_workers,
        )
        result["batch_idx"] = int(batch_idx)
        print(
            f"[DONE] train batch {batch_idx + 1}/{n_batches} "
            f"(seed_offset={seed_offset}) exit_code={result['exit_code']} "
            f"pt_out={result.get('output_pt_path') or 'N/A'}"
        )
        if (
            not args.skip_uniform_encoding
            and result["exit_code"] == 0
            and (result.get("output_pt_path") or "").strip()
        ):
            u = write_eigenfrequency_uniform_full(
                Path(result["output_pt_path"].strip()),
                patch_size=args.uniform_patch_size,
                hist_seed=args.uniform_hist_seed + batch_idx,
            )
            result["uniform_encoding"] = u
            if u.get("ok"):
                msg = (
                    f"[UNIFORM] {u['shape_in']} -> {u['shape_out']} -> {u['dst']}"
                )
                if u.get("n_nonpositive_clamped"):
                    msg += f" (clamped {u['n_nonpositive_clamped']} non-positive)"
                if u.get("hist_png"):
                    msg += f" hist={u['hist_png']}"
                print(msg)
                if u.get("hist_error"):
                    print(f"[WARN] uniform histogram: {u['hist_error']}")
            else:
                print(f"[WARN] uniform encoding failed: {u.get('error')}")
        if (
            not args.skip_fft_encoding
            and result["exit_code"] == 0
            and (result.get("output_pt_path") or "").strip()
        ):
            f = write_eigenfrequency_fft_full(
                Path(result["output_pt_path"].strip()),
                wavelet_size=args.fft_wavelet_size,
                hist_seed=args.fft_hist_seed + batch_idx,
                decode_check_seed=args.fft_decode_check_seed,
            )
            result["fft_encoding"] = f
            if f.get("ok"):
                msg = (
                    f"[FFT] {f['shape_in']} -> {f['shape_out']} -> {f['dst']} "
                    f"mean_rel_err={f['mean_rel_err']:.6e} max_rel_err={f['max_rel_err']:.6e}"
                )
                if f.get("hist_png"):
                    msg += f" hist={f['hist_png']}"
                print(msg)
                if f.get("hist_error"):
                    print(f"[WARN] FFT histogram: {f['hist_error']}")
            else:
                print(f"[WARN] FFT encoding failed: {f.get('error')}")
        manifest["train_batches"].append(result)
        if result["exit_code"] != 0:
            print(f"[STOP] train batch {batch_idx + 1} failed. See log: {result['log_path']}")
            break

    if args.run_validation and all(b["exit_code"] == 0 for b in manifest["train_batches"]):
        v_log = run_root / "logs" / "validation_batch.log"
        v_result = _run_one_batch(
            repo_root=repo_root,
            n_struct=args.validation_size,
            seed_offset=args.validation_seed_offset,
            binarize=args.binarize,
            log_path=v_log,
            parallel_workers=args.parallel_workers,
        )
        v_result["batch_idx"] = "validation"
        print(
            f"[DONE] validation batch (seed_offset={args.validation_seed_offset}) "
            f"exit_code={v_result['exit_code']} "
            f"pt_out={v_result.get('output_pt_path') or 'N/A'}"
        )
        if (
            not args.skip_uniform_encoding
            and v_result["exit_code"] == 0
            and (v_result.get("output_pt_path") or "").strip()
        ):
            u = write_eigenfrequency_uniform_full(
                Path(v_result["output_pt_path"].strip()),
                patch_size=args.uniform_patch_size,
                hist_seed=args.uniform_hist_seed + 10_000,
            )
            v_result["uniform_encoding"] = u
            if u.get("ok"):
                msg = (
                    f"[UNIFORM] validation {u['shape_in']} -> {u['shape_out']} -> {u['dst']}"
                )
                if u.get("n_nonpositive_clamped"):
                    msg += f" (clamped {u['n_nonpositive_clamped']} non-positive)"
                if u.get("hist_png"):
                    msg += f" hist={u['hist_png']}"
                print(msg)
                if u.get("hist_error"):
                    print(f"[WARN] validation uniform histogram: {u['hist_error']}")
            else:
                print(f"[WARN] validation uniform encoding failed: {u.get('error')}")
        if (
            not args.skip_fft_encoding
            and v_result["exit_code"] == 0
            and (v_result.get("output_pt_path") or "").strip()
        ):
            f = write_eigenfrequency_fft_full(
                Path(v_result["output_pt_path"].strip()),
                wavelet_size=args.fft_wavelet_size,
                hist_seed=args.fft_hist_seed + 10_000,
                decode_check_seed=args.fft_decode_check_seed,
            )
            v_result["fft_encoding"] = f
            if f.get("ok"):
                msg = (
                    f"[FFT] validation {f['shape_in']} -> {f['shape_out']} -> {f['dst']} "
                    f"mean_rel_err={f['mean_rel_err']:.6e} max_rel_err={f['max_rel_err']:.6e}"
                )
                if f.get("hist_png"):
                    msg += f" hist={f['hist_png']}"
                print(msg)
                if f.get("hist_error"):
                    print(f"[WARN] validation FFT histogram: {f['hist_error']}")
            else:
                print(f"[WARN] validation FFT encoding failed: {f.get('error')}")
        manifest["validation_batch"] = v_result

    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    successful_train = sum(1 for b in manifest["train_batches"] if b["exit_code"] == 0)
    print(
        f"[SUMMARY] successful_train_batches={successful_train}/{n_batches} "
        f"run_root={run_root}"
    )
    print(json.dumps({"manifest": str(manifest_path), "run_root": str(run_root)}, indent=2))


if __name__ == "__main__":
    main()

