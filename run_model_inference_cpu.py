"""
CPU inference: run a trained FNO on a dataset and save dense predictions.

Default mode is data-parallel across processes, which is the right pattern for a
small model evaluated over millions of independent (geometry, wavevector, band)
inputs on a many-core CPU. A single process with intra-op threading scales poorly
here because the forward is a chain of tiny ops (32x32 FFTs, per-mode complex
matmuls) and every op puts a fork/join barrier that waits on the slowest thread.
On a hybrid CPU (Intel P+E cores) that barrier repeatedly stalls the fast P-cores
on the slow E-cores, capping useful utilization.

The parallel mode instead runs N single-threaded worker processes (no intra-op
barriers), each pinned to a distinct logical core, pulling geometry chunks from a
shared dynamic work queue (so fast cores naturally process more chunks than slow
ones) and writing results straight into one shared-memory output buffer. Because
the output is geometry-major, no merge step is needed.

Modes (via --workers):
  - 0 (default): auto = number of physical cores -> parallel
  - 1          : legacy single-process path (--preload-inputs / streaming)
  - N          : N parallel workers

For CUDA inference use run_model_inference_gpu.py instead.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory

import numpy as np
import torch
from tqdm import tqdm

import model_inference_common as mic


def configure_cpu_threads(n_threads: int) -> int:
    n = max(1, int(n_threads))
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)
    print(f"CPU threads: intra_op={n}, inter_op=1")
    return n


def default_cpu_threads() -> int:
    count = os.cpu_count() or 1
    return min(count, 32)


def run_inference_cpu_streaming(
    model: torch.nn.Module,
    geometries: torch.Tensor,
    waveforms: torch.Tensor,
    band_ffts: torch.Tensor,
    batch_size: int,
    out_channels: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Assemble batches on the fly without materializing the full (N, 3, 32, 32) input tensor."""
    model.eval()
    n_geometries = geometries.shape[0]
    n_waveforms = waveforms.shape[0]
    n_bands = band_ffts.shape[0]
    total_samples = n_geometries * n_waveforms * n_bands
    print(f"Running CPU streaming inference on {total_samples} samples (batch_size={batch_size})...")
    outputs = torch.empty((total_samples, out_channels, 32, 32), dtype=output_dtype)
    geoms_f = geometries.float()
    waves_f = waveforms.float()
    bands_f = band_ffts.float()
    batch_tensor = torch.empty((batch_size, 3, 32, 32), dtype=torch.float32)
    batch_indices = [0] * batch_size
    batch_idx = 0

    with torch.inference_mode():
        for geom_idx in tqdm(range(n_geometries), desc="Geometries", unit="geom"):
            geom = geoms_f[geom_idx]
            for wave_idx in range(n_waveforms):
                wave = waves_f[wave_idx]
                for band_idx in range(n_bands):
                    batch_tensor[batch_idx, 0, :, :] = geom
                    batch_tensor[batch_idx, 1, :, :] = wave
                    batch_tensor[batch_idx, 2, :, :] = bands_f[band_idx]
                    batch_indices[batch_idx] = mic.compute_output_index(
                        geom_idx, wave_idx, band_idx, n_waveforms, n_bands
                    )
                    batch_idx += 1
                    if batch_idx >= batch_size:
                        pred = model(batch_tensor)
                        outputs[torch.as_tensor(batch_indices, dtype=torch.long)] = pred.to(output_dtype)
                        batch_idx = 0
        if batch_idx > 0:
            pred = model(batch_tensor[:batch_idx])
            outputs[torch.as_tensor(batch_indices[:batch_idx], dtype=torch.long)] = pred.to(output_dtype)
    return outputs


def run_inference_cpu(
    model: torch.nn.Module,
    inputs_cpu: torch.Tensor,
    batch_size: int,
    out_channels: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    model.eval()
    n_total = inputs_cpu.shape[0]
    print(f"Running CPU inference on {n_total} samples (batch_size={batch_size})...")
    outputs = torch.empty((n_total, out_channels, 32, 32), dtype=output_dtype)

    with torch.inference_mode():
        for start in tqdm(range(0, n_total, batch_size), desc="Batches", unit="batch"):
            end = min(start + batch_size, n_total)
            batch = inputs_cpu[start:end]
            pred = model(batch)
            outputs[start:end] = pred.to(output_dtype)

    return outputs


def default_workers() -> int:
    """Default worker count = number of physical cores (falls back to logical)."""
    n = None
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
    except Exception:
        n = None
    if not n:
        n = os.cpu_count() or 1
    return max(1, int(n))


def physical_core_logical_order(n_logical: int | None = None) -> list[int]:
    """Order logical CPU indices to spread workers across distinct physical cores first.

    On Intel hybrid parts the P-cores' hyperthread siblings are adjacent low indices,
    so taking even indices before odd ones tends to hit separate physical cores before
    doubling up on a core's sibling thread. Degrades gracefully on other topologies.
    """
    if n_logical is None:
        n_logical = os.cpu_count() or 1
    evens = list(range(0, n_logical, 2))
    odds = list(range(1, n_logical, 2))
    return evens + odds


def _numpy_dtype(out_dtype_str: str):
    return np.float16 if out_dtype_str == "float16" else np.float32


def model_cfg_to_kwargs(model_cfg: dict) -> dict:
    """Map the compact model_cfg dict to FourierNeuralOperator constructor kwargs."""
    return {
        "modes_height": model_cfg["modes_h"],
        "modes_width": model_cfg["modes_w"],
        "hidden_channels": model_cfg["hidden"],
        "n_layers": model_cfg["layers"],
        "out_channels": model_cfg["out_channels"],
    }


def _parallel_inference_worker(
    worker_id,
    affinity_logical,
    shm_name,
    out_shape,
    out_dtype_str,
    chunk_queue,
    progress,
    geometries,
    waveforms,
    band_ffts,
    model_cfg,
    model_path,
    batch_size,
    use_compile,
):
    """One single-threaded worker: pull geometry chunks, run forward, write to shared output."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if affinity_logical is not None:
        try:
            import psutil

            psutil.Process().cpu_affinity([int(affinity_logical)])
        except Exception:
            pass

    out_dtype = torch.float16 if out_dtype_str == "float16" else torch.float32
    shm = shared_memory.SharedMemory(name=shm_name)
    out_np = np.ndarray(out_shape, dtype=_numpy_dtype(out_dtype_str), buffer=shm.buf)

    model = mic.FourierNeuralOperator(
        modes_height=model_cfg["modes_h"],
        modes_width=model_cfg["modes_w"],
        hidden_channels=model_cfg["hidden"],
        n_layers=model_cfg["layers"],
        out_channels=model_cfg["out_channels"],
    )
    mic.load_model_weights(model, model_path, torch.device("cpu"))
    model.eval()
    if use_compile:
        try:
            model = torch.compile(model)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker {worker_id}] torch.compile failed, continuing uncompiled: {exc}")

    combos = int(waveforms.shape[0]) * int(band_ffts.shape[0])
    try:
        with torch.inference_mode():
            while True:
                item = chunk_queue.get()
                if item is None:
                    break
                lstart, lend = item
                inputs = mic.materialize_inputs_cpu(geometries[lstart:lend], waveforms, band_ffts)
                base = lstart * combos
                n = inputs.shape[0]
                for s in range(0, n, batch_size):
                    e = min(s + batch_size, n)
                    pred = model(inputs[s:e]).to(out_dtype)
                    out_np[base + s : base + e] = pred.numpy()
                del inputs
                with progress.get_lock():
                    progress.value += int(lend - lstart)
    finally:
        del out_np
        shm.close()


def run_inference_cpu_parallel(
    geometries: torch.Tensor,
    waveforms: torch.Tensor,
    band_ffts: torch.Tensor,
    *,
    n_workers: int,
    chunk_geoms: int,
    batch_size: int,
    out_channels: int,
    output_dtype: torch.dtype,
    model_cfg: dict,
    model_path: str,
    use_compile: bool,
    pin: bool,
) -> torch.Tensor:
    """Data-parallel CPU inference using single-threaded, core-pinned worker processes."""
    total_geoms = int(geometries.shape[0])
    combos = int(waveforms.shape[0]) * int(band_ffts.shape[0])
    total_samples = total_geoms * combos
    out_dtype_str = "float16" if output_dtype == torch.float16 else "float32"
    itemsize = 2 if output_dtype == torch.float16 else 4
    nbytes = total_samples * out_channels * 32 * 32 * itemsize
    out_shape = (total_samples, out_channels, 32, 32)

    n_workers = max(1, min(int(n_workers), total_geoms))
    chunk_geoms = max(1, int(chunk_geoms))
    print(
        f"Parallel CPU inference: {n_workers} single-threaded workers, "
        f"chunk={chunk_geoms} geoms, batch_size={batch_size}, pin={pin}, compile={use_compile}"
    )

    # Make workers' BLAS/FFT single-threaded too (inherited at spawn).
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    ctx = mp.get_context("spawn")
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        queue = ctx.Queue()
        chunks = [(s, min(s + chunk_geoms, total_geoms)) for s in range(0, total_geoms, chunk_geoms)]
        for chunk in chunks:
            queue.put(chunk)
        for _ in range(n_workers):
            queue.put(None)
        print(f"  {len(chunks)} chunks queued across {n_workers} workers")

        progress = ctx.Value("i", 0)
        order = physical_core_logical_order() if pin else None
        procs = []
        for w in range(n_workers):
            aff = order[w % len(order)] if order else None
            proc = ctx.Process(
                target=_parallel_inference_worker,
                args=(
                    w,
                    aff,
                    shm.name,
                    out_shape,
                    out_dtype_str,
                    queue,
                    progress,
                    geometries,
                    waveforms,
                    band_ffts,
                    model_cfg,
                    model_path,
                    batch_size,
                    use_compile,
                ),
            )
            proc.start()
            procs.append(proc)

        with tqdm(total=total_geoms, desc="Geometries", unit="geom") as pbar:
            last = 0
            while any(p.is_alive() for p in procs):
                cur = int(progress.value)
                if cur > last:
                    pbar.update(cur - last)
                    last = cur
                time.sleep(1.0)
            cur = int(progress.value)
            if cur > last:
                pbar.update(cur - last)

        for proc in procs:
            proc.join()
        failed = [w for w, p in enumerate(procs) if p.exitcode not in (0, None)]
        if failed:
            codes = [procs[w].exitcode for w in failed]
            raise RuntimeError(f"Parallel workers failed: ids={failed}, exit_codes={codes}")

        out_np = np.ndarray(out_shape, dtype=_numpy_dtype(out_dtype_str), buffer=shm.buf)
        outputs = torch.from_numpy(out_np).clone()
    finally:
        shm.close()
        shm.unlink()
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=str, required=True, help="Path to checkpoint (.pth).")
    p.add_argument("--input_dataset_path", type=str, required=True, help="Dataset or base directory.")
    p.add_argument("--output_path", type=str, default=None, help="Explicit output .pt path.")
    p.add_argument(
        "--inference-root",
        type=str,
        default=None,
        help="Root for default INFERENCE/<model>_<timestamp>/ output (default: repo/INFERENCE).",
    )
    p.add_argument("--batch_size", type=int, default=1024, help="Per-worker forward batch size (default: 1024).")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallel worker processes. 0=auto (physical cores, recommended), "
            "1=single-process legacy path, N=N workers."
        ),
    )
    p.add_argument(
        "--chunk-geometries",
        type=int,
        default=4,
        help="Geometries per work-queue chunk for dynamic load balancing (default: 4).",
    )
    p.add_argument(
        "--pin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin each worker to a distinct logical core (default: on).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model in torch.compile inside each worker (default: off).",
    )
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Intra-op threads for the single-process path (--workers 1 only).",
    )
    p.add_argument(
        "--preload-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize full (N, 3, 32, 32) inputs in RAM before inference (default: on).",
    )
    p.add_argument(
        "--output-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Saved prediction dtype (default: float16).",
    )
    p.add_argument(
        "--dataset_structure",
        type=str,
        default="single",
        choices=["single", "multiple"],
    )
    p.add_argument(
        "--case",
        type=str,
        default="auto",
        choices=["auto", "I3O1", "I3O4", "I3O5"],
    )
    p.add_argument("--geometries", nargs="+", type=int, default=None)
    p.add_argument("--geometry-seed", type=int, default=0)
    p.add_argument(
        "--geometry-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help=(
            "Half-open geometry index range [START, END) to run on. "
            "Takes precedence over --geometries."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    n_workers = default_workers() if args.workers == 0 else max(1, int(args.workers))
    output_dtype = torch.float16 if args.output_dtype == "float16" else torch.float32
    print(f"Using device: {device} | workers={n_workers}")

    dataset_paths = mic.resolve_dataset_paths(args.input_dataset_path, args.dataset_structure)
    print(f"Found {len(dataset_paths)} dataset(s)")

    state_dict = mic.normalize_fno_state_dict(mic.load_raw_state_dict(args.model_path, map_location="cpu"))
    if args.case == "auto":
        case, out_channels = mic.infer_case_from_state_dict(state_dict)
        print(f"Auto-inferred I/O case: {case} (out_channels={out_channels})")
    else:
        case = args.case
        out_channels = mic.CASE_OUT_CHANNELS[case]
    channel_labels = mic.CASE_CHANNEL_LABELS[case]
    del state_dict

    run_dir = os.path.dirname(os.path.abspath(args.model_path))
    hp = mic.hparams_from_resolved_config(run_dir)
    num_geometries_per_dataset = mic.infer_num_geometries_per_dataset(dataset_paths)
    if hp.get("hidden_channels") is not None and hp.get("layers") is not None:
        model_hidden = int(hp["hidden_channels"])
        model_layers = int(hp["layers"])
        arch_source = "resolved_config.json"
    else:
        state_dict = mic.normalize_fno_state_dict(mic.load_raw_state_dict(args.model_path, map_location="cpu"))
        model_hidden, model_layers = mic.infer_model_architecture(state_dict, out_channels=out_channels)
        arch_source = "state_dict inference"
        del state_dict
    modes_h = int(hp.get("modes_height", 32))
    modes_w = int(hp.get("modes_width", 32))

    geometries, waveforms, band_ffts = mic.load_input_data(dataset_paths, num_geometries_per_dataset)
    total_geometries = int(geometries.shape[0])
    if args.geometry_range is not None:
        start, end = int(args.geometry_range[0]), int(args.geometry_range[1])
        if not (0 <= start < end <= total_geometries):
            raise ValueError(
                f"--geometry-range [{start}, {end}) out of bounds for {total_geometries} geometries."
            )
        selected_indices = list(range(start, end))
        geom_mode = "range"
    else:
        selected_indices, geom_mode = mic.select_geometry_subset(
            args.geometries, args.geometry_seed, total_geometries
        )
    if selected_indices is not None:
        print(f"Geometry selection: mode={geom_mode}, {len(selected_indices)}/{total_geometries}")
        geometries = geometries[torch.as_tensor(selected_indices, dtype=torch.long)]
        gc.collect()
    else:
        print(f"Geometry selection: mode=all, {total_geometries} geometries")

    n_waveforms = waveforms.shape[0]
    n_bands = band_ffts.shape[0]
    n_samples = geometries.shape[0] * n_waveforms * n_bands
    input_gb = n_samples * 3 * 32 * 32 * 4 / (1024**3)
    output_gb = n_samples * out_channels * 32 * 32 * (2 if output_dtype == torch.float16 else 4) / (1024**3)
    print(f"Estimated RAM: inputs~{input_gb:.2f} GB, outputs~{output_gb:.2f} GB")

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    output_path, run_out_dir, timestamp = mic.resolve_output_paths(args, model_name, case)
    model_cfg = {
        "modes_h": modes_h,
        "modes_w": modes_w,
        "hidden": model_hidden,
        "layers": model_layers,
        "out_channels": out_channels,
    }

    if n_workers > 1:
        notes_mode = (
            f"parallel workers={n_workers}, chunk_geometries={args.chunk_geometries}, "
            f"pin={args.pin}, compile={args.compile}"
        )
        outputs = run_inference_cpu_parallel(
            geometries,
            waveforms,
            band_ffts,
            n_workers=n_workers,
            chunk_geoms=args.chunk_geometries,
            batch_size=args.batch_size,
            out_channels=out_channels,
            output_dtype=output_dtype,
            model_cfg=model_cfg,
            model_path=args.model_path,
            use_compile=args.compile,
            pin=args.pin,
        )
        del geometries, waveforms, band_ffts
    elif args.preload_inputs:
        n_threads = configure_cpu_threads(args.cpu_threads if args.cpu_threads > 0 else default_cpu_threads())
        notes_mode = f"single-process preload, cpu_threads={n_threads}"
        print("Materializing inputs on CPU...")
        inputs_cpu = mic.materialize_inputs_cpu(geometries, waveforms, band_ffts)
        del geometries, waveforms, band_ffts
        gc.collect()
        print("Building model...")
        model = mic.FourierNeuralOperator(**model_cfg_to_kwargs(model_cfg))
        mic.load_model_weights(model, args.model_path, device)
        outputs = run_inference_cpu(
            model,
            inputs_cpu,
            batch_size=args.batch_size,
            out_channels=out_channels,
            output_dtype=output_dtype,
        )
        del inputs_cpu, model
    else:
        n_threads = configure_cpu_threads(args.cpu_threads if args.cpu_threads > 0 else default_cpu_threads())
        notes_mode = f"single-process streaming, cpu_threads={n_threads}"
        print("Building model...")
        model = mic.FourierNeuralOperator(**model_cfg_to_kwargs(model_cfg))
        mic.load_model_weights(model, args.model_path, device)
        outputs = run_inference_cpu_streaming(
            model,
            geometries,
            waveforms,
            band_ffts,
            batch_size=args.batch_size,
            out_channels=out_channels,
            output_dtype=output_dtype,
        )
        del model
    gc.collect()

    mic.save_outputs(outputs, output_path, channel_labels, n_waveforms, n_bands)

    if selected_indices is not None:
        idx_path = os.path.join(run_out_dir, "selected_geometry_indices.pt")
        torch.save(torch.as_tensor(selected_indices, dtype=torch.long), idx_path)
        print(f"Saved selected geometry indices: {idx_path}")

    combos = n_waveforms * n_bands
    doc_path = os.path.join(run_out_dir, f"inference_info_{model_name}_{timestamp}.txt")
    mic.write_inference_doc(
        doc_path,
        {
            "timestamp": timestamp,
            "script": "run_model_inference_cpu.py",
            "model_path": os.path.abspath(args.model_path),
            "model_name": model_name,
            "case": case,
            "out_channels": out_channels,
            "channel_labels": channel_labels,
            "hidden_channels": model_hidden,
            "n_layers": model_layers,
            "modes_height": modes_h,
            "modes_width": modes_w,
            "arch_source": arch_source,
            "device": str(device),
            "batch_size": args.batch_size,
            "dataset_structure": args.dataset_structure,
            "input_dataset_path": os.path.abspath(args.input_dataset_path),
            "dataset_paths": [os.path.abspath(p) for p in dataset_paths],
            "num_geometries_per_dataset": num_geometries_per_dataset,
            "geometry_mode": geom_mode,
            "total_geometries_available": total_geometries,
            "geometry_seed": int(args.geometry_seed) if geom_mode == "random" else None,
            "selected_indices": selected_indices,
            "n_waveforms": n_waveforms,
            "n_bands": n_bands,
            "n_geometries_total": int(outputs.shape[0] // combos) if combos else 0,
            "total_samples": int(outputs.shape[0]),
            "output_filename": os.path.basename(output_path),
            "output_shape": tuple(outputs.shape),
            "output_dtype": str(outputs.dtype),
            "extra_notes": f"{notes_mode}, output_dtype={args.output_dtype}",
        },
    )
    print(f"Done! Outputs saved in: {run_out_dir}")


if __name__ == "__main__":
    main()
