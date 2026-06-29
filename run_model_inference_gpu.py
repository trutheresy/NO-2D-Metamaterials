"""
GPU inference: run a trained FNO on a dataset and save dense predictions.

Same I/O contract as the original run_model_inference.py (I3O1 / I3O4 / I3O5).
Uses CUDA by default; batches are assembled on GPU with waveforms/bands resident on device.

For CPU / many-core RAM-resident inference use run_model_inference_cpu.py instead.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import torch
from tqdm import tqdm

import model_inference_common as mic


def resolve_gpu_device(cuda_device: int, allow_cpu_fallback: bool) -> torch.device:
    if not torch.cuda.is_available():
        if allow_cpu_fallback:
            print("Warning: CUDA unavailable; falling back to CPU.", file=sys.stderr)
            return torch.device("cpu")
        raise RuntimeError("CUDA is unavailable. Use run_model_inference_cpu.py or pass --allow-cpu-fallback.")
    if cuda_device < 0 or cuda_device >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid --cuda-device {cuda_device} (visible devices: {torch.cuda.device_count()})."
        )
    torch.cuda.set_device(cuda_device)
    return torch.device(f"cuda:{cuda_device}")


def run_inference_gpu(
    model: torch.nn.Module,
    geometries: torch.Tensor,
    waveforms: torch.Tensor,
    band_ffts: torch.Tensor,
    device: torch.device,
    batch_size: int,
    out_channels: int,
    pin_memory: bool,
) -> torch.Tensor:
    model.eval()
    n_geometries = geometries.shape[0]
    n_waveforms = waveforms.shape[0]
    n_bands = band_ffts.shape[0]
    total_samples = n_geometries * n_waveforms * n_bands
    print(f"Running GPU inference on {total_samples} samples...")
    print(f"  - {n_geometries} geometries, {n_waveforms} waveforms, {n_bands} bands")
    print(f"  - batch_size={batch_size}, pin_memory={pin_memory}")

    output_dtype = torch.float16
    outputs = torch.empty((total_samples, out_channels, 32, 32), dtype=output_dtype)

    use_non_blocking = device.type == "cuda" and pin_memory
    waveforms_device = waveforms.to(device, non_blocking=use_non_blocking).float()
    band_ffts_device = band_ffts.to(device, non_blocking=use_non_blocking).float()
    batch_tensor = torch.empty((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    batch_indices = [0] * batch_size
    batch_idx = 0
    sample_count = 0

    with torch.inference_mode():
        for geom_idx in tqdm(range(n_geometries), desc="Geometries", unit="geom"):
            geometry_device = geometries[geom_idx].to(device, non_blocking=use_non_blocking).float()
            for wave_idx in range(n_waveforms):
                for band_idx in range(n_bands):
                    batch_tensor[batch_idx, 0, :, :] = geometry_device
                    batch_tensor[batch_idx, 1, :, :] = waveforms_device[wave_idx]
                    batch_tensor[batch_idx, 2, :, :] = band_ffts_device[band_idx]
                    batch_indices[batch_idx] = mic.compute_output_index(
                        geom_idx, wave_idx, band_idx, n_waveforms, n_bands
                    )
                    batch_idx += 1
                    if batch_idx >= batch_size:
                        batch_outputs = model(batch_tensor)
                        batch_outputs_cpu = batch_outputs.detach().cpu().to(output_dtype)
                        indices_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
                        outputs[indices_tensor] = batch_outputs_cpu
                        del batch_outputs, batch_outputs_cpu, indices_tensor
                        batch_idx = 0
                        sample_count += batch_size
                        if sample_count % (batch_size * 10) == 0:
                            torch.cuda.empty_cache()
            del geometry_device
            torch.cuda.empty_cache()

        if batch_idx > 0:
            batch_outputs = model(batch_tensor[:batch_idx])
            batch_outputs_cpu = batch_outputs.detach().cpu().to(output_dtype)
            indices_tensor = torch.as_tensor(batch_indices[:batch_idx], dtype=torch.long)
            outputs[indices_tensor] = batch_outputs_cpu
            del batch_outputs, batch_outputs_cpu, indices_tensor

    del waveforms_device, band_ffts_device, batch_tensor
    torch.cuda.empty_cache()
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
    p.add_argument("--batch_size", type=int, default=256, help="GPU batch size (default: 256).")
    p.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (default: 0).")
    p.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="If CUDA is missing, run on CPU instead of exiting (not recommended).",
    )
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pinned CPU tensors for faster H2D copies (default: on).",
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_gpu_device(args.cuda_device, args.allow_cpu_fallback)
    print(f"Using device: {device}")

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
    selected_indices, geom_mode = mic.select_geometry_subset(
        args.geometries, args.geometry_seed, total_geometries
    )
    if selected_indices is not None:
        print(f"Geometry selection: mode={geom_mode}, {len(selected_indices)}/{total_geometries}")
        geometries = geometries[torch.as_tensor(selected_indices, dtype=torch.long)]
        gc.collect()
    else:
        print(f"Geometry selection: mode=all, {total_geometries} geometries")

    print("Building model...")
    model = mic.FourierNeuralOperator(
        modes_height=modes_h,
        modes_width=modes_w,
        hidden_channels=model_hidden,
        n_layers=model_layers,
        out_channels=out_channels,
    )
    mic.load_model_weights(model, args.model_path, device)

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    output_path, run_out_dir, timestamp = mic.resolve_output_paths(args, model_name, case)

    if args.pin_memory and device.type == "cuda":
        geometries = geometries.pin_memory()
        waveforms = waveforms.pin_memory()
        band_ffts = band_ffts.pin_memory()

    outputs = run_inference_gpu(
        model,
        geometries,
        waveforms,
        band_ffts,
        device,
        batch_size=args.batch_size,
        out_channels=out_channels,
        pin_memory=bool(args.pin_memory),
    )

    n_waveforms = waveforms.shape[0]
    n_bands = band_ffts.shape[0]
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
            "script": "run_model_inference_gpu.py",
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
            "extra_notes": f"cuda_device={args.cuda_device}, pin_memory={args.pin_memory}",
        },
    )
    print(f"Done! Outputs saved in: {run_out_dir}")


if __name__ == "__main__":
    main()
