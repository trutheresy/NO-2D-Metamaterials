"""Shared helpers for run_model_inference_gpu.py and run_model_inference_cpu.py."""

from __future__ import annotations

import gc
import json
import os
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

try:
    from neuralop.models import FNO2d
except ImportError as e:
    raise ImportError(
        "neuralop is required. Install with: pip install neuraloperator"
    ) from e


CASE_OUT_CHANNELS = {"I3O1": 1, "I3O4": 4, "I3O5": 5}
OUT_CHANNELS_TO_CASE = {v: k for k, v in CASE_OUT_CHANNELS.items()}
CASE_CHANNEL_LABELS = {
    "I3O1": ["eigenfrequency"],
    "I3O4": ["disp_x_real", "disp_x_imag", "disp_y_real", "disp_y_imag"],
    "I3O5": ["eigenfrequency", "disp_x_real", "disp_x_imag", "disp_y_real", "disp_y_imag"],
}


def load_raw_state_dict(model_path, map_location="cpu"):
    obj = torch.load(model_path, map_location=map_location, weights_only=False)
    if hasattr(obj, "state_dict") and not isinstance(obj, dict):
        return obj.state_dict()
    return obj


def normalize_fno_state_dict(state_dict):
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    for prefix in ("fno.", "model."):
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def infer_out_channels_from_state_dict(state_dict):
    proj_pat = re.compile(r"(?:^|\.)projection\.fcs\.(\d+)\.weight$")
    best_idx, best_oc = -1, None
    for k, v in state_dict.items():
        if not hasattr(v, "shape") or len(v.shape) < 1:
            continue
        m = proj_pat.search(k)
        if m and int(m.group(1)) > best_idx:
            best_idx = int(m.group(1))
            best_oc = int(v.shape[0])
    if best_oc is None:
        fcs_pat = re.compile(r"\.fcs\.(\d+)\.weight$")
        for k, v in state_dict.items():
            if not hasattr(v, "shape") or len(v.shape) < 1:
                continue
            m = fcs_pat.search(k)
            if m and int(m.group(1)) > best_idx:
                best_idx = int(m.group(1))
                best_oc = int(v.shape[0])
    if best_oc is None:
        raise ValueError(
            "Could not locate an FNO output projection weight in the checkpoint "
            "to infer out_channels. Pass --case explicitly."
        )
    return best_oc


def infer_case_from_state_dict(state_dict):
    out_channels = infer_out_channels_from_state_dict(state_dict)
    case = OUT_CHANNELS_TO_CASE.get(out_channels)
    if case is None:
        raise ValueError(
            f"Inferred out_channels={out_channels}, which does not match a supported "
            f"case (expected one of {sorted(OUT_CHANNELS_TO_CASE)}). Pass --case explicitly."
        )
    return case, out_channels


def select_geometry_subset(geometries_arg, geometry_seed, total_geometries):
    if not geometries_arg:
        return None, "all"

    if len(geometries_arg) == 1:
        n = int(geometries_arg[0])
        if n <= 0:
            raise ValueError(f"--geometries single integer N must be positive (got {n}).")
        if n > total_geometries:
            print(
                f"Warning: requested {n} random geometries but only {total_geometries} "
                f"available; using all {total_geometries}."
            )
            n = total_geometries
        rng = random.Random(int(geometry_seed))
        idx = sorted(rng.sample(range(total_geometries), n))
        return idx, "random"

    idx_raw = [int(i) for i in geometries_arg]
    out_of_range = [i for i in idx_raw if i < 0 or i >= total_geometries]
    if out_of_range:
        raise ValueError(
            f"--geometries indices out of range [0, {total_geometries}): {out_of_range[:10]}"
            + (" ..." if len(out_of_range) > 10 else "")
        )
    seen: set[int] = set()
    idx: list[int] = []
    for i in idx_raw:
        if i not in seen:
            seen.add(i)
            idx.append(i)
    return idx, "indices"


class FourierNeuralOperator(nn.Module):
    def __init__(self, modes_height, modes_width, hidden_channels, n_layers, out_channels, in_channels=3):
        super().__init__()
        self.model = FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes_height=modes_height,
            n_modes_width=modes_width,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )

    def forward(self, x):
        return self.model(x)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)


def hparams_from_resolved_config(run_dir):
    rc = Path(run_dir) / "resolved_config.json"
    if not rc.is_file():
        return {}
    try:
        data = json.loads(rc.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    params = data.get("params") or {}
    args_ = data.get("args") or {}
    out = {}
    for key in ("out_channels", "hidden_channels", "layers", "modes_height", "modes_width"):
        if params.get(key) is not None:
            out[key] = params[key]
        elif args_.get(key) is not None:
            out[key] = args_[key]
    return out


def infer_num_geometries_per_dataset(dataset_paths):
    if not dataset_paths:
        raise ValueError("No dataset paths provided")
    first_dataset_path = dataset_paths[0]
    geometries_path = os.path.join(first_dataset_path, "geometries_full.pt")
    if not os.path.exists(geometries_path):
        raise FileNotFoundError(f"Could not find geometries_full.pt in {first_dataset_path}")
    geometries = torch.load(geometries_path, weights_only=False)
    num_geometries = geometries.shape[0]
    del geometries
    gc.collect()
    print(f"Inferred {num_geometries} geometries per dataset from {first_dataset_path}")
    return num_geometries


def infer_model_architecture(state_dict, out_channels=5):
    common_hidden = [128, 256, 512, 64, 32]
    common_layers = [2, 3, 4, 5, 6]
    print("Trying to infer model architecture from state_dict...")
    num_layers = None
    hidden_channels = None
    block_indices: set[int] = set()
    for key in state_dict.keys():
        if "fno" in key.lower():
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part in ["blocks", "layers", "fno_blocks"] and i + 1 < len(parts):
                    try:
                        block_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
                elif part in ["convs", "channel_mlp", "channel_mlp_skips", "fno_skips"] and i + 1 < len(parts):
                    try:
                        block_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
    if block_indices:
        num_layers = max(block_indices) + 1
        print(f"  Found num_layers={num_layers} from state_dict keys (indices: {sorted(block_indices)})")

    potential_hidden_values: list[int] = []
    for key in state_dict.keys():
        if "fno" in key.lower() and "weight" in key:
            shape = state_dict[key].shape
            if len(shape) >= 2:
                potential_hidden = max(shape[:2])
                if 32 <= potential_hidden <= 1024:
                    potential_hidden_values.append(potential_hidden)
    if potential_hidden_values:
        hidden_channels = Counter(potential_hidden_values).most_common(1)[0][0]
        print(f"  Found potential hidden_channels={hidden_channels} from state_dict shapes")

    if num_layers is None or hidden_channels is None:
        print("  Trying to match state_dict keys with common configurations...")
        state_dict_keys = set(state_dict.keys())
        for hidden in common_hidden:
            for layers in common_layers:
                if num_layers is not None and hidden_channels is not None:
                    break
                try:
                    model = FourierNeuralOperator(32, 32, hidden, layers, out_channels)
                    model_keys = set(model.state_dict().keys())
                    match_ratio = len(state_dict_keys & model_keys) / max(len(state_dict_keys), len(model_keys))
                    if match_ratio > 0.8:
                        if num_layers is None:
                            num_layers = layers
                        if hidden_channels is None:
                            hidden_channels = hidden
                        print(
                            f"  Found matching configuration: hidden={hidden}, layers={layers} "
                            f"(match ratio: {match_ratio:.2f})"
                        )
                    del model
                    gc.collect()
                except Exception:
                    continue
            if num_layers is not None and hidden_channels is not None:
                break

    if num_layers is None:
        print("Warning: Could not infer num_layers from state_dict, using default: 4")
        num_layers = 4
    if hidden_channels is None:
        print("Warning: Could not infer hidden_channels from state_dict, using default: 128")
        hidden_channels = 128
    print(f"Inferred model architecture: hidden_channels={hidden_channels}, num_layers={num_layers}")
    return hidden_channels, num_layers


def load_input_data(dataset_paths, num_geometries_per_dataset):
    print("Loading input data...")
    num_geometries = num_geometries_per_dataset * len(dataset_paths)
    print("Loading waveforms and bands (same across all datasets)...")
    waveforms = torch.load(
        os.path.join(dataset_paths[0], "waveforms_full.pt"),
        weights_only=False,
        map_location="cpu",
    )
    band_ffts = torch.load(
        os.path.join(dataset_paths[0], "band_fft_full.pt"),
        weights_only=False,
        map_location="cpu",
    )
    dtype = waveforms.dtype
    print(f"Waveforms - dtype: {waveforms.dtype}, shape: {waveforms.shape}")
    print(f"Band FFTs - dtype: {band_ffts.dtype}, shape: {band_ffts.shape}")
    combined_geometries = torch.empty((num_geometries, 32, 32), dtype=dtype)
    geom_offset = 0
    for i, dataset_path in enumerate(dataset_paths):
        print(f"Loading geometries from dataset {i+1}/{len(dataset_paths)}: {dataset_path}")
        geometries = torch.load(
            os.path.join(dataset_path, "geometries_full.pt"),
            weights_only=False,
            map_location="cpu",
        )
        end_idx = geom_offset + num_geometries_per_dataset
        combined_geometries[geom_offset:end_idx] = geometries[:num_geometries_per_dataset]
        geom_offset = end_idx
        del geometries
        if i % 5 == 0:
            gc.collect()
    gc.collect()
    print("Final dataset shapes:")
    print(f"  - Geometries: {combined_geometries.shape}")
    print(f"  - Waveforms: {waveforms.shape}")
    print(f"  - Band FFTs: {band_ffts.shape}")
    return combined_geometries, waveforms, band_ffts


def compute_output_index(geometry_idx, waveform_idx, band_idx, num_waveforms, num_bands):
    return geometry_idx * num_waveforms * num_bands + waveform_idx * num_bands + band_idx


def materialize_inputs_cpu(
    geometries: torch.Tensor,
    waveforms: torch.Tensor,
    band_ffts: torch.Tensor,
) -> torch.Tensor:
    """Build (N, 3, 32, 32) float32 inputs on CPU for sequential batch inference."""
    n_geometries = geometries.shape[0]
    n_waveforms = waveforms.shape[0]
    n_bands = band_ffts.shape[0]
    total = n_geometries * n_waveforms * n_bands
    chunk = n_waveforms * n_bands
    inputs = torch.empty((total, 3, 32, 32), dtype=torch.float32)
    geoms_f = geometries.float()
    waves_f = waveforms.float()
    bands_f = band_ffts.float()
    w_idx = torch.arange(n_waveforms).repeat_interleave(n_bands)
    b_idx = torch.arange(n_bands).repeat(n_waveforms)
    for g in range(n_geometries):
        sl = slice(g * chunk, (g + 1) * chunk)
        inputs[sl, 0, :, :] = geoms_f[g]
        inputs[sl, 1, :, :] = waves_f[w_idx]
        inputs[sl, 2, :, :] = bands_f[b_idx]
    return inputs


def load_model_weights(model: nn.Module, model_path: str, device: torch.device) -> None:
    print(f"Loading checkpoint from {model_path}...")
    state_dict = normalize_fno_state_dict(load_raw_state_dict(model_path, map_location="cpu"))
    model.to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully!")
    except RuntimeError:
        print("Warning: Could not load with strict=True, trying strict=False...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            preview = missing_keys[:5]
            print(f"  Missing keys: {preview}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            preview = unexpected_keys[:5]
            print(f"  Unexpected keys: {preview}{'...' if len(unexpected_keys) > 5 else ''}")
        print("Model loaded with strict=False (some keys may not match)")
    del state_dict
    if device.type == "cuda":
        torch.cuda.empty_cache()


def resolve_dataset_paths(input_dataset_path: str, dataset_structure: str) -> list[str]:
    if dataset_structure == "multiple":
        base_dir = input_dataset_path
        dataset_paths: list[str] = []
        if not os.path.isdir(base_dir):
            raise ValueError(f"Input dataset path does not exist: {base_dir}")
        items = [
            item
            for item in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, item)) and item.startswith("set ")
        ]
        for item in items:
            item_path = os.path.join(base_dir, item)
            subitems = [
                subitem
                for subitem in os.listdir(item_path)
                if os.path.isdir(os.path.join(item_path, subitem))
                and (subitem.startswith("out_continuous_") or subitem.startswith("out_binarized_"))
            ]
            for subitem in subitems:
                subitem_path = os.path.join(item_path, subitem)
                if os.path.exists(os.path.join(subitem_path, "geometries_full.pt")):
                    dataset_paths.append(subitem_path)
    else:
        if not os.path.isdir(input_dataset_path):
            raise ValueError(f"Input dataset path does not exist: {input_dataset_path}")
        dataset_paths = [input_dataset_path]
    if not dataset_paths:
        raise ValueError(f"No valid dataset paths found. Check --input_dataset_path: {input_dataset_path}")
    return dataset_paths


def save_outputs(outputs, output_path, channel_labels, num_waveforms, num_bands):
    print(f"Saving outputs to {output_path}...")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(outputs, output_path)
    combos = num_waveforms * num_bands
    n_geometries = outputs.shape[0] // combos if combos else 0
    print("Outputs saved successfully!")
    print(f"  - Format: single stacked tensor (n_samples, {outputs.shape[1]}, 32, 32)")
    print(f"  - Channels: {channel_labels}")
    print(f"  - Shape: {tuple(outputs.shape)}  dtype: {outputs.dtype}")
    print(f"  - Total size: {outputs.numel() * outputs.element_size() / (1024**3):.2f} GB")
    print(f"  - Metadata: {num_waveforms} waveforms, {num_bands} bands, {n_geometries} geometries")
    print(f"  - Index map: idx = geom*({num_waveforms}*{num_bands}) + wave*{num_bands} + band")


def write_inference_doc(doc_path, info):
    lines = [
        "INFERENCE RUN RECORD",
        "=" * 60,
        f"timestamp           : {info.get('timestamp')}",
        f"script              : {info.get('script')}",
        "",
        "MODEL",
        "-" * 60,
        f"checkpoint_path     : {info.get('model_path')}",
        f"model_name          : {info.get('model_name')}",
        f"io_case             : {info.get('case')}",
        "in_channels         : 3",
        f"out_channels        : {info.get('out_channels')}",
        f"output_channels     : {info.get('channel_labels')}",
        f"hidden_channels     : {info.get('hidden_channels')}",
        f"n_layers            : {info.get('n_layers')}",
        f"modes (h x w)       : {info.get('modes_height')} x {info.get('modes_width')}",
        f"arch_source         : {info.get('arch_source')}",
        f"device              : {info.get('device')}",
        f"batch_size          : {info.get('batch_size')}",
        "",
        "DATASETS INFERRED ON",
        "-" * 60,
        f"dataset_structure   : {info.get('dataset_structure')}",
        f"input_dataset_path  : {info.get('input_dataset_path')}",
        f"num_datasets        : {len(info.get('dataset_paths', []))}",
        f"geometries/dataset  : {info.get('num_geometries_per_dataset')}",
        "dataset_folders     :",
    ]
    for p in info.get("dataset_paths", []):
        lines.append(f"  - {p}")
        lines.append("      geometries_full.pt")
    lines.extend(
        [
            "shared_input_files (from first dataset):",
            f"  - waveforms_full.pt  (n_waveforms={info.get('n_waveforms')})",
            f"  - band_fft_full.pt   (n_bands={info.get('n_bands')})",
            "",
            "GEOMETRY SELECTION",
            "-" * 60,
            f"mode                : {info.get('geometry_mode', 'all')}",
            f"total_available     : {info.get('total_geometries_available')}",
        ]
    )
    selected = info.get("selected_indices")
    if selected is None:
        lines.append(f"num_selected        : {info.get('total_geometries_available')} (all)")
    else:
        lines.append(f"num_selected        : {len(selected)}")
        if info.get("geometry_mode") == "random":
            lines.append(f"random_seed         : {info.get('geometry_seed')}")
        if len(selected) <= 50:
            shown = ", ".join(str(i) for i in selected)
        else:
            head = ", ".join(str(i) for i in selected[:25])
            tail = ", ".join(str(i) for i in selected[-25:])
            shown = f"{head}, ... , {tail}"
        lines.append(f"selected_indices    : [{shown}]")
        lines.append("(full list saved to selected_geometry_indices.pt)")
    lines.extend(
        [
            "",
            "OUTPUT",
            "-" * 60,
            f"output_tensor       : {info.get('output_filename')}",
            f"output_shape        : {info.get('output_shape')}",
            f"output_dtype        : {info.get('output_dtype')}",
            f"n_geometries_total  : {info.get('n_geometries_total')}",
            f"total_samples       : {info.get('total_samples')}",
            (
                f"index_map           : idx = geom*({info.get('n_waveforms')}*{info.get('n_bands')}) "
                f"+ wave*{info.get('n_bands')} + band"
            ),
            "",
        ]
    )
    if info.get("extra_notes"):
        lines.extend(["NOTES", "-" * 60, info["extra_notes"], ""])
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote inference documentation: {doc_path}")


def default_inference_root() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "INFERENCE")


def resolve_output_paths(args, model_name: str, case: str) -> tuple[str, str, str]:
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    if args.output_path is not None:
        output_path = args.output_path
        run_out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(run_out_dir, exist_ok=True)
    else:
        inference_root = args.inference_root if args.inference_root is not None else default_inference_root()
        run_out_dir = os.path.join(inference_root, f"{model_name}_{timestamp}")
        os.makedirs(run_out_dir, exist_ok=True)
        output_path = os.path.join(run_out_dir, f"predictions_{case}_{model_name}.pt")
        print(f"Inference run folder: {run_out_dir}")
    return output_path, run_out_dir, timestamp
