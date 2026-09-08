"""
Script to run a specified model on a specified input dataset and save outputs efficiently.

Supports three I/O cases that mirror the training scripts (all share 3 input channels:
geometry, wavevector embedding, band embedding):

- ``I3O1`` -> 1 output channel (eigenfrequency); see ``train_from_disk_eigenfrequency.py``.
- ``I3O4`` -> 4 output channels (displacements x_real, x_imag, y_real, y_imag);
  see ``train_from_disk_displacement.py``.
- ``I3O5`` -> 5 output channels (eigenfrequency + 4 displacements);
  see ``train_from_disk.py``.

Outputs are saved as a single stacked tensor of shape ``(n_samples, out_channels, 32, 32)``.
The sample index maps as: ``index = geometry_idx * n_waveforms * n_bands + waveform_idx * n_bands + band_idx``
where ``n_waveforms`` and ``n_bands`` are inferred from the dataset (current data: 325 wavevectors, 6 bands).

By default, each run creates a folder ``INFERENCE/<model_name>_<YYMMDD-HHMMSS>/`` containing the
output tensor and an ``inference_info_*.txt`` record of the model used and the dataset files
inferred on. Pass ``--output_path`` to save the tensor to an explicit location instead.

Use ``--geometries`` to restrict inference to a subset of geometries: pass multiple indices to
run only those geometry indices, or a single integer N to randomly pick N geometries
(seeded by ``--geometry-seed``). When a subset is used, the selected indices are saved to
``selected_geometry_indices.pt`` and the output is compacted to those geometries (sample index
becomes ``subset_position * n_waveforms * n_bands + waveform_idx * n_bands + band_idx``).
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from collections import Counter

import json
import random
import re
from datetime import datetime
from pathlib import Path

# Match the training scripts (train_from_disk*.py), which build neuralop FNO2d.
try:
    from neuralop.models import FNO2d
except ImportError as e:
    print("Error: neuralop not found. Please install it with: pip install neuraloperator")
    print(f"Import error: {e}")
    sys.exit(1)


# Output-channel contract per I/O case, matching the training scripts:
#   I3O1 -> train_from_disk_eigenfrequency.py (eigenfrequency only)
#   I3O4 -> train_from_disk_displacement.py   (4 displacement channels)
#   I3O5 -> train_from_disk.py                (eigenfrequency + 4 displacements)
CASE_OUT_CHANNELS = {"I3O1": 1, "I3O4": 4, "I3O5": 5}
OUT_CHANNELS_TO_CASE = {v: k for k, v in CASE_OUT_CHANNELS.items()}
CASE_CHANNEL_LABELS = {
    "I3O1": ["eigenfrequency"],
    "I3O4": ["disp_x_real", "disp_x_imag", "disp_y_real", "disp_y_imag"],
    "I3O5": ["eigenfrequency", "disp_x_real", "disp_x_imag", "disp_y_real", "disp_y_imag"],
}


def load_raw_state_dict(model_path, map_location="cpu"):
    """Load a checkpoint and return a plain key->tensor state_dict.

    Handles both raw state_dict files and pickled module objects.
    """
    obj = torch.load(model_path, map_location=map_location, weights_only=False)
    if hasattr(obj, "state_dict") and not isinstance(obj, dict):
        return obj.state_dict()
    return obj


def normalize_fno_state_dict(state_dict):
    """Strip a common wrapper prefix so keys match the inner neuralop FNO2d.

    Different training-script versions saved checkpoints either as bare FNO2d
    keys (e.g. ``projection.fcs.1.weight``) or under a wrapper prefix
    (e.g. ``fno.projection...`` or ``model.projection...``). The inference model
    delegates state_dict I/O to its inner FNO2d, which expects bare keys.
    """
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    for prefix in ("fno.", "model."):
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def infer_out_channels_from_state_dict(state_dict):
    """Infer model output-channel count from the FNO2d output projection weight.

    The output projection is a small MLP whose final linear layer
    (``...projection.fcs.<max_idx>.weight``) has first dimension == out_channels.
    Returns the integer out_channels.
    """
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
        # Fallback: any "...fcs.<max_idx>.weight" (last MLP layer in the network).
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
    """Infer the I/O case (I3O1/I3O4/I3O5) from the checkpoint's output channels."""
    out_channels = infer_out_channels_from_state_dict(state_dict)
    case = OUT_CHANNELS_TO_CASE.get(out_channels)
    if case is None:
        raise ValueError(
            f"Inferred out_channels={out_channels}, which does not match a supported "
            f"case (expected one of {sorted(OUT_CHANNELS_TO_CASE)} -> "
            f"{list(CASE_OUT_CHANNELS)}). Pass --case explicitly."
        )
    return case, out_channels


def select_geometry_subset(geometries_arg, geometry_seed, total_geometries):
    """Resolve the --geometries argument into a list of geometry indices.

    Rules (per the requested behavior):
      - None / empty        -> run on all geometries (returns (None, 'all')).
      - a single integer N  -> randomly pick N geometries (returns (sorted idx, 'random')).
      - two or more values  -> treat as an explicit list of geometry indices.

    Indices refer to the combined geometry ordering across the provided datasets
    (dataset 0 first, then dataset 1, ...). Returns (indices_or_None, mode).
    """
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

    # Explicit list of indices: validate range and de-duplicate (preserve order).
    idx_raw = [int(i) for i in geometries_arg]
    out_of_range = [i for i in idx_raw if i < 0 or i >= total_geometries]
    if out_of_range:
        raise ValueError(
            f"--geometries indices out of range [0, {total_geometries}): {out_of_range[:10]}"
            + (" ..." if len(out_of_range) > 10 else "")
        )
    seen = set()
    idx = []
    for i in idx_raw:
        if i not in seen:
            seen.add(i)
            idx.append(i)
    return idx, "indices"


class FourierNeuralOperator(nn.Module):
    """neuralop FNO2d wrapper matching the training scripts (train_from_disk*.py).

    state_dict I/O is delegated to the inner FNO2d so checkpoints saved by the
    trainers load here with strict=True.
    """

    def __init__(self, modes_height, modes_width, hidden_channels, n_layers, out_channels, in_channels=3):
        super().__init__()
        self.modes_height = modes_height
        self.modes_width = modes_width
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
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

    # Preserve unwrapped FNO2d checkpoint keys by delegating state_dict I/O to the inner model.
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)


def hparams_from_resolved_config(run_dir):
    """Read hidden_channels/layers/modes/out_channels from resolved_config.json if present.

    Returns an empty dict when the file is missing or unreadable.
    """
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
    """
    Infer the number of geometries per dataset by loading the first dataset.
    
    Args:
        dataset_paths: List of paths to dataset directories
    
    Returns:
        Number of geometries per dataset
    """
    if not dataset_paths:
        raise ValueError("No dataset paths provided")
    
    first_dataset_path = dataset_paths[0]
    geometries_path = os.path.join(first_dataset_path, 'geometries_full.pt')
    
    if not os.path.exists(geometries_path):
        raise FileNotFoundError(f"Could not find geometries_full.pt in {first_dataset_path}")
    
    geometries = torch.load(geometries_path, weights_only=False)
    num_geometries = geometries.shape[0]
    del geometries
    gc.collect()
    
    print(f"Inferred {num_geometries} geometries per dataset from {first_dataset_path}")
    return num_geometries


def infer_model_architecture(state_dict, out_channels=5):
    """
    Infer model architecture parameters (hidden, num_layers) from the state_dict.
    Uses a trial-and-error approach: try common configurations and see which one works.
    
    Args:
        state_dict: Already-loaded (and prefix-normalized) model state_dict
        out_channels: Output channel count for the I/O case (used when building
            trial models for key matching)
    
    Returns:
        Tuple of (hidden_channels, num_layers)
    """
    # Common configurations to try
    common_hidden = [128, 256, 512, 64, 32]
    common_layers = [2, 3, 4, 5, 6]
    
    print("Trying to infer model architecture from state_dict...")
    
    # First, try to infer from state_dict keys
    num_layers = None
    hidden_channels = None
    
    # Look for FNO layer keys - they typically have names like:
    # 'fno.blocks.0.convs.0.weight', 'fno.blocks.1.convs.0.weight', etc.
    # or 'fno.fno_blocks.convs.3.weight', etc.
    block_indices = set()
    for key in state_dict.keys():
        if 'fno' in key.lower():
            # Look for patterns like 'blocks.0', 'blocks.1', 'layers.0', 'fno_blocks.convs.3', etc.
            parts = key.split('.')
            for i, part in enumerate(parts):
                # Check for 'blocks', 'layers', or 'fno_blocks' followed by a number
                if part in ['blocks', 'layers', 'fno_blocks'] and i + 1 < len(parts):
                    try:
                        block_idx = int(parts[i + 1])
                        block_indices.add(block_idx)
                    except ValueError:
                        pass
                # Also check if the part itself contains a number after 'convs', 'channel_mlp', etc.
                elif part in ['convs', 'channel_mlp', 'channel_mlp_skips', 'fno_skips'] and i + 1 < len(parts):
                    try:
                        block_idx = int(parts[i + 1])
                        block_indices.add(block_idx)
                    except ValueError:
                        pass
    
    if block_indices:
        num_layers = max(block_indices) + 1
        print(f"  Found num_layers={num_layers} from state_dict keys (indices: {sorted(block_indices)})")
    
    # Try to infer hidden_channels from weight shapes
    # Look for projection or linear layer weights in FNO blocks
    potential_hidden_values = []
    for key in state_dict.keys():
        if 'fno' in key.lower() and 'weight' in key:
            shape = state_dict[key].shape
            if len(shape) >= 2:
                # FNO layers often have shape (hidden, hidden) or (hidden, in_channels)
                # Take the larger dimension as potential hidden size
                potential_hidden = max(shape[:2])
                # Filter out obviously wrong values (too small or too large)
                if 32 <= potential_hidden <= 1024:
                    potential_hidden_values.append(potential_hidden)
    
    if potential_hidden_values:
        # Use the most common value
        counter = Counter(potential_hidden_values)
        hidden_channels = counter.most_common(1)[0][0]
        print(f"  Found potential hidden_channels={hidden_channels} from state_dict shapes")
    
    # If we couldn't infer, try to match state_dict keys with common configurations
    if num_layers is None or hidden_channels is None:
        print("  Trying to match state_dict keys with common configurations...")
        state_dict_keys = set(state_dict.keys())
        
        # Sort by most likely first (common configurations)
        for hidden in common_hidden:
            for layers in common_layers:
                # Skip if we already found both
                if num_layers is not None and hidden_channels is not None:
                    break
                try:
                    # Create a temporary model to check key compatibility
                    model = FourierNeuralOperator(
                        modes_height=32,
                        modes_width=32,
                        hidden_channels=hidden,
                        n_layers=layers,
                        out_channels=out_channels,
                    )
                    model_keys = set(model.state_dict().keys())
                    
                    # Check how many keys match
                    matching_keys = state_dict_keys.intersection(model_keys)
                    match_ratio = len(matching_keys) / max(len(state_dict_keys), len(model_keys))
                    
                    # If we have a good match (most keys match), use this config
                    if match_ratio > 0.8:  # 80% of keys match
                        if num_layers is None:
                            num_layers = layers
                        if hidden_channels is None:
                            hidden_channels = hidden
                        print(f"  Found matching configuration: hidden={hidden}, layers={layers} (match ratio: {match_ratio:.2f})")
                    del model
                    gc.collect()  # Clean up after each model creation
                except Exception as e:
                    continue
            if num_layers is not None and hidden_channels is not None:
                break
    
    # Default values if inference fails
    if num_layers is None:
        print("Warning: Could not infer num_layers from state_dict, using default: 4")
        num_layers = 4
    
    if hidden_channels is None:
        print("Warning: Could not infer hidden_channels from state_dict, using default: 256")
        hidden_channels = 128
    
    print(f"Inferred model architecture: hidden_channels={hidden_channels}, num_layers={num_layers}")
    return hidden_channels, num_layers


def load_input_data(dataset_paths, num_geometries_per_dataset):
    """
    Load input data from dataset paths, mirroring the functionality from the notebook.
    Optimized with preallocation and efficient memory management.
    
    Args:
        dataset_paths: List of paths to dataset directories
        num_geometries_per_dataset: Number of geometries per dataset
    
    Returns:
        geometries: Tensor of shape (n_geometries, 32, 32)
        waveforms: Tensor of shape (91, 32, 32)
        band_ffts: Tensor of shape (6, 32, 32)
    """
    print("Loading input data...")
    
    num_geometries = num_geometries_per_dataset * len(dataset_paths)
    
    # Load waveforms and bands only once (from first dataset)
    print("Loading waveforms and bands (same across all datasets)...")
    waveforms = torch.load(
        os.path.join(dataset_paths[0], 'waveforms_full.pt'),
        weights_only=False,
        map_location='cpu'
    )
    band_ffts = torch.load(
        os.path.join(dataset_paths[0], 'band_fft_full.pt'),
        weights_only=False,
        map_location='cpu'
    )
    
    # Infer dtype from first loaded tensor for preallocation
    dtype = waveforms.dtype
    print(f"Waveforms - dtype: {waveforms.dtype}, shape: {waveforms.shape}")
    print(f"Band FFTs - dtype: {band_ffts.dtype}, shape: {band_ffts.shape}")
    
    # Preallocate geometries tensor with correct dtype
    combined_geometries = torch.empty((num_geometries, 32, 32), dtype=dtype)
    
    # Load geometries from all datasets
    geom_offset = 0
    for i, dataset_path in enumerate(dataset_paths):
        print(f"Loading geometries from dataset {i+1}/{len(dataset_paths)}: {dataset_path}")
        geometries = torch.load(
            os.path.join(dataset_path, 'geometries_full.pt'),
            weights_only=False,
            map_location='cpu'
        )
        # Direct slice assignment (more efficient than copying)
        end_idx = geom_offset + num_geometries_per_dataset
        combined_geometries[geom_offset:end_idx] = geometries[:num_geometries_per_dataset]
        geom_offset = end_idx
        
        # Clean up immediately
        del geometries
        if i % 5 == 0:  # Periodic garbage collection (less frequent)
            gc.collect()
    
    # Final garbage collection
    gc.collect()
    
    print(f"Final dataset shapes:")
    print(f"  - Geometries: {combined_geometries.shape}")
    print(f"  - Waveforms: {waveforms.shape}")
    print(f"  - Band FFTs: {band_ffts.shape}")
    
    return combined_geometries, waveforms, band_ffts


def compute_output_index(geometry_idx, waveform_idx, band_idx, num_waveforms, num_bands):
    """
    Compute the 1D index for a given (geometry, waveform, band) combination.
    
    The mapping formula is: index = geometry_idx * num_waveforms * num_bands + 
                              waveform_idx * num_bands + band_idx
    
    Args:
        geometry_idx: Geometry index
        waveform_idx: Waveform/wavevector index (0 .. num_waveforms-1)
        band_idx: Band index (0 .. num_bands-1)
        num_waveforms: Number of waveforms (must match the dataset; current data: 325)
        num_bands: Number of bands (must match the dataset; current data: 6)
    
    Returns:
        Linear index for the output tensor
    """
    return geometry_idx * num_waveforms * num_bands + waveform_idx * num_bands + band_idx


def get_output_by_indices(outputs, geometry_idx, waveform_idx, band_idx, num_waveforms, num_bands):
    """
    Retrieve a specific output sample by geometry, waveform, and band indices.
    
    Args:
        outputs: Output tensor of shape (n_geometries * num_waveforms * num_bands, out_channels, 32, 32)
        geometry_idx: Geometry index
        waveform_idx: Waveform/wavevector index (0 .. num_waveforms-1)
        band_idx: Band index (0 .. num_bands-1)
        num_waveforms: Number of waveforms (must match the dataset; current data: 325)
        num_bands: Number of bands (must match the dataset; current data: 6)
    
    Returns:
        Output tensor of shape (out_channels, 32, 32)
    """
    index = compute_output_index(geometry_idx, waveform_idx, band_idx, num_waveforms, num_bands)
    return outputs[index]


def run_inference(model, geometries, waveforms, band_ffts, device, batch_size=256, out_channels=5):
    """
    Run model inference on all combinations of geometries, waveforms, and bands.
    Optimized for compute efficiency with preallocated tensors and vectorized operations.
    
    Args:
        model: The trained model
        geometries: Tensor of shape (n_geometries, 32, 32)
        waveforms: Tensor of shape (n_waveforms, 32, 32)
        band_ffts: Tensor of shape (n_bands, 32, 32)
        device: Device to run inference on
        batch_size: Batch size for inference
        out_channels: Number of model output channels (1 for I3O1, 4 for I3O4, 5 for I3O5)
    
    Returns:
        outputs: Tensor of shape (n_geometries * n_waveforms * n_bands, out_channels, 32, 32)
    """
    model.eval()
    
    n_geometries = geometries.shape[0]
    n_waveforms = waveforms.shape[0]
    n_bands = band_ffts.shape[0]
    
    total_samples = n_geometries * n_waveforms * n_bands
    print(f"Running inference on {total_samples} samples...")
    print(f"  - {n_geometries} geometries")
    print(f"  - {n_waveforms} waveforms")
    print(f"  - {n_bands} bands")
    
    # Determine output dtype from model's first output (preserve model precision)
    # Run a dummy forward pass to check output dtype
    # Convert inputs to float32 to match model weights (model was likely trained in float32)
    with torch.no_grad():
        dummy_input = torch.stack([
            geometries[0].to(device, non_blocking=True).float(),
            waveforms[0].to(device, non_blocking=True).float(),
            band_ffts[0].to(device, non_blocking=True).float()
        ], dim=0).unsqueeze(0)  # Add batch dimension
        dummy_output = model(dummy_input)
        model_output_dtype = dummy_output.dtype
        print(f"  - Model output dtype: {model_output_dtype}")
        del dummy_input, dummy_output
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Convert to float16 for storage efficiency (matches input precision)
    output_dtype = torch.float16
    print(f"  - Saving outputs as: {output_dtype} (for storage efficiency)")
    
    # Preallocate output tensor with desired output dtype
    outputs = torch.empty((total_samples, out_channels, 32, 32), dtype=output_dtype)
    
    # Move waveforms and band_ffts to device once (they're reused)
    # Convert to float32 to match model weights
    waveforms_device = waveforms.to(device, non_blocking=True).float()
    band_ffts_device = band_ffts.to(device, non_blocking=True).float()
    
    # Preallocate batch tensor to avoid repeated allocations (use float32 for model)
    batch_tensor = torch.empty((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    
    # Preallocate batch indices list with known maximum size
    batch_indices = [0] * batch_size
    
    sample_count = 0
    batch_idx = 0
    
    with torch.no_grad():
        for geom_idx in tqdm(range(n_geometries), desc="Processing geometries"):
            # Move geometry to device once per geometry, convert to float32
            geometry_device = geometries[geom_idx].to(device, non_blocking=True).float()
            
            for wave_idx in range(n_waveforms):
                for band_idx in range(n_bands):
                    # Directly fill preallocated batch tensor
                    batch_tensor[batch_idx, 0, :, :] = geometry_device
                    batch_tensor[batch_idx, 1, :, :] = waveforms_device[wave_idx]
                    batch_tensor[batch_idx, 2, :, :] = band_ffts_device[band_idx]
                    
                    # Precompute and store output index
                    batch_indices[batch_idx] = compute_output_index(geom_idx, wave_idx, band_idx, n_waveforms, n_bands)
                    batch_idx += 1
                    
                    # Process batch when it's full
                    if batch_idx >= batch_size:
                        # Run inference on preallocated batch tensor
                        batch_outputs = model(batch_tensor)  # Shape: (batch_size, out_channels, 32, 32)
                        
                        # Convert to CPU and dtype in one operation, then use advanced indexing
                        batch_outputs_cpu = batch_outputs.cpu().to(output_dtype)
                        # Use advanced indexing for vectorized assignment (faster than loop)
                        # Use as_tensor to avoid unnecessary copy when possible
                        indices_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
                        outputs[indices_tensor] = batch_outputs_cpu
                        
                        # Clean up
                        del batch_outputs, batch_outputs_cpu, indices_tensor
                        batch_idx = 0
                        sample_count += batch_size
                        
                        # Periodic GPU memory cleanup
                        if sample_count % (batch_size * 10) == 0 and device.type == 'cuda':
                            torch.cuda.empty_cache()
            
            # Clean up geometry from device
            del geometry_device
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Process remaining samples
        if batch_idx > 0:
            # Use only the portion of batch_tensor that's filled
            batch_outputs = model(batch_tensor[:batch_idx])
            batch_outputs_cpu = batch_outputs.cpu().to(output_dtype)
            # Use advanced indexing for vectorized assignment
            indices_tensor = torch.as_tensor(batch_indices[:batch_idx], dtype=torch.long)
            outputs[indices_tensor] = batch_outputs_cpu
            del batch_outputs, batch_outputs_cpu, indices_tensor
    
    # Clean up
    del waveforms_device, band_ffts_device, batch_tensor
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return outputs


def save_outputs(outputs, output_path, channel_labels, num_waveforms, num_bands):
    """
    Save outputs as a single stacked tensor of shape (n_samples, out_channels, 32, 32).

    This stacked-tensor format matches the training target convention (outputs_w_*.pt)
    and generalizes across the I3O1/I3O4/I3O5 cases.

    Args:
        outputs: Tensor of shape (n_samples, out_channels, 32, 32)
        output_path: Path to save the outputs
        channel_labels: List of per-channel names for the selected case
        num_waveforms: Number of waveforms (for metadata / index map)
        num_bands: Number of bands (for metadata / index map)
    """
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
    """Write a human-readable text record of an inference run.

    Records the model used, the resolved I/O case and architecture, the device,
    and the dataset folders + per-dataset input files the model was run on.
    """
    lines = []
    lines.append("INFERENCE RUN RECORD")
    lines.append("=" * 60)
    lines.append(f"timestamp           : {info.get('timestamp')}")
    lines.append("")
    lines.append("MODEL")
    lines.append("-" * 60)
    lines.append(f"checkpoint_path     : {info.get('model_path')}")
    lines.append(f"model_name          : {info.get('model_name')}")
    lines.append(f"io_case             : {info.get('case')}")
    lines.append(f"in_channels         : 3")
    lines.append(f"out_channels        : {info.get('out_channels')}")
    lines.append(f"output_channels     : {info.get('channel_labels')}")
    lines.append(f"hidden_channels     : {info.get('hidden_channels')}")
    lines.append(f"n_layers            : {info.get('n_layers')}")
    lines.append(f"modes (h x w)       : {info.get('modes_height')} x {info.get('modes_width')}")
    lines.append(f"arch_source         : {info.get('arch_source')}")
    lines.append(f"device              : {info.get('device')}")
    lines.append("")
    lines.append("DATASETS INFERRED ON")
    lines.append("-" * 60)
    lines.append(f"dataset_structure   : {info.get('dataset_structure')}")
    lines.append(f"input_dataset_path  : {info.get('input_dataset_path')}")
    lines.append(f"num_datasets        : {len(info.get('dataset_paths', []))}")
    lines.append(f"geometries/dataset  : {info.get('num_geometries_per_dataset')}")
    lines.append("dataset_folders     :")
    for p in info.get('dataset_paths', []):
        lines.append(f"  - {p}")
        lines.append(f"      geometries_full.pt")
    lines.append("shared_input_files (from first dataset):")
    lines.append(f"  - waveforms_full.pt  (n_waveforms={info.get('n_waveforms')})")
    lines.append(f"  - band_fft_full.pt   (n_bands={info.get('n_bands')})")
    lines.append("")
    lines.append("GEOMETRY SELECTION")
    lines.append("-" * 60)
    geom_mode = info.get("geometry_mode", "all")
    lines.append(f"mode                : {geom_mode}")
    lines.append(f"total_available     : {info.get('total_geometries_available')}")
    selected = info.get("selected_indices")
    if selected is None:
        lines.append(f"num_selected        : {info.get('total_geometries_available')} (all)")
    else:
        lines.append(f"num_selected        : {len(selected)}")
        if geom_mode == "random":
            lines.append(f"random_seed         : {info.get('geometry_seed')}")
        if len(selected) <= 50:
            shown = ", ".join(str(i) for i in selected)
        else:
            head = ", ".join(str(i) for i in selected[:25])
            tail = ", ".join(str(i) for i in selected[-25:])
            shown = f"{head}, ... , {tail}"
        lines.append(f"selected_indices    : [{shown}]")
        lines.append("(full list saved to selected_geometry_indices.pt)")
    lines.append("")
    lines.append("OUTPUT")
    lines.append("-" * 60)
    lines.append(f"output_tensor       : {info.get('output_filename')}")
    lines.append(f"output_shape        : {info.get('output_shape')}")
    lines.append(f"output_dtype        : {info.get('output_dtype')}")
    lines.append(f"n_geometries_total  : {info.get('n_geometries_total')}")
    lines.append(f"total_samples       : {info.get('total_samples')}")
    lines.append(
        f"index_map           : idx = geom*({info.get('n_waveforms')}*{info.get('n_bands')}) "
        f"+ wave*{info.get('n_bands')} + band"
    )
    lines.append("")

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote inference documentation: {doc_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run model inference on input dataset and save outputs efficiently'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model weights (.pth file)'
    )
    parser.add_argument(
        '--input_dataset_path',
        type=str,
        required=True,
        help='Path to the input dataset directory (or base directory containing multiple datasets)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Explicit path to save the output tensor (.pt file). If not provided, a run '
             'folder is created under --inference-root and outputs + a documentation text '
             'file are saved there.'
    )
    parser.add_argument(
        '--inference-root',
        type=str,
        default=None,
        help='Root directory for default inference outputs (default: <repo>/INFERENCE). '
             'A subfolder named <model_name>_<YYMMDD-HHMMSS> is created for each run.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for inference (default: 256)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for inference (default: auto)'
    )
    parser.add_argument(
        '--dataset_structure',
        type=str,
        default='single',
        choices=['single', 'multiple'],
        help='Dataset structure: single directory or multiple set directories (default: single)'
    )
    parser.add_argument(
        '--case',
        type=str,
        default='auto',
        choices=['auto', 'I3O1', 'I3O4', 'I3O5'],
        help='Model I/O case (output channels): auto=infer from checkpoint (default), '
             'I3O1=eigenfrequency, I3O4=displacements, I3O5=eigenfrequency+displacements'
    )
    parser.add_argument(
        '--geometries',
        nargs='+',
        type=int,
        default=None,
        help='Geometry subset to run inference on. Provide MULTIPLE indices (e.g. '
             '--geometries 3 7 12) to run only those geometry indices; provide a SINGLE '
             'integer N (e.g. --geometries 100) to randomly pick N geometries. '
             'Indices refer to the combined geometry ordering across datasets. '
             'Default: all geometries.'
    )
    parser.add_argument(
        '--geometry-seed',
        type=int,
        default=0,
        help='RNG seed used when --geometries is a single integer N (random subset). Default: 0.'
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load dataset paths (optimized with list comprehension where possible)
    if args.dataset_structure == 'multiple':
        # Multiple datasets structure (like in the notebook)
        # Look for datasets in the input_dataset_path directory
        base_dir = args.input_dataset_path
        dataset_paths = []
        
        # Try to find set directories
        if os.path.isdir(base_dir):
            # Pre-filter items to reduce iterations
            items = [item for item in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, item)) and item.startswith('set ')]
            
            # Check if it's a "set X" directory structure
            for item in items:
                item_path = os.path.join(base_dir, item)
                # Pre-filter subitems
                subitems = [subitem for subitem in os.listdir(item_path)
                           if os.path.isdir(os.path.join(item_path, subitem)) and
                           (subitem.startswith('out_continuous_') or subitem.startswith('out_binarized_'))]
                
                # Look for out_continuous and out_binarized directories
                for subitem in subitems:
                    subitem_path = os.path.join(item_path, subitem)
                    # Check if required files exist (single check)
                    geom_path = os.path.join(subitem_path, 'geometries_full.pt')
                    if os.path.exists(geom_path):
                        dataset_paths.append(subitem_path)
        else:
            raise ValueError(f"Input dataset path does not exist: {base_dir}")
    else:
        # Single dataset directory
        if not os.path.isdir(args.input_dataset_path):
            raise ValueError(f"Input dataset path does not exist: {args.input_dataset_path}")
        dataset_paths = [args.input_dataset_path]
    
    if not dataset_paths:
        raise ValueError(f"No valid dataset paths found. Check --input_dataset_path: {args.input_dataset_path}")
    
    print(f"Found {len(dataset_paths)} dataset(s)")

    # Load checkpoint once (CPU), normalize wrapper prefixes to inner FNO2d keys.
    print(f"Loading checkpoint from {args.model_path}...")
    state_dict = normalize_fno_state_dict(load_raw_state_dict(args.model_path, map_location='cpu'))

    # Resolve the I/O case -> output channels and channel semantics.
    if args.case == 'auto':
        case, out_channels = infer_case_from_state_dict(state_dict)
        print(f"Auto-inferred I/O case: {case} (out_channels={out_channels}) from checkpoint output projection")
    else:
        case = args.case
        out_channels = CASE_OUT_CHANNELS[case]
    channel_labels = CASE_CHANNEL_LABELS[case]
    print(f"I/O case: {case} (in_channels=3, out_channels={out_channels}; channels={channel_labels})")

    # Prefer architecture from the run's resolved_config.json (next to the checkpoint).
    run_dir = os.path.dirname(os.path.abspath(args.model_path))
    hp = hparams_from_resolved_config(run_dir)
    if hp.get('out_channels') is not None and int(hp['out_channels']) != out_channels:
        print(
            f"Warning: resolved_config.json out_channels={hp['out_channels']} does not match "
            f"resolved case {case} (out_channels={out_channels})."
        )

    # Infer num_geometries_per_dataset from dataset
    num_geometries_per_dataset = infer_num_geometries_per_dataset(dataset_paths)
    
    # Determine model architecture: resolved_config if available, else infer from state_dict.
    if hp.get('hidden_channels') is not None and hp.get('layers') is not None:
        model_hidden = int(hp['hidden_channels'])
        model_layers = int(hp['layers'])
        arch_source = "resolved_config.json"
        print(f"Using architecture from resolved_config.json: hidden_channels={model_hidden}, num_layers={model_layers}")
    else:
        print("Inferring model architecture from state_dict...")
        model_hidden, model_layers = infer_model_architecture(state_dict, out_channels=out_channels)
        arch_source = "state_dict inference"
    modes_h = int(hp.get('modes_height', 32))
    modes_w = int(hp.get('modes_width', 32))
    
    # Load input data
    geometries, waveforms, band_ffts = load_input_data(dataset_paths, num_geometries_per_dataset)

    # Optionally restrict to a subset of geometries (explicit indices or random N).
    total_geometries = int(geometries.shape[0])
    selected_indices, geom_mode = select_geometry_subset(
        args.geometries, args.geometry_seed, total_geometries
    )
    if selected_indices is not None:
        print(
            f"Geometry selection: mode={geom_mode}, running on "
            f"{len(selected_indices)}/{total_geometries} geometries"
        )
        geometries = geometries[torch.as_tensor(selected_indices, dtype=torch.long)]
        gc.collect()
    else:
        print(f"Geometry selection: mode=all, running on all {total_geometries} geometries")
    
    # Build model and load the (already normalized) state dict.
    print("Building model and loading weights...")
    model = FourierNeuralOperator(
        modes_height=modes_h,
        modes_width=modes_w,
        hidden_channels=model_hidden,
        n_layers=model_layers,
        out_channels=out_channels,
    ).to(device)
    
    # Try loading with strict=True first (faster if it works)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully!")
    except RuntimeError:
        print("Warning: Could not load with strict=True, trying strict=False...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  Unexpected keys: {unexpected_keys}")
        print("Model loaded with strict=False (some keys may not match)")
    
    # Clean up state_dict immediately after loading
    del state_dict
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Model name (without extension) used for the run-folder and output filename.
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # Determine output location.
    #  - If --output_path is given, save the tensor there (doc written alongside it).
    #  - Otherwise create <inference_root>/<model_name>_<YYMMDD-HHMMSS>/ and save everything there.
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    if args.output_path is not None:
        output_path = args.output_path
        run_out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(run_out_dir, exist_ok=True)
    else:
        if args.inference_root is not None:
            inference_root = args.inference_root
        else:
            inference_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "INFERENCE")
        run_out_dir = os.path.join(inference_root, f"{model_name}_{timestamp}")
        os.makedirs(run_out_dir, exist_ok=True)
        output_path = os.path.join(run_out_dir, f"predictions_{case}_{model_name}.pt")
        print(f"Inference run folder: {run_out_dir}")
    
    # Run inference
    n_waveforms = waveforms.shape[0]
    n_bands = band_ffts.shape[0]
    
    outputs = run_inference(
        model,
        geometries,
        waveforms,
        band_ffts,
        device,
        batch_size=args.batch_size,
        out_channels=out_channels,
    )
    
    # Save outputs
    save_outputs(outputs, output_path, channel_labels, n_waveforms, n_bands)

    # Persist the selected geometry indices (when a subset was used) for traceability.
    if selected_indices is not None:
        idx_path = os.path.join(run_out_dir, "selected_geometry_indices.pt")
        torch.save(torch.as_tensor(selected_indices, dtype=torch.long), idx_path)
        print(f"Saved selected geometry indices: {idx_path}")

    # Write a documentation text file recording the model and the datasets inferred on.
    combos = n_waveforms * n_bands
    doc_path = os.path.join(run_out_dir, f"inference_info_{model_name}_{timestamp}.txt")
    write_inference_doc(
        doc_path,
        {
            "timestamp": timestamp,
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
            "dataset_structure": args.dataset_structure,
            "input_dataset_path": os.path.abspath(args.input_dataset_path),
            "dataset_paths": [os.path.abspath(p) for p in dataset_paths],
            "num_geometries_per_dataset": num_geometries_per_dataset,
            "geometry_mode": geom_mode,
            "total_geometries_available": total_geometries,
            "geometry_seed": (int(args.geometry_seed) if geom_mode == "random" else None),
            "selected_indices": selected_indices,
            "n_waveforms": n_waveforms,
            "n_bands": n_bands,
            "n_geometries_total": int(outputs.shape[0] // combos) if combos else 0,
            "total_samples": int(outputs.shape[0]),
            "output_filename": os.path.basename(output_path),
            "output_shape": tuple(outputs.shape),
            "output_dtype": str(outputs.dtype),
        },
    )
    
    print(f"Done! Outputs saved in: {run_out_dir}")


if __name__ == '__main__':
    main()

