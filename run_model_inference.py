"""
Script to run a specified model on a specified input dataset and save outputs efficiently.

The outputs are saved as a 4D tensor where each index maps to a (4, 32, 32) displacement field.
The mapping is: index = geometry_idx * 91 * 6 + waveform_idx * 6 + band_idx
This avoids storing redundant wavevector and band data for each geometry.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from collections import Counter

# Import model architecture (assuming it's defined in the same directory or available)
# You may need to adjust this import based on your project structure
try:
    from neuralop.models import FNO
except ImportError as e:
    print(f"Error: neuralop not found. Please install it with: pip install neuraloperator")
    print(f"Import error: {e}")
    sys.exit(1)


class FourierNeuralOperator(nn.Module):
    """Fourier Neural Operator model architecture."""
    def __init__(self, modes_height, modes_width, in_channels, out_channels, hidden, num_layers):
        super(FourierNeuralOperator, self).__init__()
        self.modes_height = modes_height
        self.modes_width = modes_width
        self.hidden = hidden
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        # FNO layer (2D: n_modes is a tuple for 2D spatial dimensions)
        self.fno = FNO(
            n_modes=(self.modes_height, self.modes_width),
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            hidden_channels=self.hidden,
            n_layers=self.num_layers
        )

    def forward(self, x):
        x = self.fno(x)
        return x


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


def infer_model_architecture(model_path):
    """
    Infer model architecture parameters (hidden, num_layers) from the state_dict.
    Uses a trial-and-error approach: try common configurations and see which one works.
    
    Args:
        model_path: Path to the model weights file
    
    Returns:
        Tuple of (hidden_channels, num_layers)
    """
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    
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
                        in_channels=3,
                        out_channels=4,
                        hidden=hidden,
                        num_layers=layers
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


def compute_output_index(geometry_idx, waveform_idx, band_idx, num_waveforms=91, num_bands=6):
    """
    Compute the 1D index for a given (geometry, waveform, band) combination.
    
    The mapping formula is: index = geometry_idx * num_waveforms * num_bands + 
                              waveform_idx * num_bands + band_idx
    
    Args:
        geometry_idx: Geometry index
        waveform_idx: Waveform/wavevector index (0-90)
        band_idx: Band index (0-5)
        num_waveforms: Number of waveforms (default: 91)
        num_bands: Number of bands (default: 6)
    
    Returns:
        Linear index for the output tensor
    """
    return geometry_idx * num_waveforms * num_bands + waveform_idx * num_bands + band_idx


def get_output_by_indices(outputs, geometry_idx, waveform_idx, band_idx, num_waveforms=91, num_bands=6):
    """
    Retrieve a specific output tensor by geometry, waveform, and band indices.
    
    Args:
        outputs: Output tensor of shape (n_geometries * 91 * 6, 4, 32, 32)
        geometry_idx: Geometry index
        waveform_idx: Waveform/wavevector index (0-90)
        band_idx: Band index (0-5)
        num_waveforms: Number of waveforms (default: 91)
        num_bands: Number of bands (default: 6)
    
    Returns:
        Output tensor of shape (4, 32, 32)
    """
    index = compute_output_index(geometry_idx, waveform_idx, band_idx, num_waveforms, num_bands)
    return outputs[index]


def run_inference(model, geometries, waveforms, band_ffts, device, batch_size=256):
    """
    Run model inference on all combinations of geometries, waveforms, and bands.
    Optimized for compute efficiency with preallocated tensors and vectorized operations.
    
    Args:
        model: The trained model
        geometries: Tensor of shape (n_geometries, 32, 32)
        waveforms: Tensor of shape (91, 32, 32)
        band_ffts: Tensor of shape (6, 32, 32)
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns:
        outputs: Tensor of shape (n_geometries * 91 * 6, 4, 32, 32)
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
    outputs = torch.empty((total_samples, 4, 32, 32), dtype=output_dtype)
    
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
                        batch_outputs = model(batch_tensor)  # Shape: (batch_size, 4, 32, 32)
                        
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


def save_outputs(outputs, output_path, num_waveforms=91, num_bands=6):
    """
    Save outputs to the specified path in the same format as displacements_dataset.pt.
    
    Args:
        outputs: Tensor of shape (n_geometries * 91 * 6, 4, 32, 32)
        output_path: Path to save the outputs
        num_waveforms: Number of waveforms (for metadata, default: 91)
        num_bands: Number of bands (for metadata, default: 6)
    """
    print(f"Saving outputs to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Split the outputs into 4 separate tensors (matching displacements_dataset.pt format)
    # outputs shape: (n_samples, 4, 32, 32)
    # Split into 4 tensors of shape (n_samples, 32, 32) each
    tensor_0 = outputs[:, 0, :, :]  # First component
    tensor_1 = outputs[:, 1, :, :]  # Second component
    tensor_2 = outputs[:, 2, :, :]  # Third component
    tensor_3 = outputs[:, 3, :, :]  # Fourth component
    
    # Create TensorDataset matching the format of displacements_dataset.pt
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(tensor_0, tensor_1, tensor_2, tensor_3)
    
    torch.save(dataset, output_path)
    print(f"Outputs saved successfully!")
    print(f"  - Format: TensorDataset with 4 tensors")
    print(f"  - Each tensor shape: {tensor_0.shape}")
    print(f"  - Dtype: {tensor_0.dtype}")
    print(f"  - Total size: {outputs.numel() * outputs.element_size() / (1024**3):.2f} GB")
    print(f"  - Metadata: {num_waveforms} waveforms, {num_bands} bands, {outputs.shape[0] // (num_waveforms * num_bands)} geometries")


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
        help='Path to save the output tensor (.pt file). If not provided, will auto-generate as predictions_[model_name].pt in the input dataset directory'
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
    
    # Infer num_geometries_per_dataset from dataset
    num_geometries_per_dataset = infer_num_geometries_per_dataset(dataset_paths)
    
    # Infer model architecture from state_dict
    print("Inferring model architecture from state_dict...")
    model_hidden, model_layers = infer_model_architecture(args.model_path)
    
    # Load input data
    geometries, waveforms, band_ffts = load_input_data(dataset_paths, num_geometries_per_dataset)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = FourierNeuralOperator(
        modes_height=32,
        modes_width=32,
        in_channels=3,
        out_channels=4,
        hidden=model_hidden,
        num_layers=model_layers
    ).to(device)
    
    # Load state dict directly to device (more efficient)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Try loading with strict=True first (faster if it works)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully!")
    except RuntimeError as e:
        print(f"Warning: Could not load with strict=True, trying strict=False...")
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
    
    # Auto-generate output path if not provided
    if args.output_path is None:
        # Extract model name without extension
        model_basename = os.path.basename(args.model_path)
        model_name = os.path.splitext(model_basename)[0]  # Remove extension
        # Save in the first dataset directory
        output_path = os.path.join(dataset_paths[0], f'predictions_{model_name}.pt')
        print(f"Auto-generated output path: {output_path}")
    else:
        output_path = args.output_path
    
    # Run inference
    n_waveforms = waveforms.shape[0]
    n_bands = band_ffts.shape[0]
    
    outputs = run_inference(
        model,
        geometries,
        waveforms,
        band_ffts,
        device,
        batch_size=args.batch_size
    )
    
    # Save outputs
    save_outputs(outputs, output_path, n_waveforms, n_bands)
    
    print("Done!")


if __name__ == '__main__':
    main()

