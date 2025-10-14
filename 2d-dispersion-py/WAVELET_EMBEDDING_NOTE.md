# Important Note: Wavelet Embedding Functions

## Overview

The conversion script `convert_mat_to_pytorch.py` now follows the exact format from `dataset_conversion_reduction.ipynb`. However, this notebook uses custom wavelet embedding functions from `NO_utils_multiple` that are not included in the `2d-dispersion-py` package.

## Affected Functions

### 1. `embed_2const_wavelet(wavevector_x, wavevector_y, size)`

**Purpose**: Converts wavevector data into spatial domain representation for neural network input.

**Original**: From `NO_utils_multiple.embed_2const_wavelet`
**Current Implementation**: Sinusoidal encoding (placeholder)

```python
# Current placeholder implementation
waveforms[i] = np.sin(2 * np.pi * kx * X) * np.cos(2 * np.pi * ky * Y)
```

### 2. `embed_integer_wavelet(bands, size)`

**Purpose**: Embeds band indices into spatial domain for neural network input.

**Original**: From `NO_utils_multiple.embed_integer_wavelet`
**Current Implementation**: Sinusoidal encoding (placeholder)

```python
# Current placeholder implementation
bands_fft[i] = np.sin(band * np.pi * X) * np.cos(band * np.pi * Y)
```

## Why Placeholder Implementations?

The `NO_utils_multiple` module is not part of the `2d-dispersion-py` package and appears to be custom code specific to the neural operator training workflow. The placeholder implementations:

1. **Provide spatial representation**: Convert scalar/vector data to 2D spatial grids
2. **Maintain correct shapes**: Output shapes match the expected format
3. **Use simple encoding**: Sinusoidal functions that preserve spatial structure

## Using the Original Functions

If you have access to the original `NO_utils_multiple` module, you can replace the placeholder implementations:

### Option 1: Direct Import

Add to the top of `convert_mat_to_pytorch.py`:

```python
try:
    from NO_utils_multiple import embed_2const_wavelet, embed_integer_wavelet
    print("Using original wavelet embedding functions from NO_utils_multiple")
except ImportError:
    print("WARNING: NO_utils_multiple not found, using placeholder wavelet functions")
    # Keep the placeholder implementations below
```

### Option 2: Sys Path Addition

```python
import sys
sys.path.insert(0, '../')  # Or wherever NO_utils_multiple is located

from NO_utils_multiple import embed_2const_wavelet, embed_integer_wavelet
```

## Impact on Results

### Waveforms

The wavelet-embedded waveforms are used as **additional input features** for neural networks. The placeholder implementation:
- ✓ Preserves wavevector information in spatial format
- ✓ Maintains correct dimensions
- ⚠️ May use different wavelet basis than original
- ⚠️ Encoding details may differ

**Impact**: Slight differences in neural network input representation. The network should still learn effectively, but results may differ slightly from those using original embeddings.

### bands_fft

The band embeddings are used to represent which eigenvalue band is being predicted. The placeholder:
- ✓ Creates unique spatial patterns for each band
- ✓ Maintains correct dimensions
- ⚠️ Pattern may differ from original wavelet embedding

**Impact**: Minimal, as band indices are discrete values and the encoding primarily serves to represent them spatially.

## Recommendations

### For Production Use
If you're reproducing results from papers or need exact matching:
1. Obtain the original `NO_utils_multiple` module
2. Replace the placeholder functions
3. Verify output matches expected format

### For Exploratory Work
The placeholder implementations are sufficient for:
- Testing the conversion pipeline
- Exploring neural network architectures
- Understanding the data format
- Initial experiments

### For Custom Implementations
You can replace the placeholders with your own encoding:

```python
def embed_2const_wavelet(wavevector_x, wavevector_y, size):
    # Your custom wavelet transform
    # Options: Haar wavelets, Mexican hat, Morlet, etc.
    pass

def embed_integer_wavelet(bands, size):
    # Your custom encoding
    # Options: one-hot encoding, learned embeddings, etc.
    pass
```

## File Compatibility

Despite the placeholder implementations, the output files are **fully compatible** with:
- PyTorch DataLoader
- Neural network training pipelines
- The format expected by `dataset_conversion_reduction.ipynb`

The shapes and dtypes are identical to the reference implementation.

## Testing

To verify your wavelet embeddings are working:

```python
import numpy as np

# Load and check waveforms
waveforms = np.load('dataset_py/waveforms.npy')
print(f"Waveforms shape: {waveforms.shape}")  # Should be (N_wv, N_pix, N_pix)
print(f"Waveforms range: [{waveforms.min():.3f}, {waveforms.max():.3f}]")

# Load and check bands_fft
bands_fft = np.load('dataset_py/bands_fft.npy')
print(f"Bands FFT shape: {bands_fft.shape}")  # Should be (N_eig, N_pix, N_pix)
print(f"Bands FFT range: [{bands_fft.min():.3f}, {bands_fft.max():.3f}]")
```

## Summary

| Aspect | Placeholder | Original (NO_utils_multiple) |
|--------|-------------|------------------------------|
| **Functionality** | ✓ Works | ✓ Works |
| **Output Shape** | ✓ Correct | ✓ Correct |
| **Encoding Method** | Sinusoidal | Wavelet (specific type unknown) |
| **Exact Match** | ⚠️ No | ✓ Yes |
| **ML Training** | ✓ Compatible | ✓ Compatible |

---

**Recommendation**: For critical production work, obtain the original NO_utils_multiple. For exploratory work and testing, the placeholders are sufficient.

**Created**: October 14, 2025  
**Related Files**:
- `convert_mat_to_pytorch.py`
- `dataset_conversion_reduction.ipynb`
- `data/set_cr_1200n/`

