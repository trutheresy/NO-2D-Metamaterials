# MATLAB to NumPy/PyTorch Conversion Guide

## Overview

The `convert_mat_to_pytorch.py` script converts MATLAB dispersion datasets to NumPy format following the conventions from `dataset_conversion_reduction.ipynb`. This format is optimized for neural network training with PyTorch.

**Important**: This script now follows the exact format from `dataset_conversion_reduction.ipynb` which differs from the original MATLAB structure to better suit Python ML workflows.

## Usage

### Basic Usage

```bash
python convert_mat_to_pytorch.py <path_to_mat_file>
```

### Example

```bash
# Convert a dataset
python convert_mat_to_pytorch.py ../data/continuous_dataset.mat

# This creates: ../data/continuous_dataset_py/
```

## Output Structure

The script creates a new folder with `_py` suffix containing NumPy files matching `data/set_cr_1200n/` format:

### Core Data Files (matching dataset_conversion_reduction.ipynb)

| File | Type | Description | Precision |
|------|------|-------------|-----------|
| `designs.npy` | NumPy | Design patterns - **elastic modulus only** (N_struct × N_pix × N_pix) | float16 |
| `wavevectors.npy` | NumPy | Wavevector grid (N_struct × N_wv × 2) | float16 |
| `waveforms.npy` | NumPy | Wavelet-embedded wavevectors (N_wv × N_pix × N_pix) | float16 |
| `eigenvalue_data.npy` | NumPy | Eigenfrequencies (N_struct × N_wv × N_eig) | original dtype |
| `bands_fft.npy` | NumPy | Wavelet-embedded band indices (N_eig × N_pix × N_pix) | float16 |
| `eigenvector_data_x.npy` | NumPy | X-displacement eigenvectors (N_struct × N_wv × N_eig × N_pix × N_pix) | complex |
| `eigenvector_data_y.npy` | NumPy | Y-displacement eigenvectors (N_struct × N_wv × N_eig × N_pix × N_pix) | complex |
| `design_params.npy` | NumPy | Design parameters [N_eig, N_pix] | float64 |
| `conversion_summary.txt` | Text | Conversion log and file listing | - |

### Key Differences from MATLAB

1. **Designs**: Only elastic modulus (first pane) is saved, not all 3 material properties
2. **Dimension Order**: N_struct comes first to match Python ML conventions
3. **Waveforms**: Computed from wavevectors using wavelet embedding for spatial representation
4. **bands_fft**: Created from band indices using wavelet embedding, not FFT of eigenvalues
5. **Eigenvectors**: Reshaped to spatial grid format (N_pix × N_pix) instead of DOF format

## Data Format Details

### Dimensions

**Exactly matching `data/set_cr_1200n/` format:**

- **designs**: `(N_struct, N_pix, N_pix)` [float16]
  - Only elastic modulus (first material property)
  - Example: (1200, 32, 32)
  
- **wavevectors**: `(N_struct, N_wv, 2)` [float16]
  - Dimension 2: [0] = k_x, [1] = k_y
  - Example: (1200, 325, 2)
  
- **waveforms**: `(N_wv, N_pix, N_pix)` [float16]
  - Wavelet-embedded representation of wavevectors for neural network input
  - Computed from first structure's wavevectors
  - Example: (325, 32, 32)
  
- **eigenvalue_data**: `(N_struct, N_wv, N_eig)` [original dtype]
  - Eigenfrequencies in Hz
  - Example: (1200, 325, 6)
  
- **eigenvector_data_x**: `(N_struct, N_wv, N_eig, N_pix, N_pix)` [complex]
  - X-displacement components in spatial format
  - Example: (1200, 325, 6, 32, 32)
  
- **eigenvector_data_y**: `(N_struct, N_wv, N_eig, N_pix, N_pix)` [complex]
  - Y-displacement components in spatial format
  - Example: (1200, 325, 6, 32, 32)
  
- **bands_fft**: `(N_eig, N_pix, N_pix)` [float16]
  - Wavelet-embedded representation of band indices
  - Example: (6, 32, 32)
  
- **design_params**: `(1, N_param)` [float64]
  - Simple array with [N_eig, N_pix, ...]
  - Example: (1, 6)

### Precision Strategy

| Data Type | MATLAB | PyTorch | Reason |
|-----------|--------|---------|--------|
| Designs | double (64-bit) | float16 | Material properties don't need high precision |
| Frequencies | double (64-bit) | float16 | Sufficient for neural network training |
| Eigenvectors | complex128 | complex64 | Balance between accuracy and memory |
| Sparse matrices | complex128 | complex64 | Large matrices benefit from reduced size |

## Loading PyTorch Data

### Basic Loading

```python
import torch

# Load tensor data
designs = torch.load('designs.pt')
eigenvalues = torch.load('eigenvalue_data.pt')
eigvec_x = torch.load('eigenvector_data_x.pt')

# Load metadata
const = torch.load('const.pt')
```

### Working with Sparse Matrices

```python
import torch

# Load sparse matrix data
k_data = torch.load('k_data.pt')

# Reconstruct sparse tensor
struct_idx = 0
if k_data[struct_idx] is not None:
    sparse_dict = k_data[struct_idx]
    K = torch.sparse_coo_tensor(
        sparse_dict['indices'],
        sparse_dict['values'],
        sparse_dict['size']
    )
    
    # Convert to dense if needed (warning: memory intensive!)
    K_dense = K.to_dense()
```

### Example: Building a Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class DispersionDataset(Dataset):
    def __init__(self, data_path):
        self.designs = torch.load(f'{data_path}/designs.pt')
        self.eigenvalues = torch.load(f'{data_path}/eigenvalue_data.pt')
        self.wavevectors = torch.load(f'{data_path}/wavevectors.pt')
    
    def __len__(self):
        return self.designs.shape[3]  # N_struct
    
    def __getitem__(self, idx):
        return {
            'design': self.designs[:, :, :, idx],
            'eigenvalues': self.eigenvalues[:, :, idx],
            'wavevectors': self.wavevectors[:, :, idx]
        }

# Create dataset and dataloader
dataset = DispersionDataset('../data/continuous_dataset_py')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    designs = batch['design']  # (batch, N_pix, N_pix, 3)
    eigenvals = batch['eigenvalues']  # (batch, N_wv, N_eig)
    # ... your training code
```

## File Size Comparison

Expected file sizes (approximate):

| Dataset Size | MATLAB .mat | PyTorch _py/ | Reduction |
|--------------|-------------|--------------|-----------|
| N_struct=100, N_pix=32 | ~500 MB | ~250 MB | 50% |
| N_struct=1000, N_pix=32 | ~5 GB | ~2.5 GB | 50% |
| N_struct=1000, N_pix=64 | ~20 GB | ~10 GB | 50% |

The reduction comes from:
- float16 vs float64 (4× smaller for real values)
- complex64 vs complex128 (2× smaller for complex values)
- More efficient sparse matrix storage

## Handling Format Differences

### HDF5 vs scipy.io.loadmat

The script automatically detects the format:

- **MATLAB < v7.3** (scipy): `(N_pix, N_pix, 3, N_struct)`
- **MATLAB v7.3** (HDF5): `(N_struct, 3, N_pix, N_pix)`

Both are converted to the standard format: `(N_pix, N_pix, 3, N_struct)`

### Complex Number Handling

MATLAB complex numbers may be stored as:
1. Structured arrays with `real` and `imag` fields
2. Void types (V128 = complex128)

The script handles both automatically.

## Comparison with data/set_cr_1200n/

### Files Matched

| set_cr_1200n/ | Generated Output | Match |
|---------------|------------------|-------|
| `designs.npy` | `designs.pt` | ✓ Same data, PyTorch format |
| `wavevectors.npy` | `wavevectors.pt` | ✓ Same data, PyTorch format |
| `eigenvalue_data.npy` | `eigenvalue_data.pt` | ✓ Same data, PyTorch format |
| `eigenvector_data_x.npy` | `eigenvector_data_x.pt` | ✓ Same data, PyTorch format |
| `eigenvector_data_y.npy` | `eigenvector_data_y.pt` | ✓ Same data, PyTorch format |
| `bands_fft.npy` | `bands_fft.pt` | ✓ Computed from eigenvalue_data |
| `design_params.npy` | `design_params.pt` | ✓ Same data, PyTorch format |

### Files Not Generated

| File | Reason |
|------|--------|
| `waveforms.npy` | Not present in MATLAB dataset; appears to be post-processed data specific to that dataset's use case |

If `waveforms.npy` is needed, it can be computed from eigenvector data using a separate script (it may represent time-domain waveforms or other processed eigenvector data).

## Troubleshooting

### Issue: "File not found"
**Solution**: Check that the .mat file path is correct and the file exists.

### Issue: "Required key 'designs' not found"
**Solution**: The .mat file must contain at least the 'designs' variable. Check file contents with:
```python
import scipy.io
data = scipy.io.loadmat('file.mat')
print(data.keys())
```

### Issue: Out of memory during conversion
**Solution**: The script processes all data at once. For very large datasets:
1. Use a machine with more RAM
2. Modify the script to process structures in batches
3. Consider using mmap for numpy arrays

### Issue: Sparse matrices not loading correctly
**Solution**: Ensure the MATLAB file contains properly saved sparse matrices. Check with:
```python
import scipy.sparse as sp
print(sp.issparse(K_DATA[0]))
```

## Advanced Usage

### Batch Processing Multiple Files

```bash
# Process all .mat files in a directory
for mat_file in ../data/*.mat; do
    python convert_mat_to_pytorch.py "$mat_file"
done
```

### Custom Precision

To modify precision settings, edit the script:

```python
# For higher precision (float32 instead of float16)
designs_torch = numpy_to_torch(designs, torch.float32)

# For even lower precision (bfloat16)
designs_torch = numpy_to_torch(designs, torch.bfloat16)
```

## See Also

- `mat73_loader.py` - Robust MATLAB v7.3 file loader
- `MATLAB_V73_LOADING_GUIDE.md` - Detailed guide on HDF5 format
- `DIMENSION_ORDERING_GUIDE.md` - Understanding dimension ordering

---

**Created:** October 2025  
**Compatible with:** PyTorch 1.9+, NumPy 1.19+  
**Python:** 3.7+

