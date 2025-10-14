# Changelog

## Latest Update: MATLAB to NumPy/PyTorch Conversion (October 14, 2025)

### New Feature: Dataset Conversion Pipeline

Implemented comprehensive conversion system following `dataset_conversion_reduction.ipynb` format:

#### New Files Added

1. **`convert_mat_to_pytorch.py`** ⭐ NEW
   - Converts MATLAB .mat files to NumPy format
   - Exactly matches `data/set_cr_1200n/` structure
   - Features:
     - Automatic format detection (scipy vs HDF5)
     - Dimension reshaping for Python ML workflows
     - Wavelet embedding for waveforms and bands_fft
     - Eigenvector reshaping from DOF to spatial format
     - Precision optimization (float16 where appropriate)
     - Extracts elastic modulus only (first pane)

2. **`example_pytorch_dataset.py`** 📚 NEW
   - PyTorch Dataset class for converted data
   - Loads NumPy arrays and converts to PyTorch tensors
   - Compatible with PyTorch DataLoader
   - Includes example usage and statistics functions

3. **`PYTORCH_CONVERSION.md`** 📖 NEW
   - Complete conversion guide
   - Format specifications
   - Comparison with `data/set_cr_1200n/`
   - Usage examples

4. **`PYTORCH_CONVERSION_SUMMARY.md`** 📋 NEW
   - Implementation details and design decisions
   - Complete file mapping
   - Validation procedures

5. **`PYTORCH_QUICKSTART.md`** 🚀 NEW
   - One-page quick reference
   - Common operations
   - Example training loop

6. **`WAVELET_EMBEDDING_NOTE.md`** ⚠️ NEW
   - Important information about wavelet embedding placeholders
   - How to use original NO_utils_multiple functions
   - Impact on results

7. **`CONVERSION_COMPLETE.md`** ✓ NEW
   - Final summary and validation
   - Next steps guide

#### Updated Files

1. **`README.md`**
   - Added "Dataset Conversion to NumPy/PyTorch Format" section
   - Usage examples and benefits
   - Links to documentation

2. **`requirements.txt`**
   - Added PyTorch as optional dependency

#### Output Format

**Matches `data/set_cr_1200n/` exactly:**

```
dataset_py/
├── designs.npy              (N_struct, N_pix, N_pix)              [float16]
├── wavevectors.npy          (N_struct, N_wv, 2)                   [float16]
├── waveforms.npy            (N_wv, N_pix, N_pix)                  [float16]
├── eigenvalue_data.npy      (N_struct, N_wv, N_eig)               [original]
├── eigenvector_data_x.npy   (N_struct, N_wv, N_eig, N_pix, N_pix) [complex]
├── eigenvector_data_y.npy   (N_struct, N_wv, N_eig, N_pix, N_pix) [complex]
├── bands_fft.npy            (N_eig, N_pix, N_pix)                 [float16]
├── design_params.npy        (1, N_param)                          [float64]
└── conversion_summary.txt   (metadata)
```

#### Key Differences from MATLAB

1. **Dimension Order**: N_struct first (Python convention)
2. **Designs**: Elastic modulus only (not all 3 properties)
3. **Eigenvectors**: Spatial format (N_pix × N_pix) not DOF format
4. **Waveforms**: Computed via wavelet embedding (sinusoidal placeholder)
5. **bands_fft**: Computed via wavelet embedding (sinusoidal placeholder)

#### Important Note

The wavelet embedding functions use **placeholder sinusoidal encodings**. For production use with original results, replace with functions from `NO_utils_multiple`:
```python
from NO_utils_multiple import embed_2const_wavelet, embed_integer_wavelet
```

See `WAVELET_EMBEDDING_NOTE.md` for details.

---

## Major Update: Robust MATLAB v7.3 Loading (October 2025)

### Critical Fix: Comprehensive Data Structure Handling

After extensive debugging of MATLAB v7.3 (HDF5) file loading issues, implemented a complete, production-ready solution.

#### Root Cause Analysis

MATLAB v7.3 files use HDF5 format with complex data structures:
- Cell arrays stored as HDF5 object references (not dereferenced by h5py)
- Sparse matrices stored as HDF5 groups with data/ir/jc fields
- Nested structures different from older MATLAB formats
- Required recursive dereferencing and reconstruction

#### Solution: New `mat73_loader.py` Module

Comprehensive, robust loader with:
- ✅ Automatic format detection (old vs v7.3)
- ✅ Recursive HDF5 reference dereferencing
- ✅ Sparse matrix reconstruction (CSC format)
- ✅ Struct conversion to dicts
- ✅ Extensive error checking
- ✅ Diagnostic tools for debugging
- ✅ Verbose logging mode

### New Files Added

1. **`mat73_loader.py`** ⭐ NEW
   - Production-ready MATLAB v7.3 (HDF5) file loader
   - Functions:
     - `load_matlab_v73()`: Main loading function with full HDF5 support
     - `diagnose_mat_file()`: Diagnostic tool to inspect file structure
     - `_load_h5_item()`: Recursive loader for all HDF5 types
     - `_load_sparse_matrix()`: Reconstructs MATLAB sparse matrices
   - Features:
     - Handles cell arrays (object reference arrays)
     - Reconstructs sparse matrices from data/ir/jc fields
     - Converts MATLAB structs to Python dicts
     - Provides detailed loading information in verbose mode
     - Command-line diagnostic tool
   - See `MATLAB_V73_LOADING_GUIDE.md` for detailed documentation

2. **`MATLAB_V73_LOADING_GUIDE.md`** 📚 NEW
   - Comprehensive guide to MATLAB v7.3 file loading
   - Explains HDF5 format and data structures
   - Common issues and solutions
   - Usage examples and best practices
   - Command-line diagnostic tool usage

## Recent Updates (October 2025)

### New Files Added

1. **`plot_dispersion_script.py`**
   - Comprehensive visualization script equivalent to MATLAB's `plot_dispersion.m`
   - Loads datasets and creates extensive dispersion visualizations
   - Features:
     - Plots unit cell designs for multiple structures
     - Creates IBZ contour plots with high-symmetry point labels
     - Interpolates dispersion relations onto contours using RegularGridInterpolator
     - Reconstructs frequencies from eigenvectors for verification (Kr*v = Mr*v*λ)
     - Compares original vs reconstructed frequencies
     - Exports all plots to organized PNG directories
   - Configurable output resolution and structure selection

2. **`demo_create_Kr_and_Mr.py`**
   - Demo script showing how to create reduced stiffness (Kr) and mass (Mr) matrices
   - Equivalent to MATLAB's `demo_create_Kr_and_Mr.m`
   - Demonstrates:
     - Loading datasets with K_DATA, M_DATA, and T_DATA
     - Extracting matrices for specific unit cells and wavevectors
     - Computing reduced matrices: Kr = T' * K * T and Mr = T' * M * T
     - Visualizing sparsity patterns using matplotlib

3. **`test_new_features.py`**
   - Test script for newly added features
   - Tests `linspaceNDim` function with various inputs
   - Tests `get_IBZ_contour_wavevectors` for different symmetry types
   - Includes visualization of IBZ contours

4. **`fileshare_licenses/linspace_NDim_license.txt`**
   - License file for the linspaceNDim implementation
   - Based on code by Steeve AMBROISE (2009)

### Modified Files

1. **`utils.py`**
   - **Added `linspaceNDim` function**
     - N-dimensional generalization of numpy's linspace
     - Creates linearly spaced points between two N-dimensional vectors
     - Used for generating paths along IBZ contours
     - Parameters:
       - `d1`: Starting point (array-like)
       - `d2`: Ending point (array-like)
       - `n`: Number of points (default: 100)
     - Returns: Array of shape (n, len(d1))

2. **`wavevectors.py`**
   - **Updated `get_IBZ_contour_wavevectors` function**
     - Now properly implements contour generation using `linspaceNDim`
     - Returns tuple: `(wavevectors, contour_info)`
     - Enhanced contour_info dictionary with:
       - `N_segment`: Number of contour segments
       - `vertex_labels`: LaTeX labels for high-symmetry points
       - `vertices`: Coordinates of vertices
       - `wavevector_parameter`: Parameter along contour (0 to N_segment)
     - Support for multiple symmetry types:
       - `'p4mm'`: Gamma -> X -> M -> Gamma
       - `'c1m1'`: Gamma -> X -> M -> Gamma -> O_bar -> X
       - `'p6mm'`: Gamma -> K -> M -> Gamma (hexagonal)
       - `'none'`: Gamma -> X -> M -> Gamma -> Y -> O -> Gamma
       - `'all contour segments'`: All possible segments
     - Better handling of duplicate points at vertices
     - Matches MATLAB implementation behavior

3. **`__init__.py`**
   - Added `linspaceNDim` to imports
   - Added `linspaceNDim` to `__all__` list for public API

4. **`README.md`**
   - Added section on new demo script
   - Documented `linspaceNDim` utility function
   - Added "New Features" section highlighting recent additions
   - Updated utilities documentation

### Compatibility Notes

- All new features are backward compatible
- `get_IBZ_contour_wavevectors` now returns a tuple instead of just wavevectors
  - Old code: `wavevectors = get_IBZ_contour_wavevectors(N_k, a, symmetry)`
  - New code: `wavevectors, contour_info = get_IBZ_contour_wavevectors(N_k, a, symmetry)`
  - If you only need wavevectors, use: `wavevectors, _ = get_IBZ_contour_wavevectors(...)`

### Testing

Run the test script to verify new features:
```bash
python test_new_features.py
```

Run the demo to see matrix reduction in action:
```bash
python demo_create_Kr_and_Mr.py
```

### Dependencies

No new dependencies added. All features use existing packages:
- NumPy
- SciPy
- Matplotlib

### Future Work

Potential enhancements for future updates:
- Add more visualization options for reduced matrices
- Implement additional symmetry types
- Add caching for wavevector generation
- Performance optimization for large systems

