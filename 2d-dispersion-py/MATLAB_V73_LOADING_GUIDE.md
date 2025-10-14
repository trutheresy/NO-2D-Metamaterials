# MATLAB v7.3 File Loading Guide

## Overview

This guide explains how Python scripts in this package handle MATLAB v7.3 files (.mat files saved with the `-v7.3` flag), which use HDF5 format instead of the older MATLAB binary format.

## The Problem

MATLAB v7.3 files store data in HDF5 format, which requires special handling:

1. **Cell Arrays** → Stored as arrays of HDF5 object references that must be dereferenced
2. **Sparse Matrices** → Stored as HDF5 groups with separate `data`, `ir` (row indices), and `jc` (column pointers) fields
3. **Structs** → Stored as HDF5 groups with nested structure
4. **Nested Arrays** → Different nesting structure than older MATLAB files

The standard `scipy.io.loadmat()` cannot read these files and raises `NotImplementedError`.

## The Solution

### New Module: `mat73_loader.py`

A robust, comprehensive loader module that:

- ✅ **Automatically detects and loads** both old and new MATLAB formats
- ✅ **Recursively dereferences** HDF5 object references (cell arrays)
- ✅ **Reconstructs sparse matrices** from HDF5 storage format to `scipy.sparse`
- ✅ **Handles structs** as dictionaries
- ✅ **Provides diagnostic tools** to inspect file structure
- ✅ **Extensive error checking** with informative messages
- ✅ **Verbose mode** for debugging data loading issues

### Key Functions

#### `load_matlab_v73(filepath, verbose=True)`

Main loading function that handles all data types.

```python
from mat73_loader import load_matlab_v73

data = load_matlab_v73('my_file.mat', verbose=True)

# Access data
K = data['K_DATA'].flat[0]  # Already a sparse matrix!
frequencies = data['EIGENVALUE_DATA']
designs = data['designs']
```

#### `diagnose_mat_file(filepath)`

Diagnostic tool to inspect file structure without loading all data.

```python
from mat73_loader import diagnose_mat_file

# See what's in your file
diagnose_mat_file('my_file.mat')
```

**Example output:**
```
======================================================================
Diagnosing MATLAB file: continuous 13-Oct-2025 23-22-59.mat
======================================================================

Top-level variables (12 total):
├─ K_DATA: Dataset
│  ├─ Shape: (5,)
│  ├─ Dtype: object
│  └─ Type: Cell array (references)
│     └─ First element structure:
├─ M_DATA: Dataset
│  └─ Type: Cell array (references)
├─ designs: Dataset
│  ├─ Shape: (32, 32, 3, 5)
│  └─ Type: Numeric/String array
├─ const: Group (5 fields)
│  ├─ Type: Struct
│  ├─ a: Dataset
│  ├─ N_wv: Dataset
│  └─ N_eig: Dataset
...
```

### Command-Line Diagnostic Tool

You can run the diagnostic from command line:

```bash
python mat73_loader.py path/to/your/file.mat
```

## Updated Scripts

### 1. `plot_dispersion_script.py`

Comprehensive visualization script now uses robust loading:

```python
# Automatically handles both old and new MATLAB formats
data = load_dataset(data_path, verbose=False)

# Data is properly loaded - sparse matrices are already sparse!
K = data['K_DATA'].flat[struct_idx]  # scipy.sparse.csc_matrix
M = data['M_DATA'].flat[struct_idx]  # scipy.sparse.csc_matrix
T = data['T_DATA'].flat[wv_idx]      # scipy.sparse.csc_matrix
```

### 2. `demo_create_Kr_and_Mr.py`

Demo script updated with same robust loading.

### 3. All other scripts

Any script that loads MATLAB files can now use:

```python
try:
    from mat73_loader import load_matlab_v73
except ImportError:
    import mat73_loader
    load_matlab_v73 = mat73_loader.load_matlab_v73

# Then use it
data = load_matlab_v73('file.mat')
```

## Data Access Patterns

### Extracting Cell Array Elements

Cell arrays (like `K_DATA`, `M_DATA`, `T_DATA`) are loaded as numpy object arrays:

```python
# Use .flat for safe indexing (works with both 1D and 2D arrays)
K = data['K_DATA'].flat[struct_idx]

# Or direct indexing if you know the structure
K = data['K_DATA'][struct_idx]  # 1D array
K = data['K_DATA'][struct_idx, 0]  # 2D array from scipy.io.loadmat
```

### Extracting Scalars from Structs

```python
def extract_scalar(val):
    """Extract scalar from various nested structures."""
    if np.isscalar(val):
        return int(val)
    elif isinstance(val, np.ndarray):
        if val.ndim == 0:
            return int(val.item())
        else:
            return int(val.flatten()[0])
    else:
        return int(val)

# Use it
const = data['const']
if isinstance(const, dict):
    N_eig = extract_scalar(const['N_eig'])
else:
    N_eig = extract_scalar(const['N_eig'][0, 0][0, 0])  # scipy format
```

## Sparse Matrix Details

MATLAB sparse matrices in HDF5 are stored as:

```
SparseMatrix (HDF5 Group)
├─ data: [array of non-zero values]
├─ ir: [array of row indices]  
├─ jc: [array of column pointers]
├─ m: [number of rows]
└─ n: [number of columns]
```

The loader automatically reconstructs these as `scipy.sparse.csc_matrix`:

```python
# This is done automatically:
data_vals = group['data'][()].flatten()
ir = group['ir'][()].flatten().astype(int)
jc = group['jc'][()].flatten().astype(int)
m = int(group['m'][()])
n = int(group['n'][()])

matrix = sp.csc_matrix((data_vals, ir, jc), shape=(m, n))
```

## Common Issues and Solutions

### Issue: "scipy.sparse does not support dtype object"

**Cause:** Cell array references weren't dereferenced properly.

**Solution:** Use `mat73_loader.load_matlab_v73()` which handles this automatically.

### Issue: "IndexError: invalid index to scalar variable"

**Cause:** Trying to index into a scalar value with array indexing syntax.

**Solution:** Use the `extract_scalar()` helper function or check type before indexing.

### Issue: Data structure is unexpected

**Solution:** Run the diagnostic tool first:

```bash
python mat73_loader.py your_file.mat
```

This shows exactly how your data is structured.

## Requirements

```bash
pip install h5py>=3.0.0
```

Already included in `requirements.txt`.

## Testing

Test with your own MATLAB v7.3 file:

```python
from mat73_loader import diagnose_mat_file, load_matlab_v73

# 1. Diagnose structure
diagnose_mat_file('path/to/your/file.mat')

# 2. Load data
data = load_matlab_v73('path/to/your/file.mat', verbose=True)

# 3. Verify sparse matrices
import scipy.sparse as sp
K = data['K_DATA'].flat[0]
print(f"K is sparse: {sp.issparse(K)}")
print(f"K shape: {K.shape}, non-zeros: {K.nnz}")
```

## Backward Compatibility

The loader maintains full backward compatibility:

- ✅ Old MATLAB files (< v7.3) still load with `scipy.io.loadmat`
- ✅ New MATLAB files (v7.3) use robust h5py loader
- ✅ API is identical for both formats
- ✅ No code changes needed in existing scripts (just update imports)

## Performance

- **Loading time:** Comparable to `scipy.io.loadmat`
- **Memory:** Same as MATLAB format (data is sparse)
- **Cell array dereferencing:** Done once during load (not on every access)

## Further Reading

- [HDF5 Format Specification](https://www.hdfgroup.org/solutions/hdf5/)
- [MATLAB MAT-File Versions](https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html)
- [h5py Documentation](https://docs.h5py.org/)

---

**Created:** October 2025  
**Last Updated:** October 2025  
**Maintainer:** 2D Dispersion Analysis Package

