# Troubleshooting Guide

## Common Issues When Running Scripts

### Issue 1: Import Errors with Relative Imports

**Error:**
```
ImportError: attempted relative import with no known parent package
```

**Cause:** Using relative imports (`.utils`) when running scripts directly.

**Solution:** Already fixed! All modules now handle both package and script modes:
```python
try:
    from .utils import linspaceNDim
except ImportError:
    from utils import linspaceNDim
```

### Issue 2: MATLAB v7.3 Files Not Loading

**Error:**
```
NotImplementedError: Please use HDF reader for matlab v7.3 files, e.g. h5py
```

**Cause:** MATLAB v7.3 files use HDF5 format, which `scipy.io.loadmat` cannot read.

**Solution:** Install h5py and use `mat73_loader`:
```bash
pip install h5py>=3.0.0
```

Scripts now automatically detect and use the correct loader.

### Issue 3: Void128/Void Dtype Errors

**Error:**
```
ValueError: scipy.sparse does not support dtype void128
```

**Cause:** MATLAB stores complex numbers as void128 in HDF5 (binary representation of complex128).

**Solution:** Already fixed in `mat73_loader.py`:
- Detects void types
- Converts to proper complex128 using `frombuffer` or `view`
- Runtime fallback in scripts with `.astype(np.complex128)`

**Multiple layers of defense:**
1. Loader converts void → complex128 during load
2. Script detects and converts void → complex128 at runtime
3. Clear error messages if conversion fails

### Issue 4: Object Dtype in Sparse Matrices

**Error:**
```
ValueError: scipy.sparse does not support dtype object
```

**Cause:** HDF5 object references weren't dereferenced properly.

**Solution:** Use `mat73_loader.load_matlab_v73()` which:
- Recursively dereferences HDF5 object references
- Reconstructs sparse matrices from data/ir/jc fields
- Handles cell arrays properly

### Issue 5: plot_design Returns Tuple

**Error:**
```
AttributeError: 'tuple' object has no attribute 'savefig'
```

**Cause:** `plot_design()` returns `(fig, axes)` tuple, not just `fig`.

**Solution:**
```python
# Correct:
fig, _ = plot_design(design)
fig.savefig('output.png')
```

## Debugging Tools

### 1. Diagnose MATLAB File Structure

See exactly what's in your file:

```bash
python mat73_loader.py path/to/your/file.mat
```

### 2. Verbose Loading

Enable detailed loading information:

```python
from mat73_loader import load_matlab_v73

data = load_matlab_v73('file.mat', verbose=True)
```

### 3. Check Loaded Data Types

After loading, verify data types:

```python
import scipy.sparse as sp

data = load_matlab_v73('file.mat')

# Check K_DATA
K = data['K_DATA'].flat[0]
print(f"K is sparse: {sp.issparse(K)}")
print(f"K dtype: {K.dtype}")
print(f"K shape: {K.shape}")

# Check for problematic dtypes
if K.dtype.kind == 'V':
    print("WARNING: K has void dtype - need conversion!")
elif K.dtype.kind == 'O':
    print("WARNING: K has object dtype - loading failed!")
```

## Simplified Workflow

For maximum reliability, follow this pattern:

```python
# 1. Import robust loader
from mat73_loader import load_matlab_v73

# 2. Load with verbose mode for first run
data = load_matlab_v73('file.mat', verbose=True)

# 3. Extract data with .flat indexing (works for both 1D and 2D arrays)
K = data['K_DATA'].flat[struct_idx]
M = data['M_DATA'].flat[struct_idx]  
T = data['T_DATA'].flat[wv_idx]

# 4. Check and fix dtypes if needed
import scipy.sparse as sp

if sp.issparse(T) and T.dtype.kind == 'V':
    T = T.astype(np.complex128)  # Convert void to complex

# 5. Use the matrices
Kr = T.conj().T @ K @ T
```

## System Requirements

### Python Environment

Tested and working with:
- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- h5py 3.0+
- Matplotlib 3.3+

### Conda Environment

If you encounter h5py/numpy compatibility issues:

```bash
# Create fresh environment
conda create -n dispersion python=3.10
conda activate dispersion

# Install packages
pip install numpy scipy matplotlib h5py scikit-learn
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

## Performance Tips

### Memory Management

When processing many structures, close figures to free memory:

```python
fig, _ = plot_design(design)
fig.savefig('output.png')
plt.close(fig)  # Important for multiple structures!
```

### Large Datasets

For large datasets with many structures:
- Process in batches
- Use `isExportPng=True` to save and close figures
- Don't show plots with `plt.show()` until the end

## Getting Help

If you encounter a new error:

1. **Run diagnostic:**
   ```bash
   python mat73_loader.py your_file.mat
   ```

2. **Enable verbose loading:**
   ```python
   data = load_matlab_v73('file.mat', verbose=True)
   ```

3. **Check data types:**
   ```python
   print(f"K_DATA type: {type(data['K_DATA'])}")
   print(f"K_DATA shape: {data['K_DATA'].shape}")
   print(f"K_DATA[0] type: {type(data['K_DATA'].flat[0])}")
   ```

4. **Report issue** with the diagnostic output.

## Known Limitations

- MATLAB objects/classes: Not fully supported (converted to dicts)
- Function handles: Not supported
- Some esoteric MATLAB types: May require custom handling

## Success Indicators

When everything works correctly, you should see:

```
Loading dataset: your_file.mat
  File is MATLAB v7.3 format, using robust h5py loader...
  Loaded using h5py (MATLAB v7.3)

Reconstructing frequencies for structure 0...
  K: (2178, 2178), 35430 non-zeros, dtype=complex128
  M: (2178, 2178), 18818 non-zeros, dtype=complex128
  T[0]: type=<class 'scipy.sparse._csc.csc_matrix'>, sparse=True
  T[0]: shape=(2178, 2178), dtype=complex128, nnz=4356
```

All dtypes should be `complex128`, `float64`, or other numeric types (NOT `void128` or `object`).

---

**Last Updated:** October 2025  
**Maintainer:** 2D Dispersion Analysis Package

