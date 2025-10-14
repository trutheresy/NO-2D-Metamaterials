# Dimension Ordering Guide: MATLAB vs Python (HDF5)

## Critical Differences in Data Structure

When MATLAB v7.3 files are loaded with h5py, the dimension ordering is **different** from what you might expect based on the MATLAB code.

### Summary Table

| Variable | MATLAB Order | HDF5/Python Order | Python Indexing |
|----------|-------------|-------------------|-----------------|
| `designs` | `(N_pix, N_pix, 3, N_struct)` | `(N_struct, 3, N_pix, N_pix)` | `[struct_idx, :, :, :].transpose(1,2,0)` |
| `WAVEVECTOR_DATA` | `(N_wv, 2, N_struct)` | `(N_struct, 2, N_wv)` | `[struct_idx, :, :].T` |
| `EIGENVALUE_DATA` | `(N_wv, N_eig, N_struct)` | `(N_struct, N_eig, N_wv)` | `[struct_idx, :, :].T` |
| `EIGENVECTOR_DATA` | `(N_dof, N_wv, N_eig, N_struct)` | `(N_struct, N_eig, N_wv, N_dof)` | `[struct_idx, band_idx, wv_idx, :]` |
| `K_DATA` | Cell `{N_struct, 1}` | Object array `(1, N_struct)` | `[0, struct_idx]` |
| `M_DATA` | Cell `{N_struct, 1}` | Object array `(1, N_struct)` | `[0, struct_idx]` |
| `T_DATA` | Cell `{N_wv, 1}` | Object array `(1, N_wv)` | `[0, wv_idx]` |

### Why This Happens

**MATLAB** stores arrays in **column-major** order (Fortran-style).  
**HDF5** stores arrays in **row-major** order (C-style).  
**NumPy** uses **row-major** order by default.

When saving MATLAB arrays to HDF5 (v7.3 format), MATLAB transposes dimensions to maintain the logical structure, but this results in different physical ordering when loaded in Python.

### Detailed Examples

#### Example 1: Designs

**MATLAB code:**
```matlab
design = designs(:, :, :, struct_idx);  % Shape: (N_pix, N_pix, 3)
```

**Python equivalent:**
```python
# HDF5 shape: (N_struct, 3, N_pix, N_pix)
design = data['designs'][struct_idx, :, :, :].transpose(1, 2, 0)  # → (N_pix, N_pix, 3)
```

#### Example 2: Wavevectors

**MATLAB code:**
```matlab
wavevectors = WAVEVECTOR_DATA(:, :, struct_idx);  % Shape: (N_wv, 2)
```

**Python equivalent:**
```python
# HDF5 shape: (N_struct, 2, N_wv)
wavevectors = data['WAVEVECTOR_DATA'][struct_idx, :, :].T  # → (N_wv, 2)
```

#### Example 3: Eigenvalues

**MATLAB code:**
```matlab
frequencies = EIGENVALUE_DATA(:, :, struct_idx);  % Shape: (N_wv, N_eig)
```

**Python equivalent:**
```python
# HDF5 shape: (N_struct, N_eig, N_wv)
frequencies = data['EIGENVALUE_DATA'][struct_idx, :, :].T  # → (N_wv, N_eig)
```

#### Example 4: Eigenvectors

**MATLAB code:**
```matlab
eigvec = EIGENVECTOR_DATA(:, wv_idx, band_idx, struct_idx);  % Shape: (N_dof,)
```

**Python equivalent:**
```python
# HDF5 shape: (N_struct, N_eig, N_wv, N_dof)
eigvec_raw = data['EIGENVECTOR_DATA'][struct_idx, band_idx, wv_idx, :]  # → (N_dof,)

# Also handle structured dtype with real/imag fields
if eigvec_raw.dtype.names and 'real' in eigvec_raw.dtype.names:
    eigvec = eigvec_raw['real'] + 1j * eigvec_raw['imag']
else:
    eigvec = eigvec_raw
```

#### Example 5: Cell Arrays (K_DATA, M_DATA, T_DATA)

**MATLAB code:**
```matlab
K = K_DATA{struct_idx};
M = M_DATA{struct_idx};
T = T_DATA{wv_idx};
```

**Python equivalent:**
```python
# HDF5 shape: (1, N_struct) or (1, N_wv)
K = data['K_DATA'][0, struct_idx]  # Already a sparse matrix!
M = data['M_DATA'][0, struct_idx]
T = data['T_DATA'][0, wv_idx]
```

### Complex Numbers in HDF5

MATLAB complex numbers are stored as **structured arrays** in HDF5:

```python
dtype([('real', '<f8'), ('imag', '<f8')])
```

**Always convert** to proper complex:

```python
if data.dtype.names and 'real' in data.dtype.names:
    data_complex = data['real'] + 1j * data['imag']
```

### Verification Script

After loading data, verify dimensions:

```python
print("Dataset structure:")
print(f"  designs: {data['designs'].shape}")
print(f"  WAVEVECTOR_DATA: {data['WAVEVECTOR_DATA'].shape}")
print(f"  EIGENVALUE_DATA: {data['EIGENVALUE_DATA'].shape}")
print(f"  EIGENVECTOR_DATA: {data['EIGENVECTOR_DATA'].shape}")
print(f"  K_DATA: {data['K_DATA'].shape}")
print(f"  M_DATA: {data['M_DATA'].shape}")
print(f"  T_DATA: {data['T_DATA'].shape}")
```

### Quick Reference for Correct Indexing

```python
# For a specific structure index:
struct_idx = 0

# Extract design
design = data['designs'][struct_idx, :, :, :].transpose(1, 2, 0)

# Extract wavevectors
wavevectors = data['WAVEVECTOR_DATA'][struct_idx, :, :].T

# Extract frequencies
frequencies = data['EIGENVALUE_DATA'][struct_idx, :, :].T

# Extract K, M
K = data['K_DATA'][0, struct_idx]
M = data['M_DATA'][0, struct_idx]

# For specific wavevector and band:
wv_idx = 0
band_idx = 0

# Extract T
T = data['T_DATA'][0, wv_idx]

# Extract eigenvector
eigvec_raw = data['EIGENVECTOR_DATA'][struct_idx, band_idx, wv_idx, :]
if eigvec_raw.dtype.names:
    eigvec = eigvec_raw['real'] + 1j * eigvec_raw['imag']
else:
    eigvec = eigvec_raw
```

### Why Not Just Transpose Everything?

We **could** transpose all arrays after loading to match MATLAB ordering, but this:
- Uses extra memory
- Adds processing time
- Makes debugging harder

Instead, we **understand the HDF5 structure** and index correctly from the start.

### Tools for Debugging

**1. Print dimensions:**
```python
from mat73_loader import load_matlab_v73

data = load_matlab_v73('file.mat')

# Print all shapes
for key in data.keys():
    if hasattr(data[key], 'shape'):
        print(f"{key}: {data[key].shape}")
```

**2. Use diagnostic script:**
```bash
python debug_mat_file.py your_file.mat
```

**3. Enable verbose loading:**
```python
data = load_matlab_v73('file.mat', verbose=True)
```

### Summary

✅ **Always check dimensions** after loading HDF5 files  
✅ **Use transpose/reorder** as needed for correct shape  
✅ **Handle structured dtypes** for complex numbers  
✅ **Index cell arrays** correctly: `[0, idx]` for HDF5  
✅ **Verify with print statements** during development  

---

**Created:** October 2025  
**Last Updated:** October 2025  
**Related:** MATLAB_V73_LOADING_GUIDE.md, TROUBLESHOOTING.md

