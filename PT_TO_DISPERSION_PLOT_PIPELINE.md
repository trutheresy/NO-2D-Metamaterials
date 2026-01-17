# Pipeline: PyTorch Files → Dispersion Plot

## Overview

This document explains the complete pipeline from PyTorch `.pt` files to dispersion plots, including all files and steps involved.

---

## Input: PyTorch Files

The pipeline starts with a directory containing PyTorch files:

```
data/out_test_10_pt/
├── geometries_full.pt          # Single-channel designs (n_designs, N_pix, N_pix)
├── wavevectors_full.pt         # Wavevector coordinates (n_designs, n_wavevectors, 2)
├── displacements.pt            # TensorDataset with eigenvector components
│   ├── tensors[0]: eigenvector_x_real
│   ├── tensors[1]: eigenvector_x_imag
│   ├── tensors[2]: eigenvector_y_real
│   └── tensors[3]: eigenvector_y_imag
├── bands_fft_full.pt           # Band information
├── design_params_full.pt      # Design parameters
└── reduced_indices.pt          # (design_idx, wavevector_idx, band_idx) tuples
```

---

## Step-by-Step Pipeline

### STEP 1: Load PyTorch Files
**File**: `reduced_pt_to_matlab.py` (lines 250-300)

**Operations**:
1. `torch.load()` - Load each .pt file
2. `.numpy()` - Convert PyTorch tensors to NumPy arrays
3. Extract dimensions: `n_designs`, `design_res`, `n_wavevectors`, `n_bands`

**Output Variables**:
- `geometries_np`: (n_designs, design_res, design_res), dtype: float16
- `wavevectors_np`: (n_designs, n_wavevectors, 2), dtype: float16
- `eigenvector_x_real`, `eigenvector_x_imag`: (n_samples, design_res, design_res)
- `eigenvector_y_real`, `eigenvector_y_imag`: (n_samples, design_res, design_res)
- `reduced_indices`: List of (design_idx, wavevector_idx, band_idx) tuples

---

### STEP 2: Reconstruct Full EIGENVECTOR_DATA
**File**: `reduced_pt_to_matlab.py` (lines 302-361)

**Operations**:
1. **Initialize arrays**: Create zero-filled arrays for full eigenvector data
   ```python
   EIGENVECTOR_DATA_x_full = np.zeros((n_designs, n_wavevectors, n_bands, design_res, design_res), dtype=np.complex128)
   EIGENVECTOR_DATA_y_full = np.zeros((n_designs, n_wavevectors, n_bands, design_res, design_res), dtype=np.complex128)
   ```

2. **Combine real and imaginary parts**:
   ```python
   eigenvector_x_complex = (eigenvector_x_real + 1j * eigenvector_x_imag).astype(np.complex128)
   eigenvector_y_complex = (eigenvector_y_real + 1j * eigenvector_y_imag).astype(np.complex128)
   ```

3. **Place eigenvectors at correct indices**:
   ```python
   for sample_idx in range(n_samples):
       d_idx, w_idx, b_idx = indices_np[sample_idx]
       EIGENVECTOR_DATA_x_full[d_idx, w_idx, b_idx, :, :] = eigenvector_x_complex[sample_idx]
       EIGENVECTOR_DATA_y_full[d_idx, w_idx, b_idx, :, :] = eigenvector_y_complex[sample_idx]
   ```

4. **Flatten spatial dimensions**:
   ```python
   EIGENVECTOR_DATA_x_flat = EIGENVECTOR_DATA_x_full.reshape(n_designs, n_wavevectors, n_bands, -1)
   EIGENVECTOR_DATA_y_flat = EIGENVECTOR_DATA_y_full.reshape(n_designs, n_wavevectors, n_bands, -1)
   ```

5. **Interleave x and y components**:
   ```python
   EIGENVECTOR_DATA_combined[:, :, :, 0::2] = EIGENVECTOR_DATA_x_flat  # Even indices = x
   EIGENVECTOR_DATA_combined[:, :, :, 1::2] = EIGENVECTOR_DATA_y_flat  # Odd indices = y
   ```

6. **Transpose to MATLAB format**:
   ```python
   EIGENVECTOR_DATA = EIGENVECTOR_DATA_combined.transpose(0, 2, 1, 3)
   # From (n_designs, n_wavevectors, n_bands, n_dof) 
   # To   (n_designs, n_bands, n_wavevectors, n_dof)
   ```

**Output**: `EIGENVECTOR_DATA` - Shape: (n_designs, n_bands, n_wavevectors, n_dof), dtype: complex128

---

### STEP 3: Convert Design (Steel-Rubber Paradigm)
**File**: `reduced_pt_to_matlab.py` (lines 387-400)

**Function**: `apply_steel_rubber_paradigm()`

**Operations**:
1. **Extract single-channel design**:
   ```python
   design_param = geometries_np[struct_idx]  # (design_res, design_res), dtype: float16
   design_param = design_param.astype(np.float64)  # Convert to float64
   ```

2. **Apply steel-rubber paradigm**:
   - Maps single-channel design [0, 1] to 3-channel design
   - Channel 0: Elastic modulus (E_polymer=100e6 → E_steel=200e9)
   - Channel 1: Density (rho_polymer=1200 → rho_steel=8000)
   - Channel 2: Poisson's ratio (nu_polymer=0.45 → nu_steel=0.3)
   - Uses linear interpolation: `np.interp(design_flat, [0, 1], [val_polymer, val_steel])`

**Output**: `design_3ch` - Shape: (N_pix, N_pix, 3), dtype: float64

---

### STEP 4: Compute K and M Matrices
**File**: `reduced_pt_to_matlab.py` (lines 402-424)

**Function**: `get_system_matrices_VEC()` from `2d-dispersion-py/system_matrices_vec.py`

**Operations**:
1. **Create const dictionary**:
   ```python
   const_for_km = {
       'design': design_3ch,
       'N_pix': design_res,
       'N_ele': N_ele,
       'a': a_val,
       'E_min': E_min, 'E_max': E_max,
       'rho_min': rho_min, 'rho_max': rho_max,
       'poisson_min': nu_min, 'poisson_max': nu_max,
       't': t_val,
       'design_scale': 'linear',
       'isUseImprovement': True,
       'isUseSecondImprovement': False
   }
   ```

2. **Call matrix computation**:
   ```python
   K, M = get_system_matrices_VEC(const_for_km)
   ```

**Inside `get_system_matrices_VEC()`** (`system_matrices_vec.py`):

   a. **Expand design** (repelem):
      ```python
      design_expanded = np.repeat(np.repeat(design, N_ele, axis=0), N_ele, axis=1)
      ```

   b. **Extract material properties**:
      ```python
      E = (E_min + design_ch0 * (E_max - E_min)).T.astype(np.float32)
      rho = (rho_min + design_ch1 * (rho_max - rho_min)).T.astype(np.float32)
      nu = (poisson_min + design_ch2 * (poisson_max - poisson_min)).T.astype(np.float32)
      ```

   c. **Create node numbering**:
      ```python
      nodenrs = np.arange(1, (1 + N_ele_x) * (1 + N_ele_y) + 1).reshape(1 + N_ele_y, 1 + N_ele_x, order='F')
      ```

   d. **Create element DOF indices**:
      ```python
      edofVec = (2 * nodenrs[0:-1, 0:-1] - 1).reshape(N_ele_x * N_ele_y, 1, order='F').flatten()
      edofMat = np.tile(edofVec.reshape(-1, 1), (1, 8)) + np.tile(offset_array, (N_ele_x * N_ele_y, 1))
      ```

   e. **Create row/column indices**:
      ```python
      row_idxs = np.reshape(np.kron(edofMat, np.ones((8, 1))).T, 64 * N_ele_x * N_ele_y, order='F')
      col_idxs = np.reshape(np.kron(edofMat, np.ones((1, 8))).T, 64 * N_ele_x * N_ele_y, order='F')
      ```

   f. **Compute element matrices**:
      ```python
      AllLEle = get_element_stiffness_VEC(E.flatten(), nu.flatten(), t)  # From elements_vec.py
      AllLMat = get_element_mass_VEC(rho.flatten(), t, const)            # From elements_vec.py
      ```

   g. **Assemble global matrices**:
      ```python
      value_K = AllLEle.flatten(order='F')
      value_M = AllLMat.flatten(order='F')
      K = coo_matrix((value_K, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float32)
      M = coo_matrix((value_M, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float32)
      ```

**Output**: 
- `K`: Sparse matrix, shape (N_dof, N_dof), dtype: float32
- `M`: Sparse matrix, shape (N_dof, N_dof), dtype: float32

---

### STEP 5: Compute T Matrices
**File**: `reduced_pt_to_matlab.py` (lines 426-436)

**Function**: `get_transformation_matrix()` from `2d-dispersion-py/system_matrices.py`

**Operations** (for each wavevector):
1. **Extract wavevector**:
   ```python
   wv = wavevectors_np[struct_idx, wv_idx, :].astype(np.float32)  # (2,)
   ```

2. **Call transformation matrix function**:
   ```python
   T = get_transformation_matrix(wv, const_for_km)
   ```

**Inside `get_transformation_matrix()`** (`system_matrices.py`):

   a. **Compute N_node**:
      ```python
      N_node = const['N_ele'] * N_pix_val + 1
      ```

   b. **Compute phase factors**:
      ```python
      r_x = np.array([const['a'], 0], dtype=np.float32)
      r_y = np.array([0, -const['a']], dtype=np.float32)
      r_corner = np.array([const['a'], -const['a']], dtype=np.float32)
      xphase = np.exp(1j * np.dot(wavevector, r_x)).astype(np.complex64)
      yphase = np.exp(1j * np.dot(wavevector, r_y)).astype(np.complex64)
      cornerphase = np.exp(1j * np.dot(wavevector, r_corner)).astype(np.complex64)
      ```

   c. **Generate node indices**:
      ```python
      temp_x, temp_y = np.meshgrid(np.arange(1, N_node), np.arange(1, N_node), indexing='ij')
      node_idx_x = np.concatenate([temp_x.flatten(order='F'), ...])
      node_idx_y = np.concatenate([temp_y.flatten(order='F'), ...])
      ```

   d. **Compute global node and DOF indices**:
      ```python
      global_node_idx = (node_idx_y - 1) * N_node + node_idx_x
      global_dof_idxs = np.concatenate([2 * global_node_idx - 1, 2 * global_node_idx])
      ```

   e. **Compute reduced indices**:
      ```python
      reduced_global_node_idx = [...]
      reduced_global_dof_idxs = np.concatenate([2 * reduced_global_node_idx - 1, 2 * reduced_global_node_idx])
      ```

   f. **Build transformation matrix**:
      ```python
      phase_factors = np.concatenate([np.ones(...), np.full(..., xphase), ...])
      value_T = np.tile(phase_factors, 2).astype(np.complex64)
      T = csr_matrix((value_T, (row_idxs, col_idxs)), shape=(N_dof_full, N_dof_reduced), dtype=np.complex64)
      ```

**Output**: `T_data` - List of sparse matrices, one per wavevector
- Each T: Shape (N_dof_full, N_dof_reduced), dtype: complex64

---

### STEP 6: Reconstruct EIGENVALUE_DATA
**File**: `reduced_pt_to_matlab.py` (lines 438-493)

**Operations** (for each structure, wavevector, and band):

1. **Extract eigenvector**:
   ```python
   eigenvectors_struct = EIGENVECTOR_DATA[struct_idx, :, :, :]  # (n_bands, n_wavevectors, n_dof)
   eigenvectors_struct = eigenvectors_struct.transpose(2, 1, 0)  # (n_dof, n_wavevectors, n_bands)
   eigvec = eigenvectors_struct[:, wv_idx, band_idx].astype(np.complex128)
   ```

2. **Convert matrices to sparse**:
   ```python
   T_sparse = T if sp.issparse(T) else sp.csr_matrix(T)
   K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
   M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
   ```

3. **Compute reduced matrices**:
   ```python
   Kr = T_sparse.conj().T @ K_sparse @ T_sparse  # T^H * K * T
   Mr = T_sparse.conj().T @ M_sparse @ T_sparse  # T^H * M * T
   ```

4. **Compute eigenvalue using Rayleigh quotient**:
   ```python
   Kr_eigvec = Kr @ eigvec
   Mr_eigvec = Mr @ eigvec
   
   if sp.issparse(Kr_eigvec):
       Kr_eigvec = Kr_eigvec.toarray().flatten()
   if sp.issparse(Mr_eigvec):
       Mr_eigvec = Mr_eigvec.toarray().flatten()
   
   # Rayleigh quotient (CORRECT FORMULA)
   eigval = (eigvec.conj() @ (Kr @ eigvec)) / (eigvec.conj() @ (Mr @ eigvec))
   ```

5. **Convert to frequency**:
   ```python
   freq = np.sqrt(np.real(eigval)) / (2 * np.pi)
   frequencies_recon[wv_idx, band_idx] = freq
   ```

6. **Store and transpose**:
   ```python
   EIGENVALUE_DATA[struct_idx, :, :] = frequencies_recon  # (n_wavevectors, n_bands)
   EIGENVALUE_DATA = EIGENVALUE_DATA.transpose(0, 2, 1)    # (n_designs, n_bands, n_wavevectors)
   ```

**Output**: `EIGENVALUE_DATA` - Shape: (n_designs, n_bands, n_wavevectors), dtype: float64

---

### STEP 7: Save to MATLAB Format
**File**: `reduced_pt_to_matlab.py` (lines 495-550)

**Operations**:
1. **Prepare all data for saving**:
   - `EIGENVALUE_DATA`: Already in correct format
   - `EIGENVECTOR_DATA`: Convert to structured dtype (real/imag)
   - `designs`: Transpose to MATLAB format (3, H, W) → (H, W, 3)
   - `WAVEVECTOR_DATA`: Handle shape appropriately
   - `const`: Convert dict to MATLAB struct format

2. **Save using h5py**:
   ```python
   with h5py.File(output_path, 'w') as f:
       f.create_dataset('EIGENVALUE_DATA', data=EIGENVALUE_DATA, dtype=np.float64)
       f.create_dataset('EIGENVECTOR_DATA', data=eigvec_structured, dtype=eigvec_dtype)
       # ... save all other fields
   ```

**Output**: `.mat` file (MATLAB v7.3 HDF5 format)

---

### STEP 8: Generate Dispersion Plots
**File**: `plot_all_dispersion_comparisons.py` or `2d-dispersion-py/plot_dispersion_with_eigenfrequencies.py`

**Operations**:
1. **Load .mat file**:
   ```python
   with h5py.File(mat_file, 'r') as f:
       eig_data = np.array(f['EIGENVALUE_DATA'])
       wavevectors = np.array(f['WAVEVECTOR_DATA'])
       # ... load other data
   ```

2. **Get IBZ contour**:
   ```python
   from wavevectors import get_IBZ_contour_wavevectors
   contour_wv, contour_info = get_IBZ_contour_wavevectors(50, a, 'p4mm')
   ```

3. **Interpolate frequencies to contour**:
   ```python
   from scipy.interpolate import griddata
   frequencies_contour = griddata(
       wavevectors, 
       frequencies,
       contour_wv,
       method='linear',
       fill_value=np.nan
   )
   ```

4. **Plot dispersion curves**:
   ```python
   from plotting import plot_dispersion
   fig, ax = plt.subplots(figsize=(10, 6))
   plot_dispersion(
       wn=contour_info['wavevector_parameter'],
       fr=frequencies_contour,
       N_contour_segments=contour_info['N_segment'] + 1,
       ax=ax
   )
   plt.savefig(output_path, dpi=150, bbox_inches='tight')
   ```

**Output**: PNG files with dispersion plots

---

## Complete File Flow

```
PyTorch Files (.pt)
    ↓
reduced_pt_to_matlab.py
    ├── Load .pt files (Step 1)
    ├── Reconstruct EIGENVECTOR_DATA (Step 2)
    ├── Convert designs (Step 3)
    ├── Compute K, M matrices (Step 4)
    │   └── system_matrices_vec.py
    │       └── elements_vec.py (element matrices)
    ├── Compute T matrices (Step 5)
    │   └── system_matrices.py
    ├── Reconstruct EIGENVALUE_DATA (Step 6)
    └── Save to .mat file (Step 7)
    ↓
MATLAB .mat file
    ↓
plot_all_dispersion_comparisons.py
    ├── Load .mat file
    ├── Get IBZ contour
    │   └── wavevectors.py
    ├── Interpolate to contour
    └── Plot dispersion
        └── plotting.py
    ↓
Dispersion Plot PNG files
```

---

## Key Files and Their Roles

### Core Conversion Script
- **`reduced_pt_to_matlab.py`**: Main script that orchestrates the entire conversion

### Matrix Computation
- **`2d-dispersion-py/system_matrices_vec.py`**: Computes K and M matrices
- **`2d-dispersion-py/elements_vec.py`**: Computes element-level stiffness and mass matrices
- **`2d-dispersion-py/system_matrices.py`**: Computes T (transformation) matrices

### Plotting
- **`plot_all_dispersion_comparisons.py`**: Generates comparison plots
- **`2d-dispersion-py/wavevectors.py`**: Computes IBZ contour wavevectors
- **`2d-dispersion-py/plotting.py`**: Plotting utilities

### Data Format
- **Input**: PyTorch `.pt` files (float16, reduced format)
- **Intermediate**: NumPy arrays (float32/float64, complex64/complex128)
- **Output**: MATLAB `.mat` file (v7.3 HDF5 format)

---

## Data Type Conversions

1. **float16 → float64**: Designs, wavevectors (for computation)
2. **float16 → complex128**: Eigenvectors (for computation)
3. **float32**: K, M matrices (sparse, for efficiency)
4. **complex64**: T matrices (sparse, complex for phase factors)
5. **float64**: Final EIGENVALUE_DATA (for accuracy)

---

## Summary

The pipeline transforms reduced PyTorch data (stored in float16 for memory efficiency) into a complete MATLAB dataset with:
- Full eigenvector data (reconstructed from reduced indices)
- Regenerated eigenvalues (using Rayleigh quotient)
- All original data (designs, wavevectors, constants)

The key insight is that eigenvalues are **reconstructed** from K, M, T matrices and eigenvectors, not stored directly in the PyTorch format. This reconstruction uses the Rayleigh quotient formula, which is numerically stable for approximate eigenvectors.

