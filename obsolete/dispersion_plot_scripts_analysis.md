# Dispersion Plot Scripts Analysis

## MATLAB Scripts

### Case 1: Eigenfrequencies ARE Known (saved in dataset)
**Script:** `2D-dispersion-han/plot_dispersion.m`

**Description:** 
- Loads saved eigenfrequencies from dataset (.mat file)
- Uses `EIGENVALUE_DATA` field which contains frequencies
- Directly plots frequencies without reconstruction
- **Note:** Also has fallback reconstruction capability (lines 178-395) if `EIGENVALUE_DATA` is missing

**Key characteristics:**
- Primary: Reads frequencies from saved `EIGENVALUE_DATA`
- Fallback: Can reconstruct if `EIGENVALUE_DATA` is missing (requires `EIGENVECTOR_DATA`, designs, const)
- Standard plotting workflow with IBZ contour extraction

**Input:** 
- `.mat` file with `EIGENVALUE_DATA` (N_wv, N_eig, N_struct)

**Output:**
- Design plots (material property fields)
- Dispersion plots (frequencies vs. IBZ contour parameter)

---

### Case 2: Eigenfrequencies are NOT Known (need to reconstruct from eigenvectors)
**Script:** `2D-dispersion-han/plot_dispersion_from_predictions.m`

**Description:**
- Specifically designed for prediction data that has NO `EIGENVALUE_DATA`
- Loads eigenvectors from predictions/dataset
- Reconstructs frequencies from eigenvectors using K, M, T matrices
- Computes frequencies using: 
  - `eigval = norm(Kr*eigvec) / norm(Mr*eigvec)` 
  - `freq = sqrt(eigval) / (2π)`
  - Where `Kr = T'*K*T` and `Mr = T'*M*T`

**Key characteristics:**
- Requires K, M, T matrices (computed from design and wavevectors)
- Reconstructs frequencies from eigenvectors
- Used when only eigenvectors are available (e.g., from ML predictions)
- Same reconstruction logic as fallback in `plot_dispersion.m`

**Input:**
- `.mat` file with `EIGENVECTOR_DATA` (dof, wv, band, struct)
- Designs and const (for computing K, M, T matrices)

**Output:**
- Design plots (material property fields)
- Dispersion plots (reconstructed frequencies vs. IBZ contour parameter)

---

## Python Scripts

### Case 1: Eigenfrequencies ARE Known (saved in dataset)
**Primary Script:** `2d-dispersion-py/plot_dispersion_mat.py`

**Description:**
- Loads MATLAB .mat files (v7.3 HDF5 format)
- Reads saved eigenfrequencies from `EIGENVALUE_DATA`
- Directly plots frequencies without reconstruction

**Equivalent to:** `plot_dispersion.m` (primary mode)

**Key characteristics:**
- Handles MATLAB v7.3 HDF5 format loading
- Uses `mat73_loader` for compatibility
- Direct plotting of saved frequencies

**Input:**
- `.mat` file with `EIGENVALUE_DATA` (N_wv, N_eig, N_struct)

**Output:**
- Design plots (material property fields)
- Dispersion plots (frequencies vs. IBZ contour parameter)

**Alternative scripts for known frequencies (different input formats):**
- `plot_dispersion_pt.py`: For PyTorch datasets (`.pt` files)
- `plot_dispersion_np.py`: For NumPy datasets
- `plot_dispersion_numpy.py`: Another NumPy variant

---

### Case 2: Eigenfrequencies are NOT Known (need to reconstruct from eigenvectors)
**Script:** `2d-dispersion-py/plot_dispersion_infer_eigenfrequencies.py`

**Description:**
- Loads PyTorch datasets with eigenvectors
- Reconstructs frequencies from eigenvectors using K, M, T matrices
- Function: `reconstruct_frequencies_from_eigenvectors()`
- Computes frequencies using same formula as MATLAB:
  - `eigval = ||Kr @ eigvec|| / ||Mr @ eigvec||`
  - `freq = sqrt(eigval) / (2π)`
  - Where `Kr = T.T @ K @ T` and `Mr = T.T @ M @ T`

**Equivalent to:** `plot_dispersion_from_predictions.m`

**Key characteristics:**
- Can compute K, M matrices from design if not saved
- Can compute T matrices from wavevectors
- Reconstructs frequencies: `freq = sqrt(eigval) / (2π)`
- Option to save computed K, M, T matrices
- No fallback to eigenvalue data (always reconstructs when in infer mode)

**Input:**
- PyTorch dataset directory with:
  - `geometries_full.pt` (designs)
  - `wavevectors_full.pt`
  - `eigenvectors` (field format or DOF format)
  - Optionally: `K_data.pt`, `M_data.pt`, `T_data.pt`

**Output:**
- Design plots (material property fields)
- Dispersion plots (reconstructed frequencies vs. IBZ contour parameter)
- Optionally: Saved K, M, T matrices

---

## Summary Table

| Case | MATLAB Script | Python Script | Description |
|------|--------------|---------------|-------------|
| **Frequencies Known** | `plot_dispersion.m` | `plot_dispersion_mat.py` | Loads and plots saved frequencies from `EIGENVALUE_DATA` |
| **Frequencies Unknown** | `plot_dispersion_from_predictions.m` | `plot_dispersion_infer_eigenfrequencies.py` | Reconstructs frequencies from eigenvectors + K, M, T matrices |

---

## Key Functions for Reconstruction

### MATLAB
- **K, M matrices:** `get_system_matrices(const)` or `get_system_matrices_VEC(const)`
- **T matrices:** `get_transformation_matrix(wavevector, const)`
- **Reconstruction:** `eigval = norm(Kr*eigvec)/norm(Mr*eigvec)`, `freq = sqrt(eigval)/(2*pi)`

### Python
- **K, M matrices:** `get_system_matrices_VEC(const)` or `get_system_matrices(const)`
- **T matrices:** `get_transformation_matrix(wavevector, const)`
- **Reconstruction:** `reconstruct_frequencies_from_eigenvectors(K, M, T_data, eigenvectors, wavevectors, N_eig, ...)`

---

## Notes

1. **MATLAB `plot_dispersion.m`** can handle both cases - it will reconstruct if `EIGENVALUE_DATA` is missing
2. **MATLAB `plot_dispersion_from_predictions.m`** is specifically for predictions with no frequencies
3. **Python `plot_dispersion_infer_eigenfrequencies.py`** always reconstructs (no fallback to saved frequencies)
4. Both libraries use the same reconstruction formula
5. Both libraries support IBZ contour extraction for plotting
