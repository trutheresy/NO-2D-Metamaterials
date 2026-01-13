# Dispersion Plot Generation Process

This document breaks down the step-by-step process from input ingestion to dispersion plot generation in `plot_dispersion_infer_eigenfrequencies.py`.

## Overview

The script generates dispersion plots by:
1. Loading PyTorch dataset (designs, wavevectors, eigenvectors)
2. Reconstructing frequencies from eigenvectors using K, M, T matrices
3. Extracting frequencies along the IBZ (Irreducible Brillouin Zone) contour
4. Plotting the dispersion relation

---

## Step-by-Step Process

### **STEP 1: Data Loading** (`load_pt_dataset`)

**Input:**
- `data_dir`: Path to PyTorch dataset directory
- Files expected:
  - `geometries_full.pt`: Design parameters (N_struct, N_pix, N_pix)
  - `wavevectors_full.pt`: Wavevectors (N_struct, N_wv, 2)
  - `displacements_dataset.pt` or `eigenvectors_full.pt`: Eigenvectors
  - Optional: `K_data.pt`, `M_data.pt`, `T_data.pt`

**Process:**
1. Load PyTorch tensors and convert to NumPy arrays
2. Handle different eigenvector formats:
   - TensorDataset: Extract (x_real, x_imag, y_real, y_imag) → combine to complex (N_samples, 2, H, W)
   - Direct tensor: Use as-is
3. Check for reduced dataset (has `reduced_indices.pt`)
4. Load optional K, M, T matrices if available

**Output:**
- `data` dictionary with keys: `designs`, `wavevectors`, `eigenvectors`, optionally `K_data`, `M_data`, `T_data`
- Shapes:
  - `designs`: (N_struct, N_pix, N_pix)
  - `wavevectors`: (N_struct, N_wv, 2)
  - `eigenvectors`: (N_samples, 2, H, W) for field format

---

### **STEP 2: Parameter Inference**

**Process:**
1. Infer `N_pix` from `designs.shape[1]` (assumes square designs)
2. Infer `N_eig` from eigenvector shape:
   - Field format: `N_eig = total_samples / (N_struct × N_wv)`
   - DOF format: `N_eig = eigenvectors.shape[2]`
3. Set material parameter ranges:
   - `E_min, E_max = 20e6, 200e9`
   - `rho_min, rho_max = 400, 8000`
   - `nu_min, nu_max = 0.0, 0.5`
4. Set lattice parameter `a = 1.0` (default)

---

### **STEP 3: For Each Structure - Design Plotting**

**Process:**
1. Extract `design_param = designs[struct_idx, :, :]` (N_pix, N_pix) with values in [0, 1]
2. Plot raw design (before normalization):
   - Replicate single channel to 3 channels
   - Call `plot_design(design_raw_3ch)`
3. Apply steel-rubber paradigm:
   - Call `apply_steel_rubber_paradigm_single_channel()` to get normalized values
   - Convert normalized values back to actual material property values:
     - `E = E_min + (E_max - E_min) * design_normalized[:,:,0]`
     - `rho = rho_min + (rho_max - rho_min) * design_normalized[:,:,1]`
     - `nu = nu_min + (nu_max - nu_min) * design_normalized[:,:,2]`
4. Plot normalized design with actual property values

---

### **STEP 4: Frequency Reconstruction** (`reconstruct_frequencies_from_eigenvectors`)

**Input:**
- `K`: Global stiffness matrix (sparse, full DOF space)
- `M`: Global mass matrix (sparse, full DOF space)
- `T_data`: List of transformation matrices (one per wavevector)
- `eigenvectors`: Field format (N_samples, 2, H, W) or DOF format
- `wavevectors`: (N_wv, 2)
- `N_eig`: Number of eigenvalue bands

**Process:**

#### 4a. Eigenvector Format Conversion
- **Field format** (N_samples, 2, H, W):
  1. Extract x and y components: `x_field = eigenvectors[:, 0, :, :]`, `y_field = eigenvectors[:, 1, :, :]`
  2. Flatten spatial dimensions: `x_flat = x_field.reshape(N_samples, H*W)`
  3. Convert to DOF format using `convert_field_to_dof_format()`:
     - Map each pixel (i, j) to a block of (N_ele, N_ele) nodes in reduced grid
     - Interleave x and y components: `DOF[2*node_idx] = x`, `DOF[2*node_idx+1] = y`
     - Output: (N_samples, N_dof_reduced) where `N_dof_reduced = 2 * (N_ele * N_pix)^2`
  4. Reshape to (N_struct, N_wv, N_eig, N_dof_reduced)
  5. Extract structure: `eigenvectors_struct = eigenvectors_reshaped[struct_idx]`
  6. Transpose to (N_dof, N_wv, N_eig)

- **DOF format**: Already in correct format, just extract structure and transpose

#### 4b. Matrix Computation (if not loaded)
- Create `const` dictionary with design and material parameters
- Compute K, M matrices using `compute_K_M_matrices(const)`:
  - Calls `get_system_matrices_VEC()` or `get_system_matrices_VEC_simplified()`
- Compute T matrices for each wavevector:
  - For each `wv` in `wavevectors`:
    - Call `get_transformation_matrix(wv, const)`
    - Returns transformation matrix T (full_dof × reduced_dof)

#### 4c. Frequency Reconstruction (for each wavevector and band)
For each `wv_idx` in range(n_wavevectors):
1. Get transformation matrix `T = T_data[wv_idx]`
2. Transform to reduced space:
   - `Kr = T.conj().T @ K @ T` (reduced stiffness matrix)
   - `Mr = T.conj().T @ M @ T` (reduced mass matrix)
3. For each `band_idx` in range(N_eig):
   - Extract eigenvector: `eigvec = eigenvectors[:, wv_idx, band_idx]`
   - Compute matrix-vector products:
     - `Kr_eigvec = Kr @ eigvec`
     - `Mr_eigvec = Mr @ eigvec`
   - Reconstruct eigenvalue: `eigval = ||Kr_eigvec|| / ||Mr_eigvec||`
   - Convert to frequency: `freq = sqrt(real(eigval)) / (2π)`
   - Store: `frequencies_recon[wv_idx, band_idx] = freq`

**Output:**
- `frequencies_recon`: (N_wv, N_eig) array of reconstructed frequencies

**Potential Issues:**
- Dimension mismatch if field-to-DOF conversion is incorrect
- Incorrect eigenvalue reconstruction if eigenvectors don't match reduced space
- Wrong frequency if eigenvector normalization is off

---

### **STEP 5: IBZ Contour Extraction** (`extract_grid_points_on_contour`)

**Input:**
- `wavevectors`: (N_wv, 2) - actual wavevector grid points
- `frequencies`: (N_wv, N_eig) - frequencies at each wavevector
- `contour_info`: Dictionary with IBZ contour path information
- `a`: Lattice parameter
- `tolerance`: Distance tolerance for matching points (default: 2e-3)

**Process:**
1. Get IBZ contour path:
   - Call `get_IBZ_contour_wavevectors(10, a, 'p4mm')`
   - Returns `contour_info` with:
     - `vertices`: List of vertex coordinates defining contour segments
     - `N_segment`: Number of segments
     - `vertex_labels`: Labels like ['Γ', 'X', 'M', 'Γ']

2. For each contour segment (between consecutive vertices):
   - Compute segment direction: `direction = v_end - v_start`
   - Normalize: `direction_unit = direction / ||direction||`
   - For each wavevector `wv`:
     - Compute projection onto segment: `projection = dot(wv - v_start, direction_unit)`
     - Compute perpendicular distance: `perpendicular = (wv - v_start) - projection * direction_unit`
     - If `-tolerance <= projection <= segment_length + tolerance` AND `||perpendicular|| < tolerance`:
       - Point lies on segment → add to segment points
       - Store: `(wv, frequencies[i], projection)`
   - Sort points by projection distance along segment
   - Remove duplicate points at segment boundaries (keep only first occurrence)

3. Concatenate all segments:
   - `contour_points`: (N_contour, 2) - wavevectors on contour
   - `contour_freqs`: (N_contour, N_eig) - frequencies at contour points
   - `contour_param`: (N_contour,) - parameterized distance along contour (normalized to [0, N_segment])

**Output:**
- `wavevectors_contour`: (N_contour, 2)
- `frequencies_contour`: (N_contour, N_eig)
- `contour_param`: (N_contour,) - x-axis values for plotting

**Potential Issues:**
- No points found if tolerance is too small
- Missing segments if wavevector grid doesn't cover IBZ contour
- Incorrect ordering if points are not sorted properly
- Duplicate points at segment boundaries if not handled correctly

---

### **STEP 6: Dispersion Plot Generation** (`plot_dispersion_on_contour`)

**Input:**
- `contour_info`: IBZ contour information
- `frequencies_contour`: (N_contour, N_eig) - frequencies along contour
- `contour_param`: (N_contour,) - x-axis parameter values
- `title`: Plot title string
- `mark_points`: Boolean to mark data points

**Process:**
1. Create figure and axes: `fig = plt.figure(figsize=(10, 6))`, `ax = fig.add_subplot(111)`
2. For each band `band_idx` in range(N_eig):
   - Plot line: `ax.plot(contour_param, frequencies_contour[:, band_idx], linewidth=2)`
   - If `mark_points=True`: Overlay markers `'o'` at same points
3. Add vertical lines at segment boundaries:
   - `ax.axvline(i, ...)` for `i` in range(N_segment + 1)
4. Add vertex labels:
   - Set x-ticks at vertex positions
   - Set x-tick labels to `contour_info['vertex_labels']`
5. Format axes:
   - `ax.set_xlabel('Wavevector Contour Parameter')`
   - `ax.set_ylabel('Frequency [Hz]')`
   - `ax.set_title(title)`
   - `ax.grid(True, alpha=0.3)`
6. Save figure: `fig.savefig(png_path, dpi=150, bbox_inches='tight')`

**Output:**
- PNG file: `plots/<dataset_name>_recon/dispersion/<struct_idx>_recon.png`

**Potential Issues:**
- Empty plot if `frequencies_contour` is empty
- Incorrect x-axis if `contour_param` is wrong
- Missing bands if `N_eig` is incorrect
- Discontinuous curves if contour points are not properly ordered

---

## Data Flow Summary

```
Input Files (.pt)
    ↓
load_pt_dataset()
    ↓
data = {
    'designs': (N_struct, N_pix, N_pix),
    'wavevectors': (N_struct, N_wv, 2),
    'eigenvectors': (N_samples, 2, H, W)
}
    ↓
For each structure:
    ↓
    [Design Plotting]
    design_param → apply_steel_rubber_paradigm() → plot_design()
    ↓
    [Frequency Reconstruction]
    eigenvectors (field format)
        → convert_field_to_dof_format()
        → reshape to (N_dof, N_wv, N_eig)
        → For each (wv, band):
            Kr = T' @ K @ T
            Mr = T' @ M @ T
            eigval = ||Kr @ eigvec|| / ||Mr @ eigvec||
            freq = sqrt(eigval) / (2π)
    ↓
    frequencies: (N_wv, N_eig)
    ↓
    [Contour Extraction]
    wavevectors + frequencies
        → get_IBZ_contour_wavevectors()
        → extract_grid_points_on_contour()
    ↓
    frequencies_contour: (N_contour, N_eig)
    contour_param: (N_contour,)
    ↓
    [Plotting]
    plot_dispersion_on_contour()
        → ax.plot(contour_param, frequencies_contour[:, band])
    ↓
    PNG file
```

---

## Key Functions and Their Roles

1. **`load_pt_dataset()`**: Loads PyTorch data files
2. **`apply_steel_rubber_paradigm_single_channel()`**: Maps design parameter to normalized material properties
3. **`convert_field_to_dof_format()`**: Converts field-format eigenvectors to DOF format
4. **`reconstruct_frequencies_from_eigenvectors()`**: Reconstructs frequencies from eigenvectors using K, M, T
5. **`get_IBZ_contour_wavevectors()`**: Generates IBZ contour path
6. **`extract_grid_points_on_contour()`**: Extracts frequencies at points along IBZ contour
7. **`plot_dispersion_on_contour()`**: Creates the final dispersion plot

---

## Common Error Sources

1. **Eigenvector format mismatch**: Field format vs DOF format conversion errors
2. **Dimension mismatch**: Reduced DOF space size doesn't match T matrix columns
3. **Contour extraction**: No points found if tolerance is too strict or grid doesn't cover contour
4. **Frequency reconstruction**: Incorrect eigenvalues if eigenvectors are not properly normalized
5. **Ordering issues**: Contour points not sorted correctly, causing discontinuous curves

