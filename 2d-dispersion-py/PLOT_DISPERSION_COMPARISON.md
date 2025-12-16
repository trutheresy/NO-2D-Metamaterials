# MATLAB vs Python: plot_dispersion.m Equivalent

This document compares MATLAB's `plot_dispersion.m` script with its Python equivalent, including input/output types and shapes.

## MATLAB: plot_dispersion.m

### Overview
Main script that creates comprehensive dispersion visualizations from pre-computed `.mat` files.

### Input

**File Input:**
- **Type:** MATLAB `.mat` file (v7 or v7.3 HDF5 format)
- **Path:** Provided via:
  - CLI override: `temp_data_fn.txt` file containing path string
  - Default: `"D:\Research\NO-2D-Metamaterials\test_matlab\eigenvectors.mat"`

**Data Structure in .mat file:**
```matlab
data = struct with fields:
    WAVEVECTOR_DATA:     (N_wv, 2, N_struct) - Wavevectors [kx, ky] for each structure
    EIGENVALUE_DATA:     (N_wv, N_eig, N_struct) - Frequencies for each wavevector, mode, and structure
    EIGENVECTOR_DATA:    (N_dof, N_wv, N_eig, N_struct) - Eigenvectors (optional)
    K_DATA:              {1×N_struct} cell array - Stiffness matrices (optional)
    M_DATA:              {1×N_struct} cell array - Mass matrices (optional)
    T_DATA:              {1×N_wv} cell array - Transformation matrices (optional)
    CONSTITUTIVE_DATA:   containers.Map with keys:
                         - 'modulus': (N_pix, N_pix, N_struct) - Elastic modulus
                         - 'density': (N_pix, N_pix, N_struct) - Density
                         - 'poisson': (N_pix, N_pix, N_struct) - Poisson's ratio
    const:               struct with fields:
                         - a: scalar - Lattice constant
                         - N_eig: scalar - Number of eigenvalues
                         - wavevectors: (N_wv, 2) - Wavevectors
                         - symmetry_type: string (e.g., 'p4mm')
```

**Script Parameters:**
- `struct_idxs`: `1:10` (default) - Array of structure indices to plot
- `isExportPng`: `true` (default) - Whether to save PNG files
- `png_resolution`: `150` (default) - PNG resolution in DPI

### Processing Steps

1. **Load .mat file** containing dispersion data
2. **Extract material properties** (E, rho, nu) for each structure
3. **Extract wavevectors and frequencies** for each structure
4. **Reconstruct frequencies** from eigenvectors (if K, M, T available):
   - For each wavevector and mode: `eigval = norm(Kr*eigvec)/norm(Mr*eigvec)`
   - `frequencies_recon = sqrt(eigval)/(2*pi)`
5. **Create interpolants** using `scatteredInterpolant` for each eigenvalue band
6. **Get IBZ contour wavevectors** using `get_IBZ_contour_wavevectors(N_k, a, 'p4mm')`
7. **Interpolate frequencies** to IBZ contour points
8. **Generate plots**

### Output

**Generated Plots (saved as PNG files):**

1. **Material Property Fields:**
   - **Path:** `plots/<dataset_name>_mat/constitutive_fields/<struct_idx>.png`
   - **Content:** 3-panel figure showing E, rho, nu as grayscale images
   - **Type:** `figure` → `exportgraphics()`

2. **IBZ Contour Wavevectors:**
   - **Path:** `plots/<dataset_name>_mat/contour/<struct_idx>.png`
   - **Content:** Scatter plot of wavevectors along IBZ contour
   - **Type:** `figure` → `exportgraphics()`
   - **Note:** Only plotted for first structure

3. **Original Dispersion Curves:**
   - **Path:** `plots/<dataset_name>_mat/dispersion/<struct_idx>.png`
   - **Content:** Line plot of `frequencies_contour` vs `contour_info.wavevector_parameter`
   - **Type:** `figure` → `exportgraphics()`
   - **Shape:** `(N_contour_points, N_eig)` frequencies vs `(N_contour_points,)` parameter

4. **Reconstructed Dispersion Curves (if available):**
   - **Path:** `plots/<dataset_name>_mat/dispersion/<struct_idx>_recon.png`
   - **Content:** Overlay plot showing original (blue) and reconstructed (red) frequencies
   - **Type:** `figure` → `exportgraphics()`

**Intermediate Data:**
- `wavevectors_contour`: `(N_contour_points, 2)` - IBZ contour wavevectors
- `frequencies_contour`: `(N_contour_points, N_eig)` - Interpolated frequencies
- `frequencies_recon_contour`: `(N_contour_points, N_eig)` - Reconstructed frequencies (if available)
- `contour_info`: struct with:
  - `wavevector_parameter`: `(N_contour_points,)` - Parameterization along contour
  - `N_segment`: scalar - Number of contour segments

---

## Python: plot_dispersion_mat.py

### Overview
Python equivalent script that loads MATLAB `.mat` files and creates the same visualizations.

### Input

**Function: `main(cli_data_fn=None)`**

**Parameters:**
- `cli_data_fn`: `str` or `Path` (optional) - Path to `.mat` file. If `None`, uses default or CLI argument.

**Command Line Arguments:**
```bash
python plot_dispersion_mat.py [--data_fn PATH] [--struct_idxs START:END] [--export_png] [--png_resolution DPI]
```

**Data Structure (same as MATLAB):**
```python
data = {
    'WAVEVECTOR_DATA': np.ndarray,      # Shape: (N_wv, 2, N_struct), dtype: float32/float64
    'EIGENVALUE_DATA': np.ndarray,      # Shape: (N_wv, N_eig, N_struct), dtype: float32/float64
    'EIGENVECTOR_DATA': np.ndarray,     # Shape: (N_dof, N_wv, N_eig, N_struct), dtype: complex64/complex128 (optional)
    'K_DATA': list or np.ndarray,       # List of sparse/dense matrices (optional)
    'M_DATA': list or np.ndarray,       # List of sparse/dense matrices (optional)
    'T_DATA': list or np.ndarray,       # List of sparse/dense matrices (optional)
    'CONSTITUTIVE_DATA': dict,          # Dict with keys 'modulus', 'density', 'poisson'
                                        # Each: (N_pix, N_pix, N_struct), dtype: float32/float64
    'const': dict,                      # Dict with keys: 'a', 'N_eig', 'wavevectors', 'symmetry_type'
}
```

### Processing Steps

1. **Load .mat file** using `scipy.io.loadmat` or `h5py` (for v7.3)
2. **Extract material properties** (E, rho, nu) for each structure
3. **Extract wavevectors and frequencies** for each structure
4. **Reconstruct frequencies** from eigenvectors (if K, M, T available):
   ```python
   Kr = T.conj().T @ K @ T
   Mr = T.conj().T @ M @ T
   eigval = np.linalg.norm(Kr @ eigvec) / np.linalg.norm(Mr @ eigvec)
   frequencies_recon = np.sqrt(eigval) / (2 * np.pi)
   ```
5. **Create interpolators** using `scipy.interpolate.griddata` or `RegularGridInterpolator`
6. **Get IBZ contour wavevectors** using `get_IBZ_contour_wavevectors(N_k, a, 'p4mm')`
7. **Interpolate frequencies** to IBZ contour points
8. **Generate plots** using `matplotlib`

### Output

**Generated Plots (saved as PNG files):**

1. **Material Property Fields:**
   - **Path:** `plots/<dataset_name>_mat/constitutive_fields/<struct_idx>.png`
   - **Content:** 3-panel figure using `plot_design()` function
   - **Type:** `matplotlib.figure.Figure` → `fig.savefig()`

2. **IBZ Contour Wavevectors:**
   - **Path:** `plots/<dataset_name>_mat/contour/<struct_idx>.png`
   - **Content:** Scatter plot using `plt.scatter()` or `plt.plot()`
   - **Type:** `matplotlib.figure.Figure` → `fig.savefig()`

3. **Original Dispersion Curves:**
   - **Path:** `plots/<dataset_name>_mat/dispersion/<struct_idx>.png`
   - **Content:** Line plot using `plot_dispersion()` function
   - **Type:** `matplotlib.figure.Figure` → `fig.savefig()`
   - **Function Call:**
     ```python
     fig, ax, _ = plot_dispersion(
         wn=contour_info['wavevector_parameter'],  # Shape: (N_contour_points,)
         fr=frequencies_contour,                    # Shape: (N_contour_points, N_eig)
         N_contour_segments=contour_info['N_segment'] + 1,
         ax=None
     )
     ```

4. **Reconstructed Dispersion Curves (if available):**
   - **Path:** `plots/<dataset_name>_mat/dispersion/<struct_idx>_recon.png`
   - **Content:** Overlay plot using `plt.plot()` with different styles/colors
   - **Type:** `matplotlib.figure.Figure` → `fig.savefig()`

**Intermediate Data (same shapes as MATLAB):**
- `wavevectors_contour`: `np.ndarray`, shape `(N_contour_points, 2)`, dtype `float64`
- `frequencies_contour`: `np.ndarray`, shape `(N_contour_points, N_eig)`, dtype `float64`
- `frequencies_recon_contour`: `np.ndarray`, shape `(N_contour_points, N_eig)`, dtype `float64` (if available)
- `contour_info`: `dict` with:
  - `'wavevector_parameter'`: `np.ndarray`, shape `(N_contour_points,)`, dtype `float64`
  - `'N_segment'`: `int`

---

## Key Function: plot_dispersion()

### MATLAB Equivalent
MATLAB's `plot_dispersion.m` uses `plot()` directly:
```matlab
plot(ax, contour_info.wavevector_parameter, frequencies_contour)
```

### Python Function
Python's `plotting.plot_dispersion()` function:

**Function Signature:**
```python
def plot_dispersion(wn, fr, N_contour_segments, ax=None):
```

**Inputs:**
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `wn` | `np.ndarray` | `(N_wv,)` | Wavevector parameter values (1D) |
| `fr` | `np.ndarray` | `(N_wv, N_eig)` | Frequency values for each wavevector and mode |
| `N_contour_segments` | `int` | scalar | Number of contour segments (for vertical line markers) |
| `ax` | `matplotlib.axes.Axes` (optional) | - | Axes to plot on. If `None`, creates new figure. |

**Outputs:**
| Return Value | Type | Description |
|--------------|------|-------------|
| `fig` | `matplotlib.figure.Figure` | Figure handle |
| `ax` | `matplotlib.axes.Axes` | Axes handle |
| `plot_handle` | `matplotlib.lines.Line2D` or `None` | Plot handle (first line object) |

**Usage in plot_dispersion_mat.py:**
```python
fig, ax, _ = plot_dispersion(
    wn=contour_info['wavevector_parameter'],  # (N_contour_points,)
    fr=frequencies_contour,                    # (N_contour_points, N_eig)
    N_contour_segments=contour_info['N_segment'] + 1,
    ax=None
)
fig.savefig(output_path, dpi=png_resolution)
plt.close(fig)
```

---

## Comparison Table

| Aspect | MATLAB | Python |
|--------|--------|--------|
| **Main Script** | `plot_dispersion.m` | `plot_dispersion_mat.py` |
| **File Loading** | `load(data_fn)` | `scipy.io.loadmat()` or `h5py` |
| **Interpolation** | `scatteredInterpolant()` | `scipy.interpolate.griddata()` or `RegularGridInterpolator()` |
| **IBZ Contour** | `get_IBZ_contour_wavevectors()` | `get_IBZ_contour_wavevectors()` (same function) |
| **Plotting Function** | Direct `plot()` | `plotting.plot_dispersion()` |
| **Figure Export** | `exportgraphics(fig, path, 'Resolution', dpi)` | `fig.savefig(path, dpi=dpi)` |
| **Data Types** | MATLAB native (double, single, complex) | NumPy (float64, float32, complex128, complex64) |
| **Array Shapes** | Column-major (Fortran order) | Row-major (C order) by default, but uses `order='F'` when needed |

---

## Alternative Python Scripts

### plot_dispersion_infer_eigenfrequencies.py
**Purpose:** Similar functionality but for PyTorch format datasets (`.pt` files)

**Key Differences:**
- Loads PyTorch tensors instead of MATLAB `.mat` files
- Handles different data structures (TensorDataset, etc.)
- Same plotting functionality

**Input:**
- PyTorch dataset directory containing:
  - `geometries_full.pt`
  - `wavevectors_full.pt`
  - `displacements_dataset.pt` or `eigenvectors_full.pt`
  - `K_data.pt`, `M_data.pt`, `T_data.pt` (optional)

**Output:** Same plot types as `plot_dispersion_mat.py`

---

## Summary

**Python Equivalent to MATLAB's `plot_dispersion.m`:**
- **Primary:** `plot_dispersion_mat.py` (for MATLAB `.mat` files)
- **Alternative:** `plot_dispersion_infer_eigenfrequencies.py` (for PyTorch `.pt` files)

**Core Plotting Function:**
- `plotting.plot_dispersion(wn, fr, N_contour_segments, ax=None)`
  - `wn`: `(N_wv,)` - Wavevector parameter
  - `fr`: `(N_wv, N_eig)` - Frequencies
  - Returns: `(fig, ax, plot_handle)`

**Data Flow:**
```
.mat file → load_dataset() → extract data → interpolate to IBZ contour → plot_dispersion() → save PNG
```

