# Input/Output Format for `plot_dispersion_infer_eigenfrequencies.py`

## Input Data Format

### Required Inputs

The script expects a **PyTorch dataset directory** (`data_dir`) containing:

#### **Required Files** (must exist):

1. **`geometries_full.pt`** (required)
   - PyTorch tensor or NumPy array
   - Shape: `(N_struct, N_pix, N_pix)`
   - Values: Design parameters in range [0, 1]
   - Single channel (normalized design values)

2. **`wavevectors_full.pt`** (required)
   - PyTorch tensor or NumPy array
   - Shape: `(N_struct, N_wv, 2)`
   - Values: Wavevector components (kx, ky) for each structure

#### **Optional Files** (for frequency reconstruction):

3. **`K_data.pt`** (optional)
   - Stiffness matrices
   - Format: List of sparse/dense matrices, or array
   - One per structure: `K_data[struct_idx]` or `K_data[struct_idx, :, :]`

4. **`M_data.pt`** (optional)
   - Mass matrices
   - Format: List of sparse/dense matrices, or array
   - One per structure: `M_data[struct_idx]` or `M_data[struct_idx, :, :]`

5. **`T_data.pt`** (optional)
   - Transformation matrices
   - Format: List of sparse/dense matrices (one per wavevector)
   - Shared across structures: `T_data[wv_idx]` or `T_data[wv_idx, :, :]`

6. **`eigenvectors_full.pt`** (optional)
   - Eigenvectors for frequency reconstruction
   - Shape: `(N_struct, N_wv, N_eig, N_dof)` or `(N_dof, N_wv, N_eig)`
   - Complex-valued

#### **For Reduced Datasets**:

7. **`reduced_indices.pt`** (optional)
   - Indices of retained structures in reduced dataset
   - Shape: `(N_reduced,)`

### Additional Input: Eigenvalue Data

The script requires **eigenvalue data** (frequencies) which can come from:

#### **Option 1: NumPy Format** (`original_data_dir`)
- File: `eigenvalue_data.npy`
- Shape: `(N_struct, N_wv, N_eig)`
- Values: Frequencies in Hz

#### **Option 2: MATLAB Format** (`original_data_dir`)
- File: `.mat` file (MATLAB v7.3 HDF5 format)
- Variable: `EIGENVALUE_DATA`
- Shape: Automatically transposed to `(N_struct, N_wv, N_eig)`
- Values: Frequencies in Hz

### Command Line Arguments

```bash
python plot_dispersion_infer_eigenfrequencies.py [data_dir] [original_dir] [options]
```

- **`data_dir`**: Path to PyTorch dataset directory (required)
- **`original_dir`**: Path to directory/file with eigenvalue data (required)
- **`-n, --n-structs`**: Number of structures to plot (default: 5)
- **`--no-compute`**: Don't compute K, M, T matrices if not available (default: compute them)

### Example Usage

```bash
# Basic usage
python plot_dispersion_infer_eigenfrequencies.py \
    /path/to/pytorch_dataset \
    /path/to/eigenvalue_data.npy

# With MATLAB file
python plot_dispersion_infer_eigenfrequencies.py \
    /path/to/pytorch_dataset \
    /path/to/data.mat

# Plot 10 structures
python plot_dispersion_infer_eigenfrequencies.py \
    /path/to/pytorch_dataset \
    /path/to/eigenvalue_data.npy \
    -n 10

# Don't compute matrices (only use if available)
python plot_dispersion_infer_eigenfrequencies.py \
    /path/to/pytorch_dataset \
    /path/to/eigenvalue_data.npy \
    --no-compute
```

## Output Data Format

### Output Directory Structure

Outputs are saved to:
```
<current_working_directory>/plots/<dataset_name>_recon/
```

Where `<dataset_name>` is the name of the input `data_dir` directory.

### Generated Files

#### **1. Design Visualizations**

**Location**: `plots/<dataset_name>_recon/design/<struct_idx>.png`

- **Format**: PNG image
- **Resolution**: 150 DPI (default)
- **Content**: 3-panel visualization showing:
  - Elastic modulus (E) field
  - Density (ρ) field  
  - Poisson's ratio (ν) field
- **One file per structure**

#### **2. Dispersion Relation Plots**

**Location**: `plots/<dataset_name>_recon/dispersion/<struct_idx>.png` or `<struct_idx>_recon.png`

- **Format**: PNG image
- **Resolution**: 150 DPI (default)
- **Content**: 
  - **Without reconstruction**: Original dispersion curves along IBZ contour
  - **With reconstruction**: Overlay of original (solid) and reconstructed (dashed) frequencies
- **Features**:
  - X-axis: Wavevector contour parameter (0 to N_segment)
  - Y-axis: Frequency [Hz]
  - Vertical lines at segment boundaries
  - Vertex labels (Γ, X, M, etc.) for p4mm symmetry
  - Legend (if reconstruction is shown)

#### **3. Console Output**

The script prints:
- Dataset structure information
- Loading progress
- Reconstruction validation errors:
  ```
  max(abs(frequencies_recon-frequencies))/max(abs(frequencies)) = <relative_error>
  ```
- Processing status for each structure
- Final output directory location

### Output File Naming Convention

- **Design plots**: `{struct_idx}.png` (e.g., `0.png`, `1.png`, ...)
- **Dispersion plots**:
  - Without reconstruction: `{struct_idx}.png`
  - With reconstruction: `{struct_idx}_recon.png`

### Example Output Structure

```
plots/
└── my_dataset_recon/
    ├── design/
    │   ├── 0.png
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    └── dispersion/
        ├── 0_recon.png      (if reconstruction successful)
        ├── 1_recon.png
        ├── 2_recon.png
        └── ...
```

## Data Flow

```
Input:
├── PyTorch Dataset Directory
│   ├── geometries_full.pt      → designs (N_struct, N_pix, N_pix)
│   ├── wavevectors_full.pt     → wavevectors (N_struct, N_wv, 2)
│   ├── K_data.pt (optional)    → K matrices
│   ├── M_data.pt (optional)    → M matrices
│   ├── T_data.pt (optional)    → T matrices
│   └── eigenvectors_full.pt (optional) → eigenvectors
│
└── Original Dataset (for eigenvalues)
    ├── eigenvalue_data.npy OR
    └── *.mat (EIGENVALUE_DATA)

Processing:
├── Load data
├── For each structure:
│   ├── Plot design → design/{struct_idx}.png
│   ├── (Optional) Reconstruct frequencies from eigenvectors
│   ├── Extract IBZ contour points
│   └── Plot dispersion → dispersion/{struct_idx}_recon.png

Output:
└── plots/<dataset_name>_recon/
    ├── design/
    └── dispersion/
```

## Matrix Computation (if not available)

If K, M, T matrices are not provided, the script can compute them on-the-fly using:

- **Material parameters** (defaults):
  - E_min = 20e6 Pa, E_max = 200e9 Pa
  - rho_min = 400 kg/m³, rho_max = 8000 kg/m³
  - nu_min = 0.05, nu_max = 0.3
  - t = 0.01 m (thickness)
  - a = 1.0 m (lattice parameter)
  - N_ele = 4 (elements per pixel)

- **Functions used**:
  - `get_system_matrices_VEC()` for K, M
  - `get_transformation_matrix()` for T

## Notes

1. **Eigenvalue data is always required** - the script cannot run without it
2. **K, M, T, eigenvectors are optional** - if missing, reconstruction is skipped (unless `compute_matrices=True`)
3. **Reduced datasets** require `original_data_dir` to load full eigenvalue data
4. **Default behavior**: Computes matrices if not available (can be disabled with `--no-compute`)
5. **Output directory** is created automatically in the current working directory



