# MATLAB Scripts That Generate Dispersion Plots

This document lists all MATLAB scripts in the `2D-dispersion-han` library that generate dispersion plots.

## Main Plotting Scripts

### 1. **plot_dispersion.m**
**Primary script for generating comprehensive dispersion plots**

**Purpose:** Main script that creates multiple types of plots:
- Material property fields (E, rho, nu) as images
- IBZ contour wavevector visualization
- **Dispersion curves** along IBZ contour (frequency vs wavevector parameter)
- **Reconstructed dispersion curves** (if eigenvectors are available)

**Input:** 
- Loads `.mat` file containing:
  - `WAVEVECTOR_DATA`: Wavevectors
  - `EIGENVALUE_DATA`: Frequencies
  - `EIGENVECTOR_DATA`: Eigenvectors (optional)
  - `CONSTITUTIVE_DATA`: Material properties
  - `K_DATA`, `M_DATA`, `T_DATA`: System matrices (optional, for reconstruction)

**Output:**
- Saves PNG files to `plots/<dataset_name>_mat/`:
  - `constitutive_fields/<struct_idx>.png` - Material properties
  - `contour/<struct_idx>.png` - IBZ contour wavevectors
  - `dispersion/<struct_idx>.png` - Original dispersion curves
  - `dispersion/<struct_idx>_recon.png` - Reconstructed dispersion curves (if available)

**Key Features:**
- Supports CLI override via `temp_data_fn.txt` file
- Can plot multiple structures in batch
- Interpolates frequencies to IBZ contour for visualization
- Optionally reconstructs frequencies from eigenvectors using K, M, T matrices

---

### 2. **plot_dispersion_only.m**
**Simplified version focusing only on dispersion plots**

**Purpose:** Similar to `plot_dispersion.m` but focuses only on dispersion relation plots (skips material property visualization)

**Input:** Same as `plot_dispersion.m`

**Output:**
- Same dispersion plots as `plot_dispersion.m`
- Does NOT generate material property field plots

**Key Features:**
- Streamlined version for when only dispersion curves are needed
- Same interpolation and reconstruction capabilities

---

### 3. **plot_dispersion_contour.m**
**Function for 3D contour plot of dispersion surface**

**Purpose:** Creates a 3D contour plot showing frequency surface in k-space

**Function Signature:**
```matlab
function [fig_handle, ax_handle] = plot_dispersion_contour(wv, fr, N_k_x, N_k_y, ax)
```

**Inputs:**
- `wv`: `(N, 2)` array - Wavevectors [kx, ky]
- `fr`: `(N,)` array - Frequency values for a single band
- `N_k_x`: int - Number of k-points in x-direction (optional)
- `N_k_y`: int - Number of k-points in y-direction (optional)
- `ax`: axes handle (optional) - If provided, plots on existing axes

**Outputs:**
- `fig_handle`: Figure handle
- `ax_handle`: Axes handle (3D)

**Key Features:**
- Creates 3D contour plot using `contour()` with `view(ax, 3)`
- Reshapes wavevector and frequency data to grid format
- Uses `tighten_axes()` helper function to set axis limits

---

## Wrapper Scripts

### 4. **run_plot_continuous.m**
**Wrapper to run plot_dispersion.m on continuous dataset**

**Purpose:** Convenience script that sets up and runs `plot_dispersion.m` for continuous (non-binarized) datasets

**Usage:**
```matlab
% Sets data file path and runs plot_dispersion.m
data_fn_cli = 'D:\Research\NO-2D-Metamaterials\OUTPUT\test dataset\out_continuous_1.mat';
```

**Key Features:**
- Writes data file path to `temp_data_fn.txt` for `plot_dispersion.m` to read
- Automatically cleans up temp file after execution

---

### 5. **run_plot_binarized.m**
**Wrapper to run plot_dispersion.m on binarized dataset**

**Purpose:** Convenience script that sets up and runs `plot_dispersion.m` for binarized datasets

**Usage:**
```matlab
% Sets data file path and runs plot_dispersion.m
data_fn_cli = 'D:\Research\NO-2D-Metamaterials\OUTPUT\test dataset\out_binarized_1.mat';
```

**Key Features:**
- Same mechanism as `run_plot_continuous.m` but for binarized data

---

## Utility/Test Scripts

### 6. **test_batch_output.m**
**Test script that creates simple dispersion visualizations**

**Purpose:** Quick visualization script for batch output data

**What it does:**
- Plots wavevectors using `plot_wavevectors()`
- Plots design patterns using `plot_design()`
- Creates 3D scatter plots of dispersion data:
  ```matlab
  scatter3(ax, x, y, z)  % where z = frequencies
  ```

**Key Features:**
- Simple 3D scatter plot visualization (not contour plots)
- Plots first and last 3 designs from batch
- Uses `WAVEVECTOR_DATA` and `EIGENVALUE_DATA` directly

---

## Summary Table

| Script | Type | Main Output | Notes |
|--------|------|-------------|-------|
| `plot_dispersion.m` | Main script | Dispersion curves + material properties | Most comprehensive, supports reconstruction |
| `plot_dispersion_only.m` | Main script | Dispersion curves only | Simplified version |
| `plot_dispersion_contour.m` | Function | 3D contour plot | Reusable function for single band |
| `run_plot_continuous.m` | Wrapper | Calls `plot_dispersion.m` | Convenience script |
| `run_plot_binarized.m` | Wrapper | Calls `plot_dispersion.m` | Convenience script |
| `test_batch_output.m` | Test/Utility | 3D scatter plots | Quick visualization |

---

## Related Plotting Functions (Not Dispersion-Specific)

These scripts create related visualizations but are not primarily for dispersion plots:

- **plot_eigenvector.m** - Plots eigenvector components (mode shapes)
- **plot_mode.m** - Plots mode shapes with quiver plots
- **plot_wavevectors.m** - Visualizes wavevector paths in k-space
- **plot_design.m** - Visualizes material property distributions
- **visualize_designs.m** - Grid layout of multiple designs

---

## Data Flow

```
dispersion.m or dispersion_with_matrix_save_opt.m
    ↓
    Generates .mat file with:
    - WAVEVECTOR_DATA
    - EIGENVALUE_DATA
    - EIGENVECTOR_DATA (optional)
    - K_DATA, M_DATA, T_DATA (optional)
    ↓
plot_dispersion.m or plot_dispersion_only.m
    ↓
    Loads .mat file
    Interpolates to IBZ contour
    Optionally reconstructs from eigenvectors
    ↓
    Generates PNG plots:
    - dispersion/<struct_idx>.png
    - dispersion/<struct_idx>_recon.png
```

---

## Key Differences from Python Implementation

1. **MATLAB** uses IBZ (Irreducible Brillouin Zone) contour interpolation for visualization
2. **MATLAB** supports frequency reconstruction from eigenvectors using K, M, T matrices
3. **MATLAB** saves plots to organized directory structure automatically
4. **MATLAB** uses `plot()` for 2D dispersion curves, `contour()` with `view(3)` for 3D
5. **Python** uses `plot_dispersion()` for curves, `plot_dispersion_contour()` for 2D contour (not 3D)

---

## Usage Example

```matlab
% Method 1: Direct call
data_fn = 'path/to/data.mat';
% Write to temp file
fid = fopen('temp_data_fn.txt', 'w');
fprintf(fid, '%s', data_fn);
fclose(fid);
run('plot_dispersion.m');
delete('temp_data_fn.txt');

% Method 2: Use wrapper
run('run_plot_continuous.m');

% Method 3: Function call (for contour only)
wv = ...;  % (N, 2) wavevectors
fr = ...;  % (N,) frequencies for one band
[fig, ax] = plot_dispersion_contour(wv, fr, N_k_x, N_k_y);
```

