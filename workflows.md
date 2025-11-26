# Workflows Guide

This document outlines which scripts and notebooks to use for common tasks in the NO-2D-Metamaterials project.

## Table of Contents
1. [Converting MATLAB to PyTorch Data Formats](#converting-matlab-to-pytorch-data-formats)
2. [Generating Dispersion Plots](#generating-dispersion-plots)

---

## Converting MATLAB to PyTorch Data Formats

### Primary Script: `2d-dispersion-py/convert_mat_to_pytorch.py`

**When to use:** Use this script for converting individual MATLAB `.mat` files to PyTorch format. This is the recommended approach for most use cases.

**Features:**
- Processes a single `.mat` file at a time
- Uses `NO_utils.extract_data()` for loading (same as notebook)
- Uses `NO_utils_multiple.embed_2const_wavelet()` and `embed_integer_wavelet()` for wavelet embedding
- Supports dataset reduction (WVR, BR parameters)
- Saves as PyTorch `.pt` files
- Command-line interface with full control over parameters

**Usage:**
```bash
python 2d-dispersion-py/convert_mat_to_pytorch.py <path_to_mat_file> [options]

# Example: Convert with default settings (no reduction, output to data/)
python 2d-dispersion-py/convert_mat_to_pytorch.py "OUTPUT/test dataset/out_binarized_1.mat"

# Example: Convert with reduction
python 2d-dispersion-py/convert_mat_to_pytorch.py "OUTPUT/test dataset/out_binarized_1.mat" --wvr 0.66 --br 0.5

# Example: Custom output location
python 2d-dispersion-py/convert_mat_to_pytorch.py "OUTPUT/test dataset/out_binarized_1.mat" --output custom_output_folder
```

**Output:**
- Creates folder: `data/[filename_without_extension]/`
- Files saved:
  - `displacements_dataset.pt` - TensorDataset with eigenvector components
  - `reduced_indices.pt` - List of (design_idx, wavevector_idx, band_idx) tuples
  - `geometries_full.pt` - Design arrays (N_struct, N_pix, N_pix) [float16]
  - `waveforms_full.pt` - Wavelet-embedded wavevectors (N_wv, N_pix, N_pix) [float16]
  - `wavevectors_full.pt` - Wavevector data (N_struct, N_wv, 2) [float16]
  - `band_fft_full.pt` - Wavelet-embedded band indices (N_bands, N_pix, N_pix) [float16]
  - `design_params_full.pt` - Design parameters [float16]

**Default settings:**
- WVR (Wavevector Reduction Ratio): 1.0 (no reduction)
- BR (Band Reduction Ratio): 1.0 (no reduction)
- DO (Dataset Offset): 0
- Output: `data/[filename]/`
- Precision: float16

---

### Alternative: `matlab_to_reduced_pt.ipynb`

**When to use:** Use this notebook when you need to:
- Process multiple `.mat` files in batch from the same folder
- Have more interactive control during conversion
- Want to visualize the conversion process step-by-step
- Need to inspect intermediate results

**Features:**
- Processes multiple `.mat` files in a folder
- Each file gets its own output folder
- Same functionality as the Python script but in notebook format
- Better for exploratory work and debugging

**Usage:**
1. Open the notebook in Jupyter
2. Set configuration in the notebook cells:
   - `matlab_input_folder` - Folder containing `.mat` files
   - `output_base_folder` - Base output folder
   - `WVR`, `BR`, `DO` - Reduction parameters
3. Run all cells

**Output:** Same format as the Python script, but processes all files in the input folder.

---

## Generating Dispersion Plots

### MATLAB Script: `2D-dispersion-han/plot_dispersion.m`

**When to use:** Use this MATLAB script when:
- You have MATLAB `.mat` dataset files (original format)
- You want plots that match the original MATLAB workflow
- You need reconstruction plots (comparing original vs reconstructed from eigenvectors)
- You're working with datasets that have K_DATA, M_DATA, T_DATA available

**Features:**
- Loads MATLAB `.mat` files directly
- Plots constitutive fields (Elastic Modulus, Density, Poisson Ratio)
- Plots IBZ contour wavevectors
- Plots dispersion relations along IBZ high-symmetry path
- Can reconstruct frequencies from eigenvectors (if K/M/T data available)
- Saves PNG files with organized folder structure

**Usage:**
1. Create a wrapper script (e.g., `run_plot_binarized.m`):
```matlab
data_fn_cli = 'D:\Research\NO-2D-Metamaterials\OUTPUT\test dataset\out_binarized_1.mat';
run('plot_dispersion.m');
```

2. Run in MATLAB:
```bash
matlab -batch "run_plot_binarized"
```

**Output:**
- Creates folder: `OUTPUT/plots_[dataset_name]/`
- Subfolders:
  - `constitutive_fields/` - Material property visualizations (1.png, 2.png, ...)
  - `contour/` - IBZ contour wavevector plot (1.png)
  - `dispersion/` - Dispersion plots (1.png, 2.png, ..., [N]_recon.png)

**Configuration:**
- Edit `struct_idxs = 1:10;` in the script to change which structures to plot
- Set `isExportPng = true/false` to enable/disable PNG export
- Set `png_resolution = 150;` to change resolution

---

### Python Script: `2d-dispersion-py/plot_dispersion_py.py`

**When to use:** Use this script when:
- You have NumPy/PyTorch format datasets (same format as `convert_mat_to_pytorch.py` output)
- You want separate, organized plots (design, contour, dispersion)
- You need flexibility: different datasets, symmetry types, interpolation modes
- You want more control over plotting options
- You're working with structures that have 'p4mm' symmetry (default)

**Features:**
- Loads NumPy datasets from `data/` folders
- **Two plotting modes:**
  - **Interpolation mode**: Smooth curves by interpolating to contour points (more points, smoother)
  - **Grid-point mode**: Exact values from grid points on contour (fewer points, exact data)
- Creates separate plots: design, IBZ contour, dispersion
- Command-line interface with dataset path argument
- Uses 'p4mm' symmetry for IBZ contour (can be modified in code)
- More sophisticated: creates grid interpolators, extracts exact grid points on contour
- Processes first 5 structures by default

**Usage:**
```bash
# Use default dataset (set_br_1200n)
python 2d-dispersion-py/plot_dispersion_py.py

# Specify custom dataset
python 2d-dispersion-py/plot_dispersion_py.py "data/set_br_1200n"
```

**Output:**
- Creates timestamped folder: `dispersion_plots_[timestamp]_py/`
- Subfolders:
  - `design/` - Design visualizations (material properties)
  - `contour/` - IBZ contour wavevector plots (first structure only)
  - `dispersion/` - Dispersion relation plots

---

### Python Script: `2d-dispersion-py/plot_dispersion_py_reduced_data.py`

**When to use:** Use this script when:
- You have reduced PyTorch datasets (from `convert_mat_to_pytorch.py` with reduction applied)
- You need to plot from reduced datasets
- You have access to the original full dataset for eigenvalue data

**Features:**
- Loads reduced PyTorch datasets (`.pt` files)
- Merges with original eigenvalue data for full dispersion plots
- Handles reduced indices mapping

**Usage:**
```bash
python 2d-dispersion-py/plot_dispersion_py_reduced_data.py [reduced_dir] [original_dir]

# Example
python 2d-dispersion-py/plot_dispersion_py_reduced_data.py "data/out_binarized_1" "data/set_br_1200n"
```

**Output:**
- Creates timestamped folder: `dispersion_plots_[timestamp]_py_reduced/`
- Same structure as `plot_dispersion_py.py`

---

### Python Script: `2d-dispersion-py/plot_dispersion_numpy.py`

**When to use:** Use this script when:
- You have converted NumPy datasets (from `convert_matlab_datasets.ipynb` or similar)
- You want to compare original vs reconstructed eigenvalues
- You have access to the original `.mat` file with K/M/T data

**Features:**
- Loads NumPy format datasets
- Can reconstruct eigenvalues from K/M/T matrices if original `.mat` is available
- Creates comparison plots (original vs reconstructed)

**Usage:**
Edit the script to set `base_dir` path, then run:
```bash
python 2d-dispersion-py/plot_dispersion_numpy.py
```

**Output:**
- Saves directly in the dataset folder
- Creates subfolders: `design/`, `contour/`, `dispersion/`
- Includes `[N]_recon.png` and `[N]_diff.png` if reconstruction is available

---

## Quick Reference

### I want to convert MATLAB data to PyTorch:
- **Single file**: Use `2d-dispersion-py/convert_mat_to_pytorch.py`
- **Multiple files**: Use `matlab_to_reduced_pt.ipynb`

### I want to plot dispersion:
- **From MATLAB .mat files**: Use `2D-dispersion-han/plot_dispersion.m` (MATLAB)
- **From NumPy datasets (unreduced)**: Use `2d-dispersion-py/plot_dispersion_py.py`
- **From reduced PyTorch datasets**: Use `2d-dispersion-py/plot_dispersion_py_reduced_data.py`
- **Need reconstruction comparison**: Use `2d-dispersion-py/plot_dispersion_numpy.py`

---

## Notes

- All Python scripts assume the project root as the working directory
- MATLAB scripts should be run from the `2D-dispersion-han/` directory or with appropriate path setup
- Output folders are created automatically if they don't exist
- Most scripts support command-line arguments for customization

