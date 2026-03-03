# Plot Points Comparison Report

## Summary

This document reports on the implementation of plot point location saving functionality for both Python and MATLAB dispersion plotting scripts, and the comparison of saved plot points between the two implementations.

## Completed Tasks

### 1. ✅ Added Save Functionality to Python Scripts

**Modified Files:**
- `2d-dispersion-py/plot_dispersion_with_eigenfrequencies.py`
- `2d-dispersion-py/plot_dispersion_infer_eigenfrequencies.py`

**Changes:**
- Added `--save-plot-points` command-line flag
- Saves plot point locations to `plot_points.npz` in the output directory
- Saved data includes:
  - `struct_{idx}_wavevectors_contour`: Wavevector coordinates along IBZ contour (N_points × 2)
  - `struct_{idx}_frequencies_contour`: Frequencies at contour points (N_points × N_eig)
  - `struct_{idx}_contour_param`: Parameter along contour for x-axis (N_points,)
  - `struct_{idx}_use_interpolation`: Boolean flag indicating if interpolation was used

### 2. ✅ Added Save Functionality to MATLAB Scripts

**Modified Files:**
- `2D-dispersion-han/plot_dispersion.m`
- `2D-dispersion-han/plot_dispersion_from_predictions.m`

**Changes:**
- Added `isSavePlotPoints = true` flag (default: enabled)
- Saves plot point locations to `plot_points.mat` in the output directory
- Uses MATLAB v7.3 format (`-v7.3`) for HDF5 compatibility
- Saved data structure matches Python format for easy comparison

### 3. ✅ Ran Python Scripts

**Case 1 (Without Eigenfrequencies):**
- Script: `plot_dispersion_infer_eigenfrequencies.py`
- Dataset: `data/out_test_10_verify/out_binarized_1`
- Command: `python 2d-dispersion-py/plot_dispersion_infer_eigenfrequencies.py "data/out_test_10_verify/out_binarized_1" -n 1 --save-plot-points`
- Status: ✅ **SUCCESS**
- Output: `plots/out_binarized_1_recon/plot_points.npz`

**Case 2 (With Eigenfrequencies):**
- Script: `plot_dispersion_with_eigenfrequencies.py`
- Dataset: `data/out_test_10_matlab/out_binarized_1.mat`
- Command: `python 2d-dispersion-py/plot_dispersion_with_eigenfrequencies.py "data/out_test_10_matlab/out_binarized_1.mat" -n 1 --save-plot-points`
- Status: ⚠️ **PARTIAL** - Python script has issues loading this particular MATLAB v7.3 HDF5 file format
- Issue: HDF5 loader encounters a reference to string 'linear' that doesn't exist as an object
- Note: MATLAB scripts can load this file successfully, so the issue is with the Python HDF5 loader

### 4. ⏳ MATLAB Scripts Need to be Run

**To run MATLAB scripts:**

**Case 1 (Without Eigenfrequencies):**
```matlab
% In MATLAB, navigate to the project directory
cd('D:\Research\NO-2D-Metamaterials\2D-dispersion-han')

% Set the data file path (create temp file or modify script)
data_fn = "D:\Research\NO-2D-Metamaterials\data\out_test_10_verify\out_binarized_1\out_binarized_1_predictions.mat";
% Or use the temp file method:
fid = fopen('temp_data_fn.txt', 'w');
fprintf(fid, '%s', data_fn);
fclose(fid);

% Run the script
plot_dispersion_from_predictions
```

**Case 2 (With Eigenfrequencies):**
```matlab
% In MATLAB, navigate to the project directory
cd('D:\Research\NO-2D-Metamaterials\2D-dispersion-han')

% Set the data file path
data_fn = "D:\Research\NO-2D-Metamaterials\data\out_test_10_matlab\out_binarized_1.mat";
% Or use the temp file method:
fid = fopen('temp_data_fn.txt', 'w');
fprintf(fid, '%s', data_fn);
fclose(fid);

% Run the script
plot_dispersion
```

**Expected Output:**
- Case 1: `plots/out_binarized_1_mat/plot_points.mat`
- Case 2: `plots/out_binarized_1_mat/plot_points.mat` (or similar, depending on dataset name)

### 5. ✅ Created Comparison Script

**File:** `compare_plot_points.py`

**Functionality:**
- Loads Python `.npz` files and MATLAB `.mat` files
- Compares plot point locations for each structure
- Reports discrepancies in:
  - Shape mismatches
  - Value differences (absolute and relative)
- Handles both MATLAB v7.3 HDF5 and older formats

**Usage:**
```bash
python compare_plot_points.py
```

## Known Issues

1. **Python HDF5 Loader Issue (Case 2):**
   - The Python script `plot_dispersion_with_eigenfrequencies.py` cannot load `data/out_test_10_matlab/out_binarized_1.mat`
   - Error: `KeyError: "Unable to synchronously open object (object 'linear' doesn't exist)"`
   - This appears to be a reference dereferencing issue in the HDF5 loader
   - **Workaround:** Use MATLAB scripts for Case 2, or fix the HDF5 loader to handle string references properly

2. **MATLAB Scripts Not Yet Run:**
   - MATLAB scripts need to be executed to generate `plot_points.mat` files
   - Once run, the comparison script can be used to compare results

## Next Steps

1. **Run MATLAB Scripts:**
   - Execute `plot_dispersion_from_predictions.m` for Case 1
   - Execute `plot_dispersion.m` for Case 2
   - Verify that `plot_points.mat` files are created

2. **Run Comparison:**
   - Execute `python compare_plot_points.py`
   - Review discrepancies (if any)

3. **Fix Python HDF5 Loader (Optional):**
   - Investigate and fix the string reference issue in `mat73_loader.py`
   - This would allow Python scripts to handle Case 2 datasets

## File Locations

### Python Outputs:
- Case 1: `plots/out_binarized_1_recon/plot_points.npz`
- Case 2: (Not generated due to HDF5 loader issue)

### MATLAB Outputs (Expected):
- Case 1: `plots/out_binarized_1_mat/plot_points.mat` (or similar)
- Case 2: `plots/out_binarized_1_mat/plot_points.mat` (or similar)

### Comparison Script:
- `compare_plot_points.py`

## Data Structure

Both Python and MATLAB save the same data structure:

```python
{
    'struct_{idx}_wavevectors_contour': np.ndarray,  # (N_points, 2)
    'struct_{idx}_frequencies_contour': np.ndarray,  # (N_points, N_eig)
    'struct_{idx}_contour_param': np.ndarray,         # (N_points,)
    'struct_{idx}_use_interpolation': bool            # True/False
}
```

This structure allows for direct comparison between Python and MATLAB implementations.

