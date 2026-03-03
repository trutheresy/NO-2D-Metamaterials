# Instructions for Running MATLAB Scripts

## Overview
To generate `plot_points.mat` files for comparison with Python results, you need to run the MATLAB dispersion plotting scripts. The scripts have been modified to save plot point locations automatically.

## Quick Start

### Option 1: Use the Helper Script (Recommended)

1. Open MATLAB
2. Navigate to the project directory:
   ```matlab
   cd('D:\Research\NO-2D-Metamaterials')
   ```
3. Run the helper script:
   ```matlab
   run_matlab_scripts
   ```

This will automatically run both scripts and generate the plot_points.mat files.

### Option 2: Run Scripts Manually

#### Case 1: Without Eigenfrequencies (Reconstructed)

1. Navigate to the MATLAB scripts directory:
   ```matlab
   cd('D:\Research\NO-2D-Metamaterials\2D-dispersion-han')
   ```

2. Set the data file path (create a temp file):
   ```matlab
   data_fn = "D:\Research\NO-2D-Metamaterials\data\out_test_10_verify\out_binarized_1\out_binarized_1_predictions.mat";
   fid = fopen('temp_data_fn.txt', 'w');
   fprintf(fid, '%s', data_fn);
   fclose(fid);
   ```

3. Run the script:
   ```matlab
   plot_dispersion_from_predictions
   ```

4. Expected output: `plots/out_binarized_1_mat/plot_points.mat`

#### Case 2: With Eigenfrequencies

1. Navigate to the MATLAB scripts directory:
   ```matlab
   cd('D:\Research\NO-2D-Metamaterials\2D-dispersion-han')
   ```

2. Set the data file path:
   ```matlab
   data_fn = "D:\Research\NO-2D-Metamaterials\data\out_test_10_matlab\out_binarized_1.mat";
   fid = fopen('temp_data_fn.txt', 'w');
   fprintf(fid, '%s', data_fn);
   fclose(fid);
   ```

3. Run the script:
   ```matlab
   plot_dispersion
   ```

4. Expected output: `plots/out_binarized_1_mat/plot_points.mat` (may overwrite Case 1 if same output directory)

## Verification

After running the scripts, verify that the files were created:

```matlab
% Check Case 1 output
case1_file = fullfile('plots', 'out_binarized_1_mat', 'plot_points.mat');
if exist(case1_file, 'file')
    fprintf('✓ Case 1 plot_points.mat created\n');
    load(case1_file);
    fprintf('  Fields: %s\n', strjoin(fieldnames(plot_points_data), ', '));
else
    fprintf('✗ Case 1 plot_points.mat NOT FOUND\n');
end
```

## Next Steps

After generating the MATLAB plot_points.mat files, run the Python comparison script:

```bash
python compare_plot_points.py
```

This will compare the Python and MATLAB plot point locations and report any discrepancies.

## Troubleshooting

### Issue: Script can't find data file
- **Solution**: Check that the data file paths are correct
- Verify files exist:
  - Case 1: `data/out_test_10_verify/out_binarized_1/out_binarized_1_predictions.mat`
  - Case 2: `data/out_test_10_matlab/out_binarized_1.mat`

### Issue: plot_points.mat not created
- **Solution**: Check that `isSavePlotPoints = true` in the script (it should be by default)
- Check the output directory: `plots/out_binarized_1_mat/`
- Look for error messages in the MATLAB command window

### Issue: Script errors during execution
- **Solution**: Make sure all required MATLAB functions are in the path
- Check that the 2D-dispersion-han library is properly set up
- Verify the data file format is correct

