"""Investigate MATLAB saved frequencies to check for variable name or structure issues."""
import h5py
import numpy as np

mat_file = '2D-dispersion-han/plots/out_binarized_1_mat/plot_points.mat'

print("="*70)
print("Investigating MATLAB saved frequencies")
print("="*70)

with h5py.File(mat_file, 'r') as f:
    print("\nTop-level keys:")
    for k in f.keys():
        if not k.startswith('#'):
            print(f"  {k}")
    
    if 'plot_points_data' in f:
        pp = f['plot_points_data']
        print("\nKeys in plot_points_data:")
        for k in sorted(pp.keys()):
            val = np.array(pp[k])
            print(f"  {k}: shape={val.shape}, dtype={val.dtype}")
        
        # Check frequencies_contour specifically
        freq_key = 'struct_1_frequencies_contour'
        if freq_key in pp:
            freq = np.array(pp[freq_key])
            print(f"\n{freq_key} details:")
            print(f"  Raw shape: {freq.shape}")
            print(f"  After transpose: {freq.T.shape}")
            print(f"  Expected shape: (55, 6) for 55 contour points and 6 bands")
            
            # Check if it needs transposing
            if freq.shape[0] == 6 and freq.shape[1] == 55:
                print(f"  ✓ Shape suggests it's stored as (6, 55) - needs transpose")
                freq_transposed = freq.T
            elif freq.shape[0] == 55 and freq.shape[1] == 6:
                print(f"  ✓ Shape is already (55, 6) - no transpose needed")
                freq_transposed = freq
            else:
                print(f"  ⚠ Unexpected shape: {freq.shape}")
                freq_transposed = freq
            
            print(f"\n  Values (first 3 points, first 3 bands):")
            print(freq_transposed[:3, :3])
            print(f"\n  NaN count: {np.isnan(freq_transposed).sum()}/{freq_transposed.size}")
            print(f"  Min (non-NaN): {np.nanmin(freq_transposed) if not np.isnan(freq_transposed).all() else 'N/A'}")
            print(f"  Max (non-NaN): {np.nanmax(freq_transposed) if not np.isnan(freq_transposed).all() else 'N/A'}")
            
            # Check if all NaN
            if np.isnan(freq_transposed).all():
                print(f"\n  ⚠ ALL VALUES ARE NaN!")
                print(f"  This suggests MATLAB's interpolation failed or variable was not set correctly")
            else:
                print(f"\n  ✓ Has valid values")
        else:
            print(f"\n✗ {freq_key} not found in plot_points_data")
            print(f"  Available keys with 'frequencies':")
            for k in sorted(pp.keys()):
                if 'frequencies' in k.lower():
                    print(f"    {k}")
        
        # Check what MATLAB actually computed (check if frequencies_contour was set before saving)
        print(f"\nChecking MATLAB script logic:")
        print(f"  Line 572: frequencies_contour = zeros(size(wavevectors_contour,1),size(frequencies,2));")
        print(f"  Line 577: frequencies_contour(:,eig_idx) = interp_true{eig_idx}(wavevectors_contour(:,1),wavevectors_contour(:,2));")
        print(f"  Line 586: plot_points_data.(['struct_' num2str(struct_idx) '_frequencies_contour']) = frequencies_contour;")
        print(f"\n  This suggests frequencies_contour should be (55, 6) if wavevectors_contour is (55, 2)")

