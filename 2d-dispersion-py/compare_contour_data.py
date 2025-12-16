"""
Compare Python and MATLAB contour plot data to debug discrepancies.

This script loads both Python and MATLAB saved data and compares them.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path

def compare_contour_data():
    """Compare Python and MATLAB contour plot data."""
    
    test_plots_dir = Path('test_plots')
    
    # Load Python data
    python_data_path = test_plots_dir / 'plot_dispersion_contour_data.mat'
    if not python_data_path.exists():
        print(f"❌ Python data file not found: {python_data_path}")
        return
    
    python_data = sio.loadmat(str(python_data_path))
    print("=" * 80)
    print("PYTHON DATA")
    print("=" * 80)
    print(f"wv_grid shape: {python_data['wv_grid'].shape}")
    print(f"frequencies shape: {python_data['frequencies'].shape}")
    print(f"X shape: {python_data['X'].shape}")
    print(f"Y shape: {python_data['Y'].shape}")
    print(f"Z shape: {python_data['Z'].shape}")
    print(f"\nX range: [{np.min(python_data['X']):.6f}, {np.max(python_data['X']):.6f}]")
    print(f"Y range: [{np.min(python_data['Y']):.6f}, {np.max(python_data['Y']):.6f}]")
    print(f"Z range: [{np.min(python_data['Z']):.6f}, {np.max(python_data['Z']):.6f}]")
    print(f"\nZ sample (first 5x5):")
    print(python_data['Z'][:5, :5])
    
    # Load MATLAB data if available
    matlab_data_path = test_plots_dir / 'plot_dispersion_contour_data_matlab.mat'
    if matlab_data_path.exists():
        print("\n" + "=" * 80)
        print("MATLAB DATA")
        print("=" * 80)
        matlab_data = sio.loadmat(str(matlab_data_path))
        print(f"wv_grid shape: {matlab_data['wv_grid'].shape}")
        print(f"fr shape: {matlab_data['fr'].shape}")
        print(f"X shape: {matlab_data['X'].shape}")
        print(f"Y shape: {matlab_data['Y'].shape}")
        print(f"Z shape: {matlab_data['Z'].shape}")
        print(f"\nX range: [{np.min(matlab_data['X']):.6f}, {np.max(matlab_data['X']):.6f}]")
        print(f"Y range: [{np.min(matlab_data['Y']):.6f}, {np.max(matlab_data['Y']):.6f}]")
        print(f"Z range: [{np.min(matlab_data['Z']):.6f}, {np.max(matlab_data['Z']):.6f}]")
        print(f"\nZ sample (first 5x5):")
        print(matlab_data['Z'][:5, :5])
        
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        
        # Compare shapes
        if python_data['X'].shape == matlab_data['X'].shape:
            print("✅ X shapes match")
        else:
            print(f"❌ X shapes differ: Python {python_data['X'].shape} vs MATLAB {matlab_data['X'].shape}")
        
        if python_data['Y'].shape == matlab_data['Y'].shape:
            print("✅ Y shapes match")
        else:
            print(f"❌ Y shapes differ: Python {python_data['Y'].shape} vs MATLAB {matlab_data['Y'].shape}")
        
        if python_data['Z'].shape == matlab_data['Z'].shape:
            print("✅ Z shapes match")
        else:
            print(f"❌ Z shapes differ: Python {python_data['Z'].shape} vs MATLAB {matlab_data['Z'].shape}")
        
        # Compare values
        if python_data['X'].shape == matlab_data['X'].shape:
            x_diff = np.abs(python_data['X'] - matlab_data['X'])
            print(f"\nX max difference: {np.max(x_diff):.6e}")
            if np.max(x_diff) < 1e-6:
                print("✅ X values match (within tolerance)")
            else:
                print(f"❌ X values differ significantly")
                print(f"   Max diff location: {np.unravel_index(np.argmax(x_diff), x_diff.shape)}")
                print(f"   Python value: {python_data['X'][np.unravel_index(np.argmax(x_diff), x_diff.shape)]:.6f}")
                print(f"   MATLAB value: {matlab_data['X'][np.unravel_index(np.argmax(x_diff), x_diff.shape)]:.6f}")
        
        if python_data['Y'].shape == matlab_data['Y'].shape:
            y_diff = np.abs(python_data['Y'] - matlab_data['Y'])
            print(f"\nY max difference: {np.max(y_diff):.6e}")
            if np.max(y_diff) < 1e-6:
                print("✅ Y values match (within tolerance)")
            else:
                print(f"❌ Y values differ significantly")
        
        if python_data['Z'].shape == matlab_data['Z'].shape:
            z_diff = np.abs(python_data['Z'] - matlab_data['Z'])
            print(f"\nZ max difference: {np.max(z_diff):.6e}")
            print(f"Z mean difference: {np.mean(z_diff):.6e}")
            print(f"Z relative error: {np.max(z_diff) / np.max(np.abs(matlab_data['Z'])):.6e}")
            if np.max(z_diff) < 1e-3:
                print("✅ Z values match (within tolerance)")
            else:
                print(f"❌ Z values differ significantly")
                print(f"   Max diff location: {np.unravel_index(np.argmax(z_diff), z_diff.shape)}")
                print(f"   Python value: {python_data['Z'][np.unravel_index(np.argmax(z_diff), z_diff.shape)]:.6f}")
                print(f"   MATLAB value: {matlab_data['Z'][np.unravel_index(np.argmax(z_diff), z_diff.shape)]:.6f}")
                
                # Show difference map
                print(f"\n   Difference map (first 10x10):")
                print(z_diff[:10, :10])
    else:
        print(f"\n⚠️  MATLAB data file not found: {matlab_data_path}")
        print("   Run the MATLAB script: test_plot_dispersion_contour_matlab.m")
        print("   Then run this script again to compare.")


if __name__ == '__main__':
    compare_contour_data()

