"""Compare values saved from test vs values computed in plotting function vs MATLAB."""
import scipy.io as sio
import numpy as np
from pathlib import Path

test_plots_dir = Path('test_plots')

# Load test data
test_data_path = test_plots_dir / 'plot_mode_basic_data.mat'
plotting_data_path = test_plots_dir / 'plot_mode_values_from_plotting_func.mat'
matlab_data_path = test_plots_dir / 'plot_mode_basic_data_matlab.mat'

print("=" * 80)
print("COMPARING PLOT_MODE DATA")
print("=" * 80)

if not test_data_path.exists():
    print(f"❌ Test data not found: {test_data_path}")
    exit(1)
else:
    test_data = sio.loadmat(str(test_data_path))
    print(f"\n✅ Loaded test data: {test_data_path}")

if not plotting_data_path.exists():
    print(f"❌ Plotting function data not found: {plotting_data_path}")
    print("   Run the plotting test first to generate this file.")
else:
    plotting_data = sio.loadmat(str(plotting_data_path))
    print(f"✅ Loaded plotting function data: {plotting_data_path}")

if not matlab_data_path.exists():
    print(f"\n⚠️  MATLAB data file not found: {matlab_data_path}")
    print("   Run the MATLAB script: test_plot_mode_matlab.m")
    print("   Then run this script again to compare.")
    matlab_data = None
else:
    try:
        matlab_data = sio.loadmat(str(matlab_data_path))
        print(f"✅ Loaded MATLAB data: {matlab_data_path}")
    except NotImplementedError:
        # Try using h5py for v7.3 files
        try:
            import h5py
            print(f"⚠️  File is v7.3 format, attempting to load with h5py...")
            # Note: v7.3 files need special handling with h5py
            # For now, just inform the user
            print(f"   File exists but requires h5py for v7.3 format.")
            print(f"   Consider re-saving MATLAB file with '-v7' flag instead of '-v7.3'")
            matlab_data = None
        except ImportError:
            print(f"   h5py not available. Please re-save MATLAB file with '-v7' flag.")
            matlab_data = None

# Compare Python test data vs plotting function data
if plotting_data_path.exists():
    print("\n" + "=" * 80)
    print("PYTHON TEST DATA vs PLOTTING FUNCTION DATA")
    print("=" * 80)
    
    # Compare U_mat
    U_test = test_data['U_mat']
    U_plot = plotting_data['U_mat']
    print(f"\nU_mat comparison:")
    print(f"  Test shape: {U_test.shape}")
    print(f"  Plot shape: {U_plot.shape}")
    if U_test.shape == U_plot.shape:
        u_diff = np.abs(U_test - U_plot)
        print(f"  Max difference: {np.max(u_diff):.6e}")
        if np.max(u_diff) < 1e-10:
            print("  ✅ U_mat values match exactly")
        else:
            print("  ❌ U_mat values differ")
            max_idx = np.unravel_index(np.argmax(u_diff), u_diff.shape)
            print(f"    Max diff at index {max_idx}")
            print(f"    Test value: {U_test[max_idx]:.6f}")
            print(f"    Plot value: {U_plot[max_idx]:.6f}")
    
    # Compare V_mat
    V_test = test_data['V_mat']
    V_plot = plotting_data['V_mat']
    print(f"\nV_mat comparison:")
    print(f"  Test shape: {V_test.shape}")
    print(f"  Plot shape: {V_plot.shape}")
    if V_test.shape == V_plot.shape:
        v_diff = np.abs(V_test - V_plot)
        print(f"  Max difference: {np.max(v_diff):.6e}")
        if np.max(v_diff) < 1e-10:
            print("  ✅ V_mat values match exactly")
        else:
            print("  ❌ V_mat values differ")

# Compare with MATLAB if available
if matlab_data is not None:
    print("\n" + "=" * 80)
    print("PYTHON vs MATLAB COMPARISON")
    print("=" * 80)
    
    # Compare U_mat
    U_python = test_data['U_mat']
    U_matlab = matlab_data['U_mat']
    print(f"\nU_mat comparison (Python vs MATLAB):")
    print(f"  Python shape: {U_python.shape}")
    print(f"  MATLAB shape: {U_matlab.shape}")
    if U_python.shape == U_matlab.shape:
        u_diff = np.abs(U_python - U_matlab)
        print(f"  Max difference: {np.max(u_diff):.6e}")
        print(f"  Mean difference: {np.mean(u_diff):.6e}")
        print(f"  Relative error: {np.max(u_diff) / np.max(np.abs(U_matlab)):.6e}")
        if np.max(u_diff) < 1e-3:
            print("  ✅ U_mat values match (within tolerance)")
        else:
            print("  ❌ U_mat values differ significantly")
            max_idx = np.unravel_index(np.argmax(u_diff), u_diff.shape)
            print(f"    Max diff at index {max_idx}")
            print(f"    Python value: {U_python[max_idx]:.6f}")
            print(f"    MATLAB value: {U_matlab[max_idx]:.6f}")
            print(f"\n    U_mat sample (first 5x5):")
            print(f"    Python:")
            print(U_python[:5, :5])
            print(f"    MATLAB:")
            print(U_matlab[:5, :5])
    else:
        print("  ❌ U_mat shapes don't match!")
    
    # Compare V_mat
    V_python = test_data['V_mat']
    V_matlab = matlab_data['V_mat']
    print(f"\nV_mat comparison (Python vs MATLAB):")
    print(f"  Python shape: {V_python.shape}")
    print(f"  MATLAB shape: {V_matlab.shape}")
    if V_python.shape == V_matlab.shape:
        v_diff = np.abs(V_python - V_matlab)
        print(f"  Max difference: {np.max(v_diff):.6e}")
        print(f"  Mean difference: {np.mean(v_diff):.6e}")
        if np.max(v_diff) < 1e-3:
            print("  ✅ V_mat values match (within tolerance)")
        else:
            print("  ❌ V_mat values differ significantly")
    
    # Compare eigenvector_full
    if 'eigenvector_full' in test_data and 'u' in matlab_data:
        ev_python = test_data['eigenvector_full']
        ev_matlab = matlab_data['u']
        print(f"\nEigenvector (full space) comparison:")
        print(f"  Python shape: {ev_python.shape}")
        print(f"  MATLAB shape: {ev_matlab.shape}")
        if ev_python.shape == ev_matlab.shape:
            ev_diff = np.abs(ev_python - ev_matlab.flatten())
            print(f"  Max difference: {np.max(ev_diff):.6e}")
            if np.max(ev_diff) < 1e-3:
                print("  ✅ Eigenvector values match (within tolerance)")
            else:
                print("  ❌ Eigenvector values differ")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
if plotting_data_path.exists():
    print("✅ Python test and plotting function values compared")
if matlab_data is not None:
    print("✅ Python and MATLAB values compared")
    print("\n📊 Check the differences above to verify equivalence.")

