"""Compare values saved from test vs MATLAB for plot_eigenvector_components."""
import scipy.io as sio
import numpy as np
from pathlib import Path

test_plots_dir = Path('test_plots')

# Load test data
test_data_path = test_plots_dir / 'plot_eigenvector_components_data.mat'
matlab_data_path = test_plots_dir / 'plot_eigenvector_components_data_matlab.mat'

print("=" * 80)
print("COMPARING PLOT_EIGENVECTOR_COMPONENTS DATA")
print("=" * 80)

if not test_data_path.exists():
    print(f"❌ Test data not found: {test_data_path}")
    exit(1)
else:
    test_data = sio.loadmat(str(test_data_path))
    print(f"\n✅ Loaded test data: {test_data_path}")

if not matlab_data_path.exists():
    print(f"\n⚠️  MATLAB data file not found: {matlab_data_path}")
    print("   Run the MATLAB script: test_plot_eigenvector_components_matlab.m")
    print("   Then run this script again to compare.")
    matlab_data = None
else:
    try:
        matlab_data = sio.loadmat(str(matlab_data_path))
        print(f"✅ Loaded MATLAB data: {matlab_data_path}")
    except NotImplementedError:
        print(f"⚠️  File is v7.3 format, attempting to load with h5py...")
        print(f"   Consider re-saving MATLAB file with '-v7' flag instead of '-v7.3'")
        matlab_data = None

# Compare Python vs MATLAB if available
if matlab_data is not None:
    print("\n" + "=" * 80)
    print("PYTHON vs MATLAB COMPARISON")
    print("=" * 80)
    
    # Compare u_reshaped
    u_python = test_data['u_reshaped']
    u_matlab = matlab_data['u_reshaped']
    print(f"\nu_reshaped comparison (Python vs MATLAB):")
    print(f"  Python shape: {u_python.shape}")
    print(f"  MATLAB shape: {u_matlab.shape}")
    if u_python.shape == u_matlab.shape:
        u_diff = np.abs(u_python - u_matlab)
        print(f"  Max difference: {np.max(u_diff):.6e}")
        print(f"  Mean difference: {np.mean(u_diff):.6e}")
        if np.max(np.abs(u_matlab)) > 0:
            print(f"  Relative error: {np.max(u_diff) / np.max(np.abs(u_matlab)):.6e}")
        if np.max(u_diff) < 1e-3:
            print("  ✅ u_reshaped values match (within tolerance)")
        else:
            print("  ❌ u_reshaped values differ significantly")
            max_idx = np.unravel_index(np.argmax(u_diff), u_diff.shape)
            print(f"    Max diff at index {max_idx}")
            print(f"    Python value: {u_python[max_idx]:.6f}")
            print(f"    MATLAB value: {u_matlab[max_idx]:.6f}")
    else:
        print("  ❌ u_reshaped shapes don't match!")
    
    # Compare v_reshaped
    v_python = test_data['v_reshaped']
    v_matlab = matlab_data['v_reshaped']
    print(f"\nv_reshaped comparison (Python vs MATLAB):")
    print(f"  Python shape: {v_python.shape}")
    print(f"  MATLAB shape: {v_matlab.shape}")
    if v_python.shape == v_matlab.shape:
        v_diff = np.abs(v_python - v_matlab)
        print(f"  Max difference: {np.max(v_diff):.6e}")
        print(f"  Mean difference: {np.mean(v_diff):.6e}")
        if np.max(np.abs(v_matlab)) > 0:
            print(f"  Relative error: {np.max(v_diff) / np.max(np.abs(v_matlab)):.6e}")
        if np.max(v_diff) < 1e-3:
            print("  ✅ v_reshaped values match (within tolerance)")
        else:
            print("  ❌ v_reshaped values differ significantly")
    
    # Compare eigenvector_full
    if 'eigenvector_full' in test_data and 'u_full' in matlab_data:
        ev_python = test_data['eigenvector_full']
        ev_matlab = matlab_data['u_full']
        print(f"\nEigenvector (full space) comparison:")
        print(f"  Python shape: {ev_python.shape}")
        print(f"  MATLAB shape: {ev_matlab.shape}")
        if ev_python.shape == ev_matlab.shape:
            ev_diff = np.abs(ev_python.flatten() - ev_matlab.flatten())
            print(f"  Max difference: {np.max(ev_diff):.6e}")
            if np.max(ev_diff) < 1e-3:
                print("  ✅ Eigenvector values match (within tolerance)")
            else:
                print("  ❌ Eigenvector values differ")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
if matlab_data is not None:
    print("✅ Python and MATLAB values compared")
    print("\n📊 Check the differences above to verify equivalence.")
else:
    print("⚠️  MATLAB data not available for comparison")
    print("   Run test_plot_eigenvector_components_matlab.m to generate MATLAB data")

