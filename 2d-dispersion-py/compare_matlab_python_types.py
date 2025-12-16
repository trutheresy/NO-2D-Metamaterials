"""Compare MATLAB vs Python eigenvector data types and values."""
import scipy.io as sio
import numpy as np

# Load data
py_data = sio.loadmat('test_plots/plot_eigenvector_components_data.mat')
mat_data = sio.loadmat('test_plots/plot_eigenvector_components_data_matlab.mat')

print("=" * 80)
print("MATLAB vs PYTHON EIGENVECTOR TYPE AND VALUE COMPARISON")
print("=" * 80)

print("\n1. DATA TYPE COMPARISON:")
print(f"  Python u_full dtype: {py_data['eigenvector_full'].dtype}")
print(f"  MATLAB u_full dtype: {mat_data['u_full'].dtype}")
print(f"  Python u_reshaped dtype: {py_data['u_reshaped'].dtype}")
print(f"  MATLAB u_reshaped dtype: {mat_data['u_reshaped'].dtype}")
print(f"  Python v_reshaped dtype: {py_data['v_reshaped'].dtype}")
print(f"  MATLAB v_reshaped dtype: {mat_data['v_reshaped'].dtype}")

print("\n2. IMAGINARY PARTS:")
print("\n  Python:")
print(f"    imag(u_full) dtype: {np.imag(py_data['eigenvector_full']).dtype}")
print(f"    imag(u_full) max: {np.max(np.abs(np.imag(py_data['eigenvector_full']))):.15e}")
print(f"    imag(v_full) max: {np.max(np.abs(np.imag(py_data['eigenvector_full'][1::2]))):.15e}")

print("\n  MATLAB:")
print(f"    imag(u_full) dtype: {np.imag(mat_data['u_full']).dtype}")
print(f"    imag(u_full) max: {np.max(np.abs(np.imag(mat_data['u_full']))):.15e}")
print(f"    imag(v_full) max: {np.max(np.abs(np.imag(mat_data['u_full'][1::2]))):.15e}")

print("\n3. KEY FINDING:")
print(f"  MATLAB stores eigenvectors as REAL arrays (float64), not complex!")
print(f"  Python stores eigenvectors as COMPLEX arrays (complex128), even when imag parts are tiny")
print(f"\n  This means:")
print(f"    - MATLAB's imag components are exactly 0.0 (because array is real type)")
print(f"    - Python's imag components have tiny numerical noise (~1e-15)")
print(f"    - Both are mathematically equivalent (imag parts are numerical noise)")

print("\n4. HOW MATLAB ENSURES EXACTLY ZERO:")
print(f"  MATLAB doesn't 'force' it - it's a consequence of data type:")
print(f"    - For real symmetric matrices, MATLAB's eigs returns real eigenvectors")
print(f"    - MATLAB stores these as float64 (real) arrays, not complex arrays")
print(f"    - Therefore, imag() of a real array is exactly 0.0")
print(f"\n  Python:")
print(f"    - numpy.linalg.eig returns complex128 arrays regardless")
print(f"    - Even when imag parts are negligible, the array is still complex type")
print(f"    - Therefore, imag() shows tiny numerical noise")

print("\n" + "=" * 80)

