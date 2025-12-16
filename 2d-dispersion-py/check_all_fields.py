"""Check all eigenvector component fields for forcing or natural constancy."""
import scipy.io as sio
import numpy as np

# Load saved data
py_data = sio.loadmat('test_plots/plot_eigenvector_components_data.mat')
mat_data = sio.loadmat('test_plots/plot_eigenvector_components_data_matlab.mat')

u_py = py_data['u']
v_py = py_data['v']
u_mat = mat_data['u']
v_mat = mat_data['v']

u_reshaped_py = py_data['u_reshaped']
v_reshaped_py = py_data['v_reshaped']
u_reshaped_mat = mat_data['u_reshaped']
v_reshaped_mat = mat_data['v_reshaped']

print("=" * 80)
print("CHECKING ALL EIGENVECTOR COMPONENT FIELDS")
print("=" * 80)

# Check real(u)
print("\n1. real(u):")
real_u_py = np.real(u_py)
real_u_mat = np.real(u_mat)
print(f"  Python stats:")
print(f"    Min: {np.min(real_u_py):.10e}")
print(f"    Max: {np.max(real_u_py):.10e}")
print(f"    Mean: {np.mean(real_u_py):.10e}")
print(f"    Std: {np.std(real_u_py):.10e}")
print(f"    Is constant? {np.allclose(real_u_py, np.mean(real_u_py), atol=1e-10)}")
print(f"  MATLAB stats:")
print(f"    Min: {np.min(real_u_mat):.10e}")
print(f"    Max: {np.max(real_u_mat):.10e}")
print(f"    Mean: {np.mean(real_u_mat):.10e}")
print(f"    Std: {np.std(real_u_mat):.10e}")
print(f"    Is constant? {np.allclose(real_u_mat, np.mean(real_u_mat), atol=1e-10)}")
print(f"  First 10 values Python: {real_u_py[:10].flatten()}")
print(f"  First 10 values MATLAB: {real_u_mat[:10].flatten()}")

# Check imag(u)
print("\n2. imag(u):")
imag_u_py = np.imag(u_py)
imag_u_mat = np.imag(u_mat)
print(f"  Python stats:")
print(f"    Min: {np.min(imag_u_py):.10e}")
print(f"    Max: {np.max(imag_u_py):.10e}")
print(f"    Mean: {np.mean(imag_u_py):.10e}")
print(f"    Std: {np.std(imag_u_py):.10e}")
print(f"    Is constant? {np.allclose(imag_u_py, np.mean(imag_u_py), atol=1e-10)}")
print(f"  MATLAB stats:")
print(f"    Min: {np.min(imag_u_mat):.10e}")
print(f"    Max: {np.max(imag_u_mat):.10e}")
print(f"    Mean: {np.mean(imag_u_mat):.10e}")
print(f"    Std: {np.std(imag_u_mat):.10e}")
print(f"    Is constant? {np.allclose(imag_u_mat, np.mean(imag_u_mat), atol=1e-10)}")

# Check real(v)
print("\n3. real(v):")
real_v_py = np.real(v_py)
real_v_mat = np.real(v_mat)
print(f"  Python stats:")
print(f"    Min: {np.min(real_v_py):.10e}")
print(f"    Max: {np.max(real_v_py):.10e}")
print(f"    Mean: {np.mean(real_v_py):.10e}")
print(f"    Std: {np.std(real_v_py):.10e}")
print(f"    Is constant? {np.allclose(real_v_py, np.mean(real_v_py), atol=1e-10)}")
print(f"  MATLAB stats:")
print(f"    Min: {np.min(real_v_mat):.10e}")
print(f"    Max: {np.max(real_v_mat):.10e}")
print(f"    Mean: {np.mean(real_v_mat):.10e}")
print(f"    Std: {np.std(real_v_mat):.10e}")
print(f"    Is constant? {np.allclose(real_v_mat, np.mean(real_v_mat), atol=1e-10)}")
print(f"  First 10 values Python: {real_v_py[:10].flatten()}")
print(f"  First 10 values MATLAB: {real_v_mat[:10].flatten()}")

# Check imag(v)
print("\n4. imag(v):")
imag_v_py = np.imag(v_py)
imag_v_mat = np.imag(v_mat)
print(f"  Python stats:")
print(f"    Min: {np.min(imag_v_py):.10e}")
print(f"    Max: {np.max(imag_v_py):.10e}")
print(f"    Mean: {np.mean(imag_v_py):.10e}")
print(f"    Std: {np.std(imag_v_py):.10e}")
print(f"    Is constant? {np.allclose(imag_v_py, np.mean(imag_v_py), atol=1e-10)}")
print(f"  MATLAB stats:")
print(f"    Min: {np.min(imag_v_mat):.10e}")
print(f"    Max: {np.max(imag_v_mat):.10e}")
print(f"    Mean: {np.mean(imag_v_mat):.10e}")
print(f"    Std: {np.std(imag_v_mat):.10e}")
print(f"    Is constant? {np.allclose(imag_v_mat, np.mean(imag_v_mat), atol=1e-10)}")

# Check the reshaped versions too
print("\n5. CHECKING RESHAPED VERSIONS:")
print("\n  real(u_reshaped) Python std: {:.10e}".format(np.std(np.real(u_reshaped_py))))
print("  imag(u_reshaped) Python std: {:.10e}".format(np.std(np.imag(u_reshaped_py))))
print("  real(v_reshaped) Python std: {:.10e}".format(np.std(np.real(v_reshaped_py))))
print("  imag(v_reshaped) Python std: {:.10e}".format(np.std(np.imag(v_reshaped_py))))

# Check the full eigenvector to understand why real parts are constant
print("\n6. CHECKING FULL EIGENVECTOR:")
eigenvector_full_py = py_data['eigenvector_full']
eigenvector_full_mat = mat_data['u_full']

print(f"  Python eigenvector_full shape: {eigenvector_full_py.shape}")
print(f"  Python real(eigenvector_full) stats:")
real_eig_py = np.real(eigenvector_full_py)
print(f"    Min: {np.min(real_eig_py):.10e}")
print(f"    Max: {np.max(real_eig_py):.10e}")
print(f"    Mean: {np.mean(real_eig_py):.10e}")
print(f"    Std: {np.std(real_eig_py):.10e}")
print(f"    First 20 values: {real_eig_py[:20].flatten()}")

print(f"  Python imag(eigenvector_full) stats:")
imag_eig_py = np.imag(eigenvector_full_py)
print(f"    Min: {np.min(imag_eig_py):.10e}")
print(f"    Max: {np.max(imag_eig_py):.10e}")
print(f"    Mean: {np.mean(imag_eig_py):.10e}")
print(f"    Std: {np.std(imag_eig_py):.10e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("For a rigid body mode at k=0, the eigenvector should represent uniform translation.")
print("This explains why real(u) and real(v) are constant - they represent uniform x and y displacement.")
print("imag(u) and imag(v) should be zero (or near-zero due to numerical noise).")

