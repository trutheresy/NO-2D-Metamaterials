"""Check eigenvector data comparison."""
import scipy.io as sio
import numpy as np

py_data = sio.loadmat('test_plots/plot_eigenvector_components_data.mat')
mat_data = sio.loadmat('test_plots/plot_eigenvector_components_data_matlab.mat')

print('u_reshaped shapes:')
print(f'  Python: {py_data["u_reshaped"].shape}')
print(f'  MATLAB: {mat_data["u_reshaped"].shape}')

print('\nu_reshaped first 5x5:')
print('Python:')
print(py_data['u_reshaped'][:5, :5])
print('MATLAB:')
print(mat_data['u_reshaped'][:5, :5])

print('\nDifferences in first 5x5:')
diff = np.abs(py_data['u_reshaped'][:5, :5] - mat_data['u_reshaped'][:5, :5])
print(diff)
print(f'Max diff: {np.max(diff):.6e}')

