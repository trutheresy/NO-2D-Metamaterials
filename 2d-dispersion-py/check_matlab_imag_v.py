"""Check MATLAB's imag(v) values."""
import scipy.io as sio
import numpy as np
import os

if os.path.exists('test_plots/plot_eigenvector_components_data_matlab.mat'):
    data = sio.loadmat('test_plots/plot_eigenvector_components_data_matlab.mat')
    v_mat = data['v_reshaped']
    u_mat = data['u_reshaped']
    
    print("=" * 80)
    print("MATLAB DATA ANALYSIS")
    print("=" * 80)
    
    print("\n1. MATLAB imag(u_reshaped):")
    imag_u_mat = np.imag(u_mat)
    print(f"  Min: {np.min(imag_u_mat):.15e}")
    print(f"  Max: {np.max(imag_u_mat):.15e}")
    print(f"  Mean: {np.mean(imag_u_mat):.15e}")
    print(f"  Std: {np.std(imag_u_mat):.15e}")
    print(f"  Is exactly zero? {np.all(imag_u_mat == 0.0)}")
    print(f"  All close to zero? {np.allclose(imag_u_mat, 0.0, atol=1e-15)}")
    
    print("\n2. MATLAB imag(v_reshaped):")
    imag_v_mat = np.imag(v_mat)
    print(f"  Min: {np.min(imag_v_mat):.15e}")
    print(f"  Max: {np.max(imag_v_mat):.15e}")
    print(f"  Mean: {np.mean(imag_v_mat):.15e}")
    print(f"  Std: {np.std(imag_v_mat):.15e}")
    print(f"  Is exactly zero? {np.all(imag_v_mat == 0.0)}")
    print(f"  All close to zero? {np.allclose(imag_v_mat, 0.0, atol=1e-15)}")
    
    print("\n3. MATLAB full eigenvector imag components:")
    u_full_mat = data['u_full']
    imag_u_full_mat = np.imag(u_full_mat)
    print(f"  imag(u_full) min: {np.min(imag_u_full_mat):.15e}")
    print(f"  imag(u_full) max: {np.max(imag_u_full_mat):.15e}")
    print(f"  imag(u_full) mean: {np.mean(imag_u_full_mat):.15e}")
    
    # Check data types
    print("\n4. Data types:")
    print(f"  u_full dtype: {u_full_mat.dtype}")
    print(f"  u_reshaped dtype: {u_mat.dtype}")
    print(f"  v_reshaped dtype: {v_mat.dtype}")
    
    print("\n" + "=" * 80)
else:
    print("MATLAB data file not found: test_plots/plot_eigenvector_components_data_matlab.mat")
    print("Please run the MATLAB script first to generate the data.")

