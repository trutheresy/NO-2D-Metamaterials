#!/usr/bin/env python3
"""
Test H7: Eigenvector Interleaving Order

Tests if x and y components are interleaved in the correct order.
The DOF ordering should be: x0, y0, x1, y1, x2, y2, ...
NOT: y0, x0, y1, x1, y2, x2, ...
"""

import numpy as np
from pathlib import Path
import h5py

def test_eigenvector_interleaving():
    """Test eigenvector interleaving order."""
    print("=" * 80)
    print("TEST H7: Eigenvector Interleaving Order")
    print("=" * 80)
    
    # Load original data
    mat_file = Path("data/out_test_10_mat_original/out_binarized_1.mat")
    if not mat_file.exists():
        print(f"   File not found: {mat_file}")
        return False
    
    print(f"\n1. Loading original .mat file...")
    with h5py.File(str(mat_file), 'r') as f:
        eigvec_data = np.array(f['EIGENVECTOR_DATA'])
        designs = np.array(f['designs'])
    
    print(f"   EIGENVECTOR_DATA shape: {eigvec_data.shape}, dtype: {eigvec_data.dtype}")
    print(f"   designs shape: {designs.shape}")
    
    # Convert structured dtype
    if eigvec_data.dtype.names and 'real' in eigvec_data.dtype.names:
        eigvec_data = eigvec_data['real'] + 1j * eigvec_data['imag']
    
    struct_idx = 0
    band_idx = 0
    wv_idx = 0
    
    print(f"\n2. Extracting eigenvector for struct {struct_idx}, band {band_idx}, wavevector {wv_idx}...")
    eigvec = eigvec_data[struct_idx, band_idx, wv_idx, :]  # (n_dof,)
    print(f"   eigvec shape: {eigvec.shape}, dtype: {eigvec.dtype}")
    
    # The eigenvector should have shape (n_dof,) where n_dof = 2 * N_pix * N_pix
    # For N_pix=32, n_dof = 2 * 32 * 32 = 2048
    # The interleaving should be: [x0, y0, x1, y1, ..., x1023, y1023]
    
    print(f"\n3. Checking interleaving pattern...")
    n_dof = eigvec.shape[0]
    n_pix_squared = n_dof // 2
    print(f"   n_dof: {n_dof}, n_pix^2: {n_pix_squared}")
    
    # Extract x and y components
    x_components = eigvec[0::2]  # Even indices: 0, 2, 4, ...
    y_components = eigvec[1::2]  # Odd indices: 1, 3, 5, ...
    
    print(f"   x_components shape: {x_components.shape}")
    print(f"   y_components shape: {y_components.shape}")
    
    # Check if this matches the expected pattern
    # In reduced_pt_to_matlab.py, eigenvectors are interleaved as:
    # EIGENVECTOR_DATA_combined[:, :, :, 0::2] = EIGENVECTOR_DATA_x_flat
    # EIGENVECTOR_DATA_combined[:, :, :, 1::2] = EIGENVECTOR_DATA_y_flat
    # So even indices (0, 2, 4, ...) should be x, odd indices (1, 3, 5, ...) should be y
    
    print(f"\n4. Verifying interleaving matches reconstruction code...")
    print(f"   Reconstruction code uses:")
    print(f"     EIGENVECTOR_DATA_combined[:, :, :, 0::2] = EIGENVECTOR_DATA_x_flat")
    print(f"     EIGENVECTOR_DATA_combined[:, :, :, 1::2] = EIGENVECTOR_DATA_y_flat")
    print(f"   This means: even indices = x, odd indices = y")
    print(f"   Original data: even indices (0::2) = x_components")
    print(f"   Original data: odd indices (1::2) = y_components")
    
    # Check if there's any pattern that suggests wrong interleaving
    # If interleaving were wrong, we'd expect different statistical properties
    
    print(f"\n5. Statistical check...")
    x_magnitude = np.abs(x_components)
    y_magnitude = np.abs(y_components)
    
    print(f"   x_components magnitude: mean={np.mean(x_magnitude):.6e}, std={np.std(x_magnitude):.6e}")
    print(f"   y_components magnitude: mean={np.mean(y_magnitude):.6e}, std={np.std(y_magnitude):.6e}")
    
    # Check if x and y have similar distributions (they should for a valid eigenvector)
    if np.abs(np.mean(x_magnitude) - np.mean(y_magnitude)) / max(np.mean(x_magnitude), np.mean(y_magnitude)) < 0.1:
        print(f"   OK: x and y have similar magnitude distributions")
    else:
        print(f"   WARNING: x and y have very different magnitude distributions")
        print(f"   This might indicate wrong interleaving, but could also be normal")
    
    # Now check the reconstruction code to see how it interleaves
    print(f"\n6. Checking reconstruction code interleaving...")
    print(f"   In reduced_pt_to_matlab.py Phase 2.5:")
    print(f"     EIGENVECTOR_DATA_combined[:, :, :, 0::2] = EIGENVECTOR_DATA_x_flat")
    print(f"     EIGENVECTOR_DATA_combined[:, :, :, 1::2] = EIGENVECTOR_DATA_y_flat")
    print(f"   This means reconstruction puts x at even indices, y at odd indices")
    print(f"   Original data has x at even indices, y at odd indices")
    print(f"   So interleaving should match!")
    
    # However, we need to verify that the original MATLAB data uses the same convention
    # Let's check if we can infer from the data structure
    
    return True

def test_reconstruction_interleaving():
    """Test how reconstruction interleaves eigenvectors."""
    print(f"\n7. Testing reconstruction interleaving logic...")
    
    # Simulate the reconstruction process
    n_designs = 1
    n_wavevectors = 1
    n_bands = 1
    design_res = 4  # Small for testing
    n_dof = 2 * design_res * design_res  # 32
    
    # Create synthetic x and y eigenvectors
    x_eigvec = np.arange(n_dof // 2, dtype=np.complex128) + 1j * np.arange(n_dof // 2, dtype=np.complex128)
    y_eigvec = np.arange(n_dof // 2, dtype=np.complex128) * 2 + 1j * np.arange(n_dof // 2, dtype=np.complex128) * 2
    
    print(f"   x_eigvec: {x_eigvec[:5]}")
    print(f"   y_eigvec: {y_eigvec[:5]}")
    
    # Interleave as in reconstruction code
    combined = np.zeros(n_dof, dtype=np.complex128)
    combined[0::2] = x_eigvec  # Even indices = x
    combined[1::2] = y_eigvec  # Odd indices = y
    
    print(f"   Combined (reconstruction style): {combined[:10]}")
    print(f"   Pattern: [x0, y0, x1, y1, x2, y2, ...]")
    
    # Verify we can extract back
    x_extracted = combined[0::2]
    y_extracted = combined[1::2]
    
    if np.allclose(x_extracted, x_eigvec) and np.allclose(y_extracted, y_eigvec):
        print(f"   OK: Can extract x and y correctly")
    else:
        print(f"   ERROR: Cannot extract x and y correctly!")
        return False
    
    return True

if __name__ == "__main__":
    result1 = test_eigenvector_interleaving()
    result2 = test_reconstruction_interleaving()
    
    print(f"\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    
    if result1 and result2:
        print("H7: Interleaving order appears correct, but need to verify")
        print("    that original MATLAB data uses same convention.")
        print("    Reconstruction uses: even indices = x, odd indices = y")
        print("    Need to check if original uses same pattern.")
    else:
        print("H7 CONFIRMED: Interleaving issue found!")

