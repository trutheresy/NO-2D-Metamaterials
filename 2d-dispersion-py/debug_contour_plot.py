"""
Debug script to examine contour plot data generation step by step.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import sys
sys.path.insert(0, '.')

from tests.test_plotting import create_test_const
from dispersion import dispersion

def debug_contour_data():
    """Debug the contour plot data generation."""
    
    const = create_test_const()
    N_k = 20
    a = const['a']
    
    print("=" * 80)
    print("DEBUGGING CONTOUR PLOT DATA GENERATION")
    print("=" * 80)
    
    # Step 1: Create grid
    print("\n1. Creating wavevector grid...")
    kx = np.linspace(-np.pi/a, np.pi/a, N_k)
    ky = np.linspace(-np.pi/a, np.pi/a, N_k)
    print(f"   kx range: [{kx[0]:.6f}, {kx[-1]:.6f}], length: {len(kx)}")
    print(f"   ky range: [{ky[0]:.6f}, {ky[-1]:.6f}], length: {len(ky)}")
    
    # Try different meshgrid options
    print("\n2. Testing meshgrid options...")
    
    # Option 1: 'ij' indexing (matrix indexing)
    KX_ij, KY_ij = np.meshgrid(kx, ky, indexing='ij')
    wv_grid_ij = np.column_stack([KX_ij.ravel(order='F'), KY_ij.ravel(order='F')])
    print(f"   'ij' indexing: KX shape {KX_ij.shape}, wv_grid shape {wv_grid_ij.shape}")
    print(f"   First 5 wavevectors (ij):")
    print(wv_grid_ij[:5, :])
    
    # Option 2: 'xy' indexing (Cartesian indexing, MATLAB default)
    KX_xy, KY_xy = np.meshgrid(kx, ky, indexing='xy')
    wv_grid_xy = np.column_stack([KX_xy.ravel(order='F'), KY_xy.ravel(order='F')])
    print(f"\n   'xy' indexing: KX shape {KX_xy.shape}, wv_grid shape {wv_grid_xy.shape}")
    print(f"   First 5 wavevectors (xy):")
    print(wv_grid_xy[:5, :])
    
    # Calculate dispersion for both
    print("\n3. Calculating dispersion...")
    wv1, fr1, ev1, mesh1 = dispersion(const, wv_grid_ij)
    wv2, fr2, ev2, mesh2 = dispersion(const, wv_grid_xy)
    
    print(f"   Dispersion results shape: {fr1.shape}")
    print(f"   Frequency range (ij): [{np.min(fr1):.2f}, {np.max(fr1):.2f}] Hz")
    print(f"   Frequency range (xy): [{np.min(fr2):.2f}, {np.max(fr2):.2f}] Hz")
    
    # Reshape and examine
    print("\n4. Reshaping for contour plot...")
    
    # For 'ij' indexing
    X1 = wv_grid_ij[:, 0].reshape(N_k, N_k, order='F')
    Y1 = wv_grid_ij[:, 1].reshape(N_k, N_k, order='F')
    Z1 = fr1[:, 0].reshape(N_k, N_k, order='F')
    
    # For 'xy' indexing
    X2 = wv_grid_xy[:, 0].reshape(N_k, N_k, order='F')
    Y2 = wv_grid_xy[:, 1].reshape(N_k, N_k, order='F')
    Z2 = fr2[:, 0].reshape(N_k, N_k, order='F')
    
    print(f"\n   'ij' indexing reshape:")
    print(f"   X[0,0]={X1[0,0]:.6f}, X[0,-1]={X1[0,-1]:.6f}, X[-1,0]={X1[-1,0]:.6f}, X[-1,-1]={X1[-1,-1]:.6f}")
    print(f"   Y[0,0]={Y1[0,0]:.6f}, Y[0,-1]={Y1[0,-1]:.6f}, Y[-1,0]={Y1[-1,0]:.6f}, Y[-1,-1]={Y1[-1,-1]:.6f}")
    print(f"   Z[0,0]={Z1[0,0]:.2f}, Z[0,-1]={Z1[0,-1]:.2f}, Z[-1,0]={Z1[-1,0]:.2f}, Z[-1,-1]={Z1[-1,-1]:.2f}")
    
    print(f"\n   'xy' indexing reshape:")
    print(f"   X[0,0]={X2[0,0]:.6f}, X[0,-1]={X2[0,-1]:.6f}, X[-1,0]={X2[-1,0]:.6f}, X[-1,-1]={X2[-1,-1]:.6f}")
    print(f"   Y[0,0]={Y2[0,0]:.6f}, Y[0,-1]={Y2[0,-1]:.6f}, Y[-1,0]={Y2[-1,0]:.6f}, Y[-1,-1]={Y2[-1,-1]:.6f}")
    print(f"   Z[0,0]={Z2[0,0]:.2f}, Z[0,-1]={Z2[0,-1]:.2f}, Z[-1,0]={Z2[-1,0]:.2f}, Z[-1,-1]={Z2[-1,-1]:.2f}")
    
    # Check symmetry - for homogeneous design, frequencies should be symmetric
    print("\n5. Checking symmetry (homogeneous design should be symmetric)...")
    z1_symmetry = np.allclose(Z1, Z1.T, rtol=1e-4)
    z2_symmetry = np.allclose(Z2, Z2.T, rtol=1e-4)
    print(f"   Z (ij) is symmetric: {z1_symmetry}")
    print(f"   Z (xy) is symmetric: {z2_symmetry}")
    
    # Save both for comparison
    test_plots_dir = Path('test_plots')
    test_plots_dir.mkdir(exist_ok=True)
    
    sio.savemat(str(test_plots_dir / 'debug_contour_ij.mat'), {
        'X': X1, 'Y': Y1, 'Z': Z1, 'wv_grid': wv_grid_ij, 'fr': fr1
    }, oned_as='column')
    
    sio.savemat(str(test_plots_dir / 'debug_contour_xy.mat'), {
        'X': X2, 'Y': Y2, 'Z': Z2, 'wv_grid': wv_grid_xy, 'fr': fr2
    }, oned_as='column')
    
    print(f"\n💾 Saved debug data:")
    print(f"   - test_plots/debug_contour_ij.mat (ij indexing)")
    print(f"   - test_plots/debug_contour_xy.mat (xy indexing)")


if __name__ == '__main__':
    debug_contour_data()

