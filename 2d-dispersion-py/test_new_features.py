"""
Test script for new features added to 2d-dispersion-py.

This script tests:
1. linspaceNDim function
2. Updated get_IBZ_contour_wavevectors function
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import linspaceNDim
from wavevectors import get_IBZ_contour_wavevectors


def test_linspaceNDim():
    """Test the linspaceNDim function."""
    print("=" * 70)
    print("Testing linspaceNDim function")
    print("=" * 70)
    
    # Test 1: 2D points
    d1 = [0, 0]
    d2 = [1, 2]
    Y = linspaceNDim(d1, d2, 5)
    
    print("\nTest 1: 2D points")
    print(f"  Start point: {d1}")
    print(f"  End point:   {d2}")
    print(f"  Number of points: 5")
    print(f"  Result shape: {Y.shape}")
    print(f"  Result:\n{Y}")
    
    expected = np.array([
        [0.0, 0.0],
        [0.25, 0.5],
        [0.5, 1.0],
        [0.75, 1.5],
        [1.0, 2.0]
    ])
    
    if np.allclose(Y, expected):
        print("  ✓ Test 1 PASSED")
    else:
        print("  ✗ Test 1 FAILED")
        print(f"  Expected:\n{expected}")
    
    # Test 2: 3D points
    d1 = [0, 0, 0]
    d2 = [1, 1, 1]
    Y = linspaceNDim(d1, d2, 3)
    
    print("\nTest 2: 3D points")
    print(f"  Start point: {d1}")
    print(f"  End point:   {d2}")
    print(f"  Number of points: 3")
    print(f"  Result shape: {Y.shape}")
    print(f"  Result:\n{Y}")
    
    expected = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0]
    ])
    
    if np.allclose(Y, expected):
        print("  ✓ Test 2 PASSED")
    else:
        print("  ✗ Test 2 FAILED")
        print(f"  Expected:\n{expected}")
    
    # Test 3: Default n parameter
    d1 = [0]
    d2 = [1]
    Y = linspaceNDim(d1, d2)  # Should default to 100 points
    
    print("\nTest 3: Default n parameter (should be 100)")
    print(f"  Result shape: {Y.shape}")
    
    if Y.shape == (100, 1):
        print("  ✓ Test 3 PASSED")
    else:
        print("  ✗ Test 3 FAILED")
    
    print("\n" + "=" * 70)


def test_get_IBZ_contour_wavevectors():
    """Test the updated get_IBZ_contour_wavevectors function."""
    print("\n" + "=" * 70)
    print("Testing get_IBZ_contour_wavevectors function")
    print("=" * 70)
    
    a = 1.0  # Lattice parameter
    N_k = 50  # Number of points per segment
    
    # Test different symmetry types
    symmetry_types = ['p4mm', 'c1m1', 'none']
    
    for symmetry_type in symmetry_types:
        print(f"\nTest: symmetry_type = '{symmetry_type}'")
        
        try:
            wavevectors, contour_info = get_IBZ_contour_wavevectors(N_k, a, symmetry_type)
            
            print(f"  Number of wavevectors: {len(wavevectors)}")
            print(f"  Number of segments: {contour_info['N_segment']}")
            print(f"  Vertex labels: {contour_info['vertex_labels']}")
            print(f"  Wavevector shape: {wavevectors.shape}")
            print(f"  First 3 wavevectors:\n{wavevectors[:3]}")
            print(f"  Last 3 wavevectors:\n{wavevectors[-3:]}")
            
            # Check that wavevectors is 2D array
            if wavevectors.ndim == 2 and wavevectors.shape[1] == 2:
                print(f"  ✓ Wavevectors have correct shape")
            else:
                print(f"  ✗ Wavevectors have incorrect shape")
            
            # Check that contour info has required keys
            required_keys = ['N_segment', 'vertex_labels', 'vertices', 'wavevector_parameter']
            if all(key in contour_info for key in required_keys):
                print(f"  ✓ Contour info has all required keys")
            else:
                print(f"  ✗ Contour info missing keys")
            
        except Exception as e:
            print(f"  ✗ Test FAILED with error: {e}")
    
    print("\n" + "=" * 70)


def visualize_contours():
    """Visualize the IBZ contours for different symmetry types."""
    print("\n" + "=" * 70)
    print("Visualizing IBZ contours")
    print("=" * 70)
    
    a = 1.0
    N_k = 100
    
    symmetry_types = ['p4mm', 'c1m1', 'none']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, symmetry_type in enumerate(symmetry_types):
        ax = axes[idx]
        
        try:
            wavevectors, contour_info = get_IBZ_contour_wavevectors(N_k, a, symmetry_type)
            
            # Plot the contour
            ax.plot(wavevectors[:, 0], wavevectors[:, 1], 'b-', linewidth=2, label='Contour')
            
            # Mark vertices
            vertices = contour_info['vertices']
            ax.plot(vertices[:, 0], vertices[:, 1], 'ro', markersize=8, label='Vertices')
            
            # Add vertex labels
            for i, (vertex, label) in enumerate(zip(vertices, contour_info['vertex_labels'])):
                # Remove duplicate labels at same position
                if i == 0 or not np.allclose(vertex, vertices[i-1]):
                    ax.annotate(label, xy=vertex, xytext=(5, 5), 
                              textcoords='offset points', fontsize=12)
            
            ax.set_xlabel(r'$k_x$', fontsize=12)
            ax.set_ylabel(r'$k_y$', fontsize=12)
            ax.set_title(f'Symmetry: {symmetry_type}\n({len(wavevectors)} points)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Symmetry: {symmetry_type} (FAILED)')
    
    plt.tight_layout()
    plt.savefig('IBZ_contours_test.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved as 'IBZ_contours_test.png'")
    plt.show()
    
    print("=" * 70)


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TESTING NEW FEATURES IN 2D-DISPERSION-PY")
    print("=" * 70 + "\n")
    
    # Test linspaceNDim
    test_linspaceNDim()
    
    # Test get_IBZ_contour_wavevectors
    test_get_IBZ_contour_wavevectors()
    
    # Visualize contours
    visualize_contours()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

