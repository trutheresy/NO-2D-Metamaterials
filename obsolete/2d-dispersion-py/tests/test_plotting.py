"""
Unit tests for plotting functions.

Tests plotting functions and generates plots for visual verification.
These tests require manual visual inspection to verify equivalence with MATLAB plots.

IMPORTANT: This test file generates plots that must be visually compared
with MATLAB outputs to verify equivalence.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from plotting import (plot_dispersion, plot_design, plot_dispersion_contour,
                      plot_mode, visualize_designs)
from get_design import get_design
from get_mesh import get_mesh
from wavevectors import get_IBZ_wavevectors, get_IBZ_contour_wavevectors


# Create output directory for test plots
TEST_PLOTS_DIR = Path(__file__).parent.parent / 'test_plots'
TEST_PLOTS_DIR.mkdir(exist_ok=True)


def create_test_const():
    """Create a test constants structure."""
    N_pix = 5
    design = get_design('homogeneous', N_pix)
    
    const = {
        'N_pix': N_pix,
        'N_ele': 4,
        'a': 1.0,
        'design': design,
        'design_scale': 'linear',
        'E_min': 2e9,
        'E_max': 200e9,
        'rho_min': 1000,
        'rho_max': 8000,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 0.01,
        'N_eig': 5,
        'sigma_eig': 'SM',
        'isSaveEigenvectors': True,
        'isSaveMesh': False,
        'isUseGPU': False,
        'isUseParallel': False,
        'isUseImprovement': False,
        'isUseSecondImprovement': False
    }
    return const


class TestPlotDesign:
    """Test plot_design function - VISUAL VERIFICATION REQUIRED."""
    
    def test_plot_design_homogeneous(self):
        """Test plotting homogeneous design - GENERATES PLOT FOR VISUAL CHECK."""
        design = get_design('homogeneous', 5)
        
        fig, axes = plot_design(design)
        
        # Save plot for visual inspection
        output_path = TEST_PLOTS_DIR / 'plot_design_homogeneous.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✅ Generated plot: {output_path}")
        print("   📊 VISUAL CHECK REQUIRED: Compare with MATLAB plot_design.m output")
        print("   Expected: 3 subplots showing Modulus, Density, Poisson (all uniform)")
        
        assert fig is not None
        assert len(axes) == 3
    
    def test_plot_design_random(self):
        """Test plotting random design - GENERATES PLOT FOR VISUAL CHECK."""
        design = get_design(42, 8)  # Random design with seed
        
        fig, axes = plot_design(design)
        
        output_path = TEST_PLOTS_DIR / 'plot_design_random.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✅ Generated plot: {output_path}")
        print("   📊 VISUAL CHECK REQUIRED: Compare with MATLAB plot_design.m output")
        print("   Expected: 3 subplots showing binary pattern for random design")


class TestPlotDispersion:
    """Test plot_dispersion function - VISUAL VERIFICATION REQUIRED."""
    
    def test_plot_dispersion_basic(self):
        """Test basic dispersion plot - GENERATES PLOT FOR VISUAL CHECK."""
        # Generate test dispersion data
        const = create_test_const()
        wavevectors, contour_info = get_IBZ_contour_wavevectors(10, const['a'], symmetry_type='p4mm')
        
        # Calculate actual dispersion instead of random data
        from dispersion import dispersion
        wv, fr, ev, mesh = dispersion(const, wavevectors)
        wn = contour_info['wavevector_parameter']  # Wavevector parameter
        
        fig, ax, plot_handle = plot_dispersion(wn, fr, N_contour_segments=3)
        
        output_path = TEST_PLOTS_DIR / 'plot_dispersion_basic.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✅ Generated plot: {output_path}")
        print("   📊 VISUAL CHECK REQUIRED: Compare with MATLAB plot_dispersion.m output")
        print("   Expected: Line plot with dispersion curves, vertical lines at segment boundaries")
    
    def test_plot_dispersion_contour(self):
        """Test dispersion contour plot - GENERATES PLOT FOR VISUAL CHECK."""
        # Generate test data - use actual dispersion calculation
        const = create_test_const()
        
        # Create a square grid of wavevectors for contour plot (matching MATLAB usage)
        # MATLAB plot_dispersion_contour expects a rectangular grid
        N_k = 20  # Grid size
        a = const['a']
        kx = np.linspace(-np.pi/a, np.pi/a, N_k)
        ky = np.linspace(-np.pi/a, np.pi/a, N_k)
        # MATLAB meshgrid uses 'xy' indexing by default (Cartesian)
        # This means KX varies along columns, KY varies along rows
        KX, KY = np.meshgrid(kx, ky, indexing='xy')  # 'xy' matches MATLAB default
        # Flatten in column-major (Fortran) order to match MATLAB's (:)
        wv_grid = np.column_stack([KX.ravel(order='F'), KY.ravel(order='F')])
        
        # Calculate actual dispersion for the grid
        from dispersion import dispersion
        wv, fr, ev, mesh = dispersion(const, wv_grid)
        
        # Save plot data for comparison with MATLAB
        # Reshape to get X, Y, Z matrices (matching what will be plotted)
        N_k_y, N_k_x = N_k, N_k
        X_plot = wv_grid[:, 0].reshape(N_k_y, N_k_x, order='F')
        Y_plot = wv_grid[:, 1].reshape(N_k_y, N_k_x, order='F')
        Z_plot = fr[:, 0].reshape(N_k_y, N_k_x, order='F')
        
        # Save to .mat file for MATLAB comparison
        import scipy.io as sio
        plot_data = {
            'wv_grid': wv_grid,
            'frequencies': fr,
            'X': X_plot,
            'Y': Y_plot,
            'Z': Z_plot,
            'N_k_x': N_k_x,
            'N_k_y': N_k_y,
            'const': const
        }
        data_path = TEST_PLOTS_DIR / 'plot_dispersion_contour_data.mat'
        sio.savemat(str(data_path), plot_data, oned_as='column')
        print(f"\n💾 Saved plot data for comparison: {data_path}")
        print(f"   Contains: wv_grid, frequencies, X, Y, Z matrices")
        
        # Use first frequency band for contour plot
        fig, ax = plot_dispersion_contour(wv_grid, fr[:, 0], N_k, N_k)
        
        output_path = TEST_PLOTS_DIR / 'plot_dispersion_contour.png'
        # Use higher DPI and better rendering for 3D plots
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.1)
        plt.close(fig)
        
        print(f"\n✅ Generated plot: {output_path}")
        print("   📊 VISUAL CHECK REQUIRED: Compare with MATLAB plot_dispersion_contour.m output")
        print("   Expected: Contour plot of frequency surface in k-space")


class TestPlotMode:
    """Test plot_mode function - VISUAL VERIFICATION REQUIRED."""
    
    def test_plot_mode_basic(self):
        """Test basic mode shape plot - GENERATES PLOT FOR VISUAL CHECK."""
        const = create_test_const()
        design = const['design']
        
        # Use actual dispersion calculation to get real eigenvectors (matching MATLAB)
        # MATLAB plot_mode uses eigenvectors from dispersion calculation
        from dispersion import dispersion
        from wavevectors import get_IBZ_wavevectors
        
        # Get a wavevector at Gamma point (k=0) for testing
        wv_gamma = np.array([[0.0, 0.0]])
        wv, fr, ev, mesh = dispersion(const, wv_gamma)
        
        # ev shape: (N_dof, N_k, N_eig)
        # Extract eigenvector for first wavevector (k=0) and first mode
        k_idx = 0
        eig_idx = 0
        eigenvector_reduced = ev[:, k_idx, eig_idx]  # Reduced space eigenvector
        
        # Transform to full space (matching MATLAB: u = T*u_reduced)
        from system_matrices import get_transformation_matrix
        wv_vec = wv_gamma[k_idx, :]
        T = get_transformation_matrix(wv_vec, const)
        eigenvector_full = T @ eigenvector_reduced  # Full space eigenvector
        
        # Normalize (matching MATLAB: u = u/max(abs(u))*(1/10)*const.a)
        scale_factor = (1/10) * const['a']
        eigenvector_full = eigenvector_full / np.max(np.abs(eigenvector_full)) * scale_factor
        
        # Save plotting values for comparison
        N_nodes = const['N_ele'] * const['N_pix'] + 1
        U_vec = eigenvector_full[::2]  # x-displacements
        V_vec = eigenvector_full[1::2]  # y-displacements
        U_mat = U_vec.reshape(N_nodes, N_nodes)
        V_mat = V_vec.reshape(N_nodes, N_nodes)
        
        # Create coordinate grids
        x_coords = np.linspace(0, const['a'], N_nodes)
        y_coords = np.linspace(0, const['a'], N_nodes)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Save data for comparison
        import scipy.io as sio
        plot_data = {
            'wv': wv_gamma,
            'fr': fr,
            'eigenvector_reduced': eigenvector_reduced,
            'eigenvector_full': eigenvector_full,
            'U_mat': U_mat,
            'V_mat': V_mat,
            'X': X,
            'Y': Y,
            'N_nodes': N_nodes,
            'k_idx': k_idx + 1,  # MATLAB uses 1-based indexing
            'eig_idx': eig_idx + 1,  # MATLAB uses 1-based indexing
            'const': const
        }
        data_path = TEST_PLOTS_DIR / 'plot_mode_basic_data.mat'
        sio.savemat(str(data_path), plot_data, oned_as='column')
        print(f"\n💾 Saved plot data for comparison: {data_path}")
        print(f"   Contains: eigenvectors, U_mat, V_mat, X, Y matrices")
        
        # Use full space eigenvector for plotting
        # plot_mode expects eigenvector in format (N_dof, N_modes)
        eigenvector_for_plot = np.zeros((len(eigenvector_full), 1), dtype=complex)
        eigenvector_for_plot[:, 0] = eigenvector_full
        
        # Call plot_mode with save_values=True to save values for comparison
        fig, ax = plot_mode(design, eigenvector_for_plot, const, mode_idx=0, save_values=True)
        
        output_path = TEST_PLOTS_DIR / 'plot_mode_basic.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✅ Generated plot: {output_path}")
        print("   📊 VISUAL CHECK REQUIRED: Compare with MATLAB plot_mode.m output")
        print("   Expected: Quiver plot showing displacement field (mode shape)")


class TestVisualizeDesigns:
    """Test visualize_designs function - VISUAL VERIFICATION REQUIRED."""
    
    def test_visualize_designs_multiple(self):
        """Test visualizing multiple designs - GENERATES PLOT FOR VISUAL CHECK."""
        designs = [
            get_design('homogeneous', 5),
            get_design('dispersive-tetragonal', 5),
            get_design(42, 5)  # Random
        ]
        titles = ['Homogeneous', 'Tetragonal', 'Random']
        
        fig, axes = visualize_designs(designs, titles)
        
        output_path = TEST_PLOTS_DIR / 'visualize_designs_multiple.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✅ Generated plot: {output_path}")
        print("   📊 VISUAL CHECK REQUIRED: Compare with MATLAB visualize_designs.m output")
        print("   Expected: Grid of design visualizations")


class TestPlotEigenvector:
    """Test eigenvector plotting (equivalent to plot_eigenvector.m) - VISUAL VERIFICATION REQUIRED."""
    
    def test_plot_eigenvector_components(self):
        """Test plotting eigenvector components - GENERATES PLOTS FOR VISUAL CHECK."""
        const = create_test_const()
        
        # Use actual dispersion calculation to get real eigenvectors (matching MATLAB)
        from dispersion import dispersion
        from system_matrices import get_transformation_matrix
        
        # Get a wavevector at Gamma point (k=0) for testing
        wv_gamma = np.array([[0.0, 0.0]])
        wv, fr, ev, mesh = dispersion(const, wv_gamma)
        
        # Extract eigenvector for first wavevector (k=0) and first mode
        k_idx = 0
        eig_idx = 0
        eigenvector_reduced = ev[:, k_idx, eig_idx]  # Reduced space eigenvector
        
        # Transform to full space (matching MATLAB: u = T*u_reduced)
        wv_vec = wv_gamma[k_idx, :]
        T = get_transformation_matrix(wv_vec, const)
        eigenvector_full = T @ eigenvector_reduced  # Full space eigenvector
        
        # Match MATLAB's data type behavior: if eigenvector_reduced is real and T is effectively real (k=0),
        # the result should be real (float64), not complex
        if np.isrealobj(eigenvector_reduced) and np.allclose(wv_vec, 0.0):
            # At k=0, T is effectively real (phase factors are 1.0), so result should be real
            eigenvector_full = np.real(eigenvector_full).astype(np.float64)
        
        # Extract u and v components
        u = eigenvector_full[0::2]  # Horizontal displacement (x)
        v = eigenvector_full[1::2]  # Vertical displacement (y)
        
        # Reshape to spatial grid
        # MATLAB: u = reshape(u, design_size(1:2))
        # MATLAB uses column-major (Fortran) order
        # The eigenvector corresponds to the mesh, not the design pixels
        N_node = const['N_ele'] * const['N_pix'] + 1
        # Ensure we have enough elements
        if len(u) >= N_node * N_node:
            u_reshaped = u[:N_node*N_node].reshape(N_node, N_node, order='F')
            v_reshaped = v[:N_node*N_node].reshape(N_node, N_node, order='F')
        else:
            # Fallback: use available data
            N_actual = int(np.sqrt(len(u)))
            u_reshaped = u[:N_actual*N_actual].reshape(N_actual, N_actual, order='F')
            v_reshaped = v[:N_actual*N_actual].reshape(N_actual, N_actual, order='F')
        
        # Save plotting values for comparison
        import scipy.io as sio
        plot_data = {
            'wv': wv_gamma,
            'fr': fr,
            'eigenvector_reduced': eigenvector_reduced,
            'eigenvector_full': eigenvector_full,
            'u': u,
            'v': v,
            'u_reshaped': u_reshaped,
            'v_reshaped': v_reshaped,
            'N_node': N_node,
            'k_idx': k_idx + 1,  # MATLAB uses 1-based indexing
            'eig_idx': eig_idx + 1,  # MATLAB uses 1-based indexing
            'const': const
        }
        data_path = TEST_PLOTS_DIR / 'plot_eigenvector_components_data.mat'
        sio.savemat(str(data_path), plot_data, oned_as='column')
        print(f"\n💾 Saved plot data for comparison: {data_path}")
        print(f"   Contains: eigenvectors, u, v, u_reshaped, v_reshaped matrices")
        
        # Plot real and imaginary parts (matching MATLAB imagesc with YDir='normal')
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Set extent to match actual coordinates (0 to N_node-1 in both directions)
        # origin='lower' puts (0,0) at bottom-left (like MATLAB imagesc with YDir='normal')
        extent = [0, N_node-1, 0, N_node-1]
        
        im1 = axes[0, 0].imshow(np.real(u_reshaped), cmap='viridis', origin='lower', 
                                extent=extent, aspect='equal', interpolation='nearest')
        axes[0, 0].set_title('real(u)')
        axes[0, 0].set_xlabel('x index')
        axes[0, 0].set_ylabel('y index')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        im2 = axes[0, 1].imshow(np.imag(u_reshaped), cmap='viridis', origin='lower',
                                extent=extent, aspect='equal', interpolation='nearest')
        axes[0, 1].set_title('imag(u)')
        axes[0, 1].set_xlabel('x index')
        axes[0, 1].set_ylabel('y index')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        im3 = axes[1, 0].imshow(np.real(v_reshaped), cmap='viridis', origin='lower',
                                extent=extent, aspect='equal', interpolation='nearest')
        axes[1, 0].set_title('real(v)')
        axes[1, 0].set_xlabel('x index')
        axes[1, 0].set_ylabel('y index')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        im4 = axes[1, 1].imshow(np.imag(v_reshaped), cmap='viridis', origin='lower',
                                extent=extent, aspect='equal', interpolation='nearest')
        axes[1, 1].set_title('imag(v)')
        axes[1, 1].set_xlabel('x index')
        axes[1, 1].set_ylabel('y index')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Reduce spacing between subplots
        plt.tight_layout(pad=1.0)
        
        output_path = TEST_PLOTS_DIR / 'plot_eigenvector_components.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✅ Generated plot: {output_path}")
        print("   📊 VISUAL CHECK REQUIRED: Compare with MATLAB plot_eigenvector.m output")
        print("   Expected: 4 subplots showing real(u), imag(u), real(v), imag(v)")


class TestPlotMesh:
    """Test mesh plotting (equivalent to plot_mesh.m) - VISUAL VERIFICATION REQUIRED."""
    
    def test_plot_mesh_basic(self):
        """Test basic mesh plot - GENERATES PLOT FOR VISUAL CHECK."""
        const = create_test_const()
        mesh = get_mesh(const)
        
        # Plot mesh (simplified version - full implementation needed)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Extract node coordinates
        node_coords_x = mesh['node_coords'][0]
        node_coords_y = mesh['node_coords'][1]
        
        # Plot nodes
        ax.scatter(node_coords_x.flatten(), node_coords_y.flatten(), 
                  c='black', s=20, zorder=3)
        
        # Plot elements (simplified - would need full implementation)
        # This is a placeholder - full mesh plotting needs to be implemented
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Mesh')
        
        output_path = TEST_PLOTS_DIR / 'plot_mesh_basic.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✅ Generated plot: {output_path}")
        print("   📊 VISUAL CHECK REQUIRED: Compare with MATLAB plot_mesh.m output")
        print("   ⚠️  NOTE: Full mesh element plotting not yet implemented")


class TestPlotWavevectors:
    """Test wavevector plotting (equivalent to plot_wavevectors.m) - VISUAL VERIFICATION REQUIRED."""
    
    def test_plot_wavevectors_basic(self):
        """Test basic wavevector plot - GENERATES PLOT FOR VISUAL CHECK."""
        a = 1.0
        wavevectors = get_IBZ_wavevectors([10, 10], a, symmetry_type='none')
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(wavevectors[:, 0], wavevectors[:, 1], 
                  c='black', s=30, alpha=0.5, zorder=2)
        
        # Add labels
        for i in range(len(wavevectors)):
            ax.text(wavevectors[i, 0] + 0.05, wavevectors[i, 1] + 0.05, 
                   str(i + 1), fontsize=8)
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$\gamma_x$')
        ax.set_ylabel(r'$\gamma_y$')
        ax.set_title('Wavevectors')
        ax.grid(True, alpha=0.3)
        
        output_path = TEST_PLOTS_DIR / 'plot_wavevectors_basic.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✅ Generated plot: {output_path}")
        print("   📊 VISUAL CHECK REQUIRED: Compare with MATLAB plot_wavevectors.m output")
        print("   Expected: Scatter plot with numbered wavevector points")


def run_all_plotting_tests():
    """Run all plotting tests and generate summary."""
    print("=" * 80)
    print("PLOTTING TESTS - VISUAL VERIFICATION REQUIRED")
    print("=" * 80)
    print(f"\nPlots will be saved to: {TEST_PLOTS_DIR}")
    print("\nThese plots must be visually compared with MATLAB outputs.")
    print("=" * 80)
    
    # Run tests
    test_design = TestPlotDesign()
    test_design.test_plot_design_homogeneous()
    test_design.test_plot_design_random()
    
    test_dispersion = TestPlotDispersion()
    test_dispersion.test_plot_dispersion_basic()
    test_dispersion.test_plot_dispersion_contour()
    
    test_mode = TestPlotMode()
    test_mode.test_plot_mode_basic()
    
    test_visualize = TestVisualizeDesigns()
    test_visualize.test_visualize_designs_multiple()
    
    test_eigenvector = TestPlotEigenvector()
    test_eigenvector.test_plot_eigenvector_components()
    
    test_mesh = TestPlotMesh()
    test_mesh.test_plot_mesh_basic()
    
    test_wavevectors = TestPlotWavevectors()
    test_wavevectors.test_plot_wavevectors_basic()
    
    print("\n" + "=" * 80)
    print("✅ All plotting tests completed!")
    print(f"📁 Check plots in: {TEST_PLOTS_DIR}")
    print("📊 Please visually compare each plot with corresponding MATLAB output")
    print("=" * 80)


if __name__ == '__main__':
    run_all_plotting_tests()

