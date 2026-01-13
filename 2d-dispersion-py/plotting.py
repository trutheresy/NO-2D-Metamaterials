"""
Plotting and visualization functions.

This module provides functions for visualizing dispersion relations,
design patterns, and other results from the 2D dispersion analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_dispersion(wn, fr, N_contour_segments, ax=None):
    """
    Plot dispersion relations.
    
    Parameters
    ----------
    wn : array_like
        Wavevector parameter values
    fr : array_like
        Frequency values
    N_contour_segments : int
        Number of contour segments
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
        
    Returns
    -------
    fig_handle : matplotlib.figure.Figure
        Figure handle
    ax_handle : matplotlib.axes.Axes
        Axes handle
    plot_handle : matplotlib.lines.Line2D
        Plot handle
    """
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    # Plot dispersion curves - each band should be a separate line
    # fr is (N_wv, N_eig) - plot each column (band) separately
    plot_handles = []
    for band_idx in range(fr.shape[1]):
        # Plot each band as a separate line to avoid crossing/entanglement
        p = ax.plot(wn, fr[:, band_idx], 'k.-', linewidth=1.5, markersize=3)
        plot_handles.extend(p)
    
    # Add grid
    ax.grid(True, which='minor', alpha=0.3)
    ax.grid(True, which='major', alpha=0.5)
    
    # Add vertical lines for contour segments
    for i in range(1, N_contour_segments):
        ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('wavevector parameter')
    ax.set_ylabel('frequency [Hz]')
    
    return fig, ax, plot_handles[0] if plot_handles else None


def plot_design(design):
    """
    Plot design pattern showing material properties.
    
    Parameters
    ----------
    design : array_like
        3D array containing design (N_pix x N_pix x 3)
        Third dimension: [0] = elastic modulus, [1] = density, [2] = Poisson's ratio
        
    Returns
    -------
    fig_handle : matplotlib.figure.Figure
        Figure handle
    subax_handle : list
        List of subplot axes handles
    """
    
    fig = plt.figure(figsize=(14, 4))
    
    N_prop = 3
    titles = ['Modulus', 'Density', 'Poisson']
    
    subax_handle = []
    im_list = []
    for prop_idx in range(N_prop):
        ax = fig.add_subplot(1, 3, prop_idx + 1)
        
        # Get data range for this property (don't force vmin/vmax to [0, 1] if values exceed that)
        prop_data = design[:, :, prop_idx]
        vmin = np.min(prop_data)
        vmax = np.max(prop_data)
        
        # If all values are the same, use a small range around that value
        if vmin == vmax:
            vmin = vmin - 0.1 if vmin > 0 else 0
            vmax = vmax + 0.1 if vmax < 1 else 1
        
        im = ax.imshow(prop_data, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        
        # Get the unique value(s) in this property to display
        unique_vals = np.unique(prop_data)
        if len(unique_vals) == 1:
            # Uniform value - show it in the title
            ax.set_title(f'{titles[prop_idx]} = {unique_vals[0]:.3f}', fontsize=12)
        else:
            ax.set_title(titles[prop_idx], fontsize=12)
        
        ax.axis('off')  # Remove axis ticks for cleaner look
        
        # Add individual colorbar for each subplot
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', 
                           fraction=0.046, pad=0.04)
        cbar.set_label('Value', fontsize=10)
        
        subax_handle.append(ax)
        im_list.append(im)
    
    plt.tight_layout()
    
    return fig, subax_handle


def plot_dispersion_surface(wv, fr, N_k_x=None, N_k_y=None, ax=None):
    """
    Plot 3D dispersion surface.
    
    Parameters
    ----------
    wv : array_like
        Wavevectors (N x 2)
    fr : array_like
        Frequencies (N x N_eig)
    N_k_x : int, optional
        Number of k-points in x-direction. If None, inferred from data.
    N_k_y : int, optional
        Number of k-points in y-direction. If None, inferred from data.
    ax : matplotlib.axes.Axes, optional
        3D axes to plot on. If None, creates new figure and axes.
        
    Returns
    -------
    fig_handle : matplotlib.figure.Figure
        Figure handle
    ax_handle : matplotlib.axes.Axes
        Axes handle
    """
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    if N_k_x is None or N_k_y is None:
        N_k_y = int(np.sqrt(wv.shape[0]))
        N_k_x = int(np.sqrt(wv.shape[0]))
    
    # Reshape data for surface plot
    X = wv[:, 0].reshape(N_k_y, N_k_x)
    Y = wv[:, 1].reshape(N_k_y, N_k_x)
    Z = fr.reshape(N_k_y, N_k_x)
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    
    ax.set_xlabel(r'$\gamma_x$')
    ax.set_ylabel(r'$\gamma_y$')
    ax.set_zlabel(r'$\omega$')
    
    # Set equal aspect ratio
    ax.set_box_aspect([np.pi, np.pi, np.max(Z)])
    
    return fig, ax


def plot_dispersion_contour(wv, fr, N_k_x=None, N_k_y=None, ax=None):
    """
    Plot dispersion contour plot (2D version).
    
    This function creates a 2D contour plot matching MATLAB's plot_dispersion_contour.m.
    Uses 2D contour instead of 3D for better visualization when saved.
    
    Parameters
    ----------
    wv : array_like
        Wavevectors (N x 2)
    fr : array_like
        Frequencies (N,) - single frequency band for contour
    N_k_x : int, optional
        Number of k-points in x-direction. If None, inferred from data.
    N_k_y : int, optional
        Number of k-points in y-direction. If None, inferred from data.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
        
    Returns
    -------
    fig_handle : matplotlib.figure.Figure
        Figure handle
    ax_handle : matplotlib.axes.Axes
        2D Axes handle
    """
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    if N_k_x is None or N_k_y is None:
        N_k_y = int(np.sqrt(wv.shape[0]))
        N_k_x = int(np.sqrt(wv.shape[0]))
    
    # Reshape data for contour plot
    # MATLAB uses column-major (Fortran) order for reshape
    # MATLAB: X = reshape(squeeze(wv(:,1)),N_k_y,N_k_x)
    X = wv[:, 0].reshape(N_k_y, N_k_x, order='F')
    Y = wv[:, 1].reshape(N_k_y, N_k_x, order='F')
    Z = fr.reshape(N_k_y, N_k_x, order='F')
    
    # Create 2D contour plot
    n_levels = 20
    contour = ax.contour(X, Y, Z, levels=n_levels, cmap='viridis', linewidths=1.2)
    contour_filled = ax.contourf(X, Y, Z, levels=n_levels, cmap='viridis', alpha=0.7)
    
    # Add colorbar
    plt.colorbar(contour_filled, ax=ax, label=r'$\omega$')
    
    ax.set_xlabel(r'$\gamma_x$')
    ax.set_ylabel(r'$\gamma_y$')
    
    # Tighten axes (matching MATLAB tighten_axes function)
    ax.set_xlim([np.min(X), np.max(X)])
    ax.set_ylim([np.min(Y), np.max(Y)])
    
    # Set equal aspect ratio for x and y axes
    ax.set_aspect('equal')
    
    return fig, ax


def plot_mode(design, eigenvector, const, mode_idx=0, ax=None, save_values=False):
    """
    Plot mode shape for a given eigenvector.
    
    Parameters
    ----------
    design : array_like
        Design pattern
    eigenvector : array_like
        Eigenvector (mode shape) - full space eigenvector (already transformed)
    const : dict
        Constants structure
    mode_idx : int, optional
        Mode index to plot (default: 0)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    save_values : bool, optional
        If True, save plotting values for comparison (default: False)
        
    Returns
    -------
    fig_handle : matplotlib.figure.Figure
        Figure handle
    ax_handle : matplotlib.axes.Axes
        Axes handle
    """
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    # Extract mode shape
    mode_shape = eigenvector[:, mode_idx]
    
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_val = N_pix[0]
    else:
        N_pix_val = N_pix
    
    # Reshape to spatial grid (matching MATLAB)
    # MATLAB: U_vec = u(1:2:end); U_mat = reshape(U_vec, const.N_ele*const.N_pix + 1, const.N_ele*const.N_pix + 1)'
    # The ' (transpose) in MATLAB means we need to transpose after reshape in Python
    N_nodes = const['N_ele'] * N_pix_val + 1
    U_vec = mode_shape[::2]  # x-displacements (every other element starting at 0)
    V_vec = mode_shape[1::2]  # y-displacements (every other element starting at 1)
    
    # Reshape and transpose to match MATLAB (MATLAB uses ' transpose operator)
    U_mat = U_vec.reshape(N_nodes, N_nodes).T  # Transpose to match MATLAB
    V_mat = V_vec.reshape(N_nodes, N_nodes).T  # Transpose to match MATLAB
    
    # Create coordinate grids (matching MATLAB)
    # MATLAB: original_nodal_locations = linspace(0,const.a,const.N_ele*const.N_pix + 1)
    # MATLAB: [X,Y] = meshgrid(original_nodal_locations,flip(original_nodal_locations))
    original_nodal_locations = np.linspace(0, const['a'], N_nodes)
    X, Y = np.meshgrid(original_nodal_locations, np.flip(original_nodal_locations))
    
    # Save values if requested
    if save_values:
        from pathlib import Path
        import scipy.io as sio
        test_plots_dir = Path('test_plots')
        test_plots_dir.mkdir(exist_ok=True)
        plot_values = {
            'U_mat': U_mat,
            'V_mat': V_mat,
            'X': X,
            'Y': Y,
            'U_vec': U_vec,
            'V_vec': V_vec,
            'mode_shape': mode_shape,
            'N_nodes': N_nodes
        }
        sio.savemat(str(test_plots_dir / 'plot_mode_values_from_plotting_func.mat'), 
                   plot_values, oned_as='column')
        print(f"💾 TEMPORARY: Saved plot values from plotting function: {test_plots_dir / 'plot_mode_values_from_plotting_func.mat'}")
    
    # Create quiver plot
    ax.quiver(X, Y, np.real(U_mat), np.real(V_mat), 
             scale=1.0, scale_units='xy', alpha=0.7)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Mode Shape {mode_idx + 1}')
    ax.set_aspect('equal')
    
    return fig, ax


def visualize_designs(designs, titles=None):
    """
    Visualize multiple designs in a grid layout.
    
    Parameters
    ----------
    designs : list
        List of design arrays
    titles : list, optional
        List of titles for each design
        
    Returns
    -------
    fig_handle : matplotlib.figure.Figure
        Figure handle
    axes_handle : list
        List of axes handles
    """
    
    n_designs = len(designs)
    n_cols = min(3, n_designs)
    n_rows = (n_designs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    # Ensure axes is always 2D for consistent indexing
    if n_designs == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1) if hasattr(axes, 'reshape') else np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1) if hasattr(axes, 'reshape') else np.array([[ax] for ax in axes])
    
    axes_handle = []
    for i, design in enumerate(designs):
        row = i // n_cols
        col = i % n_cols
        
        ax = axes[row, col] if axes.ndim == 2 else axes[col if n_rows == 1 else row]
        
        im = ax.imshow(design[:, :, 0], cmap='gray', vmin=0, vmax=1)
        ax.set_aspect('equal')
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
        else:
            ax.set_title(f'Design {i+1}')
        
        axes_handle.append(ax)
    
    # Hide unused subplots
    for i in range(n_designs, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if axes.ndim == 2:
            axes[row, col].set_visible(False)
        else:
            axes[col if n_rows == 1 else row].set_visible(False)
    
    plt.tight_layout()
    
    return fig, axes_handle

