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
    
    # Plot dispersion curves
    plot_handle = ax.plot(wn, fr, 'k.-')
    
    # Add grid
    ax.grid(True, which='minor', alpha=0.3)
    ax.grid(True, which='major', alpha=0.5)
    
    # Add vertical lines for contour segments
    for i in range(1, N_contour_segments):
        ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('wavevector parameter')
    ax.set_ylabel('frequency [Hz]')
    
    return fig, ax, plot_handle[0]


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
        im = ax.imshow(design[:, :, prop_idx], cmap='viridis', vmin=0, vmax=1)
        ax.set_aspect('equal')
        ax.set_title(titles[prop_idx], fontsize=12)
        ax.axis('off')  # Remove axis ticks for cleaner look
        subax_handle.append(ax)
        im_list.append(im)
    
    # Add colorbar on the right side, same height as the plots
    # Use the last subplot's image for colorbar
    cbar = fig.colorbar(im_list[-1], ax=subax_handle, orientation='vertical', 
                       fraction=0.046, pad=0.04)
    cbar.set_label('Property Value', fontsize=11)
    
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
    Plot dispersion contour plot.
    
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
        Axes to plot on. If None, creates new figure and axes.
        
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
    
    if N_k_x is None or N_k_y is None:
        N_k_y = int(np.sqrt(wv.shape[0]))
        N_k_x = int(np.sqrt(wv.shape[0]))
    
    # Reshape data for contour plot
    X = wv[:, 0].reshape(N_k_y, N_k_x)
    Y = wv[:, 1].reshape(N_k_y, N_k_x)
    Z = fr.reshape(N_k_y, N_k_x)
    
    # Create contour plot
    contour = ax.contour(X, Y, Z, levels=20)
    ax.clabel(contour, inline=True, fontsize=8)
    
    ax.set_xlabel(r'$\gamma_x$')
    ax.set_ylabel(r'$\gamma_y$')
    ax.set_aspect('equal')
    
    # Tighten axes
    ax.set_xlim([np.min(X), np.max(X)])
    ax.set_ylim([np.min(Y), np.max(Y)])
    
    return fig, ax


def plot_mode(design, eigenvector, const, mode_idx=0, ax=None):
    """
    Plot mode shape for a given eigenvector.
    
    Parameters
    ----------
    design : array_like
        Design pattern
    eigenvector : array_like
        Eigenvector (mode shape)
    const : dict
        Constants structure
    mode_idx : int, optional
        Mode index to plot (default: 0)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
        
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
    
    # Reshape to spatial grid
    N_nodes = const['N_ele'] * N_pix_val + 1
    u_disp = mode_shape[::2].reshape(N_nodes, N_nodes)  # x-displacements
    v_disp = mode_shape[1::2].reshape(N_nodes, N_nodes)  # y-displacements
    
    # Create quiver plot
    x_coords, y_coords = np.meshgrid(np.linspace(0, 1, N_nodes), 
                                    np.linspace(0, 1, N_nodes))
    
    ax.quiver(x_coords, y_coords, u_disp, v_disp, 
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
    if n_designs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_handle = []
    for i, design in enumerate(designs):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
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
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    return fig, axes_handle

