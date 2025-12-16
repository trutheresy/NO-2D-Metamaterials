# Plotting API Reference

This document describes the expected input and output types and shapes for all plotting functions in the Python dispersion library.

## Table of Contents
1. [plot_dispersion](#plot_dispersion)
2. [plot_dispersion_contour](#plot_dispersion_contour)
3. [plot_dispersion_surface](#plot_dispersion_surface)
4. [plot_design](#plot_design)
5. [plot_mode](#plot_mode)
6. [visualize_designs](#visualize_designs)

---

## plot_dispersion

Plot dispersion curves (frequency vs wavevector parameter).

### Inputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `wn` | `np.ndarray` (float) | `(N_wv,)` | Wavevector parameter values (1D array) |
| `fr` | `np.ndarray` (float) | `(N_wv, N_eig)` | Frequency values for each wavevector and mode |
| `N_contour_segments` | `int` | scalar | Number of contour segments (for vertical line markers) |
| `ax` | `matplotlib.axes.Axes` (optional) | - | Axes to plot on. If `None`, creates new figure. |

### Outputs

| Return Value | Type | Description |
|--------------|------|-------------|
| `fig` | `matplotlib.figure.Figure` | Figure handle |
| `ax` | `matplotlib.axes.Axes` | Axes handle |
| `plot_handle` | `matplotlib.lines.Line2D` or `None` | Plot handle (first line object, or None if no lines) |

### Example
```python
wn = np.linspace(0, 10, 100)  # Shape: (100,)
fr = np.random.rand(100, 5)   # Shape: (100, 5) - 100 wavevectors, 5 modes
fig, ax, line = plot_dispersion(wn, fr, N_contour_segments=5)
```

---

## plot_dispersion_contour

Plot 2D contour plot of frequency surface in k-space.

### Inputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `wv` | `np.ndarray` (float) | `(N, 2)` | Wavevectors - each row is [kx, ky] |
| `fr` | `np.ndarray` (float) | `(N,)` | Frequency values for a single band (1D array) |
| `N_k_x` | `int` (optional) | scalar | Number of k-points in x-direction. If `None`, inferred as `sqrt(N)`. |
| `N_k_y` | `int` (optional) | scalar | Number of k-points in y-direction. If `None`, inferred as `sqrt(N)`. |
| `ax` | `matplotlib.axes.Axes` (optional) | - | Axes to plot on. If `None`, creates new figure. |

### Outputs

| Return Value | Type | Description |
|--------------|------|-------------|
| `fig` | `matplotlib.figure.Figure` | Figure handle |
| `ax` | `matplotlib.axes.Axes` | 2D Axes handle |

### Notes
- `wv` is reshaped to `(N_k_y, N_k_x, 2)` using Fortran (column-major) order to match MATLAB
- `fr` is reshaped to `(N_k_y, N_k_x)` using Fortran order
- Creates a 2D contour plot (not 3D) for better visualization when saved

### Example
```python
wv = np.random.rand(100, 2)  # Shape: (100, 2) - 100 wavevectors
fr = np.random.rand(100)      # Shape: (100,) - frequencies for one band
fig, ax = plot_dispersion_contour(wv, fr, N_k_x=10, N_k_y=10)
```

---

## plot_dispersion_surface

Plot 3D surface plot of dispersion relation.

### Inputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `wv` | `np.ndarray` (float) | `(N, 2)` | Wavevectors - each row is [kx, ky] |
| `fr` | `np.ndarray` (float) | `(N,)` | Frequency values for a single band (1D array) |
| `N_k_x` | `int` (optional) | scalar | Number of k-points in x-direction. If `None`, inferred as `sqrt(N)`. |
| `N_k_y` | `int` (optional) | scalar | Number of k-points in y-direction. If `None`, inferred as `sqrt(N)`. |
| `ax` | `matplotlib.axes.Axes` (optional, 3D) | - | 3D axes to plot on. If `None`, creates new 3D figure. |

### Outputs

| Return Value | Type | Description |
|--------------|------|-------------|
| `fig` | `matplotlib.figure.Figure` | Figure handle |
| `ax` | `matplotlib.axes.Axes` (3D) | 3D Axes handle |

### Example
```python
wv = np.random.rand(100, 2)  # Shape: (100, 2)
fr = np.random.rand(100)      # Shape: (100,)
fig, ax = plot_dispersion_surface(wv, fr, N_k_x=10, N_k_y=10)
```

---

## plot_design

Plot design pattern showing material properties (modulus, density, Poisson's ratio).

### Inputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `design` | `np.ndarray` (float) | `(N_pix, N_pix, 3)` | Design array where:<br>- `[:, :, 0]` = elastic modulus<br>- `[:, :, 1]` = density<br>- `[:, :, 2]` = Poisson's ratio |

### Outputs

| Return Value | Type | Description |
|--------------|------|-------------|
| `fig` | `matplotlib.figure.Figure` | Figure handle |
| `subax_handle` | `list` of `matplotlib.axes.Axes` | List of 3 subplot axes handles (one per property) |

### Example
```python
N_pix = 32
design = np.random.rand(N_pix, N_pix, 3)  # Shape: (32, 32, 3)
fig, axes = plot_design(design)
```

---

## plot_mode

Plot mode shape (eigenvector visualization) using quiver plot.

### Inputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `design` | `np.ndarray` (float) | `(N_pix, N_pix, 3)` or any | Design pattern (used for context, not directly plotted) |
| `eigenvector` | `np.ndarray` (float or complex) | `(N_dof, N_modes)` or `(N_dof,)` | Full space eigenvector(s). If 2D, each column is a mode. |
| `const` | `dict` | - | Constants structure containing:<br>- `'N_ele'`: number of elements per side<br>- `'N_pix'`: number of pixels per side (int or list/tuple)<br>- `'a'`: lattice constant |
| `mode_idx` | `int` (optional) | scalar | Mode index to plot (default: 0). Used if `eigenvector` is 2D. |
| `ax` | `matplotlib.axes.Axes` (optional) | - | Axes to plot on. If `None`, creates new figure. |
| `save_values` | `bool` (optional) | scalar | If `True`, saves plotting values to `.mat` file for comparison (default: `False`) |

### Outputs

| Return Value | Type | Description |
|--------------|------|-------------|
| `fig` | `matplotlib.figure.Figure` | Figure handle |
| `ax` | `matplotlib.axes.Axes` | Axes handle |

### Notes
- `eigenvector` should be in **full space** (already transformed from reduced space)
- If `eigenvector` is 2D `(N_dof, N_modes)`, `mode_idx` selects which column to plot
- If `eigenvector` is 1D `(N_dof,)`, it's treated as a single mode
- The eigenvector is split into x and y components: `u = eigenvector[0::2]`, `v = eigenvector[1::2]`
- Reshaped to `(N_nodes, N_nodes)` where `N_nodes = N_ele * N_pix + 1`
- Uses quiver plot to show displacement vectors

### Example
```python
const = {'N_ele': 4, 'N_pix': 8, 'a': 1.0}
N_dof = 2 * (const['N_ele'] * const['N_pix'] + 1)**2
eigenvector = np.random.rand(N_dof, 3)  # Shape: (N_dof, 3) - 3 modes
design = np.random.rand(8, 8, 3)        # Shape: (8, 8, 3)
fig, ax = plot_mode(design, eigenvector, const, mode_idx=0)
```

---

## visualize_designs

Visualize multiple designs in a grid layout.

### Inputs

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `designs` | `list` of `np.ndarray` | `[(N_pix, N_pix, 3), ...]` | List of design arrays. Each design is `(N_pix, N_pix, 3)`. |
| `titles` | `list` of `str` (optional) | `[str, ...]` | List of titles for each design. If `None`, uses "Design 1", "Design 2", etc. |

### Outputs

| Return Value | Type | Description |
|--------------|------|-------------|
| `fig` | `matplotlib.figure.Figure` | Figure handle |
| `axes_handle` | `list` of `matplotlib.axes.Axes` | List of axes handles (one per design) |

### Notes
- Creates a grid with up to 3 columns
- Only plots the first property (modulus) from each design: `design[:, :, 0]`
- Unused subplots are hidden

### Example
```python
designs = [
    np.random.rand(32, 32, 3),  # Design 1
    np.random.rand(32, 32, 3),  # Design 2
    np.random.rand(32, 32, 3),  # Design 3
]
titles = ['Homogeneous', 'Random', 'Tetragonal']
fig, axes = visualize_designs(designs, titles=titles)
```

---

## Common Data Types

### Eigenvector Data Types
- **At k=0 (real matrices)**: Eigenvectors are stored as `np.float64` (real arrays)
- **At k≠0 (complex matrices)**: Eigenvectors are stored as `np.complex128` (complex arrays)
- This matches MATLAB's behavior where real eigenvectors are stored as `float64`, not `complex128`

### Frequency Arrays
- Always `np.float32` or `np.float64` (real)
- Shape: `(N_wavevectors, N_eig)` for multiple modes
- Shape: `(N_wavevectors,)` for single mode

### Wavevector Arrays
- Always `np.float32` or `np.float64` (real)
- Shape: `(N_wavevectors, 2)` where each row is `[kx, ky]`

### Design Arrays
- Always `np.float32` or `np.float64` (real)
- Shape: `(N_pix, N_pix, 3)` where:
  - `[:, :, 0]` = elastic modulus
  - `[:, :, 1]` = density
  - `[:, :, 2]` = Poisson's ratio

---

## Integration with dispersion() Function

The plotting functions are typically used with output from `dispersion()`:

```python
from dispersion import dispersion
from plotting import plot_dispersion, plot_mode, plot_design

# Run dispersion calculation
const = {...}  # Constants dict
wavevectors = np.array([[0.0, 0.0], [np.pi, 0.0], ...])  # Shape: (N_wv, 2)
wv, fr, ev, mesh = dispersion(const, wavevectors)

# fr shape: (N_wv, N_eig)
# ev shape: (N_dof_reduced, N_wv, N_eig) if isSaveEigenvectors=True, else None

# Plot dispersion curves
wn = np.arange(len(wavevectors))  # Wavevector parameter
fig, ax, line = plot_dispersion(wn, fr, N_contour_segments=5)

# Plot mode shape (requires transformation to full space)
from get_transformation_matrix import get_transformation_matrix
T = get_transformation_matrix(wavevectors[0, :], const)
eigenvector_full = T @ ev[:, 0, 0]  # Transform first mode of first wavevector
design = get_design('homogeneous', const['N_pix'])
fig, ax = plot_mode(design, eigenvector_full, const, mode_idx=0)
```

---

## Notes

1. **Matplotlib Backend**: All functions use `matplotlib.pyplot` and return figure/axes handles for further customization.

2. **Data Type Matching**: The library matches MATLAB's data type behavior:
   - Real eigenvectors (at k=0) are stored as `float64`, not `complex128`
   - Complex eigenvectors (at k≠0) are stored as `complex128`

3. **Coordinate Systems**: 
   - `plot_mode` uses flipped Y-coordinates to match MATLAB: `Y = np.flip(original_nodal_locations)`
   - `plot_dispersion_contour` uses Fortran (column-major) order for reshaping to match MATLAB

4. **Saving Plots**: To save plots, use:
   ```python
   fig.savefig('output.png', dpi=300)
   plt.close(fig)
   ```

