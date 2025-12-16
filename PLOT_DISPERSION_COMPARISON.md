# Functionality Comparison: MATLAB vs Python Plotting Scripts

## Overview

This document compares `2D-dispersion-han/plot_dispersion.m` (MATLAB) with `2d-dispersion-py/plot_dispersion_pt.py` (Python) in terms of functionality.

## Feature Comparison Matrix

| Feature | MATLAB `plot_dispersion.m` | Python `plot_dispersion_pt.py` | Status |
|---------|---------------------------|-------------------------------|--------|
| **Data Loading** |
| Load from .mat file | ✅ Yes | ❌ No (PyTorch format only) | Different |
| Load PyTorch format | ❌ No | ✅ Yes | Different |
| Load NumPy format | ❌ No | ✅ Yes (via original_data_dir) | Python only |
| Handle reduced datasets | ❌ No | ✅ Yes | Python only |
| **Material Property Visualization** |
| Plot E, ρ, ν fields | ✅ Yes (3-panel) | ✅ Yes (via plot_design) | ✅ Equivalent |
| Export PNG | ✅ Yes | ✅ Yes | ✅ Equivalent |
| **Dispersion Data** |
| Load wavevectors | ✅ Yes | ✅ Yes | ✅ Equivalent |
| Load frequencies | ✅ Yes | ✅ Yes | ✅ Equivalent |
| Load eigenvectors | ✅ Optional | ✅ Optional | ✅ Equivalent |
| **K, M, T Matrix Usage** |
| Load K_DATA | ✅ Yes | ❌ No | ⚠️ Missing |
| Load M_DATA | ✅ Yes | ❌ No | ⚠️ Missing |
| Load T_DATA | ✅ Yes | ❌ No | ⚠️ Missing |
| Reconstruct frequencies from eigenvectors | ✅ Yes (Lines 100-116) | ❌ No (commented TODO) | ⚠️ Missing |
| Validate eigenvectors | ✅ Yes | ❌ No | ⚠️ Missing |
| **Interpolation** |
| Scattered interpolation | ✅ Yes (scatteredInterpolant) | ✅ Yes (griddata) | ✅ Equivalent |
| Grid interpolation | ❌ No | ✅ Yes (RegularGridInterpolator) | Python only |
| **IBZ Contour** |
| Generate contour wavevectors | ✅ Yes (p4mm) | ✅ Yes (p4mm) | ✅ Equivalent |
| Extract grid points on contour | ❌ No | ✅ Yes | Python only |
| Plot contour path | ✅ Yes | ✅ Yes | ✅ Equivalent |
| **Dispersion Plotting** |
| Plot original frequencies | ✅ Yes | ✅ Yes | ✅ Equivalent |
| Plot reconstructed frequencies | ✅ Yes (if K,M,T available) | ❌ No | ⚠️ Missing |
| Overlay comparison plot | ✅ Yes | ❌ No | ⚠️ Missing |
| Mark points on curves | ❌ No | ✅ Yes (optional) | Python only |
| **Output** |
| Export PNG | ✅ Yes | ✅ Yes | ✅ Equivalent |
| Custom output directory | ✅ Yes | ✅ Yes | ✅ Equivalent |
| Batch processing | ✅ Yes (loop) | ✅ Yes (loop) | ✅ Equivalent |

## Detailed Feature Analysis

### 1. Data Loading

#### MATLAB (`plot_dispersion.m`)
```matlab
data = load(data_fn);  % Loads .mat file
E_all = data.CONSTITUTIVE_DATA('modulus');
wavevectors = data.WAVEVECTOR_DATA(:,:,struct_idx);
frequencies = data.EIGENVALUE_DATA(:,:,struct_idx);
```

**Features**:
- Direct .mat file loading
- Accesses data via structure fields
- Handles single structure at a time

#### Python (`plot_dispersion_pt.py`)
```python
data = load_pt_dataset(data_dir, original_data_dir)
# Supports:
# - PyTorch format (.pt files)
# - Reduced datasets (with reduced_indices.pt)
# - Merging with original NumPy/MATLAB data
```

**Features**:
- PyTorch format support
- Reduced dataset handling
- Automatic merging with original eigenvalue data
- More flexible data source options

**Verdict**: Python version is more flexible but doesn't support direct .mat loading.

---

### 2. Material Property Visualization

#### MATLAB
```matlab
% 3-panel tiled layout
imagesc(ax,E)    % Young's modulus
imagesc(ax,rho)  % Density
imagesc(ax,nu)   % Poisson's ratio
```

#### Python
```python
# Reconstructs material properties from design
geometry = np.stack([elastic_modulus, density, poisson_ratio], axis=-1)
fig_design, _ = plot_design(geometry)
```

**Verdict**: ✅ **Functionally equivalent** - both visualize E, ρ, ν fields

---

### 3. K, M, T Matrix Usage

#### MATLAB - **IMPLEMENTED** ✅
```matlab
% Lines 95-121: Frequency reconstruction
can_reconstruct = isfield(data, 'K_DATA') && isfield(data, 'M_DATA') && ...
                  isfield(data, 'T_DATA') && ...

if can_reconstruct
    K = data.K_DATA{struct_idx};
    M = data.M_DATA{struct_idx};
    for wv_idx = 1:size(data.const.wavevectors,1)
        T = data.T_DATA{wv_idx};
        Kr = T'*K*T;
        Mr = T'*M*T;
        for band_idx = 1:data.const.N_eig
            eigvec = data.EIGENVECTOR_DATA(:,wv_idx,band_idx,struct_idx);
            eigval = norm(Kr*eigvec)/norm(Mr*eigvec);
            frequencies_recon(wv_idx,band_idx) = sqrt(eigval)/(2*pi);
        end
    end
    % Compare original vs reconstructed
    disp(['max(abs(frequencies_recon-frequencies))/max(abs(frequencies)) = ' ...])
end
```

**Features**:
- ✅ Loads K, M, T from data
- ✅ Reconstructs frequencies from eigenvectors
- ✅ Validates eigenvectors against original eigenvalues
- ✅ Creates comparison plots (original vs reconstructed)

#### Python - **NOT IMPLEMENTED** ❌
```python
# Lines 615-626: Commented TODO
# Process outline 
## Load PyTorch dataset
### Load K matrices (N, 2000+, 2000+) 
### Load M matrices (N, 2000+, 2000+)
### Load T matrices (N, 2000+, 2000+)

## Reconstruct frequencies from eigenvectors (plot dispersion.m)
###
```

**Current Status**:
- ❌ Does not load K, M, T matrices
- ❌ Does not reconstruct frequencies
- ❌ Does not validate eigenvectors
- ❌ No comparison plots

**Verdict**: ⚠️ **Major functionality gap** - Python version lacks the validation/reconstruction feature

---

### 4. Interpolation Methods

#### MATLAB
```matlab
% Uses scatteredInterpolant (always scattered)
interp_true{eig_idx} = scatteredInterpolant(wavevectors, frequencies(:,eig_idx), ...);
frequencies_contour = interp_true{eig_idx}(wavevectors_contour(:,1), wavevectors_contour(:,2));
```

**Features**:
- Always uses scattered interpolation
- Works with any wavevector distribution

#### Python
```python
# Two modes:
# 1. Grid interpolation (if wavevectors form regular grid)
grid_interp, wavevectors_grid = create_grid_interpolators(wavevectors, frequencies, None)

# 2. Scattered interpolation (fallback)
frequencies_contour = griddata(wavevectors, frequencies[:, eig_idx], wavevectors_contour, ...)
```

**Features**:
- ✅ Grid interpolation (more efficient for regular grids)
- ✅ Scattered interpolation (fallback)
- ✅ Automatic detection of grid vs scattered
- ✅ Extract exact grid points on contour (no interpolation)

**Verdict**: ✅ **Python is more advanced** - supports both grid and scattered interpolation

---

### 5. IBZ Contour Handling

#### MATLAB
```matlab
% Generate contour with interpolation points
[wavevectors_contour, contour_info] = get_IBZ_contour_wavevectors(N_k, data.const.a, 'p4mm');
% Always interpolates to contour points
frequencies_contour = interp_true{eig_idx}(wavevectors_contour(:,1), wavevectors_contour(:,2));
```

**Features**:
- Always interpolates to contour
- Fixed number of interpolation points per segment

#### Python
```python
# Two modes:
if use_interpolation:
    # Mode 1: Interpolate to smooth contour
    wavevectors_contour, contour_info = get_IBZ_contour_wavevectors(N_k_interp, a, 'p4mm')
else:
    # Mode 2: Extract exact grid points on contour (no interpolation)
    wavevectors_contour, frequencies_contour_grid, contour_param_grid = \
        extract_grid_points_on_contour(wavevectors, frequencies, contour_info, a)
```

**Features**:
- ✅ Interpolation mode (smooth curves)
- ✅ Grid-points-only mode (exact values, no interpolation)
- ✅ Automatic extraction of grid points on contour path

**Verdict**: ✅ **Python is more flexible** - supports both interpolated and exact grid point modes

---

### 6. Plotting Features

#### MATLAB
```matlab
% Original frequencies
plot(ax, contour_info.wavevector_parameter, frequencies_contour)

% Reconstructed frequencies (overlay)
plot(ax, contour_info.wavevector_parameter, frequencies_contour, ...)  % Original
plot(ax, contour_info.wavevector_parameter, frequencies_recon_contour, ...)  % Reconstructed
legend(ax, p)
```

**Features**:
- Basic line plots
- Overlay comparison (if reconstruction available)
- Vertical lines at segment boundaries

#### Python
```python
# Enhanced plotting
plot_dispersion_on_contour(ax_disp, contour_info, frequencies_contour, ...,
                           title=f'Dispersion Relation ({mode_str})', 
                           mark_points=mark_points)
```

**Features**:
- ✅ Optional point markers
- ✅ Better title formatting
- ✅ Grid overlay
- ❌ No reconstruction comparison plots

**Verdict**: ⚠️ **Mixed** - Python has better visualization options but lacks reconstruction plots

---

## Summary of Differences

### ✅ Python Advantages

1. **More flexible data loading**: Supports PyTorch, NumPy, and can merge with MATLAB data
2. **Reduced dataset support**: Can handle downsampled datasets
3. **Better interpolation**: Grid interpolation + scattered fallback
4. **Exact grid point extraction**: Can plot without interpolation
5. **Enhanced visualization**: Point markers, better formatting

### ⚠️ Python Missing Features

1. **K, M, T matrix loading**: Not implemented
2. **Frequency reconstruction**: Not implemented (commented TODO)
3. **Eigenvector validation**: Not implemented
4. **Reconstruction comparison plots**: Not implemented
5. **Direct .mat file support**: Requires conversion to PyTorch format first

### ✅ MATLAB Advantages

1. **Direct .mat file support**: Native MATLAB format
2. **K, M, T validation**: Full reconstruction and validation
3. **Comparison plots**: Original vs reconstructed overlay

### ⚠️ MATLAB Limitations

1. **Fixed interpolation**: Always uses scattered interpolation
2. **No grid point extraction**: Always interpolates to contour
3. **Less flexible data loading**: Only .mat files

---

## Recommendations

### For Python Version (`plot_dispersion_pt.py`)

**High Priority**:
1. ✅ **Implement K, M, T matrix loading** from PyTorch dataset
2. ✅ **Implement frequency reconstruction** from eigenvectors (match MATLAB lines 100-116)
3. ✅ **Add reconstruction comparison plots** (overlay original vs reconstructed)

**Medium Priority**:
4. Add direct .mat file loading support
5. Add validation error reporting (like MATLAB's max error calculation)

**Low Priority**:
6. Add support for other symmetry types (currently only p4mm)

### Implementation Notes for Python

To match MATLAB functionality, add:

```python
# In load_pt_dataset():
if (data_dir / 'K_data.pt').exists():
    data['K_data'] = torch.load(data_dir / 'K_data.pt', map_location='cpu')
if (data_dir / 'M_data.pt').exists():
    data['M_data'] = torch.load(data_dir / 'M_data.pt', map_location='cpu')
if (data_dir / 'T_data.pt').exists():
    data['T_data'] = torch.load(data_dir / 'T_data.pt', map_location='cpu')

# In main(), after loading data:
if 'K_data' in data and 'M_data' in data and 'T_data' in data:
    # Reconstruct frequencies (match MATLAB lines 100-116)
    frequencies_recon = reconstruct_frequencies(
        data['K_data'], data['M_data'], data['T_data'],
        data['eigenvector_data'], struct_idx
    )
    # Create comparison plot
    plot_comparison(ax, frequencies_contour, frequencies_recon_contour, ...)
```

---

## Conclusion

The Python version (`plot_dispersion_pt.py`) is **more advanced** in terms of:
- Data loading flexibility
- Interpolation methods
- Visualization options

However, it is **missing critical functionality**:
- K, M, T matrix usage
- Frequency reconstruction from eigenvectors
- Validation and comparison plots

The MATLAB version is **more complete** for validation purposes but less flexible for different data formats.

**Overall**: Python version needs the K, M, T reconstruction feature to be functionally equivalent to MATLAB.



