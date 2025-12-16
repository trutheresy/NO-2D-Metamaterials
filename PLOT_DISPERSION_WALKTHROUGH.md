# Walkthrough of `plot_dispersion.m`

## Overview
This script visualizes dispersion relations for 2D metamaterials. It loads pre-computed dispersion data and creates plots showing material properties, dispersion curves, and optionally reconstructs frequencies from eigenvectors using K, M, and T matrices.

## Code Structure

### 1. **Initialization (Lines 1-44)**

```matlab
% Load data file (either from CLI override or default path)
data = load(data_fn);

% Extract material properties
E_all = data.CONSTITUTIVE_DATA('modulus');
rho_all = data.CONSTITUTIVE_DATA('density');
nu_all = data.CONSTITUTIVE_DATA('poisson');
```

**Purpose**: Loads a `.mat` file containing pre-computed dispersion data for multiple structures.

**Data Structure**:
- `WAVEVECTOR_DATA`: Wavevectors used in the dispersion calculation
- `EIGENVALUE_DATA`: Computed frequencies (eigenvalues)
- `EIGENVECTOR_DATA`: Computed eigenvectors (if saved)
- `K_DATA`: Stiffness matrices (one per structure)
- `M_DATA`: Mass matrices (one per structure)
- `T_DATA`: Transformation matrices (one per wavevector, shared across structures)
- `CONSTITUTIVE_DATA`: Material property fields (E, rho, nu)

### 2. **Material Property Visualization (Lines 47-83)**

```matlab
for struct_idx = struct_idxs
    E = E_all(:,:,struct_idx);
    rho = rho_all(:,:,struct_idx);
    nu = nu_all(:,:,struct_idx);
    
    % Create 3-panel figure showing E, rho, nu fields
    imagesc(ax,E)
```

**Purpose**: Visualizes the material property distributions (Young's modulus, density, Poisson's ratio) for each structure as grayscale images.

### 3. **Load Dispersion Data (Lines 85-93)**

```matlab
wavevectors = data.WAVEVECTOR_DATA(:,:,struct_idx);
frequencies = data.EIGENVALUE_DATA(:,:,struct_idx);
```

**Purpose**: Extracts the wavevectors and corresponding frequencies for the current structure.

**Dimensions**:
- `wavevectors`: `[N_wavevectors × 2]` (kx, ky pairs)
- `frequencies`: `[N_wavevectors × N_eig]` (frequency for each wavevector and eigenvalue band)

### 4. **Frequency Reconstruction Using K, M, T Matrices (Lines 95-121)**

**This is the key section that uses K, M, and T matrices!**

```matlab
can_reconstruct = isfield(data, 'K_DATA') && isfield(data, 'M_DATA') && ...
                  isfield(data, 'T_DATA') && ...
                  ~isempty(data.K_DATA) && ~isempty(data.M_DATA) && ...
                  ~isempty(data.T_DATA) && ...
                  length(data.K_DATA) >= struct_idx && ...
                  length(data.M_DATA) >= struct_idx;
```

**Purpose**: Checks if K, M, and T matrices are available for reconstruction.

#### If reconstruction is possible:

```matlab
K = data.K_DATA{struct_idx};  % Stiffness matrix for this structure
M = data.M_DATA{struct_idx};  % Mass matrix for this structure

for wv_idx = 1:size(data.const.wavevectors,1)
    T = data.T_DATA{wv_idx};  % Transformation matrix for this wavevector
    
    % Transform to reduced space (periodic boundary conditions)
    Kr = T'*K*T;  % Reduced stiffness matrix
    Mr = T'*M*T;  % Reduced mass matrix
    
    for band_idx = 1:data.const.N_eig
        eigvec = data.EIGENVECTOR_DATA(:,wv_idx,band_idx,struct_idx);
        
        % Reconstruct eigenvalue from eigenvector
        eigval = norm(Kr*eigvec)/norm(Mr*eigvec);
        
        % Convert to frequency
        frequencies_recon(wv_idx,band_idx) = sqrt(eigval)/(2*pi);
    end
end
```

**How it works**:

1. **K and M matrices**: These are the global stiffness and mass matrices for the full unit cell (before applying periodic boundary conditions). They are structure-specific (one per structure).

2. **T matrix**: This is the transformation matrix that applies periodic boundary conditions. It transforms from the full DOF space to the reduced DOF space. It is wavevector-specific (one per wavevector) but shared across structures.

3. **Reduced matrices**: 
   - `Kr = T'*K*T`: Transforms the stiffness matrix to the reduced space
   - `Mr = T'*M*T`: Transforms the mass matrix to the reduced space
   
   These are the matrices used in the eigenvalue problem: `Kr * eigvec = eigval * Mr * eigvec`

4. **Eigenvalue reconstruction**: 
   - Given an eigenvector `eigvec` from the stored data
   - The eigenvalue can be computed using the Rayleigh quotient: `eigval = norm(Kr*eigvec)/norm(Mr*eigvec)`
   - This is equivalent to: `eigval = (eigvec'*Kr*eigvec)/(eigvec'*Mr*eigvec)`
   - The frequency is then: `f = sqrt(eigval)/(2*pi)`

**Why reconstruct?**
- **Validation**: Verifies that the stored eigenvectors are correct by checking if reconstructed frequencies match the original eigenvalues
- **Debugging**: Helps identify numerical issues or data corruption
- **Visualization**: Allows comparison between original and reconstructed frequencies

### 5. **Interpolation (Lines 123-135)**

```matlab
for eig_idx = 1:data.const.N_eig
    interp_true{eig_idx} = scatteredInterpolant(wavevectors, frequencies(:,eig_idx), ...);
    if can_reconstruct
        interp_recon{eig_idx} = scatteredInterpolant(wavevectors, frequencies_recon(:,eig_idx), ...);
    end
end
```

**Purpose**: Creates interpolants to evaluate frequencies at arbitrary wavevector points (needed for plotting along the IBZ contour).

### 6. **IBZ Contour Generation (Lines 137-158)**

```matlab
[wavevectors_contour, contour_info] = get_IBZ_contour_wavevectors(N_k, data.const.a, 'p4mm');
```

**Purpose**: Generates wavevectors along the boundary of the Irreducible Brillouin Zone (IBZ) for p4mm symmetry. This creates a closed path through high-symmetry points (Γ → X → M → Γ).

### 7. **Contour Evaluation (Lines 160-170)**

```matlab
for eig_idx = 1:size(frequencies,2)
    frequencies_contour(:,eig_idx) = interp_true{eig_idx}(wavevectors_contour(:,1), wavevectors_contour(:,2));
    if can_reconstruct
        frequencies_recon_contour(:,eig_idx) = interp_recon{eig_idx}(wavevectors_contour(:,1), wavevectors_contour(:,2));
    end
end
```

**Purpose**: Evaluates frequencies along the IBZ contour using the interpolants.

### 8. **Plotting (Lines 172-223)**

**Two types of plots**:

1. **Original dispersion** (Lines 172-191): Plots the frequencies as originally computed
2. **Reconstructed dispersion** (Lines 193-223): Overlays reconstructed frequencies (if available) to compare with original

## Key Points About K, M, T Usage

### When are K, M, T used?

✅ **YES - Used in reconstruction** (Lines 100-116):
- K and M are loaded from `K_DATA{struct_idx}` and `M_DATA{struct_idx}`
- T is loaded from `T_DATA{wv_idx}` for each wavevector
- They are used to reconstruct eigenvalues from stored eigenvectors

❌ **NO - Not used for original computation**:
- The original frequencies in `EIGENVALUE_DATA` were computed earlier (during dataset generation)
- The plotting script only uses K, M, T for **validation/reconstruction**, not for the primary computation

### Matrix Dimensions

- **K, M**: `[N_dof_full × N_dof_full]` where `N_dof_full = 2 * (N_nodes_x * N_nodes_y)`
- **T**: `[N_dof_full × N_dof_reduced]` where `N_dof_reduced = 2 * (N_node-1)^2`
- **Kr, Mr**: `[N_dof_reduced × N_dof_reduced]` (after transformation)

### Why T is shared across structures

The transformation matrix T depends only on:
- The wavevector (periodic boundary condition phase factors)
- The mesh geometry (node numbering)

It does NOT depend on:
- Material properties (E, rho, nu)
- The design/structure

Therefore, T is the same for all structures when using the same wavevector grid, so it's stored once per wavevector rather than once per structure.

## Summary

1. **Loads pre-computed data** including K, M, T matrices
2. **Visualizes material properties** (E, rho, nu)
3. **Optionally reconstructs frequencies** from eigenvectors using K, M, T for validation
4. **Interpolates** frequencies to evaluate along IBZ contour
5. **Plots dispersion relations** along the IBZ boundary, comparing original and reconstructed frequencies

The K, M, and T matrices are used **only for validation/reconstruction**, not for the primary dispersion calculation, which was done earlier during dataset generation.



