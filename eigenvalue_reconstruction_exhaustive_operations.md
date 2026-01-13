# Exhaustive List of Operations in Eigenvalue Reconstruction Chain

## Starting Quantities Required

### 1. Input Data from PyTorch Files
- `geometries_full.pt`: Single-channel design array `(n_designs, design_res, design_res)` - dtype: float16
- `wavevectors_full.pt`: Wavevector coordinates `(n_designs, n_wavevectors, 2)` - dtype: float16
- `displacements_dataset`: TensorDataset containing:
  - `eigenvector_x_real`: Real part of x-displacements
  - `eigenvector_x_imag`: Imaginary part of x-displacements
  - `eigenvector_y_real`: Real part of y-displacements
  - `eigenvector_y_imag`: Imaginary part of y-displacements
- `reduced_indices.pt` or full indices: List/tensor of `(design_idx, wavevector_idx, band_idx)` tuples
- `bands_fft_full.pt`: Band information
- `design_params_full.pt`: Design parameters

### 2. Constant Parameters
- `N_pix`: Number of pixels per element (scalar or list)
- `N_ele`: Number of elements per pixel (scalar, typically 1)
- `a`: Lattice constant (scalar)
- `t`: Thickness (scalar)
- `E_min`, `E_max`: Young's modulus bounds
- `rho_min`, `rho_max`: Density bounds
- `nu_min`, `nu_max` or `poisson_min`, `poisson_max`: Poisson's ratio bounds
- `design_scale`: 'linear' or 'log'
- `isUseImprovement`: Boolean flag
- `isUseSecondImprovement`: Boolean flag

---

## Complete Operation Chain

### PHASE 1: Data Loading and Conversion

#### 1.1 Load PyTorch Files
**Operation**: `torch.load(file_path, map_location='cpu', weights_only=False)`
- Load geometries: `geometries = torch.load("geometries_full.pt")`
- Load wavevectors: `wavevectors = torch.load("wavevectors_full.pt")`
- Load displacements: `displacements_dataset = torch.load("displacements.pt")`
- Load indices: `reduced_indices = torch.load("reduced_indices.pt")` or create full indices

**Variable Renaming/Extraction**:
- `geometries_np = geometries.numpy()` - Convert to numpy
- `wavevectors_np = wavevectors.numpy()` - Convert to numpy
- Extract tensor components:
  - `eigenvector_x_real = displacements_dataset.tensors[0].numpy()`
  - `eigenvector_x_imag = displacements_dataset.tensors[1].numpy()`
  - `eigenvector_y_real = displacements_dataset.tensors[2].numpy()`
  - `eigenvector_y_imag = displacements_dataset.tensors[3].numpy()`

**Dimension Extraction**:
- `n_designs = geometries_np.shape[0]`
- `design_res = geometries_np.shape[1]`
- `n_wavevectors = wavevectors_np.shape[1]`
- `n_bands = bands_fft.shape[0]`

#### 1.2 Convert Indices Format
**Operation**: Convert indices to numpy array format
```python
if isinstance(reduced_indices, torch.Tensor):
    indices_np = reduced_indices.numpy()
else:
    # Convert list of tuples to numpy array
    n_indices = len(reduced_indices)
    indices_np = np.zeros((n_indices, 3), dtype=np.int64)
    for i, (d, w, b) in enumerate(reduced_indices):
        indices_np[i, 0] = int(d) if isinstance(d, torch.Tensor) else int(d)
        indices_np[i, 1] = int(w) if isinstance(w, torch.Tensor) else int(w)
        indices_np[i, 2] = int(b) if isinstance(b, torch.Tensor) else int(b)
```

**Variable**: `indices_np` - shape `(n_samples, 3)`, dtype: int64

---

### PHASE 2: Eigenvector Reconstruction

#### 2.1 Initialize Full Eigenvector Arrays
**Operation**: Create zero-filled arrays
```python
EIGENVECTOR_DATA_x_full = np.zeros((n_designs, n_wavevectors, n_bands, design_res, design_res), 
                                    dtype=np.complex128)
EIGENVECTOR_DATA_y_full = np.zeros((n_designs, n_wavevectors, n_bands, design_res, design_res), 
                                    dtype=np.complex128)
```

**Variable**: 
- `EIGENVECTOR_DATA_x_full` - shape `(n_designs, n_wavevectors, n_bands, design_res, design_res)`, dtype: complex128
- `EIGENVECTOR_DATA_y_full` - shape `(n_designs, n_wavevectors, n_bands, design_res, design_res)`, dtype: complex128

#### 2.2 Combine Real and Imaginary Parts
**Operation**: Create complex arrays from real and imaginary parts
```python
eigenvector_x_complex = (eigenvector_x_real + 1j * eigenvector_x_imag).astype(np.complex128)
eigenvector_y_complex = (eigenvector_y_real + 1j * eigenvector_y_imag).astype(np.complex128)
```

**Operations**:
- Addition: `eigenvector_x_real + 1j * eigenvector_x_imag`
- Multiplication: `1j * eigenvector_x_imag` (complex unit multiplication)
- Type conversion: `.astype(np.complex128)`

**Variable**:
- `eigenvector_x_complex` - dtype: complex128
- `eigenvector_y_complex` - dtype: complex128

#### 2.3 Place Eigenvectors at Correct Indices
**Operation**: Index assignment loop
```python
for sample_idx in range(n_samples):
    d_idx, w_idx, b_idx = indices_np[sample_idx]
    EIGENVECTOR_DATA_x_full[d_idx, w_idx, b_idx, :, :] = eigenvector_x_complex[sample_idx]
    EIGENVECTOR_DATA_y_full[d_idx, w_idx, b_idx, :, :] = eigenvector_y_complex[sample_idx]
```

**Operations**:
- Array indexing: `indices_np[sample_idx]` - extracts tuple `(d_idx, w_idx, b_idx)`
- Tuple unpacking: `d_idx, w_idx, b_idx = ...`
- Multi-dimensional indexing: `EIGENVECTOR_DATA_x_full[d_idx, w_idx, b_idx, :, :]`
- Slice assignment: `[:, :] = eigenvector_x_complex[sample_idx]`

#### 2.4 Reshape Eigenvectors
**Operation**: Flatten spatial dimensions
```python
EIGENVECTOR_DATA_x_flat = EIGENVECTOR_DATA_x_full.reshape(n_designs, n_wavevectors, n_bands, -1)
EIGENVECTOR_DATA_y_flat = EIGENVECTOR_DATA_y_full.reshape(n_designs, n_wavevectors, n_bands, -1)
```

**Operations**:
- Reshape: `.reshape(...)` - changes array shape without changing data
- Automatic dimension: `-1` - infers dimension from other sizes

**Variable**:
- `EIGENVECTOR_DATA_x_flat` - shape `(n_designs, n_wavevectors, n_bands, design_res*design_res)`
- `EIGENVECTOR_DATA_y_flat` - shape `(n_designs, n_wavevectors, n_bands, design_res*design_res)`

#### 2.5 Interleave x and y Components
**Operation**: Combine x and y DOF into single array
```python
n_dof = 2 * design_res * design_res
EIGENVECTOR_DATA_combined = np.zeros((n_designs, n_wavevectors, n_bands, n_dof), dtype=np.complex128)
EIGENVECTOR_DATA_combined[:, :, :, 0::2] = EIGENVECTOR_DATA_x_flat
EIGENVECTOR_DATA_combined[:, :, :, 1::2] = EIGENVECTOR_DATA_y_flat
```

**Operations**:
- Multiplication: `2 * design_res * design_res` - compute total DOF
- Array creation: `np.zeros(...)` - create zero-filled array
- Strided indexing: `[:, :, :, 0::2]` - every other element starting at 0 (even indices)
- Strided indexing: `[:, :, :, 1::2]` - every other element starting at 1 (odd indices)
- Strided assignment: assign to strided slices

**Variable**: `EIGENVECTOR_DATA_combined` - shape `(n_designs, n_wavevectors, n_bands, n_dof)`, dtype: complex128

#### 2.6 Transpose to MATLAB Format
**Operation**: Reorder dimensions
```python
EIGENVECTOR_DATA = EIGENVECTOR_DATA_combined.transpose(0, 2, 1, 3)
```

**Operations**:
- Transpose: `.transpose(0, 2, 1, 3)` - reorders dimensions from `(n_designs, n_wavevectors, n_bands, n_dof)` to `(n_designs, n_bands, n_wavevectors, n_dof)`

**Variable**: `EIGENVECTOR_DATA` - shape `(n_designs, n_bands, n_wavevectors, n_dof)`, dtype: complex128

---

### PHASE 3: Design Conversion (Steel-Rubber Paradigm)

#### 3.1 Extract Design for Structure
**Operation**: Index single design
```python
design_param = geometries_np[struct_idx]  # (design_res, design_res)
```

**Operations**:
- Array indexing: `geometries_np[struct_idx]` - extracts single design
- Variable: `design_param` - shape `(design_res, design_res)`, dtype: float16 (from original)

#### 3.2 Type Conversion
**Operation**: Convert to float64
```python
design_param = design_param.astype(np.float64)
```

**Operations**:
- Type conversion: `.astype(np.float64)` - converts float16 to float64

**Variable**: `design_param` - dtype: float64

#### 3.3 Apply Steel-Rubber Paradigm
**Operation**: Convert single-channel to 3-channel design
```python
const_for_paradigm = {
    'E_min': E_min,
    'E_max': E_max,
    'rho_min': rho_min,
    'rho_max': rho_max,
    'poisson_min': nu_min,
    'poisson_max': nu_max
}
design_3ch = apply_steel_rubber_paradigm(design_param, const_for_paradigm)
```

**Inside `apply_steel_rubber_paradigm` function**:

**3.3.1 Hardcoded Material Values**
```python
E_polymer = 100e6
E_steel = 200e9
rho_polymer = 1200.0
rho_steel = 8e3
nu_polymer = 0.45
nu_steel = 0.3
```

**3.3.2 Compute Normalized Design Values**
```python
design_out_polymer_E = (E_polymer - E_min) / (E_max - E_min)
design_out_polymer_rho = (rho_polymer - rho_min) / (rho_max - rho_min)
design_out_polymer_nu = (nu_polymer - poisson_min) / (poisson_max - poisson_min)
design_out_steel_E = (E_steel - E_min) / (E_max - E_min)
design_out_steel_rho = (rho_steel - rho_min) / (rho_max - rho_min)
design_out_steel_nu = (nu_steel - poisson_min) / (poisson_max - poisson_min)
```

**Operations**:
- Subtraction: `E_polymer - E_min`
- Subtraction: `E_max - E_min`
- Division: `(E_polymer - E_min) / (E_max - E_min)`
- (Repeated for rho and nu)

**3.3.3 Create Design Values Array**
```python
design_vals = np.array([
    [design_out_polymer_E, design_out_steel_E],
    [design_out_polymer_rho, design_out_steel_rho],
    [design_out_polymer_nu, design_out_steel_nu]
])
```

**Operations**:
- Array creation: `np.array([...])` - creates 2D array

**3.3.4 Interpolate Design Values**
```python
N_pix = design_single.shape[0]
design_out = np.zeros((N_pix, N_pix, 3))
for prop_idx in range(3):
    design_flat = design_single.flatten()
    design_out[:, :, prop_idx] = np.interp(
        design_flat,
        [design_in_polymer, design_in_steel],
        design_vals[prop_idx, :]
    ).reshape(N_pix, N_pix)
```

**Operations**:
- Flatten: `.flatten()` - converts 2D to 1D
- Interpolation: `np.interp(design_flat, [0, 1], [val_polymer, val_steel])` - linear interpolation
- Reshape: `.reshape(N_pix, N_pix)` - converts back to 2D

**Variable**: `design_3ch` - shape `(N_pix, N_pix, 3)`, dtype: float64

---

### PHASE 4: K and M Matrix Computation

#### 4.1 Create Const Dictionary for Matrix Computation
**Operation**: Assemble parameters
```python
const_for_km = {
    'design': design_3ch,
    'N_pix': design_res,
    'N_ele': N_ele,
    'a': a_val,
    'E_min': E_min,
    'E_max': E_max,
    'rho_min': rho_min,
    'rho_max': rho_max,
    'poisson_min': nu_min,
    'poisson_max': nu_max,
    't': t_val,
    'design_scale': 'linear',
    'isUseImprovement': True,
    'isUseSecondImprovement': False
}
```

#### 4.2 Compute K and M Matrices
**Operation**: `K, M = get_system_matrices_VEC(const_for_km)`

**Inside `get_system_matrices_VEC` function**:

**4.2.1 Extract N_pix**
```python
N_pix = const['N_pix']
if isinstance(N_pix, (list, tuple)):
    N_ele_x = N_pix[0] * const['N_ele']
    N_ele_y = N_pix[1] * const['N_ele']
else:
    N_ele_x = N_pix * const['N_ele']
    N_ele_y = N_pix * const['N_ele']
```

**Operations**:
- Type check: `isinstance(N_pix, (list, tuple))`
- Conditional branching: `if/else`
- Multiplication: `N_pix * const['N_ele']`

**4.2.2 Expand Design (Repelem)**
```python
design_expanded = np.repeat(
    np.repeat(const['design'], const['N_ele'], axis=0), 
    const['N_ele'], axis=1
)
```

**Operations**:
- Repeat along axis 0: `np.repeat(design, N_ele, axis=0)` - repeats rows
- Repeat along axis 1: `np.repeat(..., N_ele, axis=1)` - repeats columns
- Nested function calls

**Variable**: `design_expanded` - shape `(N_pix*N_ele, N_pix*N_ele, 3)`

**4.2.3 Extract Material Properties**
```python
if const['design_scale'] == 'linear':
    design_ch0 = design_expanded[:, :, 0].astype(np.float64)
    design_ch1 = design_expanded[:, :, 1].astype(np.float64)
    design_ch2 = design_expanded[:, :, 2].astype(np.float64)
    E = (const['E_min'] + design_ch0 * (const['E_max'] - const['E_min'])).T.astype(np.float32)
    nu = (const['poisson_min'] + design_ch2 * (const['poisson_max'] - const['poisson_min'])).T.astype(np.float32)
    t = const['t']
    rho = (const['rho_min'] + design_ch1 * (const['rho_max'] - const['rho_min'])).T.astype(np.float32)
```

**Operations**:
- String comparison: `const['design_scale'] == 'linear'`
- Array slicing: `design_expanded[:, :, 0]` - extract channel
- Type conversion: `.astype(np.float64)` and `.astype(np.float32)`
- Subtraction: `const['E_max'] - const['E_min']`
- Multiplication: `design_ch0 * (const['E_max'] - const['E_min'])`
- Addition: `const['E_min'] + design_ch0 * ...`
- Transpose: `.T` - matrix transpose
- (Repeated for nu and rho)

**Variable**:
- `E` - shape `(N_ele_x, N_ele_y)`, dtype: float32
- `nu` - shape `(N_ele_x, N_ele_y)`, dtype: float32
- `rho` - shape `(N_ele_x, N_ele_y)`, dtype: float32
- `t` - scalar

**4.2.4 Create Node Numbering**
```python
nodenrs = np.arange(1, (1 + N_ele_x) * (1 + N_ele_y) + 1).reshape(1 + N_ele_y, 1 + N_ele_x, order='F')
```

**Operations**:
- Range creation: `np.arange(1, (1 + N_ele_x) * (1 + N_ele_y) + 1)` - creates 1D array [1, 2, ..., N]
- Multiplication: `(1 + N_ele_x) * (1 + N_ele_y)` - compute total nodes
- Addition: `1 + N_ele_x`
- Reshape: `.reshape(..., order='F')` - Fortran (column-major) order

**Variable**: `nodenrs` - shape `(1 + N_ele_y, 1 + N_ele_x)`, dtype: int

**4.2.5 Create Element DOF Vector**
```python
edofVec = (2 * nodenrs[0:-1, 0:-1] - 1).reshape(N_ele_x * N_ele_y, 1, order='F').flatten()
```

**Operations**:
- Array slicing: `nodenrs[0:-1, 0:-1]` - all but last row/column
- Multiplication: `2 * nodenrs[...]`
- Subtraction: `... - 1`
- Reshape: `.reshape(..., order='F')` - column-major
- Flatten: `.flatten()` - convert to 1D

**Variable**: `edofVec` - shape `(N_ele_x * N_ele_y,)`, dtype: int

**4.2.6 Create Element DOF Matrix**
```python
offset_array = np.concatenate([
    2*(N_ele_y+1) + np.array([0, 1, 2, 3]),
    np.array([2, 3, 0, 1])
])
edofMat = np.tile(edofVec.reshape(-1, 1), (1, 8)) + np.tile(offset_array, (N_ele_x * N_ele_y, 1))
```

**Operations**:
- Array creation: `np.array([0, 1, 2, 3])`
- Addition: `2*(N_ele_y+1) + np.array([...])`
- Concatenation: `np.concatenate([...])`
- Reshape: `.reshape(-1, 1)` - column vector
- Tile: `np.tile(..., (1, 8))` - repeat horizontally
- Tile: `np.tile(..., (N_ele_x * N_ele_y, 1))` - repeat vertically
- Addition: `np.tile(...) + np.tile(...)` - element-wise addition

**Variable**: `edofMat` - shape `(N_ele_x * N_ele_y, 8)`, dtype: int

**4.2.7 Create Row and Column Indices**
```python
row_idxs = np.reshape(np.kron(edofMat, np.ones((8, 1))).T, 64 * N_ele_x * N_ele_y, order='F')
col_idxs = np.reshape(np.kron(edofMat, np.ones((1, 8))).T, 64 * N_ele_x * N_ele_y, order='F')
```

**Operations**:
- Array creation: `np.ones((8, 1))` - column vector of ones
- Kronecker product: `np.kron(edofMat, np.ones((8, 1)))` - block replication
- Transpose: `.T`
- Reshape: `.reshape(..., order='F')` - column-major

**Variable**:
- `row_idxs` - shape `(64 * N_ele_x * N_ele_y,)`, dtype: int
- `col_idxs` - shape `(64 * N_ele_x * N_ele_y,)`, dtype: int

**4.2.8 Compute Element Stiffness Matrices**
```python
AllLEle = get_element_stiffness_VEC(E.flatten(), nu.flatten(), t)
```

**Inside `get_element_stiffness_VEC`**:
- Flatten: `E.flatten()`, `nu.flatten()`
- Vectorized element stiffness computation
- Returns: `AllLEle` - shape `(64, N_ele_x * N_ele_y)`, dtype: float32

**4.2.9 Compute Element Mass Matrices**
```python
AllLMat = get_element_mass_VEC(rho.flatten(), t, const)
```

**Inside `get_element_mass_VEC`**:
- Flatten: `rho.flatten()`
- Vectorized element mass computation
- Returns: `AllLMat` - shape `(64, N_ele_x * N_ele_y)`, dtype: float32

**4.2.10 Flatten Element Matrix Values**
```python
value_K = AllLEle.flatten(order='F')
value_M = AllLMat.flatten(order='F')
```

**Operations**:
- Flatten: `.flatten(order='F')` - column-major flattening

**Variable**:
- `value_K` - shape `(64 * N_ele_x * N_ele_y,)`, dtype: float32
- `value_M` - shape `(64 * N_ele_x * N_ele_y,)`, dtype: float32

**4.2.11 Assemble Global Matrices**
```python
N_dof = 2 * (1 + N_ele_x) * (1 + N_ele_y)
K = coo_matrix((value_K, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float32)
M = coo_matrix((value_M, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float32)
```

**Operations**:
- Multiplication: `2 * (1 + N_ele_x) * (1 + N_ele_y)` - compute total DOF
- Sparse matrix creation: `coo_matrix((data, (row, col)), shape=...)` - coordinate format

**Variable**:
- `K` - sparse matrix, shape `(N_dof, N_dof)`, dtype: float32
- `M` - sparse matrix, shape `(N_dof, N_dof)`, dtype: float32

---

### PHASE 5: T Matrix Computation

#### 5.1 Extract Wavevector for Structure
**Operation**: Index wavevectors
```python
wavevectors_struct = wavevectors_np[struct_idx, :, :]  # (n_wavevectors, 2)
```

**Operations**:
- Array indexing: `wavevectors_np[struct_idx, :, :]`

**Variable**: `wavevectors_struct` - shape `(n_wavevectors, 2)`, dtype: float16 (from original)

#### 5.2 Loop Over Wavevectors
**Operation**: `for wv_idx, wv in enumerate(wavevectors_struct):`

#### 5.3 Type Conversion
**Operation**: Convert to float32
```python
wv = wv.astype(np.float32)
```

**Operations**:
- Type conversion: `.astype(np.float32)` - float16 to float32

#### 5.4 Compute T Matrix
**Operation**: `T = get_transformation_matrix(wv, const_for_km)`

**Inside `get_transformation_matrix` function**:

**5.4.1 Extract N_pix and Compute N_node**
```python
N_pix = const['N_pix']
if isinstance(N_pix, (list, tuple)):
    N_pix_val = N_pix[0]
else:
    N_pix_val = N_pix
N_node = const['N_ele'] * N_pix_val + 1
```

**Operations**:
- Type check: `isinstance(N_pix, (list, tuple))`
- Conditional: `if/else`
- Multiplication: `const['N_ele'] * N_pix_val`
- Addition: `... + 1`

**5.4.2 Compute Phase Factors**
```python
r_x = np.array([const['a'], 0], dtype=np.float32)
r_y = np.array([0, -const['a']], dtype=np.float32)
r_corner = np.array([const['a'], -const['a']], dtype=np.float32)
xphase = np.exp(1j * np.dot(wavevector, r_x)).astype(np.complex64)
yphase = np.exp(1j * np.dot(wavevector, r_y)).astype(np.complex64)
cornerphase = np.exp(1j * np.dot(wavevector, r_corner)).astype(np.complex64)
```

**Operations**:
- Array creation: `np.array([const['a'], 0])`
- Negation: `-const['a']`
- Dot product: `np.dot(wavevector, r_x)` - vector dot product
- Complex unit: `1j`
- Multiplication: `1j * np.dot(...)`
- Exponential: `np.exp(...)` - complex exponential
- Type conversion: `.astype(np.complex64)`

**Variable**:
- `xphase` - dtype: complex64
- `yphase` - dtype: complex64
- `cornerphase` - dtype: complex64

**5.4.3 Generate Node Indices**
```python
temp_x, temp_y = np.meshgrid(np.arange(1, N_node), np.arange(1, N_node), indexing='ij')
node_idx_x = np.concatenate([
    temp_x.flatten(order='F'),
    np.full(N_node - 1, N_node),
    np.arange(1, N_node),
    [N_node]
])
node_idx_y = np.concatenate([
    temp_y.flatten(order='F'),
    np.arange(1, N_node),
    np.full(N_node - 1, N_node),
    [N_node]
])
```

**Operations**:
- Range: `np.arange(1, N_node)` - creates [1, 2, ..., N_node-1]
- Meshgrid: `np.meshgrid(..., indexing='ij')` - matrix indexing
- Flatten: `.flatten(order='F')` - column-major
- Array creation: `np.full(N_node - 1, N_node)` - fill array
- Concatenation: `np.concatenate([...])`

**Variable**:
- `node_idx_x` - shape `(N_node^2,)`, dtype: int
- `node_idx_y` - shape `(N_node^2,)`, dtype: int

**5.4.4 Compute Global Node Indices**
```python
global_node_idx = (node_idx_y - 1) * N_node + node_idx_x
```

**Operations**:
- Subtraction: `node_idx_y - 1` - convert to 0-based
- Multiplication: `(node_idx_y - 1) * N_node`
- Addition: `... + node_idx_x`

**Variable**: `global_node_idx` - shape `(N_node^2,)`, dtype: int

**5.4.5 Compute Global DOF Indices**
```python
global_dof_idxs = np.concatenate([
    2 * global_node_idx - 1,  # x-displacements
    2 * global_node_idx        # y-displacements
])
```

**Operations**:
- Multiplication: `2 * global_node_idx`
- Subtraction: `... - 1`
- Concatenation: `np.concatenate([...])`

**Variable**: `global_dof_idxs` - shape `(2 * N_node^2,)`, dtype: int

**5.4.6 Define Index Ranges**
```python
unch_idxs = np.arange((N_node - 1)**2)
x_idxs = slice((N_node - 1)**2, (N_node - 1)**2 + N_node - 1)
y_idxs = slice((N_node - 1)**2 + N_node - 1, (N_node - 1)**2 + 2 * (N_node - 1))
```

**Operations**:
- Exponentiation: `(N_node - 1)**2`
- Range: `np.arange(...)`
- Slice creation: `slice(...)`

**5.4.7 Compute Reduced Global Node Indices**
```python
reduced_global_node_idx = np.concatenate([
    (node_idx_y[unch_idxs] - 1) * (N_node - 1) + node_idx_x[unch_idxs],
    (node_idx_y[x_idxs] - 1) * (N_node - 1) + node_idx_x[x_idxs] - (N_node - 1),
    node_idx_x[y_idxs],
    [1]  # Corner node
])
```

**Operations**:
- Array indexing: `node_idx_y[unch_idxs]`
- Subtraction: `... - 1`
- Multiplication: `(node_idx_y[...] - 1) * (N_node - 1)`
- Addition: `... + node_idx_x[...]`
- Subtraction: `... - (N_node - 1)`
- Concatenation: `np.concatenate([...])`
- Array creation: `[1]`

**Variable**: `reduced_global_node_idx` - shape `((N_node-1)^2 + ...)`, dtype: int

**5.4.8 Compute Reduced Global DOF Indices**
```python
reduced_global_dof_idxs = np.concatenate([
    2 * reduced_global_node_idx - 1,  # x-displacements
    2 * reduced_global_node_idx       # y-displacements
])
```

**Operations**:
- Multiplication: `2 * reduced_global_node_idx`
- Subtraction: `... - 1`
- Concatenation: `np.concatenate([...])`

**Variable**: `reduced_global_dof_idxs` - shape `(2 * (N_node-1)^2,)`, dtype: int

**5.4.9 Build Transformation Matrix**
```python
row_idxs = (global_dof_idxs - 1).astype(int)
col_idxs = (reduced_global_dof_idxs - 1).astype(int)
phase_factors = np.concatenate([
    np.ones((N_node - 1)**2),
    np.full(N_node - 1, xphase),
    np.full(N_node - 1, yphase),
    [cornerphase]
])
value_T = np.tile(phase_factors, 2).astype(np.complex64)
N_dof_full = 2 * N_node * N_node
N_dof_reduced = 2 * (N_node - 1) * (N_node - 1)
T = csr_matrix((value_T, (row_idxs, col_idxs)), 
               shape=(N_dof_full, N_dof_reduced), 
               dtype=np.complex64)
```

**Operations**:
- Subtraction: `global_dof_idxs - 1` - convert to 0-based
- Type conversion: `.astype(int)`
- Array creation: `np.ones((N_node - 1)**2)`
- Array creation: `np.full(N_node - 1, xphase)`
- Concatenation: `np.concatenate([...])`
- Tile: `np.tile(phase_factors, 2)` - repeat array
- Type conversion: `.astype(np.complex64)`
- Multiplication: `2 * N_node * N_node`
- Sparse matrix creation: `csr_matrix((data, (row, col)), shape=...)` - compressed sparse row format

**Variable**: `T` - sparse matrix, shape `(N_dof_full, N_dof_reduced)`, dtype: complex64

**5.4.10 Store T Matrix**
**Operation**: `T_data.append(T)`

**Variable**: `T_data` - list of sparse matrices

---

### PHASE 6: Reduced Matrix Computation

#### 6.1 Extract Eigenvectors for Structure
**Operation**: Index and transpose
```python
eigenvectors_struct = EIGENVECTOR_DATA[struct_idx, :, :, :]  # (n_eig, n_wv, n_dof)
eigenvectors_struct = eigenvectors_struct.transpose(2, 1, 0)  # (n_dof, n_wv, n_eig)
```

**Operations**:
- Array indexing: `EIGENVECTOR_DATA[struct_idx, :, :, :]`
- Transpose: `.transpose(2, 1, 0)` - reorder dimensions

**Variable**: `eigenvectors_struct` - shape `(n_dof, n_wv, n_eig)`, dtype: complex128

#### 6.2 Loop Over Wavevectors
**Operation**: `for wv_idx in range(n_wavevectors):`

#### 6.3 Get T Matrix
**Operation**: `T = T_data[wv_idx]`

**Operations**:
- List indexing: `T_data[wv_idx]`

#### 6.4 Convert to Sparse Format
**Operation**: Ensure sparse format and type
```python
T_sparse = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.float32))
K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
```

**Operations**:
- Type check: `sp.issparse(T)`
- Conditional: `if/else`
- Type conversion: `.astype(np.float32)` - complex64 to float32 (NOTE: This is a potential issue!)
- Sparse conversion: `sp.csr_matrix(...)`

**Variable**:
- `T_sparse` - sparse matrix, dtype: complex64 or float32 (depending on original)
- `K_sparse` - sparse matrix, dtype: float32
- `M_sparse` - sparse matrix, dtype: float32

#### 6.5 Compute Reduced Matrices
**Operation**: Matrix triple product
```python
Kr = T_sparse.conj().T @ K_sparse @ T_sparse
Mr = T_sparse.conj().T @ M_sparse @ T_sparse
```

**Operations**:
- Complex conjugate: `.conj()` - complex conjugation
- Transpose: `.T` - matrix transpose
- Hermitian transpose: `.conj().T` - combined operation
- Matrix multiplication: `@` - matrix product (sparse)
- Triple product: `A @ B @ C` - two matrix multiplications

**Variable**:
- `Kr` - sparse matrix, shape `(N_dof_reduced, N_dof_reduced)`, dtype: complex (from T) or float32
- `Mr` - sparse matrix, shape `(N_dof_reduced, N_dof_reduced)`, dtype: complex (from T) or float32

---

### PHASE 7: Eigenvalue Reconstruction

#### 7.1 Loop Over Bands
**Operation**: `for band_idx in range(n_bands):`

#### 7.2 Extract Eigenvector
**Operation**: Index eigenvector
```python
eigvec = eigenvectors_struct[:, wv_idx, band_idx].astype(np.complex128)
```

**Operations**:
- Array indexing: `eigenvectors_struct[:, wv_idx, band_idx]` - extract vector
- Type conversion: `.astype(np.complex128)` - ensure complex128

**Variable**: `eigvec` - shape `(n_dof_reduced,)`, dtype: complex128

#### 7.3 Compute Matrix-Vector Products
**Operation**: Sparse matrix-vector multiplication
```python
Kr_eigvec = Kr @ eigvec
Mr_eigvec = Mr @ eigvec
```

**Operations**:
- Matrix-vector multiplication: `@` - sparse matrix times dense vector
- (Performed twice)

**Variable**:
- `Kr_eigvec` - shape `(n_dof_reduced,)`, dtype: complex (matches Kr)
- `Mr_eigvec` - shape `(n_dof_reduced,)`, dtype: complex (matches Mr)

#### 7.4 Convert to Dense
**Operation**: Convert sparse results to dense
```python
if sp.issparse(Kr_eigvec):
    Kr_eigvec = Kr_eigvec.toarray().flatten()
if sp.issparse(Mr_eigvec):
    Mr_eigvec = Mr_eigvec.toarray().flatten()
```

**Operations**:
- Type check: `sp.issparse(...)`
- Conditional: `if`
- Convert to dense: `.toarray()` - sparse to dense conversion
- Flatten: `.flatten()` - ensure 1D

**Variable**:
- `Kr_eigvec` - shape `(n_dof_reduced,)`, dtype: complex (matches Kr)
- `Mr_eigvec` - shape `(n_dof_reduced,)`, dtype: complex (matches Mr)

#### 7.5 Compute Eigenvalue
**Operation**: Norm ratio
```python
eigval = np.linalg.norm(Kr_eigvec) / np.linalg.norm(Mr_eigvec)
```

**Operations**:
- L2 norm: `np.linalg.norm(Kr_eigvec)` - computes ||Kr_eigvec||_2
- L2 norm: `np.linalg.norm(Mr_eigvec)` - computes ||Mr_eigvec||_2
- Division: `/` - scalar division

**Variable**: `eigval` - dtype: float64 (real scalar)

#### 7.6 Convert to Frequency
**Operation**: Square root and scaling
```python
frequencies_recon[wv_idx, band_idx] = np.sqrt(np.real(eigval)) / (2 * np.pi)
```

**Operations**:
- Real part: `np.real(eigval)` - extract real component (discards imaginary)
- Square root: `np.sqrt(...)` - compute square root
- Multiplication: `2 * np.pi` - compute 2π
- Division: `/` - scalar division
- Array assignment: `frequencies_recon[wv_idx, band_idx] = ...`

**Variable**: `frequencies_recon` - shape `(n_wavevectors, n_bands)`, dtype: float64

#### 7.7 Store Results
**Operation**: Assign to output array
```python
EIGENVALUE_DATA[struct_idx, :, :] = frequencies_recon
```

**Operations**:
- Array assignment: `EIGENVECTOR_DATA[struct_idx, :, :] = frequencies_recon`

#### 7.8 Final Transpose
**Operation**: Transpose to MATLAB format
```python
EIGENVALUE_DATA = EIGENVALUE_DATA.transpose(0, 2, 1)
```

**Operations**:
- Transpose: `.transpose(0, 2, 1)` - reorder from `(n_designs, n_wavevectors, n_bands)` to `(n_designs, n_bands, n_wavevectors)`

**Variable**: `EIGENVALUE_DATA` - shape `(n_designs, n_bands, n_wavevectors)`, dtype: float64

---

## Potential Causes of Discrepancies (Non-Precision Related)

### 1. **Type Conversion Issues**

#### 1.1 Complex to Real Conversion in T Matrix
**Location**: Phase 6.4
```python
T_sparse = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.float32))
```
**Issue**: If T is complex64 and gets converted to float32, this loses the imaginary part entirely. This would cause massive errors in Kr and Mr computation.

**Check**: Verify T matrix dtype before and after conversion.

#### 1.2 Inconsistent Type Conversions
**Location**: Throughout
- Eigenvectors: complex128
- T matrix: complex64
- K, M matrices: float32
- Kr, Mr: Result of mixing complex and real types

**Issue**: Type mismatches in matrix operations can cause unexpected behavior or errors.

### 2. **Indexing and Dimension Mismatches**

#### 2.1 Eigenvector Dimension Mismatch
**Location**: Phase 7.2
```python
eigvec = eigenvectors_struct[:, wv_idx, band_idx]
```
**Issue**: If `eigenvectors_struct` shape doesn't match `n_dof_reduced`, this will cause dimension errors or use wrong eigenvector.

**Check**: Verify `eigenvectors_struct.shape[0] == N_dof_reduced`.

#### 2.2 T Matrix Shape Mismatch
**Location**: Phase 6.5
**Issue**: If T matrix shape `(N_dof_full, N_dof_reduced)` doesn't match K/M shapes `(N_dof, N_dof)`, matrix multiplication will fail or produce wrong results.

**Check**: Verify `T.shape[0] == K.shape[0] == M.shape[0]` and `T.shape[1] == n_dof_reduced`.

#### 2.3 Wavevector Index Mismatch
**Location**: Phase 5.1, 6.2
**Issue**: If wavevector indices don't align between T_data and eigenvectors, wrong T matrix is used for each eigenvector.

**Check**: Verify wavevector ordering is consistent.

### 3. **Matrix Operation Order Issues**

#### 3.1 Matrix Multiplication Associativity
**Location**: Phase 6.5
```python
Kr = T_sparse.conj().T @ K_sparse @ T_sparse
```
**Issue**: While mathematically `(T^H @ K) @ T == T^H @ (K @ T)`, numerically with sparse matrices and different precisions, order can matter.

**Check**: Test both orderings and compare.

#### 3.2 Sparse Matrix Format Differences
**Location**: Phase 6.4
**Issue**: CSR vs CSC vs COO formats can have different numerical behavior in matrix multiplication.

**Check**: Ensure consistent sparse format throughout.

### 4. **Design Conversion Issues**

#### 4.1 Steel-Rubber Paradigm Parameter Mismatch
**Location**: Phase 3.3
**Issue**: If E_min, E_max, rho_min, rho_max, nu_min, nu_max don't match between original and reconstruction, design_3ch will be different, leading to different K and M.

**Check**: Verify all material property bounds match exactly.

#### 4.2 Design Expansion (Repelem) Issues
**Location**: Phase 4.2.2
**Issue**: If N_ele differs between original and reconstruction, design expansion will be different.

**Check**: Verify N_ele matches.

### 5. **Node and DOF Indexing Issues**

#### 5.1 1-Based vs 0-Based Indexing
**Location**: Throughout Phase 4 and 5
**Issue**: MATLAB uses 1-based indexing, Python uses 0-based. If conversion is incorrect, node/DOF indices will be wrong.

**Check**: Verify all index conversions (subtracting 1 where needed).

#### 5.2 Meshgrid Indexing Order
**Location**: Phase 5.4.3
```python
temp_x, temp_y = np.meshgrid(..., indexing='ij')
```
**Issue**: If `indexing='ij'` (matrix) vs `indexing='xy'` (Cartesian) is wrong, node ordering will be incorrect.

**Check**: Verify meshgrid indexing matches MATLAB's behavior.

#### 5.3 Reshape Order (Fortran vs C)
**Location**: Throughout Phase 4
**Issue**: MATLAB uses column-major (Fortran) order, Python defaults to row-major (C). If `order='F'` is missing, arrays will be wrong.

**Check**: Verify all reshapes use `order='F'` where MATLAB reshape is used.

### 6. **Eigenvalue Computation Formula Issues**

#### 6.1 Norm Computation
**Location**: Phase 7.5
```python
eigval = np.linalg.norm(Kr_eigvec) / np.linalg.norm(Mr_eigvec)
```
**Issue**: This formula assumes `Kr @ eigvec = eigval * Mr @ eigvec`. If eigenvector is not exact (from float16 conversion), this relationship doesn't hold.

**Check**: Verify if original uses same formula or solves generalized eigenvalue problem.

#### 6.2 Real Part Extraction
**Location**: Phase 7.6
```python
np.sqrt(np.real(eigval))
```
**Issue**: If eigval has significant imaginary part, discarding it causes error.

**Check**: Verify `np.imag(eigval)` is near zero.

### 7. **Eigenvector Reconstruction Issues**

#### 7.1 Interleaving Order
**Location**: Phase 2.5
```python
EIGENVECTOR_DATA_combined[:, :, :, 0::2] = EIGENVECTOR_DATA_x_flat
EIGENVECTOR_DATA_combined[:, :, :, 1::2] = EIGENVECTOR_DATA_y_flat
```
**Issue**: If x and y are interleaved in wrong order compared to original, eigenvectors won't match DOF ordering.

**Check**: Verify DOF ordering matches original (x0, y0, x1, y1, ... vs y0, x0, y1, x1, ...).

#### 7.2 Missing Eigenvector Entries
**Location**: Phase 2.3
**Issue**: If reduced_indices don't cover all (design, wavevector, band) combinations, some eigenvectors remain zero, causing wrong eigenvalues.

**Check**: Verify all combinations are covered or zeros are handled correctly.

### 8. **Constant Parameter Mismatches**

#### 8.1 N_pix Handling
**Location**: Throughout
**Issue**: If N_pix is list vs scalar handled differently, dimensions will be wrong.

**Check**: Verify N_pix format matches original.

#### 8.2 Lattice Constant 'a'
**Location**: Phase 5.4.2
**Issue**: If 'a' differs, phase factors in T matrix will be wrong.

**Check**: Verify 'a' matches exactly.

### 9. **Sparse Matrix Storage and Operations**

#### 9.1 Sparse Matrix Conversion
**Location**: Phase 6.4, 7.4
**Issue**: Converting between sparse formats (COO → CSR → CSC) or sparse ↔ dense can introduce numerical differences.

**Check**: Verify sparse format consistency.

#### 9.2 Sparse Matrix Multiplication Implementation
**Location**: Phase 6.5
**Issue**: Different sparse matrix libraries or versions may have different numerical implementations.

**Check**: Verify scipy version and sparse matrix backend.

### 10. **Division by Small Numbers**

#### 10.1 Small Denominator in Eigenvalue Formula
**Location**: Phase 7.5
```python
eigval = np.linalg.norm(Kr_eigvec) / np.linalg.norm(Mr_eigvec)
```
**Issue**: If `||Mr_eigvec||` is very small, division amplifies errors.

**Check**: Check for near-zero denominators.

#### 10.2 Small Eigenvalues
**Location**: Phase 7.6
**Issue**: If eigval is very small, square root can amplify relative errors.

**Check**: Check distribution of eigval values.

---

## Recommended Investigation Order

1. **Type Conversion Issues** (Highest Priority)
   - Check T matrix dtype in Phase 6.4
   - Verify no complex → real conversion happens

2. **Dimension Mismatches**
   - Verify all matrix/vector dimensions match
   - Check eigenvector length vs reduced DOF

3. **Indexing Issues**
   - Verify 1-based → 0-based conversions
   - Check meshgrid and reshape orders

4. **Eigenvector Interleaving**
   - Verify x/y interleaving order matches original

5. **Constant Parameter Mismatches**
   - Verify all const parameters match exactly

6. **Matrix Operation Order**
   - Test different multiplication orders
   - Verify sparse format consistency

