"""
Check if element stiffness matrix needs to be scaled by element area.

The formula (1/48)*E*t/(1-nu^2)*[...] is for a UNIT square element.
For non-unit elements, stiffness should be scaled by element area (h^2).
"""
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import sys
import h5py

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC_simplified

# Configuration
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
struct_idx = 0

print("="*70)
print("CHECKING ELEMENT SIZE SCALING")
print("="*70)

# Load Python matrices
K_data = torch.load(python_data_dir / 'K_data.pt', map_location='cpu')
M_data = torch.load(python_data_dir / 'M_data.pt', map_location='cpu')
K_py = K_data[struct_idx]
M_py = M_data[struct_idx]
if not sp.issparse(K_py):
    K_py = sp.csr_matrix(K_py)
if not sp.issparse(M_py):
    M_py = sp.csr_matrix(M_py)

# Load MATLAB matrices
with h5py.File(str(matlab_mat_file), 'r') as f:
    K_DATA_ref = f['K_DATA'][struct_idx, 0]
    M_DATA_ref = f['M_DATA'][struct_idx, 0]
    K_data_matlab = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir = np.array(f[K_DATA_ref]['ir']).flatten().astype(int)
    K_jc = np.array(f[K_DATA_ref]['jc']).flatten().astype(int)
    M_data_matlab = np.array(f[M_DATA_ref]['data']).flatten()
    M_ir = np.array(f[M_DATA_ref]['ir']).flatten().astype(int)
    M_jc = np.array(f[M_DATA_ref]['jc']).flatten().astype(int)
    n = len(K_jc) - 1

K_ml = sp.csr_matrix((K_data_matlab, K_ir, K_jc), shape=(n, n))
M_ml = sp.csr_matrix((M_data_matlab, M_ir, M_jc), shape=(n, n))

# Compare a sample of values
K_py_dense = K_py.toarray()
K_ml_dense = K_ml.toarray()

# Find common non-zero locations
py_nnz = (K_py_dense != 0)
ml_nnz = (K_ml_dense != 0)
common_nnz = py_nnz & ml_nnz

if np.any(common_nnz):
    py_vals = K_py_dense[common_nnz]
    ml_vals = K_ml_dense[common_nnz]
    
    # Calculate ratio
    ratio = ml_vals / (py_vals + 1e-15)  # Avoid division by zero
    
    print(f"\nCommon non-zero locations: {np.sum(common_nnz)}")
    print(f"Ratio (MATLAB/Python) statistics:")
    print(f"  Mean: {np.mean(ratio):.6e}")
    print(f"  Median: {np.median(ratio):.6e}")
    print(f"  Min: {np.min(ratio):.6e}")
    print(f"  Max: {np.max(ratio):.6e}")
    print(f"  Std: {np.std(ratio):.6e}")
    
    # Check if ratio is approximately constant (indicating a scaling factor)
    if np.std(ratio) / np.abs(np.mean(ratio)) < 0.1:  # Less than 10% variation
        scaling_factor = np.mean(ratio)
        print(f"\n✅ Ratio is approximately constant: {scaling_factor:.6e}")
        print(f"   This suggests a uniform scaling factor is missing in Python")
        
        # Try to identify what this scaling factor corresponds to
        # Load design to compute element size
        try:
            designs_dataset = torch.load(python_data_dir / 'designs_dataset.pt', map_location='cpu')
            if hasattr(designs_dataset, 'tensors') and len(designs_dataset.tensors) > 0:
                design = designs_dataset.tensors[0][struct_idx].numpy()
            else:
                design = designs_dataset[struct_idx].numpy()
            
            if design.ndim == 2:
                N_pix = design.shape[0]
            else:
                N_pix = design.shape[0]
            
            # Element size calculation
            a = 1.0  # Default
            N_ele = 1
            element_size = a / (N_ele * N_pix)
            element_area = element_size ** 2
            
            print(f"\n   Element size calculation:")
            print(f"   a = {a}, N_ele = {N_ele}, N_pix = {N_pix}")
            print(f"   element_size = {element_size:.6e} m")
            print(f"   element_area = {element_area:.6e} m²")
            
            if np.isclose(scaling_factor, 1.0 / element_area, rtol=0.1):
                print(f"\n   ⚠️  Scaling factor ({scaling_factor:.6e}) is approximately 1/element_area ({1.0/element_area:.6e})")
                print(f"   This suggests Python should scale stiffness by element_area!")
            elif np.isclose(scaling_factor, element_area, rtol=0.1):
                print(f"\n   ⚠️  Scaling factor ({scaling_factor:.6e}) is approximately element_area ({element_area:.6e})")
                print(f"   This suggests Python should NOT scale, but something else is wrong")
            else:
                print(f"\n   ⚠️  Scaling factor doesn't match element_area directly")
                print(f"   Need further investigation")
                
        except Exception as e:
            print(f"\n   Could not load design to check element size: {e}")
    else:
        print(f"\n⚠️  Ratio varies significantly (std/mean = {np.std(ratio)/np.abs(np.mean(ratio)):.2%})")
        print(f"   This suggests more than just a scaling factor difference")

else:
    print("\n⚠️  No common non-zero locations found - matrices have different sparsity patterns")

