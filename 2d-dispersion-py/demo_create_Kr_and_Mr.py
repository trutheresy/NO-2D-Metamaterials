"""
Demo: Create Reduced Stiffness and Mass Matrices (Kr and Mr)

This script demonstrates how to:
1. Load a dataset with K_DATA, M_DATA, and T_DATA
2. Pick a unit cell and wavevector
3. Create reduced stiffness (Kr) and mass (Mr) matrices
4. Visualize their sparsity patterns

The reduced matrices are created using:
    Kr = T' * K * T
    Mr = T' * M * T

where:
    - K is the global stiffness matrix
    - M is the global mass matrix
    - T is the transformation matrix for a specific wavevector
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio

# Import robust MATLAB loader
try:
    from mat73_loader import load_matlab_v73
except ImportError:
    import mat73_loader
    load_matlab_v73 = mat73_loader.load_matlab_v73


def load_dataset(data_path):
    """
    Load dataset from .mat file (handles both v7.3 HDF5 and older formats).
    
    Parameters
    ----------
    data_path : str or Path
        Path to the .mat file containing the dataset
        
    Returns
    -------
    data : dict
        Dictionary containing the dataset variables
    """
    print(f"Loading dataset from: {data_path}")
    
    # Try to load with scipy first (for older MATLAB files)
    try:
        data = sio.loadmat(data_path, squeeze_me=True)
        # Remove MATLAB metadata keys
        data = {k: v for k, v in data.items() if not k.startswith('__')}
        print("  Loaded using scipy.io.loadmat (MATLAB < v7.3)")
        
    except NotImplementedError:
        # Use robust h5py loader
        print("  File is MATLAB v7.3 format, using robust h5py loader...")
        try:
            data = load_matlab_v73(data_path, verbose=False)
        except ImportError:
            raise ImportError(
                "h5py is required to read MATLAB v7.3 files. "
                "Install it with: pip install h5py"
            )
    
    print("\nDataset contents:")
    print(f"  - Number of structures: {len(data.get('K_DATA', []))}")
    if 'T_DATA' in data:
        print(f"  - Number of wavevectors: {len(data['T_DATA'])}")
    print(f"  - Available keys: {[k for k in data.keys() if not k.startswith('__')]}")
    
    return data


def create_reduced_matrices(K, M, T):
    """
    Create reduced stiffness and mass matrices.
    
    Parameters
    ----------
    K : scipy.sparse matrix
        Global stiffness matrix
    M : scipy.sparse matrix
        Global mass matrix
    T : scipy.sparse matrix
        Transformation matrix for a specific wavevector
        
    Returns
    -------
    Kr : scipy.sparse matrix
        Reduced stiffness matrix
    Mr : scipy.sparse matrix
        Reduced mass matrix
    """
    # Create reduced matrices: Kr = T' * K * T, Mr = T' * M * T
    Kr = T.conj().T @ K @ T
    Mr = T.conj().T @ M @ T
    
    print(f"\nMatrix dimensions:")
    print(f"  - K:  {K.shape}")
    print(f"  - M:  {M.shape}")
    print(f"  - T:  {T.shape}")
    print(f"  - Kr: {Kr.shape}")
    print(f"  - Mr: {Mr.shape}")
    
    print(f"\nMatrix sparsity:")
    print(f"  - K:  {K.nnz} non-zero elements ({100*K.nnz/np.prod(K.shape):.2f}%)")
    print(f"  - M:  {M.nnz} non-zero elements ({100*M.nnz/np.prod(M.shape):.2f}%)")
    print(f"  - Kr: {Kr.nnz} non-zero elements ({100*Kr.nnz/np.prod(Kr.shape):.2f}%)")
    print(f"  - Mr: {Mr.nnz} non-zero elements ({100*Mr.nnz/np.prod(Mr.shape):.2f}%)")
    
    return Kr, Mr


def plot_sparsity_patterns(K, M, Kr, Mr, save_fig=False):
    """
    Visualize sparsity patterns of matrices.
    
    Parameters
    ----------
    K : scipy.sparse matrix
        Global stiffness matrix
    M : scipy.sparse matrix
        Global mass matrix
    Kr : scipy.sparse matrix
        Reduced stiffness matrix
    Mr : scipy.sparse matrix
        Reduced mass matrix
    save_fig : bool, optional
        Whether to save the figure (default: False)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot K sparsity
    axes[0, 0].spy(K, markersize=1)
    axes[0, 0].set_title(f'K (Global Stiffness Matrix)\nShape: {K.shape}, NNZ: {K.nnz}')
    axes[0, 0].set_xlabel('Column Index')
    axes[0, 0].set_ylabel('Row Index')
    
    # Plot M sparsity
    axes[0, 1].spy(M, markersize=1)
    axes[0, 1].set_title(f'M (Global Mass Matrix)\nShape: {M.shape}, NNZ: {M.nnz}')
    axes[0, 1].set_xlabel('Column Index')
    axes[0, 1].set_ylabel('Row Index')
    
    # Plot Kr sparsity
    axes[1, 0].spy(Kr, markersize=1, color='red')
    axes[1, 0].set_title(f'Kr (Reduced Stiffness Matrix)\nShape: {Kr.shape}, NNZ: {Kr.nnz}')
    axes[1, 0].set_xlabel('Column Index')
    axes[1, 0].set_ylabel('Row Index')
    
    # Plot Mr sparsity
    axes[1, 1].spy(Mr, markersize=1, color='green')
    axes[1, 1].set_title(f'Mr (Reduced Mass Matrix)\nShape: {Mr.shape}, NNZ: {Mr.nnz}')
    axes[1, 1].set_xlabel('Column Index')
    axes[1, 1].set_ylabel('Row Index')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('sparsity_patterns.png', dpi=150, bbox_inches='tight')
        print("\nFigure saved as 'sparsity_patterns.png'")
    
    plt.show()


def main():
    """
    Main demonstration function.
    """
    print("=" * 70)
    print("Demo: Create Reduced Stiffness and Mass Matrices (Kr and Mr)")
    print("=" * 70)
    
    # Example path - update this to point to your actual dataset
    # You can use either a binarized or continuous dataset
    example_paths = [
        "generate_dispersion_dataset_Han/OUTPUT/output 15-Sep-2025 15-33-28/binarized 15-Sep-2025 15-33-28.mat",
        "generate_dispersion_dataset_Han/OUTPUT/output 15-Sep-2025 15-36-03/continuous 15-Sep-2025 15-36-03.mat",
    ]
    
    # Try to find an existing dataset
    data_fn = None
    for path in example_paths:
        test_path = Path("../2D-dispersion_alex") / path
        if test_path.exists():
            data_fn = test_path
            break
    
    if data_fn is None:
        print("\nERROR: Could not find dataset file.")
        print("Please update the path in this script to point to your dataset.")
        print("\nExpected format: A .mat file containing K_DATA, M_DATA, and T_DATA")
        return
    
    # Load the dataset
    data = load_dataset(data_fn)
    
    # Check if required data is present
    required_keys = ['K_DATA', 'M_DATA', 'T_DATA']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        print(f"\nERROR: Dataset is missing required keys: {missing_keys}")
        print("This dataset may not have been generated with matrix saving enabled.")
        return
    
    # Pick a unit cell (struct_idx)
    struct_idx = 0  # Python uses 0-based indexing (MATLAB uses 1-based)
    print(f"\n{'='*70}")
    print(f"Selected unit cell: struct_idx = {struct_idx} (0-based)")
    print(f"{'='*70}")
    
    # Pick a wavevector (wv_idx)
    wv_idx = 0  # Python uses 0-based indexing
    print(f"Selected wavevector: wv_idx = {wv_idx} (0-based)")
    
    # Extract K, M, and T
    # Note: Properly loaded data should already have sparse matrices
    K = data['K_DATA'].flat[struct_idx] if data['K_DATA'].ndim > 0 else data['K_DATA']
    M = data['M_DATA'].flat[struct_idx] if data['M_DATA'].ndim > 0 else data['M_DATA']
    T = data['T_DATA'].flat[wv_idx] if data['T_DATA'].ndim > 0 else data['T_DATA']
    
    # Verify they are sparse matrices (should already be after robust loading)
    if not sp.issparse(K):
        if hasattr(K, 'dtype') and K.dtype == object:
            raise TypeError(f"K is object type - HDF5 loading may have failed. Type: {type(K)}")
        K = sp.csr_matrix(K)
    if not sp.issparse(M):
        if hasattr(M, 'dtype') and M.dtype == object:
            raise TypeError(f"M is object type - HDF5 loading may have failed. Type: {type(M)}")
        M = sp.csr_matrix(M)
    if not sp.issparse(T):
        if hasattr(T, 'dtype') and T.dtype == object:
            raise TypeError(f"T is object type - HDF5 loading may have failed. Type: {type(T)}")
        T = sp.csr_matrix(T)
    
    print(f"\nExtracted matrices:")
    print(f"  - K: {type(K)}")
    print(f"  - M: {type(M)}")
    print(f"  - T: {type(T)}")
    
    # Create reduced stiffness and mass matrices
    print(f"\n{'='*70}")
    print("Creating reduced matrices...")
    print(f"{'='*70}")
    Kr, Mr = create_reduced_matrices(K, M, T)
    
    # Visualize sparsity patterns
    print(f"\n{'='*70}")
    print("Plotting sparsity patterns...")
    print(f"{'='*70}")
    plot_sparsity_patterns(K, M, Kr, Mr, save_fig=True)
    
    print(f"\n{'='*70}")
    print("Demo completed successfully!")
    print(f"{'='*70}")
    print("\nThe reduced matrices Kr and Mr can now be used for eigenvalue analysis:")
    print("  - Solve: Kr * u = lambda * Mr * u")
    print("  - Where lambda = omega^2 (eigenfrequencies)")
    print("  - And u are the eigenmodes")


if __name__ == "__main__":
    main()

