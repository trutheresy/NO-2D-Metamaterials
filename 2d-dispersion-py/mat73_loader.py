"""
Robust MATLAB v7.3 (HDF5) file loader with comprehensive error handling.

This module provides utilities to load MATLAB v7.3 files saved with the -v7.3 flag,
which uses HDF5 format. It handles:
- Cell arrays (stored as HDF5 object references)
- Sparse matrices (stored as groups with data/ir/jc fields)
- Structs (stored as HDF5 groups)
- Regular arrays

Author: Python translation for 2D dispersion analysis
"""

import numpy as np
import scipy.sparse as sp
import h5py
from pathlib import Path


def load_matlab_v73(filepath, verbose=True):
    """
    Load a MATLAB v7.3 (.mat) file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .mat file
    verbose : bool, optional
        If True, print detailed loading information (default: True)
        
    Returns
    -------
    data : dict
        Dictionary containing all variables from the MATLAB file
    """
    filepath = Path(filepath)
    
    if verbose:
        print(f"Loading MATLAB v7.3 file: {filepath.name}")
    
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        if verbose:
            print(f"  File contains {len(f.keys())} top-level variables")
        
        for key in f.keys():
            if key.startswith('#'):
                continue  # Skip HDF5 internal references
            
            if verbose:
                print(f"  Loading variable: {key}")
            
            try:
                data[key] = _load_h5_item(f, f[key], key, verbose=verbose)
            except Exception as e:
                print(f"    ERROR loading {key}: {e}")
                raise
    
    if verbose:
        print("  Loading complete!")
    
    return data


def _load_h5_item(f, item, name="", indent=2, verbose=False):
    """
    Recursively load an HDF5 item (dataset, group, or reference).
    
    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle
    item : h5py object
        Item to load (Dataset, Group, or Reference)
    name : str
        Name of the item (for debugging)
    indent : int
        Indentation level for verbose output
    verbose : bool
        Print detailed information
        
    Returns
    -------
    result : various
        Loaded and converted data
    """
    prefix = " " * indent
    
    # Handle HDF5 references
    if isinstance(item, h5py.Reference):
        if verbose:
            print(f"{prefix}├─ Dereferencing {name}")
        try:
            return _load_h5_item(f, f[item], name, indent + 2, verbose)
        except (KeyError, OSError, ValueError) as e:
            # Reference might point to a string in #refs# group or be invalid
            error_msg = str(e)
            if verbose:
                print(f"{prefix}│  └─ Reference failed: {error_msg}")
            
            # Try to extract string name from error message (e.g., "object 'linear' doesn't exist")
            # This is a workaround for MATLAB string references
            if "doesn't exist" in error_msg and "'" in error_msg:
                # Extract the string name from the error
                try:
                    start = error_msg.find("'") + 1
                    end = error_msg.find("'", start)
                    string_name = error_msg[start:end]
                    if verbose:
                        print(f"{prefix}│  └─ Extracted string name: '{string_name}', returning as string")
                    return string_name
                except:
                    pass
            
            # Try to find in #refs# group
            if '#refs#' in f:
                refs_group = f['#refs#']
                # Try all refs to find matching string (this is a fallback)
                # For now, return None to allow loading to continue
                if verbose:
                    print(f"{prefix}│  └─ Could not dereference reference, returning None")
                return None
            else:
                if verbose:
                    print(f"{prefix}│  └─ No #refs# group found, returning None")
                return None
    
    # Handle datasets
    elif isinstance(item, h5py.Dataset):
        data = item[()]
        
        if verbose:
            print(f"{prefix}├─ Dataset '{name}': shape={data.shape}, dtype={data.dtype}")
        
        # Check if this is a cell array (array of references)
        if data.dtype == h5py.ref_dtype:
            if verbose:
                print(f"{prefix}│  └─ Cell array detected, dereferencing {data.size} elements")
            
            # This is a cell array - dereference all elements
            result = np.empty(data.shape, dtype=object)
            for idx in np.ndindex(data.shape):
                ref = data[idx]
                if ref:  # Valid reference
                    try:
                        result[idx] = _load_h5_item(f, f[ref], f"{name}[{idx}]", indent + 4, verbose)
                    except (KeyError, OSError, ValueError) as e:
                        # Reference might be invalid or point to a string
                        error_msg = str(e)
                        if verbose:
                            print(f"{prefix}│     └─ Could not dereference reference at {idx}: {error_msg}")
                        
                        # Try to extract string name from error message
                        if "doesn't exist" in error_msg and "'" in error_msg:
                            try:
                                start = error_msg.find("'") + 1
                                end = error_msg.find("'", start)
                                string_name = error_msg[start:end]
                                if verbose:
                                    print(f"{prefix}│        └─ Extracted string: '{string_name}'")
                                result[idx] = string_name
                            except:
                                result[idx] = None
                        else:
                            # Try to find in #refs# group
                            if '#refs#' in f:
                                refs_group = f['#refs#']
                                # For now, set to None to allow loading to continue
                                result[idx] = None
                            else:
                                result[idx] = None
                else:
                    result[idx] = None
            
            return result
        
        # Regular numeric or string data
        else:
            return data
    
    # Handle groups (could be struct or sparse matrix)
    elif isinstance(item, h5py.Group):
        keys = list(item.keys())
        
        if verbose:
            print(f"{prefix}├─ Group '{name}': {len(keys)} fields")
        
        # Check if this is a MATLAB sparse matrix
        if _is_sparse_matrix(item):
            if verbose:
                print(f"{prefix}│  └─ Sparse matrix detected")
            return _load_sparse_matrix(item, verbose)
        
        # Regular struct - convert to dict
        else:
            if verbose:
                print(f"{prefix}│  └─ Struct detected")
            result = {}
            for key in keys:
                result[key] = _load_h5_item(f, item[key], key, indent + 4, verbose)
            return result
    
    # Unknown type
    else:
        if verbose:
            print(f"{prefix}├─ Unknown type: {type(item)}")
        return item


def _is_sparse_matrix(group):
    """
    Check if an HDF5 group represents a MATLAB sparse matrix.
    
    Parameters
    ----------
    group : h5py.Group
        HDF5 group to check
        
    Returns
    -------
    is_sparse : bool
        True if the group represents a sparse matrix
    """
    required_fields = {'data', 'ir', 'jc'}
    return required_fields.issubset(set(group.keys()))


def _load_sparse_matrix(group, verbose=False):
    """
    Load a MATLAB sparse matrix from an HDF5 group.
    
    MATLAB sparse matrices are stored in CSC format with:
    - data: non-zero values (may be complex stored as structured array)
    - ir: row indices
    - jc: column pointers
    - m, n: dimensions (optional)
    
    Parameters
    ----------
    group : h5py.Group
        HDF5 group containing sparse matrix data
    verbose : bool
        Print detailed information
        
    Returns
    -------
    matrix : scipy.sparse.csc_matrix
        Reconstructed sparse matrix
    """
    # Load components
    data_raw = group['data'][()]
    ir = group['ir'][()].flatten().astype(int)
    jc = group['jc'][()].flatten().astype(int)
    
    # Get dimensions
    if 'm' in group and 'n' in group:
        m = int(group['m'][()])
        n = int(group['n'][()])
    else:
        # Infer dimensions
        m = int(ir.max() + 1) if len(ir) > 0 else 0
        n = int(len(jc) - 1)
    
    # Handle different data formats
    # MATLAB complex numbers can be stored as structured arrays with 'real' and 'imag' fields
    # or as void128/void64 dtype
    if data_raw.dtype.names:
        # Structured array with real and imag parts
        if 'real' in data_raw.dtype.names and 'imag' in data_raw.dtype.names:
            data = data_raw['real'].flatten() + 1j * data_raw['imag'].flatten()
        else:
            # Unknown structure - try to extract
            data = data_raw.flatten()
    elif data_raw.dtype.kind == 'V':  # void type (complex stored as bytes)
        # This is likely complex data stored as void - convert to complex128
        # void128 = complex128 (2 float64), void64 = float64
        if data_raw.dtype.itemsize == 16:  # 128 bits = 2*float64 = complex128
            # Flatten first then convert entire byte buffer
            data_flat = data_raw.flatten()
            data = np.frombuffer(data_flat.tobytes(), dtype=np.complex128)
        elif data_raw.dtype.itemsize == 8:  # 64 bits = float64
            data_flat = data_raw.flatten()
            data = np.frombuffer(data_flat.tobytes(), dtype=np.float64)
        else:
            # Unknown void type - try view as complex128 (most common)
            print(f"        WARNING: Unknown void type with itemsize {data_raw.dtype.itemsize}, trying complex128...")
            try:
                data_flat = data_raw.flatten()
                data = data_flat.view(np.complex128).flatten()
            except:
                data = data_raw.flatten().astype(np.complex128)
    else:
        # Regular numeric array
        data = data_raw.flatten()
    
    # Additional check for 'imag' field stored separately
    if 'imag' in group:
        imag_data = group['imag'][()].flatten()
        if not np.iscomplexobj(data):  # Only add if not already complex
            data = data + 1j * imag_data
    
    if verbose:
        nnz = len(data)
        density = 100 * nnz / (m * n) if m * n > 0 else 0
        print(f"      └─ Sparse matrix: {m}×{n}, {nnz} non-zeros ({density:.2f}% dense), dtype={data.dtype}")
    
    # Construct CSC matrix
    matrix = sp.csc_matrix((data, ir, jc), shape=(m, n))
    
    return matrix


def diagnose_mat_file(filepath):
    """
    Diagnose the structure of a MATLAB v7.3 file without fully loading it.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .mat file
    """
    filepath = Path(filepath)
    print("=" * 70)
    print(f"Diagnosing MATLAB file: {filepath.name}")
    print("=" * 70)
    
    with h5py.File(filepath, 'r') as f:
        print(f"\nTop-level variables ({len(f.keys())} total):")
        
        for key in f.keys():
            if key.startswith('#'):
                continue
            
            item = f[key]
            _diagnose_item(f, item, key, indent=0)
    
    print("\n" + "=" * 70)


def _diagnose_item(f, item, name, indent=0):
    """
    Recursively diagnose an HDF5 item structure.
    
    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle
    item : h5py object
        Item to diagnose
    name : str
        Name of the item
    indent : int
        Indentation level
    """
    prefix = "  " * indent
    
    if isinstance(item, h5py.Dataset):
        data = item[()]
        print(f"{prefix}├─ {name}: Dataset")
        print(f"{prefix}│  ├─ Shape: {data.shape}")
        print(f"{prefix}│  ├─ Dtype: {data.dtype}")
        
        if data.dtype == h5py.ref_dtype:
            print(f"{prefix}│  └─ Type: Cell array (references)")
            # Show first element structure
            if data.size > 0:
                first_ref = data.flat[0]
                if first_ref:
                    print(f"{prefix}│     └─ First element structure:")
                    _diagnose_item(f, f[first_ref], "element[0]", indent + 2)
        else:
            print(f"{prefix}│  └─ Type: Numeric/String array")
    
    elif isinstance(item, h5py.Group):
        keys = list(item.keys())
        print(f"{prefix}├─ {name}: Group ({len(keys)} fields)")
        
        if _is_sparse_matrix(item):
            data = item['data'][()]
            ir = item['ir'][()]
            jc = item['jc'][()]
            m = int(item['m'][()]) if 'm' in item else ir.max() + 1
            n = int(item['n'][()]) if 'n' in item else len(jc) - 1
            print(f"{prefix}│  └─ Type: Sparse matrix ({m}×{n})")
        else:
            print(f"{prefix}│  └─ Type: Struct")
            for key in keys[:3]:  # Show first 3 fields
                _diagnose_item(f, item[key], key, indent + 2)
            if len(keys) > 3:
                print(f"{prefix}{'  ' * (indent + 2)}... and {len(keys) - 3} more fields")


if __name__ == "__main__":
    # Self-test
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        diagnose_mat_file(filepath)
    else:
        print("Usage: python mat73_loader.py <filepath.mat>")
        print("  Run diagnosis on a MATLAB v7.3 file")

