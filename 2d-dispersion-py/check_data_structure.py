"""Check if data folders are compatible with plot_dispersion_infer_eigenfrequencies.py"""
import torch
from pathlib import Path

data_base = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10')

folders = ['out_binarized_1', 'out_continuous_1']

print("=" * 80)
print("CHECKING DATA STRUCTURE FOR plot_dispersion_infer_eigenfrequencies.py")
print("=" * 80)

for folder_name in folders:
    data_dir = data_base / folder_name
    print(f"\n{'='*80}")
    print(f"Folder: {folder_name}")
    print(f"Path: {data_dir}")
    print(f"{'='*80}")
    
    if not data_dir.exists():
        print(f"  ❌ Folder does not exist!")
        continue
    
    # Required files
    print("\nRequired Files:")
    required = {
        'geometries_full.pt': False,
        'wavevectors_full.pt': False,
        'displacements_dataset.pt': False,  # For infer mode
    }
    
    for filename in required.keys():
        exists = (data_dir / filename).exists()
        required[filename] = exists
        status = "✅" if exists else "❌"
        print(f"  {status} {filename}")
    
    # Optional files
    print("\nOptional Files (for frequency reconstruction):")
    optional = {
        'K_data.pt': False,
        'M_data.pt': False,
        'T_data.pt': False,
        'reduced_indices.pt': False,
    }
    
    for filename in optional.keys():
        exists = (data_dir / filename).exists()
        optional[filename] = exists
        status = "✅" if exists else "⚠️  (will compute on-the-fly)"
        print(f"  {status} {filename}")
    
    # Check if all required files exist
    all_required = all(required.values())
    
    if not all_required:
        print(f"\n❌ MISSING REQUIRED FILES - Script will NOT work")
        missing = [k for k, v in required.items() if not v]
        print(f"   Missing: {', '.join(missing)}")
        continue
    
    print(f"\n✅ All required files present!")
    
    # Load and check shapes
    print("\nData Shapes:")
    try:
        geo = torch.load(data_dir / 'geometries_full.pt', map_location='cpu')
        wv = torch.load(data_dir / 'wavevectors_full.pt', map_location='cpu')
        disp = torch.load(data_dir / 'displacements_dataset.pt', map_location='cpu')
        
        if hasattr(geo, 'shape'):
            print(f"  geometries: {geo.shape}, dtype={geo.dtype}")
        else:
            print(f"  geometries: type={type(geo)}")
        
        if hasattr(wv, 'shape'):
            print(f"  wavevectors: {wv.shape}, dtype={wv.dtype}")
        else:
            print(f"  wavevectors: type={type(wv)}")
        
        print(f"  displacements: type={type(disp)}")
        if hasattr(disp, 'tensors'):
            print(f"    TensorDataset with {len(disp.tensors)} tensors")
            if len(disp.tensors) >= 4:
                print(f"    Tensor shapes:")
                for i, t in enumerate(disp.tensors[:4]):
                    print(f"      [{i}]: {t.shape}, dtype={t.dtype}")
        
        # Check if reduced dataset
        is_reduced = (data_dir / 'reduced_indices.pt').exists()
        if is_reduced:
            red_idx = torch.load(data_dir / 'reduced_indices.pt', map_location='cpu')
            print(f"\n  Reduced dataset detected")
            print(f"    reduced_indices: {red_idx.shape if hasattr(red_idx, 'shape') else type(red_idx)}")
        
        # Summary
        print(f"\n{'='*80}")
        print("COMPATIBILITY ASSESSMENT:")
        print(f"{'='*80}")
        
        if all_required:
            print("✅ Script WILL WORK with this folder")
            print("\n  Mode: INFER (default)")
            print("    - Will reconstruct frequencies from eigenvectors")
            print("    - Will compute K, M, T matrices on-the-fly (not in dataset)")
            print("    - Does NOT require eigenvalue_data")
            
            if not optional['K_data.pt'] or not optional['M_data.pt'] or not optional['T_data.pt']:
                print("\n  ⚠️  Note: K, M, T matrices not found - will be computed (slower)")
        else:
            print("❌ Script will NOT work - missing required files")
            
    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print("Both folders should work with plot_dispersion_infer_eigenfrequencies.py")
print("in INFER mode (default), which reconstructs frequencies from eigenvectors.")
print("\nUsage:")
print("  python plot_dispersion_infer_eigenfrequencies.py <data_dir>")
print("\nExample:")
print("  python plot_dispersion_infer_eigenfrequencies.py D:\\Research\\NO-2D-Metamaterials\\data\\out_test_10\\out_binarized_1")

