"""Analyze MATLAB's saved contour to determine which symmetry type was used."""
import h5py
import numpy as np
import matplotlib.pyplot as plt

mat_file = '2D-dispersion-han/plots/out_binarized_1_mat/plot_points.mat'

with h5py.File(mat_file, 'r') as f:
    pp = f['plot_points_data']
    wv_key = 'struct_1_wavevectors_contour'
    param_key = 'struct_1_contour_param'
    
    wv = np.array(pp[wv_key]).T  # (55, 2)
    param = np.array(pp[param_key]).flatten()  # (55,)
    
    print(f"MATLAB contour analysis:")
    print(f"  Number of points: {len(wv)}")
    print(f"  contour_param range: [{param.min():.2f}, {param.max():.2f}]")
    print(f"  N_segment (inferred): {param.max():.0f}")
    print(f"\n  First 5 points:")
    for i in range(min(5, len(wv))):
        print(f"    [{wv[i,0]:8.4f}, {wv[i,1]:8.4f}] (param={param[i]:.4f})")
    
    print(f"\n  Last 5 points:")
    for i in range(max(0, len(wv)-5), len(wv)):
        print(f"    [{wv[i,0]:8.4f}, {wv[i,1]:8.4f}] (param={param[i]:.4f})")
    
    # Check for symmetry patterns
    print(f"\n  Checking for symmetry patterns:")
    print(f"    p4mm: Gamma->X->M->Gamma (3 segments, N_segment=3)")
    print(f"    none: Gamma->X->M->Gamma->Y->O->Gamma (6 segments, N_segment=6)")
    
    # Check segment boundaries (where param is integer)
    segment_boundaries = []
    for i in range(len(param)):
        if abs(param[i] - round(param[i])) < 0.01:  # Close to integer
            segment_boundaries.append((i, param[i], wv[i]))
    
    print(f"\n  Segment boundaries (param ≈ integer):")
    for idx, p, pt in segment_boundaries[:10]:  # First 10
        print(f"    Index {idx}: param={p:.2f}, point=[{pt[0]:.4f}, {pt[1]:.4f}]")
    
    # Count segments
    unique_integers = sorted(set([round(p) for p in param if abs(p - round(p)) < 0.01]))
    print(f"\n  Unique integer param values: {unique_integers}")
    print(f"  Number of segments (N_segment): {len(unique_integers) - 1}")
    
    # For p4mm: should have vertices at param 0, 1, 2, 3 (4 vertices, 3 segments)
    # For none: should have vertices at param 0, 1, 2, 3, 4, 5, 6 (7 vertices, 6 segments)
    if len(unique_integers) == 4:
        print(f"  ✓ Matches p4mm (3 segments)")
    elif len(unique_integers) == 7:
        print(f"  ✓ Matches none (6 segments)")
    else:
        print(f"  ⚠ Unexpected number of segments: {len(unique_integers) - 1}")

