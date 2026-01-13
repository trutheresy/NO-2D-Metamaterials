"""
Analyze remaining potential causes of eigenvalue discrepancies.
"""

print("=" * 80)
print("REMAINING POTENTIAL CAUSES OF EIGENVALUE DISCREPANCIES")
print("=" * 80)

print("\nELIMINATED HYPOTHESES:")
print("  H1: Norm formula (formula is correct)")
print("  H3: Frequency conversion formula (f = sqrt(eigval)/(2*pi) is correct)")
print("  H4: T matrix transformation (T matrices match exactly)")
print("  H5: K/M matrices (K/M matrices match when computed from same design)")
print("  H6: Eigenvector validity (eigenvectors satisfy eigenvector equation)")
print("  H7 (EIGENVALUE_DATA precision): Fixed (float32->float64)")
print("  H2 (Eigenvector precision): Implemented (float16->float64)")

print("\nOBSERVED ISSUES:")
print("  - Large errors (thousands of percent)")
print("  - Reconstructed frequencies are much larger than original")
print("  - Original: 4.07e-04 Hz at Gamma point")
print("  - Reconstructed: 6.85 Hz at Gamma point")
print("  - This suggests a fundamental issue, not just precision")

print("\n" + "=" * 80)
print("REMAINING POTENTIAL CAUSES")
print("=" * 80)

print("\n1. EIGENVECTOR FORMAT/ORDERING MISMATCH:")
print("   - Eigenvectors from .pt files: spatial format (N_pix, N_pix) for x/y")
print("   - Eigenvectors in MATLAB: DOF format (N_dof) with interleaved x/y")
print("   - Need to verify conversion from spatial to DOF format is correct")
print("   - Need to verify interleaving (x at even indices, y at odd indices)")

print("\n2. EIGENVECTOR INDEXING/ORDERING:")
print("   - Band ordering: Are eigenvectors in the same band order?")
print("   - Wavevector ordering: Are eigenvectors in the same wavevector order?")
print("   - DOF ordering: Are DOFs in the same order (spatial to DOF mapping)?")

print("\n3. K/M MATRIX COMPUTATION DIFFERENCES:")
print("   - K/M matrices match when computed from same design")
print("   - But are we using the same design during reconstruction?")
print("   - Need to verify design used for K/M computation matches original")

print("\n4. EIGENVECTOR SCALING/NORMALIZATION:")
print("   - Eigenvectors can be multiplied by any complex phase factor")
print("   - Normalization: Are eigenvectors normalized the same way?")
print("   - Phase alignment: Are eigenvectors in the same phase?")

print("\n5. DESIGN/GEOMETRY DIFFERENCES:")
print("   - Designs in .pt files vs original .mat file")
print("   - Material property mapping (steel-rubber paradigm)")
print("   - Design resolution, N_pix, N_ele differences")

print("\n6. DOF SPACE MISMATCH:")
print("   - Full DOF space vs reduced DOF space")
print("   - N_dof = 2 * (N_ele * N_pix)^2 for reduced space")
print("   - Need to verify eigenvectors are in the correct DOF space")

print("\n7. EIGENVECTOR SOURCE DIFFERENCES:")
print("   - Eigenvectors from .pt files (reduced dataset)")
print("   - Original eigenvectors from MATLAB (full dataset)")
print("   - Are they computed from the same problem?")
print("   - Are they the same eigenvectors (just stored differently)?")

print("\n8. UNIT CONVERSION ISSUES:")
print("   - Frequency units (Hz) - formula is correct")
print("   - But are there scale factors in K/M matrices?")
print("   - Are there scale factors in eigenvectors?")

print("\n9. FORMULA IMPLEMENTATION DIFFERENCES:")
print("   - Formula looks the same: eigval = ||Kr*eigvec|| / ||Mr*eigvec||")
print("   - But are Kr and Mr computed the same way?")
print("   - Are matrix multiplications done the same way?")

print("\n10. EIGENVECTOR PHASE/NORMALIZATION:")
print("    - Eigenvectors are not unique (can multiply by complex phase)")
print("    - But eigenvalue should be the same regardless of phase")
print("    - However, if normalization differs, might affect computation")

print("\n" + "=" * 80)
print("MOST LIKELY CAUSES (based on observed large errors):")
print("=" * 80)

print("\n1. EIGENVECTOR FORMAT/ORDERING MISMATCH (HIGH PRIORITY):")
print("   - Spatial format vs DOF format conversion")
print("   - Interleaving order (x at even/odd indices?)")
print("   - DOF ordering (spatial to DOF mapping)")

print("\n2. EIGENVECTOR SOURCE DIFFERENCES (HIGH PRIORITY):")
print("   - Are eigenvectors from .pt files the same as original?")
print("   - Are they computed from the same problem/design?")
print("   - Are they the same eigenvectors (just stored differently)?")

print("\n3. K/M MATRIX DESIGN MISMATCH (MEDIUM PRIORITY):")
print("   - Design used for K/M computation vs original")
print("   - Material property mapping differences")

print("\n4. DOF SPACE MISMATCH (MEDIUM PRIORITY):")
print("   - Full vs reduced DOF space")
print("   - N_dof size differences")

print("\n" + "=" * 80)
print("NEXT STEPS FOR INVESTIGATION:")
print("=" * 80)

print("\n1. Compare eigenvector formats:")
print("   - Check eigenvector shape and format in .pt files")
print("   - Check eigenvector shape and format in original .mat file")
print("   - Verify conversion from spatial to DOF format")

print("\n2. Compare eigenvector values:")
print("   - Compare eigenvectors at specific (struct, wv, band) indices")
print("   - Check if they're the same eigenvectors (up to phase)")
print("   - Check normalization")

print("\n3. Compare designs:")
print("   - Compare designs used for K/M computation")
print("   - Verify material property mapping")

print("\n4. Test with stored eigenvectors:")
print("   - Use stored eigenvectors from original .mat file")
print("   - Compute eigenvalues using Python reconstruction")
print("   - Compare with original eigenvalues")
print("   - This would isolate if issue is with eigenvectors or computation")

