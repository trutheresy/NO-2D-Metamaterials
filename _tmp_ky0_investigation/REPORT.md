# ky=0 Hypothesis Investigation (read-only)

Temporary scripts in `_tmp_ky0_investigation/`. No datasets or inference files modified.

## Executive summary

### c_test group NMAE (group-weighted) and second-peak rate

| Group | N samples | Mean NMAE | Second-peak % | ch0 NMAE | disp NMAE |
|-------|-----------|-----------|---------------|----------|-----------|
| ky0 | 150000 | 0.6597 | 70.5 | 0.0184 | 1.3009 |
| kx0_only | 72000 | 0.5853 | 49.6 | 0.0026 | 1.1679 |
| kx0 | 78000 | 0.7102 | 53.4 | 0.0327 | 1.3877 |
| edge | 432000 | 0.4477 | 37.8 | 0.0079 | 0.8875 |
| interior | 1518000 | 0.1271 | 9.7 | 0.0022 | 0.2519 |

## Hypothesis verdicts

### H1: Half-plane IBZ puts full Γ–X path on domain boundary
- **Verdict:** SUPPORTED
- ky=0 has 25 of 325 waves (7.7%) on the IBZ bottom edge; designs use p4mm, wavevectors use symmetry_type=none.

### H4: Training under-exposure of ky=0 in reduced_indices
- **Verdict:** WEAK / INCONCLUSIVE
- ky0/interior mean appearance count ratio ≈ 1.001 (expected ~1.0 under uniform 65/325 downselect per geom×band).

### H2: Embedding degeneracy (shared ky=0 on ky=0 row)
- **Verdict:** PARTIAL
- All ky=0 share ky=0 in embed_2const_wavelet; mean off-diag cos ≈ 0.604. Correlation(mean_sim, NMAE) on ky=0: -0.081.

### H3/H9: Displacement dominates failures (c_test)
- **Verdict:** PARTIAL
- ky0 disp/ch0=70.62 vs interior 112.59; ky0 second-peak=70.5% vs interior 9.7%.

### H3/H9: Displacement dominates failures (b_test)
- **Verdict:** STRONG
- ky0 disp/ch0=42.15 vs interior 24.79; ky0 second-peak=74.0% vs interior 23.7%.

### H5/H12: Signed-kx mirror asymmetry on Γ–X
- **Verdict:** WEAK
- c_test mean |ΔNMAE| mirror pairs = 0.0082; 13/13 pairs both >50% second peak.

### H6/H8: Truth displacement complexity higher on ky=0
- **Verdict:** WEAK
- c_test truth disp RMS ky0=0.0154, kx0=0.0155, interior=0.0156.

### H6: Eigenvalue path roughness along Γ–X
- **Verdict:** CHECK
- Mean |Δf| along ky0 path = 52.83 vs kx0 = 52.83.

### H7: Higher bands worse on ky=0
- **Verdict:** CHECK REPORT
- c_test worst ky0 band b5 nmae=0.757; interior b5=0.237.

## New hypotheses surfaced

- **H13:** Geometry-specific Γ–X failures — see `geom_ky0_summary` in wave_stats JSON.
- **H14:** float16 displacement targets amplify error on high-|u| ky=0 modes.
- **H15:** Mismatch between p4mm design symmetry and non-reduced BZ sampling creates physically distinct modes on Γ–X.

## Scripts run

1. `h07_compute_wave_stats.py` — per-wave NMAE from existing predictions
2. `h01_geometry_training.py` — IBZ layout + train downselect exposure
3. `h02_encoding_similarity.py` — waveform similarity vs error
4. `h03_channel_decomposition.py` — ch0 vs displacement
5. `h04_band_by_group.py` — band-resolved NMAE by group
6. `h05_mirror_pairs.py` — ±kx mirror asymmetry
7. `h06_truth_physics.py` — truth field complexity

Raw JSON in `_tmp_ky0_investigation\cache/`.