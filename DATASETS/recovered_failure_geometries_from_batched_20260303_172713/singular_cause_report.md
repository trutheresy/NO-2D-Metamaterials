# Singular-Factor Cause Test Report

## Scope/Subset Used
- Cause 2: 32 cases (12 failures + 20 controls), matrix-level K/M comparisons.
- Cause 1: representative subset (5 failure cases, 10 wavevectors).
- Cause 3: same representative subset (5 failure cases, 10 wavevectors).
- Cause 4: tiny backend subset (2 failure cases, 5 wavevectors).

## Cause 2 (Pruning)
- prune_on mean_rel_avg: 0.011429897649580491
- prune_off mean_rel_avg: 0.011429897649580491
- inf cases: on=0 off=0
- Observation: no material difference between prune on/off on tested matrix metrics.

## Cause 1 (Precision)
- current mean_rel_avg: 85832638296.78577
- high_precision mean_rel_avg: 561847.9848928439
- inf cases: current=0 high=0
- Observation: high-precision path dramatically reduces MATLAB-relative error.

## Cause 3 (Symmetry + Quantization)
- current_float16: mean_rel_avg=85832638296.78577 inf_cases=0
- guarded_float16: mean_rel_avg=85832636707.52426 inf_cases=0
- guarded_float64_noquant: mean_rel_avg=91933858812.55432 inf_cases=0
- Observation: guarded symmetry did not materially reduce error in tested subset; float64 no-quantization was not better in this run.

## Cause 4 (Backend Variants)
- sparse_shiftinvert_sigma: mean_rel_avg=2809238.739909641 inf_cases=0
- sparse_which_SM: mean_rel_avg=41298402.28880578 inf_cases=0
- dense_SM: mean_rel_avg=44451690.51382662 inf_cases=0
- Observation: MATLAB baseline was closest to sparse shift-invert with numeric sigma among tested backend variants.

## Ranked Impact (observed reduction score)
1. cause1_precision (score=85832076448.80087)
2. cause3_symmetry_quantization (score=1589.2615051269531)
3. cause2_pruning (score=0.0)
4. cause4_backend (score=0.0)

## Notes
- Per requirement, NaN/singularity were treated as Inf errors; no Inf cases occurred in these completed subsets.
- Runtime constraints required subset reduction for Causes 1/3/4; rerun larger sets for tighter confidence intervals.