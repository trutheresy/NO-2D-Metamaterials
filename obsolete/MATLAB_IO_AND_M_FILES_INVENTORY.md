# MATLAB I/O and MATLAB-Format Script Inventory

This inventory lists (1) non-MATLAB scripts that ingest/emit MATLAB formats or invoke MATLAB, and (2) every MATLAB `.m` script/function in the repository.

## Counts

- Python MATLAB-I/O/bridge scripts (active): `51`
- Python MATLAB-I/O/bridge scripts (obsolete): `8`
- MATLAB `.m` files (active): `70`
- MATLAB `.m` files (obsolete): `70`

## Python Scripts With MATLAB Inputs/Outputs or MATLAB Bridge Behavior

| Scope | Script | MATLAB Interaction | What It Does |
|---|---|---|---|
| active | `2d-dispersion-py/convert_mat_to_pytorch.py` | uses MATLAB-style dataset field names | Convert MATLAB dataset to Reduced PyTorch Format (matching matlab_to_reduced_pt.ipynb) |
| active | `2d-dispersion-py/demo_create_Kr_and_Mr.py` | reads .mat/HDF5-MAT | Demo: Create Reduced Stiffness and Mass Matrices (Kr and Mr) |
| active | `2d-dispersion-py/mat73_loader.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Robust MATLAB v7.3 (HDF5) file loader with comprehensive error handling. |
| active | `2d-dispersion-py/plot_dispersion_infer_eigenfrequencies.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Plot Dispersion Script with Eigenfrequency Reconstruction |
| active | `2d-dispersion-py/plot_dispersion_with_eigenfrequencies_reduced_set.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Plot Dispersion Script with Eigenfrequencies (Consolidated) |
| active | `2d-dispersion-py/plotting.py` | writes .mat/HDF5-MAT | Plotting and visualization functions. |
| active | `2d-dispersion-py/utils_han.py` | uses MATLAB-style dataset field names | Utility functions matching 2D-dispersion-Han MATLAB library. |
| active | `NO_utils.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Get the shape of the input array |
| active | `NO_utils_multiple.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Initialize containers for combined data if multiple datafiles is True |
| active | `OUTPUT/output_2026-03-02_04-15-15/generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `OUTPUT/output_2026-03-02_11-39-20/generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `OUTPUT/output_2026-03-02_21-28-29/generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `OUTPUT/output_2026-03-02_21-40-24/generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `OUTPUT/output_2026-03-02_21-47-46/generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `OUTPUT/output_2026-03-02_21-53-33/generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `OUTPUT/output_2026-03-02_21-59-07/generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `OUTPUT/output_2026-03-02_22-04-30/generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `OUTPUT/output_2026-03-02_22-10-45/generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `OUTPUT/output_2026-03-03_12-02-25/generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `compare_KMT_python_matlab.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare K, M, T matrices computed by Python vs MATLAB. |
| active | `compare_K_M_float64_vs_matlab.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare K and M matrices computed in Python with float64 precision vs MATLAB. |
| active | `compare_K_M_matrices.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare Python-generated K & M matrices with MATLAB ground truth. |
| active | `compare_K_construction_intermediates.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare intermediate steps of K matrix construction between MATLAB and Python. |
| active | `compare_K_intermediates.py` | reads .mat/HDF5-MAT | Compare intermediate variables from MATLAB and Python K matrix computation. |
| active | `compare_K_intermediates_detailed.py` | reads .mat/HDF5-MAT | Compare intermediate values from Python and MATLAB K matrix assembly. |
| active | `compare_K_matrix_statistics.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare the scale and statistics of K matrices between MATLAB and Python. |
| active | `compare_case2_plot_points.py` | reads .mat/HDF5-MAT | Compare Case 2 (with eigenfrequencies) plot points between Python and MATLAB. |
| active | `compare_case2_with_eigenfrequencies.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare plot points for Case 2 (with eigenfrequencies) between Python and MATLAB. |
| active | `compare_debug_to_original.py` | reads .mat/HDF5-MAT, uses MATLAB-style dataset field names | Compare debug reconstruction output to original MATLAB data. |
| active | `compare_dispersion_plots.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare Dispersion Plots Between MATLAB and Python Libraries |
| active | `compare_eigenvalue_full_pt_vs_matlab.py` | reads .mat/HDF5-MAT, uses MATLAB-style dataset field names | PT data |
| active | `compare_geometries.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare geometries between two MATLAB .mat files |
| active | `compare_infer_plot_debug_vs_matlab.py` | reads .mat/HDF5-MAT | Compare infer-plot debug intermediates (Python) against MATLAB debug artifacts. |
| active | `compare_material_properties.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare material property values (E, nu, rho, t) between Python and MATLAB. |
| active | `compare_matlab_files.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Compare original and reconstructed MATLAB .mat files |
| active | `compare_plot_debug_outputs.py` | reads .mat/HDF5-MAT | Compare Python/MATLAB plot-debug intermediates and final plot points. |
| active | `compare_plot_points.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare plot point locations between Python and MATLAB scripts. |
| active | `compare_with_saved_matlab_K.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Compare Python K and M matrices with saved MATLAB matrices using MATLAB design and const. |
| active | `crossflow_function_level_tests.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | ref/tgt shape: (N_dof, N_wv, N_eig) |
| active | `crossflow_native_audit.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Save MATLAB-native copy (.mat) |
| active | `detailed_comparison.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Detailed comparison of Python and MATLAB plot points. |
| active | `generate_dispersion_dataset_Han_Alex.py` | writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Python translation of generate_dispersion_dataset_Han_Alex.m |
| active | `generate_dispersion_dataset_fixed_geometry.py` | reads .mat/HDF5-MAT | Deterministic Python generation flow for fixed-geometry equivalence checks. |
| active | `generate_matlab_inventory_md.py` | reads .mat/HDF5-MAT, invokes MATLAB runtime, uses MATLAB-style dataset field names | No top-level docstring/comment; likely utility or bridge script. |
| active | `generate_p4mm_fixed10_crossflow.py` | writes .mat/HDF5-MAT | Build deterministic p4mm geometry bank (single-channel values in [0,1]). |
| active | `mat_data_utils.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Find the .mat file in the specified folder |
| active | `matlab_to_reduced_pt.py` | uses MATLAB-style dataset field names | MATLAB to Reduced PyTorch Dataset Converter |
| active | `pt_tensor_to_mat.py` | writes .mat/HDF5-MAT | Best effort for non-tensor objects. |
| active | `reduced_pt_to_matlab.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Reduced PyTorch to MATLAB Dataset Converter (Inverse Function) |
| active | `run_fixed_geometry_crossflow_equivalence.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Orchestrate fixed-geometry MATLAB/Python generation and strict comparison. |
| active | `tensor_to_eigenvector_mat.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | Script to convert a tensor of shape (4, n, 32, 32) to EIGENVECTOR_DATA.mat file. |
| obsolete | `obsolete/2d-dispersion-py/tests/test_plotting.py` | writes .mat/HDF5-MAT | Unit tests for plotting functions. |
| obsolete | `obsolete/2d-dispersion-py/tests/test_utils.py` | uses MATLAB-style dataset field names | Unit tests for utility functions. |
| obsolete | `obsolete/compare_eigenvalue_full_pt_vs_matlab_debug.py` | reads .mat/HDF5-MAT, uses MATLAB-style dataset field names | DEBUG ONLY: delete after debugging is complete. |
| obsolete | `obsolete/compare_intermediates_fixed_geometry_debug.py` | reads .mat/HDF5-MAT | DEBUG ONLY: delete after debugging is complete. |
| obsolete | `obsolete/compare_saved_outputs_full_fixed_debug.py` | reads .mat/HDF5-MAT, uses MATLAB-style dataset field names | DEBUG ONLY: delete after debugging is complete. |
| obsolete | `obsolete/generate_dispersion_dataset_Han_Alex_fixed_geometry_debug.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT, uses MATLAB-style dataset field names | DEBUG ONLY: delete after debugging is complete. |
| obsolete | `obsolete/run_fixed_geometry_csv_tests_debug.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | DEBUG ONLY: delete after debugging is complete. |
| obsolete | `obsolete/test_K_matrix_difference_pattern.py` | reads .mat/HDF5-MAT, writes .mat/HDF5-MAT | Analyze the pattern of differences between Python and MATLAB K matrices. |

## All MATLAB `.m` Files

| Scope | MATLAB File | What It Does |
|---|---|---|
| active | `2D-dispersion-mat/OUTPUT/output with system matrices/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| active | `2D-dispersion-mat/apply_p4mm_symmetry.m` | Measure original data range of A |
| active | `2D-dispersion-mat/apply_steel_rubber_paradigm.m` | design should be a N_pix matrix (with only one pane) |
| active | `2D-dispersion-mat/cellofsparse_to_full.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/compute_KMT_from_mat.m` | compute_KMT_from_mat.m |
| active | `2D-dispersion-mat/convert_design.m` | initial_format and target_format can be 'linear','log','explicit' |
| active | `2D-dispersion-mat/demo_create_Kr_and_Mr.m` | data_fn = "generate_dispersion_dataset_Han\OUTPUT\output 15-Sep-2025 15-36-03\continuous 15-Sep-2025 15-36-03.mat"; |
| active | `2D-dispersion-mat/design_parameters.m` | design_number |
| active | `2D-dispersion-mat/dispersion.m` | if const.N_eig>N_dof |
| active | `2D-dispersion-mat/dispersion_with_matrix_save_opt.m` | if const.N_eig>N_dof |
| active | `2D-dispersion-mat/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| active | `2D-dispersion-mat/generate_dispersion_dataset_Han.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| active | `2D-dispersion-mat/get_IBZ_contour_wavevectors.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/get_IBZ_wavevectors.m` | IBZ_shape = 'square'; |
| active | `2D-dispersion-mat/get_design.m` | Dispersive cell - Tetragonal |
| active | `2D-dispersion-mat/get_design2.m` | switch design_parameters.property_coupling |
| active | `2D-dispersion-mat/get_element_mass.m` | dof are [ u1 v1 u2 v2 u3 v3 u4 v4 ] (indexing starts with lower left |
| active | `2D-dispersion-mat/get_element_mass_VEC.m` | dof are [ u1 v1 u2 v2 u3 v3 u4 v4 ] (indexing starts with lower left |
| active | `2D-dispersion-mat/get_element_stiffness.m` | dof are [ u1 v1 u2 v2 u3 v3 u4 v4 ] (indexing starts with lower left |
| active | `2D-dispersion-mat/get_element_stiffness_VEC.m` | dof are [ u1 v1 u2 v2 u3 v3 u4 v4 ] (indexing starts with lower left |
| active | `2D-dispersion-mat/get_global_idxs.m` | local node 1 |
| active | `2D-dispersion-mat/get_mask.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/get_mesh.m` | Dimensionality of physical space |
| active | `2D-dispersion-mat/get_pixel_properties.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/get_prop.m` | case 'matern52' |
| active | `2D-dispersion-mat/get_system_matrices.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/get_system_matrices_VEC.m` | % Linear quadrilateral elements |
| active | `2D-dispersion-mat/get_system_matrices_VEC_simplified.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/get_transformation_matrix.m` | should we change to ndgrid? they are the unch_idxs --> node_idx_x = [reshape(ndgrid(1:(N_node-1),1:(N_node - 1)),[],1)' N_node*ones(1,N_node - 1) 1:(N_node-1) N_node]; |
| active | `2D-dispersion-mat/gridMesh.m` | MATLAB script/function (no header comment). |
| active | `2D-dispersion-mat/init_storage.m` | Initialize storage arrays |
| active | `2D-dispersion-mat/kernel_prop.m` | scatter(points(:,1),points(:,2)) |
| active | `2D-dispersion-mat/make_chunks.m` | MAKECHUNKS Split 1:N into consecutive chunks of length <= M. |
| active | `2D-dispersion-mat/matern52.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/matern52_kernel.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/matern52_prop.m` | scatter(points(:,1),points(:,2)) |
| active | `2D-dispersion-mat/periodic_kernel.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/periodic_kernel_not_squared.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/plot_design.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/plot_dispersion.m` | dispersion_library_path = '../../'; |
| active | `2D-dispersion-mat/plot_dispersion_contour.m` | zlabel(ax,'\omega') |
| active | `2D-dispersion-mat/plot_dispersion_debug_v.m` | dispersion_library_path = '../../'; |
| active | `2D-dispersion-mat/plot_dispersion_from_predictions.m` | plot_dispersion_from_predictions.m |
| active | `2D-dispersion-mat/plot_dispersion_only.m` | dispersion_library_path = '../../'; |
| active | `2D-dispersion-mat/plot_eigenvector.m` | demonstrates how eigenvectors are indexed (dof ordering) |
| active | `2D-dispersion-mat/plot_mesh.m` | inds = cell(1,mesh.dim); |
| active | `2D-dispersion-mat/plot_mode.m` | mesh = get_mesh(const); |
| active | `2D-dispersion-mat/plot_mode_ui.m` | axes(ax); |
| active | `2D-dispersion-mat/plot_node_labels.m` | MATLAB function file (no header comment). |
| active | `2D-dispersion-mat/plot_wavevectors.m` | axis([-pi/const.a pi/const.a -pi/const.a pi/const.a]) |
| active | `2D-dispersion-mat/run_plot_binarized.m` | Wrapper script to run plot_dispersion.m on binarized dataset |
| active | `2D-dispersion-mat/run_plot_continuous.m` | Wrapper script to run plot_dispersion.m on continuous dataset |
| active | `2D-dispersion-mat/trace_K_construction_matlab.m` | trace_K_construction_matlab.m |
| active | `2D-dispersion-mat/visualize_designs.m` | Generate designs |
| active | `2D-dispersion-mat/visualize_mesh_ordering.m` | Generate coordinates of nodes |
| active | `analyze_matlab_precision_chain.m` | analyze_matlab_precision_chain.m |
| active | `data/out_test_10_mat_original/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| active | `debug_K_matrix_assembly_matlab.m` | Detailed MATLAB debugging script to generate intermediate values |
| active | `example_visualizations.m` | Visualization options |
| active | `generate_dispersion_dataset.m` | Output flags |
| active | `generate_dispersion_dataset_Han_Alex.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| active | `generate_dispersion_dataset_fixed_geometry.m` | Deterministic MATLAB generation flow for fixed-geometry equivalence checks. |
| active | `generate_dispersion_from_prescribed_geometries.m` | Generate MATLAB dispersion outputs from prescribed geometries loaded from |
| active | `p4mm_symmetry_demo.m` | MATLAB script/function (no header comment). |
| active | `plot_dispersion_mod.m` | flags |
| active | `run_matlab_case2_with_eigenfrequencies.m` | MATLAB script to run plot_dispersion.m for Case 2 (with eigenfrequencies) |
| active | `run_matlab_scripts.m` | Script to run MATLAB dispersion plotting scripts and generate plot_points.mat files |
| active | `test_matlab/OUTPUT/output 12-Jan-2026 12-22-04/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| active | `test_matlab/finalize_conversion_to_matlab.m` | MATLAB script/function (no header comment). |
| active | `unpack_struct.m` | MATLAB function file (no header comment). |
| obsolete | `obsolete/2D-dispersion-mat/check_contour_analysis.m` | Create list of wavevectors that define the IBZ contour |
| obsolete | `obsolete/check_matlab_interpolation_inputs.m` | Check what inputs MATLAB uses for scatteredInterpolant |
| obsolete | `obsolete/data_old_251030/OUTPUT/test dataset/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| obsolete | `obsolete/data_old_251030/OUTPUT/train dataset 1/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| obsolete | `obsolete/data_old_251030/OUTPUT/train dataset 2/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| obsolete | `obsolete/data_old_251030/OUTPUT/train dataset 3/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| obsolete | `obsolete/data_old_251030/OUTPUT/train dataset 4/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| obsolete | `obsolete/data_old_251030/OUTPUT/train dataset 5/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| obsolete | `obsolete/data_old_251030/OUTPUT/train dataset 6/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| obsolete | `obsolete/data_old_251030/OUTPUT/train dataset 7/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| obsolete | `obsolete/data_old_251030/OUTPUT/train dataset 8/ex_dispersion_batch_save.m` | dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex'; |
| obsolete | `obsolete/data_old_251030/br_5n_mat/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b10_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b11_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b12_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b13_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b14_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b15_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b16_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b17_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b18_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b19_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b1_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b20_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b2_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b3_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b4_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b5_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b6_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b7_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b8_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_b9_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_br_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c10_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c11_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c12_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c13_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c14_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c15_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c16_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c17_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c18_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c19_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c1_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c20_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c2_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c3_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c4_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c5_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c6_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c7_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c8_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_c9_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/data_old_251030/set_cr_1200n/generate_dispersion_dataset_Han.m` | Output flags |
| obsolete | `obsolete/dispersion tests/generate_dispersion_dataset_Han.m` | dispersion_library_path = 'C:\Users\alex\OneDrive - California Institute of Technology\Documents\Graduate\Research\2D-dispersion'; |
| obsolete | `obsolete/generate_dispersion_dataset_Han_check.m` | Output flags |
| obsolete | `obsolete/generate_dispersion_from_prescribed_geometries_debug.m` | DEBUG ONLY: delete after debugging is complete. |
| obsolete | `obsolete/plot_dispersion_dataset_debug.m` | DEBUG ONLY: save intermediate plotting arrays for parity comparison. |
| obsolete | `obsolete/test_batch_output.m` | Run your own clear, then manually load whatever mat file |
| obsolete | `obsolete/test_check_matlab_get_element_stiffness_output.m` | Check what get_element_stiffness_VEC actually returns |
| obsolete | `obsolete/test_matlab_K_matrix_detailed.m` | Detailed MATLAB test script for K matrix computation |
| obsolete | `obsolete/test_matlab_contour_behavior.m` | Test MATLAB's get_IBZ_contour_wavevectors behavior |
| obsolete | `obsolete/test_matlab_functions.m` | Test script for MATLAB 2D-dispersion-han library |
| obsolete | `obsolete/test_matlab_get_element_stiffness_shape.m` | Test what shape get_element_stiffness_VEC returns for vectorized inputs |
| obsolete | `obsolete/test_matlab_step_by_step.m` | Test script to compare intermediate variables step-by-step with Python |
| obsolete | `obsolete/test_matlab_transpose_behavior.m` | Test MATLAB transpose behavior on 3D arrays |
| obsolete | `obsolete/test_matlab_vectorized_stiffness.m` | Test how MATLAB's get_element_stiffness handles vectorized inputs |
| obsolete | `obsolete/test_plot_dispersion_contour_matlab.m` | MATLAB script to generate and save plot_dispersion_contour data for comparison |
| obsolete | `obsolete/test_plot_eigenvector_components_matlab.m` | MATLAB script to generate and save plot_eigenvector_components data for comparison |
| obsolete | `obsolete/test_plot_mode_matlab.m` | MATLAB script to generate and save plot_mode data for comparison |

## Notes

- `active` means outside `obsolete/`; `obsolete` means already archived under `obsolete/`.
- This is a static code scan based on file extension and MATLAB-related patterns.
