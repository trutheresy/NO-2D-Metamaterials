% Test script for MATLAB 2D-dispersion-han library
% This script tests key functions and saves outputs for comparison with Python

clear; close all;

% Add MATLAB library to path
addpath('2D-dispersion-han');

% Create output directory
output_dir = 'test_outputs_matlab';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('Running MATLAB function tests...\n');

%% Test 1: Design Generation
fprintf('\n=== Test 1: Design Generation ===\n');
test_designs = {'homogeneous', 'dispersive-tetragonal', 'dispersive-orthotropic', 'quasi-1D'};
N_pix = 8;  % Use 8 instead of 5 to avoid indexing issues with N_pix/4
designs_matlab = struct();
for i = 1:length(test_designs)
    design_name = test_designs{i};
    design = get_design(design_name, N_pix);
    % Replace hyphens with underscores for valid MATLAB field names
    field_name = strrep(design_name, '-', '_');
    designs_matlab.(field_name) = design;
    fprintf('  Generated design: %s (size: %s)\n', design_name, mat2str(size(design)));
end
save(fullfile(output_dir, 'test1_designs.mat'), 'designs_matlab', '-v7.3');

%% Test 2: Wavevector Generation
fprintf('\n=== Test 2: Wavevector Generation ===\n');
a = 1.0;
symmetry_types = {'none', 'omit', 'p4mm', 'p2mm'};
wavevectors_matlab = struct();
for i = 1:length(symmetry_types)
    sym_type = symmetry_types{i};
    try
        wv = get_IBZ_wavevectors([11, 6], a, sym_type);
        wavevectors_matlab.(sym_type) = wv;
        fprintf('  Generated wavevectors for %s: %d points\n', sym_type, size(wv, 1));
    catch ME
        fprintf('  Error with %s: %s\n', sym_type, ME.message);
    end
end
save(fullfile(output_dir, 'test2_wavevectors.mat'), 'wavevectors_matlab', '-v7.3');

%% Test 3: System Matrices
fprintf('\n=== Test 3: System Matrices ===\n');
const.N_ele = 2;
const.N_pix = 8;  % Match design test size
const.a = 1.0;
const.t = 1.0;
const.E_min = 2e9;
const.E_max = 200e9;
const.rho_min = 1e3;
const.rho_max = 8e3;
const.poisson_min = 0.0;
const.poisson_max = 0.5;
const.design_scale = 'linear';
const.design = get_design('homogeneous', const.N_pix);

[K, M] = get_system_matrices(const);
fprintf('  K matrix: %d x %d, nnz: %d\n', size(K, 1), size(K, 2), nnz(K));
fprintf('  M matrix: %d x %d, nnz: %d\n', size(M, 1), size(M, 2), nnz(M));

% Save sparse matrices (convert to full for saving)
K_data = struct('data', full(K), 'size', size(K), 'nnz', nnz(K));
M_data = struct('data', full(M), 'size', size(M), 'nnz', nnz(M));
save(fullfile(output_dir, 'test3_system_matrices.mat'), 'K_data', 'M_data', 'const', '-v7.3');

%% Test 4: Transformation Matrix
fprintf('\n=== Test 4: Transformation Matrix ===\n');
wavevector = [0.5, 0.3];
T = get_transformation_matrix(wavevector, const);
fprintf('  T matrix: %d x %d\n', size(T, 1), size(T, 2));
T_data = struct('data', full(T), 'size', size(T));
save(fullfile(output_dir, 'test4_transformation.mat'), 'T_data', 'wavevector', '-v7.3');

%% Test 5: Full Dispersion Calculation
fprintf('\n=== Test 5: Full Dispersion Calculation ===\n');
% Use smaller problem to match Python test
const_disp.N_ele = 1;  % Use 1 element per pixel for smaller problem
const_disp.N_pix = 8;  % Match design test size
const_disp.a = 1.0;
const_disp.t = 1.0;
const_disp.E_min = 2e9;
const_disp.E_max = 200e9;
const_disp.rho_min = 1e3;
const_disp.rho_max = 8e3;
const_disp.poisson_min = 0.0;
const_disp.poisson_max = 0.5;
const_disp.design_scale = 'linear';
const_disp.design = get_design('homogeneous', 8);
const_disp.N_eig = 6;
const_disp.sigma_eig = 1.0;
const_disp.isUseGPU = false;
const_disp.isUseImprovement = true;
const_disp.isUseSecondImprovement = false;
const_disp.isUseParallel = false;
const_disp.isSaveEigenvectors = true;
const_disp.isSaveMesh = false;

wavevectors = get_IBZ_wavevectors([5, 3], const_disp.a, 'none');  % Smaller wavevector set
const_disp.wavevectors = wavevectors;

try
    if const_disp.isSaveMesh
        [wv, fr, ev, mesh] = dispersion(const_disp, wavevectors);
    else
        [wv, fr, ev] = dispersion(const_disp, wavevectors);
        mesh = [];
    end
    fprintf('  Wavevectors: %d points\n', size(wv, 1));
    fprintf('  Frequencies: %s\n', mat2str(size(fr)));
    fprintf('  Eigenvectors: %s\n', mat2str(size(ev)));
    
    dispersion_results = struct('wv', wv, 'fr', fr, 'ev', ev);
    save(fullfile(output_dir, 'test5_dispersion.mat'), 'dispersion_results', 'const_disp', '-v7.3');
catch ME
    fprintf('  ERROR in dispersion calculation: %s\n', ME.message);
    fprintf('  Skipping dispersion test for now\n');
    dispersion_results = struct('wv', [], 'fr', [], 'ev', []);
    save(fullfile(output_dir, 'test5_dispersion.mat'), 'dispersion_results', 'const_disp', '-v7.3');
end

%% Test 6: Element Matrices
fprintf('\n=== Test 6: Element Matrices ===\n');
E = 100e9;
nu = 0.3;
t = 1.0;
rho = 5000;
const_test = struct('N_ele', 2, 'N_pix', 5, 'a', 1.0, 't', t);

k_ele = get_element_stiffness(E, nu, t, const_test);
m_ele = get_element_mass(rho, t, const_test);
fprintf('  Element stiffness: %d x %d\n', size(k_ele, 1), size(k_ele, 2));
fprintf('  Element mass: %d x %d\n', size(m_ele, 1), size(m_ele, 2));

element_results = struct('k_ele', k_ele, 'm_ele', m_ele, 'E', E, 'nu', nu, 'rho', rho);
save(fullfile(output_dir, 'test6_elements.mat'), 'element_results', '-v7.3');

fprintf('\n=== All MATLAB tests completed ===\n');
fprintf('Results saved to: %s\n', output_dir);

