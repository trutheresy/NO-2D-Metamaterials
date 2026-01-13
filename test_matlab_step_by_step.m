% Test script to compare intermediate variables step-by-step with Python
% Based on get_system_matrices_VEC_simplified.m

clear; close all;

% Load data
data_dir = 'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10';
mat_file = fullfile(data_dir, 'out_binarized_1.mat');

% Load from .mat file
fprintf('Loading data from: %s\n', mat_file);
data = load(mat_file);

struct_idx = 1; % MATLAB uses 1-based indexing

% Extract design
if isfield(data, 'designs')
    if ndims(data.designs) == 4
        design_3d = data.designs(:,:,:,struct_idx);
    else
        design_3d = data.designs(:,:,:);
    end
else
    error('designs not found in .mat file');
end

fprintf('\nDesign shape: %s\n', mat2str(size(design_3d)));

% Set up const (matching ex_dispersion_batch_save.m)
const = struct();
const.N_pix = size(design_3d, 1);
const.N_ele = 1;
const.a = 1;
const.design = design_3d;
const.design_scale = 'linear';
const.E_min = 20e6;
const.E_max = 200e9;
const.rho_min = 1200;  % NOTE: Different from Python default!
const.rho_max = 8e3;
const.poisson_min = 0;
const.poisson_max = 0.5;
const.t = 1;  % NOTE: Different from Python default (0.01)!

fprintf('\nconst parameters:\n');
fprintf('  N_pix = %d\n', const.N_pix);
fprintf('  N_ele = %d\n', const.N_ele);
fprintf('  a = %.6f\n', const.a);
fprintf('  E: [%.6e, %.6e]\n', const.E_min, const.E_max);
fprintf('  rho: [%.2f, %.2f]\n', const.rho_min, const.rho_max);
fprintf('  nu: [%.2f, %.2f]\n', const.poisson_min, const.poisson_max);
fprintf('  t = %.6f\n', const.t);

% Step 1: Expand design
fprintf('\n=== STEP 1: Expand design ===\n');
const.design = repelem(const.design, const.N_ele, const.N_ele, 1);
fprintf('Expanded design shape: %s\n', mat2str(size(const.design)));

% Step 2: Extract material properties
fprintf('\n=== STEP 2: Extract material properties ===\n');
if strcmp(const.design_scale, 'linear')
    E = (const.E_min + const.design(:,:,1).*(const.E_max - const.E_min))';
    nu = (const.poisson_min + const.design(:,:,3).*(const.poisson_max - const.poisson_min))';
    t = const.t';
    rho = (const.rho_min + const.design(:,:,2).*(const.rho_max - const.rho_min))';
else
    error('Only linear design_scale supported');
end

fprintf('E: shape=%s, min=%.6e, max=%.6e, mean=%.6e\n', ...
    mat2str(size(E)), min(E(:)), max(E(:)), mean(E(:)));
fprintf('nu: shape=%s, min=%.6f, max=%.6f, mean=%.6f\n', ...
    mat2str(size(nu)), min(nu(:)), max(nu(:)), mean(nu(:)));
fprintf('rho: shape=%s, min=%.2f, max=%.2f, mean=%.2f\n', ...
    mat2str(size(rho)), min(rho(:)), max(rho(:)), mean(rho(:)));
fprintf('t: %.6f\n', t);

% Step 3: Compute element size
fprintf('\n=== STEP 3: Compute element size ===\n');
N_ele_x = const.N_pix * const.N_ele;
N_ele_y = const.N_pix * const.N_ele;
element_size = const.a / (const.N_ele * const.N_pix);
element_area = element_size^2;
fprintf('N_ele_x = %d, N_ele_y = %d\n', N_ele_x, N_ele_y);
fprintf('element_size = %.6e m\n', element_size);
fprintf('element_area = %.6e m²\n', element_area);

% Step 4: Get element matrices for first element
fprintf('\n=== STEP 4: Compute element matrices (first element) ===\n');
E_first = E(1, 1);
nu_first = nu(1, 1);
rho_first = rho(1, 1);
t_first = t;

fprintf('First element material properties:\n');
fprintf('  E = %.6e Pa\n', E_first);
fprintf('  nu = %.6f\n', nu_first);
fprintf('  rho = %.2f kg/m³\n', rho_first);
fprintf('  t = %.6f m\n', t_first);

% Get element stiffness matrix
k_ele = get_element_stiffness_VEC(E_first, nu_first, t_first);
fprintf('\nElement stiffness matrix:\n');
fprintf('  Shape: %s\n', mat2str(size(k_ele)));
fprintf('  Min: %.6e, Max: %.6e, Mean: %.6e\n', ...
    min(k_ele(:)), max(k_ele(:)), mean(k_ele(:)));
fprintf('  Sample (first 3x3):\n');
disp(k_ele(1:3, 1:3));

% Get element mass matrix
m_ele = get_element_mass(rho_first, t_first, const);
fprintf('\nElement mass matrix:\n');
fprintf('  Shape: %s\n', mat2str(size(m_ele)));
fprintf('  Min: %.6e, Max: %.6e, Mean: %.6e\n', ...
    min(m_ele(:)), max(m_ele(:)), mean(m_ele(:)));
fprintf('  Sample (first 3x3):\n');
disp(m_ele(1:3, 1:3));

% Step 5: Compute system matrices
fprintf('\n=== STEP 5: Compute system matrices ===\n');
[K, M] = get_system_matrices_VEC_simplified(const);

fprintf('K matrix: shape=%s, nnz=%d\n', mat2str(size(K)), nnz(K));
fprintf('  Min: %.6e, Max: %.6e, Mean: %.6e\n', ...
    min(K(K~=0)), max(K(:)), mean(K(K~=0)));

fprintf('M matrix: shape=%s, nnz=%d\n', mat2str(size(M)), nnz(M));
fprintf('  Min: %.6e, Max: %.6e, Mean: %.6e\n', ...
    min(M(M~=0)), max(M(:)), mean(M(M~=0)));

% Save intermediate variables for comparison
save('test_matlab_intermediates.mat', ...
    'const', 'E', 'nu', 'rho', 't', ...
    'element_size', 'element_area', ...
    'k_ele', 'm_ele', 'K', 'M', ...
    'E_first', 'nu_first', 'rho_first', 't_first');

fprintf('\n✅ Saved intermediate variables to test_matlab_intermediates.mat\n');

