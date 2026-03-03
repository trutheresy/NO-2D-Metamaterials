% Detailed MATLAB test script for K matrix computation
% Based on get_system_matrices_VEC_simplified.m
% Saves all intermediate variables for comparison

clear; close all;

% Load data
data_dir = 'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10';
mat_file = fullfile(data_dir, 'out_binarized_1.mat');

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
const.rho_min = 1200;
const.rho_max = 8e3;
const.poisson_min = 0;
const.poisson_max = 0.5;
const.t = 1;

fprintf('\nconst parameters:\n');
fprintf('  N_pix = %d, N_ele = %d, a = %.6f\n', const.N_pix, const.N_ele, const.a);
fprintf('  E: [%.6e, %.6e]\n', const.E_min, const.E_max);
fprintf('  rho: [%.2f, %.2f]\n', const.rho_min, const.rho_max);
fprintf('  nu: [%.2f, %.2f]\n', const.poisson_min, const.poisson_max);
fprintf('  t = %.6f\n', const.t);

% Step 1: Expand design
fprintf('\n=== STEP 1: Expand design ===\n');
const.design = repelem(const.design, const.N_ele, const.N_ele, 1);
fprintf('Expanded design shape: %s\n', mat2str(size(const.design)));
fprintf('Design range: [%.6f, %.6f]\n', min(const.design(:)), max(const.design(:)));

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

% Step 3: Compute element size and indices
fprintf('\n=== STEP 3: Compute indices ===\n');
N_ele_x = const.N_pix * const.N_ele;
N_ele_y = const.N_pix * const.N_ele;
fprintf('N_ele_x = %d, N_ele_y = %d\n', N_ele_x, N_ele_y);

nodenrs = reshape(1:(1+N_ele_x)*(1+N_ele_y),1+N_ele_y,1+N_ele_x);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)-1,N_ele_x*N_ele_y,1);
offset_array = [2 3 2*(N_ele_x+1)+[2 3 0 1] 0 1];
edofMat = repmat(edofVec,1,8)+repmat(offset_array,N_ele_x*N_ele_y,1);

fprintf('edofVec shape: %s, range: [%d, %d]\n', mat2str(size(edofVec)), min(edofVec(:)), max(edofVec(:)));
fprintf('edofMat shape: %s, range: [%d, %d]\n', mat2str(size(edofMat)), min(edofMat(:)), max(edofMat(:)));

row_idxs = reshape(kron(edofMat,ones(8,1))',64*N_ele_x*N_ele_y,1);
col_idxs = reshape(kron(edofMat,ones(1,8))',64*N_ele_x*N_ele_y,1);

fprintf('row_idxs shape: %s, range: [%d, %d]\n', mat2str(size(row_idxs)), min(row_idxs(:)), max(row_idxs(:)));
fprintf('col_idxs shape: %s, range: [%d, %d]\n', mat2str(size(col_idxs)), min(col_idxs(:)), max(col_idxs(:)));

% Step 4: Get element stiffness matrices
fprintf('\n=== STEP 4: Compute element stiffness matrices ===\n');
AllLEle = get_element_stiffness_VEC(E(:),nu(:),t)';
fprintf('AllLEle shape: %s\n', mat2str(size(AllLEle)));
fprintf('AllLEle range: [%.6e, %.6e], mean: %.6e\n', ...
    min(AllLEle(:)), max(AllLEle(:)), mean(AllLEle(:)));

% Check first element
E_first = E(1, 1);
nu_first = nu(1, 1);
t_first = t;
k_ele_first = get_element_stiffness_VEC(E_first, nu_first, t_first);
fprintf('\nFirst element (E=%.6e, nu=%.6f, t=%.6f):\n', E_first, nu_first, t_first);
fprintf('  k_ele shape: %s\n', mat2str(size(k_ele_first)));
fprintf('  k_ele range: [%.6e, %.6e], mean: %.6e\n', ...
    min(k_ele_first(:)), max(k_ele_first(:)), mean(k_ele_first(:)));
fprintf('  k_ele sample (first 3x3):\n');
disp(k_ele_first(1:3, 1:3));

% Step 5: Flatten and assemble
fprintf('\n=== STEP 5: Flatten and assemble K matrix ===\n');
value_K = AllLEle(:);
fprintf('value_K shape: %s\n', mat2str(size(value_K)));
fprintf('value_K range: [%.6e, %.6e], mean: %.6e\n', ...
    min(value_K(:)), max(value_K(:)), mean(value_K(:)));
fprintf('value_K nnz: %d\n', nnz(value_K));

K = sparse(row_idxs,col_idxs,value_K);
fprintf('\nAssembled K matrix:\n');
fprintf('  Shape: %s, nnz: %d\n', mat2str(size(K)), nnz(K));
K_dense = full(K);
K_nonzero = K_dense(K_dense ~= 0);
fprintf('  Non-zero values: min=%.6e, max=%.6e, mean=%.6e\n', ...
    min(K_nonzero(:)), max(K_nonzero(:)), mean(K_nonzero(:)));
fprintf('  All values: min=%.6e, max=%.6e, mean=%.6e\n', ...
    min(K_dense(:)), max(K_dense(:)), mean(K_dense(:)));

% Save intermediate variables
save('test_matlab_K_intermediates.mat', ...
    'const', 'E', 'nu', 'rho', 't', ...
    'nodenrs', 'edofVec', 'edofMat', 'row_idxs', 'col_idxs', ...
    'AllLEle', 'value_K', 'K', 'K_dense', ...
    'E_first', 'nu_first', 't_first', 'k_ele_first');

fprintf('\n✅ Saved intermediate variables to test_matlab_K_intermediates.mat\n');

