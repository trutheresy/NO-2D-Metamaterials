% Detailed MATLAB debugging script to generate intermediate values
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

fprintf('Design shape: %s\n', mat2str(size(design_3d)));

% Set up const
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
fprintf('  N_pix = %d, N_ele = %d, t = %.6f\n', const.N_pix, const.N_ele, const.t);

% Step-by-step computation matching Python
N_ele_x = const.N_pix * const.N_ele;
N_ele_y = const.N_pix * const.N_ele;

% Expand design
const.design = repelem(const.design, const.N_ele, const.N_ele, 1);

% Extract material properties
E = (const.E_min + const.design(:,:,1).*(const.E_max - const.E_min))';
nu = (const.poisson_min + const.design(:,:,3).*(const.poisson_max - const.poisson_min))';
t = const.t';

% Node numbering
nodenrs = reshape(1:(1+N_ele_x)*(1+N_ele_y),1+N_ele_y,1+N_ele_x);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)-1,N_ele_x*N_ele_y,1);

% Offset array
offset_array = [2*(N_ele_y+1)+[0 1 2 3] 2 3 0 1];
edofMat = repmat(edofVec,1,8)+repmat(offset_array,N_ele_x*N_ele_y,1);

% Row and column indices
row_idxs = reshape(kron(edofMat,ones(8,1))',64*N_ele_x*N_ele_y,1);
col_idxs = reshape(kron(edofMat,ones(1,8))',64*N_ele_x*N_ele_y,1);

% Element stiffness matrices
% MATLAB: get_element_stiffness_VEC(E(:),nu(:),t) returns (N_ele, 64)
AllLEle = get_element_stiffness_VEC(E(:),nu(:),t)';  % Transpose: (64, N_ele)
value_K = AllLEle(:);  % Flatten column-wise (interleaves elements)

fprintf('\nElement matrix info:\n');
fprintf('  AllLEle_raw shape: %s\n', mat2str(size(AllLEle_raw)));
fprintf('  AllLEle shape: %s\n', mat2str(size(AllLEle)));
fprintf('  value_K length: %d\n', length(value_K));
fprintf('  value_K range: [%.6e, %.6e]\n', min(value_K(:)), max(value_K(:)));

% Compute K matrix
K = sparse(row_idxs,col_idxs,value_K);

fprintf('\nComputed K matrix:\n');
fprintf('  Shape: %s, nnz: %d\n', mat2str(size(K)), nnz(K));

% Save intermediate values
save('debug_K_intermediates_matlab.mat', ...
    'row_idxs', 'col_idxs', 'value_K', ...
    'AllLEle_raw', 'AllLEle', ...
    'edofMat', 'offset_array', ...
    'E', 'nu', 't', ...
    '-v7');

fprintf('\nSaved intermediate values to debug_K_intermediates_matlab.mat\n');

