% analyze_matlab_precision_chain.m
% 
% Script to check the precision (class) of variables in MATLAB's K and M computation chain

clear; close all;

% Add path to MATLAB functions
addpath('D:\Research\NO-2D-Metamaterials\2D-dispersion-han');

% Load test data
input_mat_file = "D:\Research\NO-2D-Metamaterials\data\KMT_py_matlab_reconstruction_test\mat\pt.mat";

fprintf('=========================================================================\n');
fprintf('MATLAB K and M Matrix Computation Precision Analysis\n');
fprintf('=========================================================================\n');

% Load data
info = h5info(input_mat_file);
data = struct();

for i = 1:length(info.Datasets)
    ds_name = info.Datasets(i).Name;
    if ~strncmp(ds_name, '#', 1)
        ds_data = h5read(input_mat_file, ['/' ds_name]);
        if isstruct(ds_data) && isfield(ds_data, 'real') && isfield(ds_data, 'imag')
            data.(ds_name) = complex(ds_data.real, ds_data.imag);
        else
            data.(ds_name) = ds_data;
        end
    end
end

group_names = {};
if ~isempty(info.Groups)
    group_names = {info.Groups.Name};
end
if any(contains(group_names, 'const'))
    const_group_idx = find(contains(group_names, 'const'), 1);
    const_group_path = ['/' info.Groups(const_group_idx).Name];
    const_info = h5info(input_mat_file, const_group_path);
    data.const = struct();
    
    if ~isempty(const_info.Datasets)
        for i = 1:length(const_info.Datasets)
            ds_name = const_info.Datasets(i).Name;
            ds_data = h5read(input_mat_file, [const_group_path '/' ds_name]);
            if isa(ds_data, 'uint16') && ndims(ds_data) <= 2
                data.const.(ds_name) = char(ds_data(:)');
            else
                data.const.(ds_name) = ds_data;
            end
        end
    end
    
    if ~isempty(const_info.Attributes)
        for i = 1:length(const_info.Attributes)
            attr_name = const_info.Attributes(i).Name;
            attr_data = h5readatt(input_mat_file, const_group_path, attr_name);
            data.const.(attr_name) = attr_data;
        end
    end
end

% Extract first structure for testing
struct_idx = 1;
if ndims(data.designs) == 4
    design_3d = data.designs(:, :, :, struct_idx);
else
    design_3d = data.designs(:, :, :);
end

const = data.const;
const.design = design_3d;
const.N_pix = size(design_3d, 1);
const.N_ele = 1;

fprintf('\n1. Material Property Extraction:\n');
const.design = repelem(const.design, const.N_ele, const.N_ele, 1);
fprintf('   const.design class: %s\n', class(const.design));

if strcmp(const.design_scale, 'linear')
    E = (const.E_min + const.design(:,:,1).*(const.E_max - const.E_min))';
    nu = (const.poisson_min + const.design(:,:,3).*(const.poisson_max - const.poisson_min))';
    t = const.t';
    rho = (const.rho_min + const.design(:,:,2).*(const.rho_max - const.rho_min))';
end

fprintf('   E class: %s\n', class(E));
fprintf('   nu class: %s\n', class(nu));
fprintf('   t class: %s\n', class(t));
fprintf('   rho class: %s\n', class(rho));

fprintf('\n2. Element Matrix Computation:\n');
AllLEle = get_element_stiffness_VEC(E(:), nu(:), t)';
AllLMat = get_element_mass_VEC(rho(:), t, const)';
fprintf('   AllLEle class: %s\n', class(AllLEle));
fprintf('   AllLMat class: %s\n', class(AllLMat));

fprintf('\n3. Matrix Assembly:\n');
N_ele_x = const.N_pix * const.N_ele;
N_ele_y = const.N_pix * const.N_ele;
nodenrs = reshape(1:(1+N_ele_x)*(1+N_ele_y), 1+N_ele_y, 1+N_ele_x);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)-1, N_ele_x*N_ele_y, 1);
edofMat = repmat(edofVec,1,8)+repmat([2*(N_ele_y+1)+[0 1 2 3] 2 3 0 1], N_ele_x*N_ele_y,1);
row_idxs = reshape(kron(edofMat,ones(8,1))', 64*N_ele_x*N_ele_y, 1);
col_idxs = reshape(kron(edofMat,ones(1,8))', 64*N_ele_x*N_ele_y, 1);

value_K = AllLEle(:);
value_M = AllLMat(:);
fprintf('   value_K class: %s\n', class(value_K));
fprintf('   value_M class: %s\n', class(value_M));

K = sparse(row_idxs, col_idxs, value_K);
M = sparse(row_idxs, col_idxs, value_M);
fprintf('   K class: %s\n', class(K));
fprintf('   M class: %s\n', class(M));

% MATLAB sparse matrices store data internally, check by extracting non-zeros
[K_i, K_j, K_vals] = find(K);
[M_i, M_j, M_vals] = find(M);
fprintf('   K non-zero values class: %s\n', class(K_vals));
fprintf('   M non-zero values class: %s\n', class(M_vals));

fprintf('\n=========================================================================\n');
fprintf('Summary:\n');
fprintf('=========================================================================\n');
fprintf('MATLAB defaults:\n');
fprintf('  - Numeric literals (E_min, etc.): double\n');
fprintf('  - Computations (E, nu, rho): Inherit from inputs (likely double)\n');
fprintf('  - Element matrices: Inherit from inputs\n');
fprintf('  - sparse() matrices: Store data as input type (double by default)\n');
fprintf('=========================================================================\n');

