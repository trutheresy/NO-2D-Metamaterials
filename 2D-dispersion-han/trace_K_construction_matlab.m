% trace_K_construction_matlab.m
% 
% Diagnostic script to trace K matrix construction in MATLAB and save all intermediate steps.
% This will be compared with Python to find where discrepancies occur.

clear; close all;

% Load test data
input_mat_file = "D:\Research\NO-2D-Metamaterials\data\KMT_py_matlab_reconstruction_test\mat\pt.mat";
output_dir = "D:\Research\NO-2D-Metamaterials\data\KMT_py_matlab_reconstruction_test\matlab_intermediates";

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('=========================================================================\n');
fprintf('Tracing K Matrix Construction in MATLAB\n');
fprintf('=========================================================================\n');
fprintf('Input:  %s\n', input_mat_file);
fprintf('Output: %s\n', output_dir);
fprintf('\n');

% Load data
fprintf('1. Loading data...\n');
try
    data = load(input_mat_file);
catch
    % Try h5read for HDF5 files
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

fprintf('   Structure %d: design size = %s\n', struct_idx, mat2str(size(design_3d)));
fprintf('\n');

% Now trace through get_system_matrices_VEC step by step
fprintf('2. Tracing K matrix construction...\n');

% Step 1: N_ele_x, N_ele_y
N_ele_x = const.N_pix * const.N_ele;
N_ele_y = const.N_pix * const.N_ele;
fprintf('   Step 1: N_ele_x = %d, N_ele_y = %d\n', N_ele_x, N_ele_y);
save(fullfile(output_dir, 'step1_N_ele.mat'), 'N_ele_x', 'N_ele_y');

% Step 2: Expand design (repelem)
const.design = repelem(const.design, const.N_ele, const.N_ele, 1);
design_expanded = const.design;
fprintf('   Step 2: design_expanded size = %s\n', mat2str(size(design_expanded)));
save(fullfile(output_dir, 'step2_design_expanded.mat'), 'design_expanded');

% Step 3: Extract material properties
if strcmp(const.design_scale, 'linear')
    E = (const.E_min + const.design(:,:,1).*(const.E_max - const.E_min))';
    nu = (const.poisson_min + const.design(:,:,3).*(const.poisson_max - const.poisson_min))';
    t = const.t';
    rho = (const.rho_min + const.design(:,:,2).*(const.rho_max - const.rho_min))';
elseif strcmp(const.design_scale, 'log')
    E = exp(const.design(:,:,1))';
    nu = (const.poisson_min + const.design(:,:,3).*(const.poisson_max - const.poisson_min))';
    t = const.t';
    rho = exp(const.design(:,:,2))';
end
fprintf('   Step 3: Material properties\n');
fprintf('     E: size = %s, range = [%.6e, %.6e]\n', mat2str(size(E)), min(E(:)), max(E(:)));
fprintf('     nu: size = %s, range = [%.6e, %.6e]\n', mat2str(size(nu)), min(nu(:)), max(nu(:)));
fprintf('     rho: size = %s, range = [%.6e, %.6e]\n', mat2str(size(rho)), min(rho(:)), max(rho(:)));
save(fullfile(output_dir, 'step3_material_props.mat'), 'E', 'nu', 't', 'rho');

% Step 4: Node numbering
nodenrs = reshape(1:(1+N_ele_x)*(1+N_ele_y), 1+N_ele_y, 1+N_ele_x);
fprintf('   Step 4: nodenrs size = %s, range = [%d, %d]\n', mat2str(size(nodenrs)), min(nodenrs(:)), max(nodenrs(:)));
save(fullfile(output_dir, 'step4_nodenrs.mat'), 'nodenrs');

% Step 5: edofVec
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)-1, N_ele_x*N_ele_y, 1);
fprintf('   Step 5: edofVec size = %s, range = [%d, %d]\n', mat2str(size(edofVec)), min(edofVec(:)), max(edofVec(:)));
save(fullfile(output_dir, 'step5_edofVec.mat'), 'edofVec');

% Step 6: edofMat
edofMat = repmat(edofVec, 1, 8) + repmat([2*(N_ele_y+1)+[0 1 2 3] 2 3 0 1], N_ele_x*N_ele_y, 1);
fprintf('   Step 6: edofMat size = %s\n', mat2str(size(edofMat)));
fprintf('     First row: %s\n', mat2str(edofMat(1, :)));
save(fullfile(output_dir, 'step6_edofMat.mat'), 'edofMat');

% Step 7: row_idxs and col_idxs
row_idxs = reshape(kron(edofMat, ones(8,1))', 64*N_ele_x*N_ele_y, 1);
col_idxs = reshape(kron(edofMat, ones(1,8))', 64*N_ele_x*N_ele_y, 1);
fprintf('   Step 7: Indices\n');
fprintf('     row_idxs: size = %s, range = [%d, %d]\n', mat2str(size(row_idxs)), min(row_idxs(:)), max(row_idxs(:)));
fprintf('     col_idxs: size = %s, range = [%d, %d]\n', mat2str(size(col_idxs)), min(col_idxs(:)), max(col_idxs(:)));
save(fullfile(output_dir, 'step7_indices.mat'), 'row_idxs', 'col_idxs');

% Step 8: Element matrices
AllLEle = get_element_stiffness_VEC(E(:), nu(:), t)';
AllLMat = get_element_mass_VEC(rho(:), t, const)';
fprintf('   Step 8: Element matrices\n');
fprintf('     AllLEle: size = %s, range = [%.6e, %.6e]\n', mat2str(size(AllLEle)), min(AllLEle(:)), max(AllLEle(:)));
fprintf('     AllLMat: size = %s, range = [%.6e, %.6e]\n', mat2str(size(AllLMat)), min(AllLMat(:)), max(AllLMat(:)));
save(fullfile(output_dir, 'step8_element_matrices.mat'), 'AllLEle', 'AllLMat');

% Step 9: Flattened values
value_K = AllLEle(:);
value_M = AllLMat(:);
fprintf('   Step 9: Flattened values\n');
fprintf('     value_K: size = %s, range = [%.6e, %.6e]\n', mat2str(size(value_K)), min(value_K(:)), max(value_K(:)));
fprintf('     value_M: size = %s, range = [%.6e, %.6e]\n', mat2str(size(value_M)), min(value_M(:)), max(value_M(:)));
save(fullfile(output_dir, 'step9_values.mat'), 'value_K', 'value_M');

% Step 10: Final sparse matrices
K = sparse(row_idxs, col_idxs, value_K);
M = sparse(row_idxs, col_idxs, value_M);
fprintf('   Step 10: Final matrices\n');
fprintf('     K: size = %s, nnz = %d\n', mat2str(size(K)), nnz(K));
fprintf('     M: size = %s, nnz = %d\n', mat2str(size(M)), nnz(M));
save(fullfile(output_dir, 'step10_final_matrices.mat'), 'K', 'M', '-v7.3');

fprintf('\n');
fprintf('=========================================================================\n');
fprintf('All intermediate steps saved to: %s\n', output_dir);
fprintf('=========================================================================\n');

