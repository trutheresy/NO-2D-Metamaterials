% compute_KMT_from_mat.m
% 
% Diagnostic script to compute K, M, T matrices from a .mat file and save them
% for comparison with Python reconstruction.
%
% This script:
% 1. Loads a .mat file with designs and wavevectors
% 2. Computes K, M, T matrices using MATLAB functions
% 3. Saves K, M, T matrices to a new .mat file for comparison

clear; close all;

% Input and output paths
input_mat_file = "D:\Research\NO-2D-Metamaterials\data\KMT_py_matlab_reconstruction_test\mat\pt.mat";
output_mat_file = "D:\Research\NO-2D-Metamaterials\data\KMT_py_matlab_reconstruction_test\mat\computed_K_M_T_matrices.mat";

fprintf('=========================================================================\n');
fprintf('Computing K, M, T Matrices from MATLAB .mat file\n');
fprintf('=========================================================================\n');
fprintf('Input:  %s\n', input_mat_file);
fprintf('Output: %s\n', output_mat_file);
fprintf('\n');

% Load data
fprintf('1. Loading data from .mat file...\n');
try
    data = load(input_mat_file);
    fprintf('   Successfully loaded .mat file\n');
catch ME
    fprintf('   ERROR loading file: %s\n', ME.message);
    fprintf('   Trying h5read...\n');
    try
        % Try h5read for HDF5 files
        info = h5info(input_mat_file);
        data = struct();
        
        % Debug: Print what's in the file
        fprintf('   Top-level datasets: ');
        if ~isempty(info.Datasets)
            fprintf('%s ', info.Datasets.Name);
        end
        fprintf('\n');
        fprintf('   Top-level groups: ');
        if ~isempty(info.Groups)
            fprintf('%s ', info.Groups.Name);
        end
        fprintf('\n');
        
        % Read top-level datasets
        for i = 1:length(info.Datasets)
            ds_name = info.Datasets(i).Name;
            if ~strncmp(ds_name, '#', 1)
                ds_data = h5read(input_mat_file, ['/' ds_name]);
                
                % Handle complex data stored as structured array
                if isstruct(ds_data) && isfield(ds_data, 'real') && isfield(ds_data, 'imag')
                    data.(ds_name) = complex(ds_data.real, ds_data.imag);
                else
                    data.(ds_name) = ds_data;
                end
            end
        end
        
        % Read const struct
        group_names = {};
        if ~isempty(info.Groups)
            group_names = {info.Groups.Name};
        end
        
        % Check if const group exists (could be '/const' or just 'const')
        const_exists = any(contains(group_names, 'const'));
        
        if const_exists
            % Find the const group path
            const_group_idx = find(contains(group_names, 'const'), 1);
            const_group_path = ['/' info.Groups(const_group_idx).Name];
            
            const_info = h5info(input_mat_file, const_group_path);
            data.const = struct();
            
            % Read datasets
            if ~isempty(const_info.Datasets)
                for i = 1:length(const_info.Datasets)
                    ds_name = const_info.Datasets(i).Name;
                    ds_data = h5read(input_mat_file, [const_group_path '/' ds_name]);
                    
                    % Handle character arrays (stored as uint16)
                    if isa(ds_data, 'uint16') && ndims(ds_data) <= 2
                        % Convert from uint16 to char
                        data.const.(ds_name) = char(ds_data(:)');
                    else
                        data.const.(ds_name) = ds_data;
                    end
                end
            end
            
            % Read attributes (scalar values stored as attributes)
            if ~isempty(const_info.Attributes)
                for i = 1:length(const_info.Attributes)
                    attr_name = const_info.Attributes(i).Name;
                    attr_data = h5readatt(input_mat_file, const_group_path, attr_name);
                    data.const.(attr_name) = attr_data;
                end
            end
        else
            fprintf('   Warning: const group not found in HDF5 file\n');
        end
        
        fprintf('   Successfully loaded with h5read\n');
    catch ME2
        fprintf('   ERROR with h5read: %s\n', ME2.message);
        error('Could not load .mat file');
    end
end

% Extract dimensions
if isfield(data, 'designs')
    designs = data.designs;
    n_designs = size(designs, 4); % MATLAB: (n_panes, H, W, n_designs)
else
    error('designs field not found in .mat file');
end

if isfield(data, 'WAVEVECTOR_DATA')
    wavevectors_data = data.WAVEVECTOR_DATA;
    if ndims(wavevectors_data) == 3
        n_wavevectors = size(wavevectors_data, 3); % (2, n_wavevectors, n_designs) or (n_designs, 2, n_wavevectors)
    else
        n_wavevectors = size(wavevectors_data, 2);
    end
else
    error('WAVEVECTOR_DATA field not found in .mat file');
end

% Extract const parameters
if isfield(data, 'const')
    const_base = data.const;
else
    error('const field not found in .mat file');
end

fprintf('   Found %d structures\n', n_designs);
fprintf('   Found %d wavevectors per structure\n', n_wavevectors);

% Initialize storage for K, M, T matrices
K_DATA = cell(1, n_designs);
M_DATA = cell(1, n_designs);
T_DATA = cell(1, n_wavevectors); % T matrices are the same for all structures (same wavevectors)

fprintf('\n');
fprintf('2. Computing K, M, T matrices for each structure...\n');

for struct_idx = 1:n_designs
    fprintf('\n   Structure %d/%d...\n', struct_idx, n_designs);
    
    % Extract design for this structure
    if ndims(designs) == 4
        design_3d = designs(:, :, :, struct_idx); % (n_panes, H, W)
    else
        design_3d = designs(:, :, :);
    end
    
    % Set up const struct for this structure
    const = const_base;
    const.design = design_3d; % (n_panes, H, W)
    const.N_pix = size(design_3d, 1); % Assuming square
    const.N_ele = 1; % Match Python default
    
    % Expand design if needed (N_ele > 1)
    if const.N_ele > 1
        const.design = repelem(const.design, const.N_ele, const.N_ele, 1);
    end
    
    % Compute K and M matrices
    fprintf('      Computing K and M...\n');
    [K, M] = get_system_matrices_VEC(const);
    
    % Store K and M
    K_DATA{struct_idx} = K;
    M_DATA{struct_idx} = M;
    
    fprintf('        K: size=%s, nnz=%d\n', mat2str(size(K)), nnz(K));
    fprintf('        M: size=%s, nnz=%d\n', mat2str(size(M)), nnz(M));
    
    % Compute T matrices for each wavevector (only for first structure to avoid redundancy)
    if struct_idx == 1
        fprintf('      Computing T matrices for %d wavevectors...\n', n_wavevectors);
        
        % Get wavevectors for this structure
        if ndims(wavevectors_data) == 3
            if size(wavevectors_data, 1) == 2 && size(wavevectors_data, 2) == n_wavevectors
                % Format: (2, n_wavevectors, n_designs)
                wv_struct = wavevectors_data(:, :, struct_idx)'; % Transpose to (n_wavevectors, 2)
            else
                % Format: (n_designs, 2, n_wavevectors)
                wv_struct = squeeze(wavevectors_data(struct_idx, :, :))'; % (n_wavevectors, 2)
            end
        else
            wv_struct = wavevectors_data'; % (n_wavevectors, 2)
        end
        
        for wv_idx = 1:n_wavevectors
            wavevector = wv_struct(wv_idx, :); % (1, 2)
            
            T = get_transformation_matrix(wavevector, const);
            if isempty(T)
                error('Failed to compute T matrix for wavevector %d', wv_idx);
            end
            
            T_DATA{wv_idx} = T;
            
            if wv_idx == 1
                fprintf('        T[%d]: size=%s, dtype=%s, sparse=%d\n', ...
                    wv_idx, mat2str(size(T)), class(T), issparse(T));
            end
        end
    end
end

% Save K, M, T matrices to .mat file
fprintf('\n');
fprintf('3. Saving K, M, T matrices to %s...\n', output_mat_file);

% Convert sparse matrices to full for saving (or keep sparse if MATLAB supports it)
% MATLAB can save sparse matrices directly in .mat files
save(output_mat_file, 'K_DATA', 'M_DATA', 'T_DATA', '-v7.3');

fprintf('   Saved successfully!\n');
fprintf('\n');
fprintf('=========================================================================\n');
fprintf('MATLAB K, M, T computation complete!\n');
fprintf('=========================================================================\n');

