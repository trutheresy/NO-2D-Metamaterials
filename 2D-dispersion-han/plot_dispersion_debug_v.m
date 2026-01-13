clear; close all;

% dispersion_library_path = '../../';
% addpath(dispersion_library_path)

% Check for temp file with CLI override path
data_fn_preserve = '';
if exist('temp_data_fn.txt', 'file')
    fid = fopen('temp_data_fn.txt', 'r');
    if fid ~= -1
        data_fn_preserve = fgetl(fid);
        fclose(fid);
    end
end

clearvars -except data_fn_preserve; close all;

% Use the saved path if it was provided, otherwise use default
if ~isempty(data_fn_preserve) && ischar(data_fn_preserve)
    data_fn = data_fn_preserve;
    fprintf('Using CLI override: %s\n', data_fn);
else
    % Use default if not provided or invalid
    data_fn = "D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1\out_binarized_1_predictions.mat";
    fprintf('Using default path: %s\n', data_fn);
end
[~,fn,~] = fileparts(data_fn);
fn = char(fn);

% Try to load the file - if it's v7.3 HDF5, MATLAB should detect it automatically
% If the file was created with h5py (pure HDF5), we need to use h5read or matfile
try
    % First try regular load (works for MATLAB-created v7.3 files)
    data = load(data_fn);
catch ME1
    % If regular load fails, try using h5read to read the HDF5 file directly
    try
        fprintf('Regular load failed, trying h5read...\n');
        % Read all top-level datasets from the HDF5 file
        info = h5info(data_fn);
        data = struct();
        
        % Read top-level datasets first
        for i = 1:length(info.Datasets)
            ds_name = info.Datasets(i).Name;
            if ~strncmp(ds_name, '#', 1)  % Skip MATLAB internal groups
                % Read the full dataset (h5read reads the entire dataset by default)
                ds_data = h5read(data_fn, ['/' ds_name]);
                % Handle complex data stored as structured array
                if isstruct(ds_data) && isfield(ds_data, 'real') && isfield(ds_data, 'imag')
                    % Reconstruct complex array from real/imag parts
                    data.(ds_name) = complex(ds_data.real, ds_data.imag);
                else
                    data.(ds_name) = ds_data;
                end
            end
        end
        
        % Read const struct group
        for i = 1:length(info.Groups)
            group_name = info.Groups(i).Name;
            if strcmp(group_name, '/const')
                % Handle const struct group
                const_info = h5info(data_fn, '/const');
                const_data = struct();
                for j = 1:length(const_info.Datasets)
                    ds_name = const_info.Datasets(j).Name;
                    ds_data = h5read(data_fn, ['/const/' ds_name]);
                    % Handle complex data if needed
                    if isstruct(ds_data) && isfield(ds_data, 'real') && isfield(ds_data, 'imag')
                        const_data.(ds_name) = complex(ds_data.real, ds_data.imag);
                    else
                        const_data.(ds_name) = ds_data;
                    end
                end
                data.const = const_data;
            end
        end
        fprintf('Successfully loaded with h5read\n');
    catch ME2
        fprintf('Error with h5read: %s\n', ME2.message);
        fprintf('Trying matfile as last resort...\n');
        try
            m = matfile(data_fn);
            var_names = who(m);
            data = struct();
            for i = 1:length(var_names)
                var_name = var_names{i};
                if ~strcmp(var_name, 'Properties')
                    data.(var_name) = m.(var_name);
                end
            end
            fprintf('Successfully loaded with matfile\n');
        catch ME3
            fprintf('All loading methods failed. Error: %s\n', ME3.message);
            rethrow(ME1);
        end
    end
end

% flags
isExportPng = true;
png_resolution = 150;
isSavePlotPoints = true;  % Save plot point locations for comparison with Python

% Output root: save under plots/<dataset_name>_mat/... (in current directory)
output_root = fullfile(pwd, 'plots', [fn '_mat']);

% DEBUG: Create debug output folder in main directory
debug_output_root = fullfile(pwd, 'debug');
if ~isfolder(debug_output_root)
    mkdir(debug_output_root);
end
fprintf('\n========================================\n');
fprintf('DEBUG MODE: Saving computed objects to: %s\n', debug_output_root);
fprintf('========================================\n\n');

% Make plots for one unit cell or multiple
struct_idxs = 1:10;

% DEBUG: Initialize debug data structures
debug_K_M_data = struct();
debug_T_data = struct();
debug_eigenvalue_recon_data = struct();
debug_const_data = struct();
debug_design_data = struct();

% Initialize plot points data structure if saving
if isSavePlotPoints
    plot_points_data = struct();
end

% Extract or compute constitutive data
if isfield(data, 'CONSTITUTIVE_DATA')
    % CONSTITUTIVE_DATA exists as containers.Map
    E_all = data.CONSTITUTIVE_DATA('modulus');
    rho_all = data.CONSTITUTIVE_DATA('density');
    nu_all = data.CONSTITUTIVE_DATA('poisson');
else
    % Compute CONSTITUTIVE_DATA from designs and const values
    fprintf('CONSTITUTIVE_DATA not found, computing from designs...\n');
    if ~isfield(data, 'designs') || ~isfield(data, 'const')
        error('Cannot compute CONSTITUTIVE_DATA: missing designs or const');
    end
    
    % Extract const values (handle both struct and array formats)
    if isstruct(data.const)
        E_min = data.const.E_min(1,1);
        E_max = data.const.E_max(1,1);
        rho_min = data.const.rho_min(1,1);
        rho_max = data.const.rho_max(1,1);
        poisson_min = data.const.poisson_min(1,1);
        poisson_max = data.const.poisson_max(1,1);
    else
        E_min = data.const.E_min(1,1);
        E_max = data.const.E_max(1,1);
        rho_min = data.const.rho_min(1,1);
        rho_max = data.const.rho_max(1,1);
        poisson_min = data.const.poisson_min(1,1);
        poisson_max = data.const.poisson_max(1,1);
    end
    
    % Compute material properties from designs
    % designs is (N_pix, N_pix, 3, N_struct) or (N_struct, 3, N_pix, N_pix) depending on format
    designs = data.designs;
    if ndims(designs) == 4
        % Check dimension order
        if size(designs, 1) == size(designs, 2) && size(designs, 3) == 3
            % MATLAB order: (N_pix, N_pix, 3, N_struct)
            N_struct = size(designs, 4);
            E_all = zeros(size(designs, 1), size(designs, 2), N_struct);
            rho_all = zeros(size(designs, 1), size(designs, 2), N_struct);
            nu_all = zeros(size(designs, 1), size(designs, 2), N_struct);
            for s = 1:N_struct
                E_all(:,:,s) = E_min + (E_max - E_min) * designs(:,:,1,s);
                rho_all(:,:,s) = rho_min + (rho_max - rho_min) * designs(:,:,2,s);
                nu_all(:,:,s) = poisson_min + (poisson_max - poisson_min) * designs(:,:,3,s);
            end
        else
            % HDF5 order: (N_struct, 3, N_pix, N_pix)
            N_struct = size(designs, 1);
            N_pix = size(designs, 3);
            E_all = zeros(N_pix, N_pix, N_struct);
            rho_all = zeros(N_pix, N_pix, N_struct);
            nu_all = zeros(N_pix, N_pix, N_struct);
            for s = 1:N_struct
                % Permute from (N_pix, N_pix, 3) to access each pane
                design_3d = permute(designs(s, :, :, :), [3, 4, 2, 1]); % (N_pix, N_pix, 3)
                E_all(:,:,s) = E_min + (E_max - E_min) * design_3d(:,:,1);
                rho_all(:,:,s) = rho_min + (rho_max - rho_min) * design_3d(:,:,2);
                nu_all(:,:,s) = poisson_min + (poisson_max - poisson_min) * design_3d(:,:,3);
            end
        end
    else
        error('Unexpected designs array shape');
    end
    fprintf('Computed CONSTITUTIVE_DATA from designs\n');
end

% DEBUG: Always compute and save K, M, T matrices for comparison
% Extract const values and set up const structure for each structure
if isstruct(data.const)
    N_pix = data.const.N_pix(1,1);
    N_ele = data.const.N_ele(1,1);
    N_eig = data.const.N_eig(1,1);
    a_val = data.const.a(1,1);
    t_val = data.const.t(1,1);
    E_min = data.const.E_min(1,1);
    E_max = data.const.E_max(1,1);
    rho_min = data.const.rho_min(1,1);
    rho_max = data.const.rho_max(1,1);
    poisson_min = data.const.poisson_min(1,1);
    poisson_max = data.const.poisson_max(1,1);
    sigma_eig = data.const.sigma_eig(1,1);
    if isfield(data.const, 'design_scale')
        % Convert from uint16 character array to string
        if isnumeric(data.const.design_scale)
            if size(data.const.design_scale, 2) == 1
                design_scale = char(data.const.design_scale(:,1)');
            else
                design_scale = char(data.const.design_scale(1,:));
            end
        else
            design_scale = char(data.const.design_scale);
        end
        design_scale = strtrim(design_scale(:)');
    else
        design_scale = 'linear';
    end
    if isfield(data.const, 'symmetry_type')
        if isnumeric(data.const.symmetry_type)
            if size(data.const.symmetry_type, 2) == 1
                symmetry_type = char(data.const.symmetry_type(:,1)');
            else
                symmetry_type = char(data.const.symmetry_type(1,:));
            end
        else
            symmetry_type = char(data.const.symmetry_type);
        end
        symmetry_type = strtrim(symmetry_type(:)');
    else
        symmetry_type = 'p4mm';
    end
    if isfield(data.const, 'wavevectors')
        wavevectors_const = data.const.wavevectors;
    else
        wavevectors_const = data.WAVEVECTOR_DATA(:,:,1)';
    end
else
    N_pix = data.const.N_pix(1,1);
    N_ele = data.const.N_ele(1,1);
    N_eig = data.const.N_eig(1,1);
    a_val = data.const.a(1,1);
    t_val = data.const.t(1,1);
    E_min = data.const.E_min(1,1);
    E_max = data.const.E_max(1,1);
    rho_min = data.const.rho_min(1,1);
    rho_max = data.const.rho_max(1,1);
    poisson_min = data.const.poisson_min(1,1);
    poisson_max = data.const.poisson_max(1,1);
    sigma_eig = data.const.sigma_eig(1,1);
    if isfield(data.const, 'design_scale')
        if isnumeric(data.const.design_scale)
            if size(data.const.design_scale, 2) == 1
                design_scale = char(data.const.design_scale(:,1)');
            else
                design_scale = char(data.const.design_scale(1,:));
            end
        else
            design_scale = char(data.const.design_scale);
        end
        design_scale = strtrim(design_scale(:)');
    else
        design_scale = 'linear';
    end
    if isfield(data.const, 'symmetry_type')
        if isnumeric(data.const.symmetry_type)
            if size(data.const.symmetry_type, 2) == 1
                symmetry_type = char(data.const.symmetry_type(:,1)');
            else
                symmetry_type = char(data.const.symmetry_type(1,:));
            end
        else
            symmetry_type = char(data.const.symmetry_type);
        end
        symmetry_type = strtrim(symmetry_type(:)');
    else
        symmetry_type = 'p4mm';
    end
    if isfield(data.const, 'wavevectors')
        wavevectors_const = data.const.wavevectors;
    else
        wavevectors_const = data.WAVEVECTOR_DATA(:,:,1)';
    end
end

% Ensure design_scale is valid
if ~strcmp(design_scale, 'linear') && ~strcmp(design_scale, 'log')
    warning('design_scale is "%s", expected "linear" or "log". Forcing to "linear".', design_scale);
    design_scale = 'linear';
end

% Determine number of structures and wavevectors
if size(data.designs, 1) == size(data.designs, 2) && size(data.designs, 3) == 3
    N_struct_total = size(data.designs, 4);
else
    N_struct_total = size(data.designs, 1);
end
N_wv = size(wavevectors_const, 1);

% Set up const structure template (will be modified for each structure)
const_template = struct();
const_template.N_pix = N_pix;
const_template.N_ele = N_ele;
const_template.N_eig = N_eig;
const_template.a = a_val;
const_template.t = t_val;
const_template.E_min = E_min;
const_template.E_max = E_max;
const_template.rho_min = rho_min;
const_template.rho_max = rho_max;
const_template.poisson_min = poisson_min;
const_template.poisson_max = poisson_max;
const_template.sigma_eig = sigma_eig;
const_template.design_scale = design_scale;
const_template.symmetry_type = symmetry_type;
const_template.wavevectors = wavevectors_const;
const_template.isUseGPU = false;
const_template.isUseImprovement = true;
const_template.isUseSecondImprovement = false;
const_template.isUseParallel = false; % Don't use parallel for reconstruction
const_template.isSaveEigenvectors = false;
const_template.eigenvector_dtype = 'single';
const_template.isSaveMesh = false;

% DEBUG: Always compute and save K, M, T matrices
fprintf('\n========================================\n');
fprintf('DEBUG: Computing K, M, T matrices from designs...\n');
fprintf('========================================\n');

for struct_idx_recon = 1:N_struct_total
    fprintf('\n========================================\n');
    fprintf('DEBUG: Computing matrices for structure %d/%d...\n', struct_idx_recon, N_struct_total);
    fprintf('========================================\n');
    
    % Extract geometry for this structure
    fprintf('DEBUG: Extracting design geometry...\n');
    if size(data.designs, 1) == size(data.designs, 2) && size(data.designs, 3) == 3
        % MATLAB order: (N_pix, N_pix, 3, N_struct)
        design_3d = data.designs(:,:,:,struct_idx_recon);
        fprintf('  Design shape (MATLAB order): %s\n', mat2str(size(design_3d)));
    else
        % HDF5 order: (N_struct, 3, N_pix, N_pix)
        design_3d = permute(data.designs(struct_idx_recon, :, :, :), [3, 4, 2, 1]); % (N_pix, N_pix, 3)
        fprintf('  Design shape (HDF5 order, after permute): %s\n', mat2str(size(design_3d)));
    end
    fprintf('  Design range: [%.6e, %.6e]\n', min(design_3d(:)), max(design_3d(:)));
    
    % DEBUG: Save design
    debug_design_data.(['struct_' num2str(struct_idx_recon)]) = design_3d;
    
    % Set up const structure for this structure
    fprintf('DEBUG: Setting up const structure...\n');
    const = const_template;
    const.design = design_3d;
    fprintf('  const.N_pix = %d\n', const.N_pix);
    fprintf('  const.N_ele = %d\n', const.N_ele);
    fprintf('  const.N_eig = %d\n', const.N_eig);
    
    % DEBUG: Save const
    debug_const_data.(['struct_' num2str(struct_idx_recon)]) = const;
    
    % Compute K and M matrices using library function
    fprintf('DEBUG: Computing K and M matrices...\n');
    [K, M] = get_system_matrices(const);
    fprintf('  K matrix: size = %s, nnz = %d, range = [%.6e, %.6e]\n', ...
        mat2str(size(K)), nnz(K), full(min(K(:))), full(max(K(:))));
    fprintf('  M matrix: size = %s, nnz = %d, range = [%.6e, %.6e]\n', ...
        mat2str(size(M)), nnz(M), full(min(M(:))), full(max(M(:))));
    
    % DEBUG: Save K and M matrices
    debug_K_M_data.(['struct_' num2str(struct_idx_recon) '_K']) = K;
    debug_K_M_data.(['struct_' num2str(struct_idx_recon) '_M']) = M;
    
    % Compute T matrices for each wavevector
    T_cell = cell(N_wv, 1);  % DEBUG: Store T matrices
    fprintf('DEBUG: Computing T matrices for %d wavevectors...\n', N_wv);
    for wv_idx = 1:N_wv
        if mod(wv_idx, 10) == 0 || wv_idx == 1
            fprintf('  Computing T matrix for wavevector %d/%d...\n', wv_idx, N_wv);
        end
        
        % Ensure wavevector is a row vector of doubles
        wavevector = double(wavevectors_const(wv_idx, :));
        if size(wavevector, 1) > size(wavevector, 2)
            wavevector = wavevector';
        end
        
        % Compute T matrix using library function
        T = get_transformation_matrix(wavevector, const);
        
        % DEBUG: Save T matrix
        T_cell{wv_idx} = T;
    end
    
    % DEBUG: Save T matrices for this structure
    debug_T_data.(['struct_' num2str(struct_idx_recon) '_T']) = T_cell;
end

% DEBUG: Save computed matrices
fprintf('\n========================================\n');
fprintf('DEBUG: Saving computed K, M, T matrices to debug folder...\n');
fprintf('========================================\n');

debug_save_path = fullfile(debug_output_root, 'computed_K_M_T_matrices.mat');
save(debug_save_path, 'debug_K_M_data', 'debug_T_data', 'debug_const_data', ...
    'debug_design_data', 'wavevectors_const', 'const_template', ...
    'N_struct_total', 'N_wv', 'N_eig', '-v7.3');
fprintf('DEBUG: Saved computed matrices to: %s\n', debug_save_path);

% Check if EIGENVALUE_DATA is missing and needs to be reconstructed
if ~isfield(data, 'EIGENVALUE_DATA') || isempty(data.EIGENVALUE_DATA)
    fprintf('\n========================================\n');
    fprintf('EIGENVALUE_DATA not found. Reconstructing from eigenvectors...\n');
    fprintf('========================================\n');
    
    % Check that required data is available
    if ~isfield(data, 'EIGENVECTOR_DATA') || isempty(data.EIGENVECTOR_DATA)
        error('Cannot reconstruct EIGENVALUE_DATA: EIGENVECTOR_DATA is missing');
    end
    if ~isfield(data, 'designs') || isempty(data.designs)
        error('Cannot reconstruct EIGENVALUE_DATA: designs is missing');
    end
    if ~isfield(data, 'const') || isempty(data.const)
        error('Cannot reconstruct EIGENVALUE_DATA: const is missing');
    end
    
    % Extract const values and set up const structure for each structure
    if isstruct(data.const)
        N_pix = data.const.N_pix(1,1);
        N_ele = data.const.N_ele(1,1);
        N_eig = data.const.N_eig(1,1);
        a_val = data.const.a(1,1);
        t_val = data.const.t(1,1);
        E_min = data.const.E_min(1,1);
        E_max = data.const.E_max(1,1);
        rho_min = data.const.rho_min(1,1);
        rho_max = data.const.rho_max(1,1);
        poisson_min = data.const.poisson_min(1,1);
        poisson_max = data.const.poisson_max(1,1);
        sigma_eig = data.const.sigma_eig(1,1);
        if isfield(data.const, 'design_scale')
            % Convert from uint16 character array to string
            if isnumeric(data.const.design_scale)
                if size(data.const.design_scale, 2) == 1
                    design_scale = char(data.const.design_scale(:,1)');
                else
                    design_scale = char(data.const.design_scale(1,:));
                end
            else
                design_scale = char(data.const.design_scale);
            end
            design_scale = strtrim(design_scale(:)');
        else
            design_scale = 'linear';
        end
        if isfield(data.const, 'symmetry_type')
            if isnumeric(data.const.symmetry_type)
                if size(data.const.symmetry_type, 2) == 1
                    symmetry_type = char(data.const.symmetry_type(:,1)');
                else
                    symmetry_type = char(data.const.symmetry_type(1,:));
                end
            else
                symmetry_type = char(data.const.symmetry_type);
            end
            symmetry_type = strtrim(symmetry_type(:)');
        else
            symmetry_type = 'p4mm';
        end
        if isfield(data.const, 'wavevectors')
            wavevectors_const = data.const.wavevectors;
        else
            wavevectors_const = data.WAVEVECTOR_DATA(:,:,1)';
        end
    else
        N_pix = data.const.N_pix(1,1);
        N_ele = data.const.N_ele(1,1);
        N_eig = data.const.N_eig(1,1);
        a_val = data.const.a(1,1);
        t_val = data.const.t(1,1);
        E_min = data.const.E_min(1,1);
        E_max = data.const.E_max(1,1);
        rho_min = data.const.rho_min(1,1);
        rho_max = data.const.rho_max(1,1);
        poisson_min = data.const.poisson_min(1,1);
        poisson_max = data.const.poisson_max(1,1);
        sigma_eig = data.const.sigma_eig(1,1);
        if isfield(data.const, 'design_scale')
            if isnumeric(data.const.design_scale)
                if size(data.const.design_scale, 2) == 1
                    design_scale = char(data.const.design_scale(:,1)');
                else
                    design_scale = char(data.const.design_scale(1,:));
                end
            else
                design_scale = char(data.const.design_scale);
            end
            design_scale = strtrim(design_scale(:)');
        else
            design_scale = 'linear';
        end
        if isfield(data.const, 'symmetry_type')
            if isnumeric(data.const.symmetry_type)
                if size(data.const.symmetry_type, 2) == 1
                    symmetry_type = char(data.const.symmetry_type(:,1)');
                else
                    symmetry_type = char(data.const.symmetry_type(1,:));
                end
            else
                symmetry_type = char(data.const.symmetry_type);
            end
            symmetry_type = strtrim(symmetry_type(:)');
        else
            symmetry_type = 'p4mm';
        end
        if isfield(data.const, 'wavevectors')
            wavevectors_const = data.const.wavevectors;
        else
            wavevectors_const = data.WAVEVECTOR_DATA(:,:,1)';
        end
    end
    
    % Ensure design_scale is valid
    if ~strcmp(design_scale, 'linear') && ~strcmp(design_scale, 'log')
        warning('design_scale is "%s", expected "linear" or "log". Forcing to "linear".', design_scale);
        design_scale = 'linear';
    end
    
    % Determine number of structures and wavevectors
    if size(data.designs, 1) == size(data.designs, 2) && size(data.designs, 3) == 3
        N_struct_total = size(data.designs, 4);
    else
        N_struct_total = size(data.designs, 1);
    end
    N_wv = size(wavevectors_const, 1);
    
    % Initialize EIGENVALUE_DATA array: (N_wv, N_eig, N_struct) in MATLAB order
    EIGENVALUE_DATA = zeros(N_wv, N_eig, N_struct_total);
    
    % Set up const structure template (will be modified for each structure)
    const_template = struct();
    const_template.N_pix = N_pix;
    const_template.N_ele = N_ele;
    const_template.N_eig = N_eig;
    const_template.a = a_val;
    const_template.t = t_val;
    const_template.E_min = E_min;
    const_template.E_max = E_max;
    const_template.rho_min = rho_min;
    const_template.rho_max = rho_max;
    const_template.poisson_min = poisson_min;
    const_template.poisson_max = poisson_max;
    const_template.sigma_eig = sigma_eig;
    const_template.design_scale = design_scale;
    const_template.symmetry_type = symmetry_type;
    const_template.wavevectors = wavevectors_const;
    const_template.isUseGPU = false;
    const_template.isUseImprovement = true;
    const_template.isUseSecondImprovement = false;
    const_template.isUseParallel = false; % Don't use parallel for reconstruction
    const_template.isSaveEigenvectors = false;
    const_template.eigenvector_dtype = 'single';
    const_template.isSaveMesh = false;
    
    % Reconstruct EIGENVALUE_DATA for each structure
    for struct_idx_recon = 1:N_struct_total
        fprintf('\n========================================\n');
        fprintf('DEBUG: Reconstructing EIGENVALUE_DATA for structure %d/%d...\n', struct_idx_recon, N_struct_total);
        fprintf('========================================\n');
        
        % Extract geometry for this structure
        fprintf('DEBUG: Extracting design geometry...\n');
        if size(data.designs, 1) == size(data.designs, 2) && size(data.designs, 3) == 3
            % MATLAB order: (N_pix, N_pix, 3, N_struct)
            design_3d = data.designs(:,:,:,struct_idx_recon);
            fprintf('  Design shape (MATLAB order): %s\n', mat2str(size(design_3d)));
        else
            % HDF5 order: (N_struct, 3, N_pix, N_pix)
            design_3d = permute(data.designs(struct_idx_recon, :, :, :), [3, 4, 2, 1]); % (N_pix, N_pix, 3)
            fprintf('  Design shape (HDF5 order, after permute): %s\n', mat2str(size(design_3d)));
        end
        fprintf('  Design range: [%.6e, %.6e]\n', min(design_3d(:)), max(design_3d(:)));
        
        % DEBUG: Save design
        debug_design_data.(['struct_' num2str(struct_idx_recon)]) = design_3d;
        
        % Set up const structure for this structure
        fprintf('DEBUG: Setting up const structure...\n');
        const = const_template;
        const.design = design_3d;
        fprintf('  const.N_pix = %d\n', const.N_pix);
        fprintf('  const.N_ele = %d\n', const.N_ele);
        fprintf('  const.N_eig = %d\n', const.N_eig);
        fprintf('  const.a = %.6e\n', const.a);
        fprintf('  const.t = %.6e\n', const.t);
        fprintf('  const.E_min = %.6e, E_max = %.6e\n', const.E_min, const.E_max);
        fprintf('  const.rho_min = %.6e, rho_max = %.6e\n', const.rho_min, const.rho_max);
        fprintf('  const.poisson_min = %.6e, poisson_max = %.6e\n', const.poisson_min, const.poisson_max);
        
        % DEBUG: Save const
        debug_const_data.(['struct_' num2str(struct_idx_recon)]) = const;
        
        % Compute K and M matrices using library function
        fprintf('DEBUG: Computing K and M matrices...\n');
        [K, M] = get_system_matrices(const);
        fprintf('  K matrix: size = %s, nnz = %d, range = [%.6e, %.6e]\n', ...
            mat2str(size(K)), nnz(K), full(min(K(:))), full(max(K(:))));
        fprintf('  M matrix: size = %s, nnz = %d, range = [%.6e, %.6e]\n', ...
            mat2str(size(M)), nnz(M), full(min(M(:))), full(max(M(:))));
        
        % DEBUG: Save K and M matrices
        debug_K_M_data.(['struct_' num2str(struct_idx_recon) '_K']) = K;
        debug_K_M_data.(['struct_' num2str(struct_idx_recon) '_M']) = M;
        
        % Reconstruct frequencies for each wavevector
        frequencies_recon = zeros(N_wv, N_eig);
        T_cell = cell(N_wv, 1);  % DEBUG: Store T matrices
        Kr_cell = cell(N_wv, 1); % DEBUG: Store Kr matrices
        Mr_cell = cell(N_wv, 1); % DEBUG: Store Mr matrices
        
        fprintf('DEBUG: Processing %d wavevectors...\n', N_wv);
        for wv_idx = 1:N_wv
            if mod(wv_idx, 10) == 0 || wv_idx == 1
                fprintf('  Processing wavevector %d/%d...\n', wv_idx, N_wv);
            end
            
            % Ensure wavevector is a row vector of doubles
            wavevector = double(wavevectors_const(wv_idx, :));
            if size(wavevector, 1) > size(wavevector, 2)
                wavevector = wavevector';
            end
            if wv_idx == 1 || mod(wv_idx, 20) == 0
                fprintf('    Wavevector %d: [%.6e, %.6e]\n', wv_idx, wavevector(1), wavevector(2));
            end
            
            % Compute T matrix using library function
            T = get_transformation_matrix(wavevector, const);
            if wv_idx == 1 || mod(wv_idx, 20) == 0
                fprintf('    T matrix: size = %s, nnz = %d\n', mat2str(size(T)), nnz(T));
            end
            
            % DEBUG: Save T matrix
            T_cell{wv_idx} = T;
            
            % Transform to reduced space
            Kr = T'*K*T;
            Mr = T'*M*T;
            if wv_idx == 1 || mod(wv_idx, 20) == 0
                fprintf('    Kr matrix: size = %s, nnz = %d\n', mat2str(size(Kr)), nnz(Kr));
                fprintf('    Mr matrix: size = %s, nnz = %d\n', mat2str(size(Mr)), nnz(Mr));
            end
            
            % DEBUG: Save Kr and Mr matrices
            Kr_cell{wv_idx} = Kr;
            Mr_cell{wv_idx} = Mr;
            
            % Reconstruct eigenvalues from eigenvectors
            % EIGENVECTOR_DATA is in MATLAB order: (dof, wv, band, struct) after h5read transpose
            % So we access it as: data.EIGENVECTOR_DATA(:, wv_idx, band_idx, struct_idx_recon)
            for band_idx = 1:N_eig
                eigvec = data.EIGENVECTOR_DATA(:, wv_idx, band_idx, struct_idx_recon);
                eigvec = double(eigvec); % Cast to double for computation
                
                % Reconstruct eigenvalue: eigval = norm(Kr*eigvec)/norm(Mr*eigvec)
                % This is equivalent to: Kr*eigvec = eigval*Mr*eigvec
                eigval = norm(Kr*eigvec)/norm(Mr*eigvec);
                
                % Convert to frequency
                frequencies_recon(wv_idx, band_idx) = sqrt(eigval)/(2*pi);
            end
        end
        
        % DEBUG: Save T, Kr, Mr matrices for this structure
        debug_T_data.(['struct_' num2str(struct_idx_recon) '_T']) = T_cell;
        debug_T_data.(['struct_' num2str(struct_idx_recon) '_Kr']) = Kr_cell;
        debug_T_data.(['struct_' num2str(struct_idx_recon) '_Mr']) = Mr_cell;
        
        fprintf('DEBUG: Reconstructed frequencies range: [%.6e, %.6e] Hz\n', ...
            min(frequencies_recon(:)), max(frequencies_recon(:)));
        
        % DEBUG: Save reconstructed frequencies
        debug_eigenvalue_recon_data.(['struct_' num2str(struct_idx_recon) '_frequencies']) = frequencies_recon;
        
        % Store in EIGENVALUE_DATA (MATLAB order: (N_wv, N_eig, N_struct))
        EIGENVALUE_DATA(:, :, struct_idx_recon) = frequencies_recon;
    end
    
    % Store reconstructed EIGENVALUE_DATA in data structure
    data.EIGENVALUE_DATA = EIGENVALUE_DATA;
    
    fprintf('\n========================================\n');
    fprintf('DEBUG: Saving computed objects to debug folder...\n');
    fprintf('========================================\n');
    
    % DEBUG: Save all computed objects
    debug_save_path = fullfile(debug_output_root, 'reconstruction_debug_data.mat');
    save(debug_save_path, 'debug_K_M_data', 'debug_T_data', 'debug_eigenvalue_recon_data', ...
        'debug_const_data', 'debug_design_data', 'EIGENVALUE_DATA', 'wavevectors_const', ...
        'const_template', 'N_struct_total', 'N_wv', 'N_eig', '-v7.3');
    fprintf('DEBUG: Saved debug data to: %s\n', debug_save_path);
    
    fprintf('Successfully reconstructed EIGENVALUE_DATA!\n');
    fprintf('========================================\n\n');
end

for struct_idx = struct_idxs
    %% Plot the material property fields (actual values, not design variables)
    fig = figure();
    tlo = tiledlayout(1,3,'Parent',fig);

    E = E_all(:,:,struct_idx);
    rho = rho_all(:,:,struct_idx);
    nu = nu_all(:,:,struct_idx);

    ax = nexttile;
    imagesc(ax,E)
    daspect(ax,[1 1 1])
    colormap(ax,'gray')
    colorbar(ax)
    title('E [Pa]')

    ax = nexttile;
    imagesc(ax,rho)
    daspect(ax,[1 1 1])
    colormap(ax,'gray')
    colorbar(ax)
    title('rho [kg/m^3]')

    ax = nexttile;
    imagesc(ax,nu)
    daspect(ax,[1 1 1])
    colormap(ax,'gray')
    colorbar(ax)
    title('nu [-]')
%     ax.CLim = []; % optional: set color limits if desired

    if isExportPng
        png_path = fullfile(output_root, 'constitutive_fields', [num2str(struct_idx) '.png']);
        if ~isfolder(fileparts(png_path))
            mkdir(fileparts(png_path))
        end
        exportgraphics(fig,png_path,'Resolution',png_resolution);
    end

    %% Get relevant dispersion data

    disp('size(data.WAVEVECTOR_DATA)')
    disp(size(data.WAVEVECTOR_DATA))
    % Extract wavevectors - handle different possible shapes
    % MATLAB may see shape as (91, 2, 10) or (10, 2, 91) depending on how data was saved
    wv_size = size(data.WAVEVECTOR_DATA);
    if length(wv_size) == 3
        if wv_size(3) == struct_idx || wv_size(3) >= struct_idx
            % Shape is likely (N_wv, 2, N_struct) or (2, N_wv, N_struct)
            wavevectors = double(data.WAVEVECTOR_DATA(:,:,struct_idx));
            if size(wavevectors, 1) == 2 && size(wavevectors, 2) > 2
                % Need to transpose: (2, N_wv) -> (N_wv, 2)
                wavevectors = wavevectors';
            end
        else
            % Shape is likely (N_struct, 2, N_wv)
            wavevectors = double(data.WAVEVECTOR_DATA(struct_idx, :, :));
            if size(wavevectors, 1) == 2 && size(wavevectors, 2) > 2
                % Need to transpose: (2, N_wv) -> (N_wv, 2)
                wavevectors = wavevectors';
            elseif size(wavevectors, 2) == 2 && size(wavevectors, 1) > 2
                % Already correct: (N_wv, 2)
            end
        end
    else
        error('Unexpected WAVEVECTOR_DATA shape: %s', mat2str(wv_size));
    end
    disp(['wavevectors shape after extraction: ' mat2str(size(wavevectors))]);

    % EIGENVALUE_DATA should now be available (either from file or reconstructed)
    if isfield(data, 'EIGENVALUE_DATA') && ~isempty(data.EIGENVALUE_DATA)
        disp('size(data.EIGENVALUE_DATA)')
        disp(size(data.EIGENVALUE_DATA))
        % Extract frequencies - handle different possible shapes
        freq_size = size(data.EIGENVALUE_DATA);
        if length(freq_size) == 3
            if freq_size(3) == struct_idx || freq_size(3) >= struct_idx
                % Shape is likely (N_wv, N_eig, N_struct)
                frequencies = double(data.EIGENVALUE_DATA(:,:,struct_idx));
            else
                % Shape is likely (N_struct, N_eig, N_wv)
                frequencies = double(data.EIGENVALUE_DATA(struct_idx, :, :));
                if size(frequencies, 1) < size(frequencies, 2) && size(frequencies, 1) <= 10
                    % Need to transpose: (N_eig, N_wv) -> (N_wv, N_eig)
                    frequencies = frequencies';
                end
            end
        else
            error('Unexpected EIGENVALUE_DATA shape: %s', mat2str(freq_size));
        end
        disp(['frequencies shape after extraction: ' mat2str(size(frequencies))]);
    else
        error('EIGENVALUE_DATA is still missing after reconstruction attempt');
    end

    %% Reconstruct frequencies from eigenvectors (if K_DATA, M_DATA, T_DATA available)
    can_reconstruct = isfield(data, 'K_DATA') && isfield(data, 'M_DATA') && isfield(data, 'T_DATA') && ...
                      ~isempty(data.K_DATA) && ~isempty(data.M_DATA) && ~isempty(data.T_DATA) && ...
                      length(data.K_DATA) >= struct_idx && length(data.M_DATA) >= struct_idx;
    
    % Get N_eig from const (handle both struct and array formats)
    if isstruct(data.const)
        N_eig = data.const.N_eig(1,1);
    else
        N_eig = data.const.N_eig(1,1);
    end
    
    if can_reconstruct
        frequencies_recon = zeros(size(data.const.wavevectors,1), N_eig);
        K = data.K_DATA{struct_idx};
        M = data.M_DATA{struct_idx};
        for wv_idx = 1:size(data.const.wavevectors,1)
            if length(data.T_DATA) >= wv_idx && ~isempty(data.T_DATA{wv_idx})
                T = data.T_DATA{wv_idx};
                Kr = T'*K*T;
                Mr = T'*M*T;
                for band_idx = 1:N_eig
                    eigvec = data.EIGENVECTOR_DATA(:,wv_idx,band_idx,struct_idx);
                    eigvec = double(eigvec); % NEW: Cast to double so that the following multiplication doesn't complain
                    eigval = norm(Kr*eigvec)/norm(Mr*eigvec); % eigval = eigs(Kr,Mr) solves Kr*eigvec = Mr*eigvec*eigval ==> eigval = norm(Kr*eigvec)/norm(Mr*eigvec)
                    frequencies_recon(wv_idx,band_idx) = sqrt(eigval)/(2*pi);
                end
            end
        end
        disp(['max(abs(frequencies_recon-frequencies))/max(abs(frequencies)) = ' num2str(max(abs(frequencies_recon-frequencies),[],'all')/max(abs(frequencies),[],'all'))])
    else
        frequencies_recon = frequencies; % Use original if reconstruction not possible
        disp('Reconstruction data (K_DATA, M_DATA, T_DATA) not available, skipping reconstruction')
    end

    % Create an interpolant for each eigenvalue band
    interp_method = 'linear';
    extrap_method = 'linear';
    % Get N_eig from const (handle both struct and array formats)
    if isstruct(data.const)
        N_eig = data.const.N_eig(1,1);
    else
        N_eig = data.const.N_eig(1,1);
    end
    interp_true = cell(N_eig,1);
    if can_reconstruct
        interp_recon = cell(N_eig,1);
    end
    for eig_idx = 1:N_eig
        interp_true{eig_idx} = scatteredInterpolant(wavevectors,frequencies(:,eig_idx),interp_method,extrap_method);
        if can_reconstruct
            interp_recon{eig_idx} = scatteredInterpolant(wavevectors,frequencies_recon(:,eig_idx),interp_method,extrap_method);
        end
    end

    % Get the IBZ contour wave vectors. NOTE: These only exist for some symmetry
    % groups. symmetry = 'p4mm' has a valid IBZ contour.
    N_k = 10; % Number of interpolation query points per contour segment
    % Get symmetry_type and a from const (handle both struct and array formats)
    if isstruct(data.const)
        a_val = data.const.a(1,1);
        if isfield(data.const, 'symmetry_type')
            % symmetry_type might be stored as string, char array, or uint16 character array
            if ischar(data.const.symmetry_type) || isstring(data.const.symmetry_type)
                sym_type = char(data.const.symmetry_type);
            elseif isnumeric(data.const.symmetry_type)
                % Handle column vector of uint16 characters
                if size(data.const.symmetry_type, 2) == 1
                    sym_type = char(data.const.symmetry_type(:,1)');
                else
                    sym_type = char(data.const.symmetry_type(1,:));
                end
            else
                sym_type = char(data.const.symmetry_type);
            end
            % Ensure it's a row vector string and trim whitespace
            sym_type = strtrim(sym_type(:)');
        else
            sym_type = 'p4mm';  % Default
        end
    else
        a_val = data.const.a(1,1);
        if isfield(data.const, 'symmetry_type')
            if ischar(data.const.symmetry_type) || isstring(data.const.symmetry_type)
                sym_type = char(data.const.symmetry_type);
            elseif isnumeric(data.const.symmetry_type)
                if size(data.const.symmetry_type, 2) == 1
                    sym_type = char(data.const.symmetry_type(:,1)');
                else
                    sym_type = char(data.const.symmetry_type(1,:));
                end
            else
                sym_type = char(data.const.symmetry_type);
            end
            sym_type = strtrim(sym_type(:)');
        else
            sym_type = 'p4mm';  % Default
        end
    end
    [wavevectors_contour, contour_info] = get_IBZ_contour_wavevectors(N_k, a_val, sym_type);

    % Plot the IBZ contour wavevectors
    if struct_idx == struct_idxs(1)
        fig = figure();
        ax = axes(fig);
        plot(ax,wavevectors_contour(:,1),wavevectors_contour(:,2),'k.')
        axis(ax,'padded')
        daspect(ax,[1 1 1])
        xlabel(ax,'wavevector x component [1/m]')
        ylabel(ax,'wavevector y component [1/m]')
        if isExportPng
            png_path = fullfile(output_root, 'contour', [num2str(struct_idx) '.png']);
            if ~isfolder(fileparts(png_path))
                mkdir(fileparts(png_path))
            end
            exportgraphics(fig,png_path,'Resolution',png_resolution);
        end
    end

    % Evaluate frequencies on the desired wavevectors using the interpolant
    frequencies_contour = zeros(size(wavevectors_contour,1),size(frequencies,2));
    if can_reconstruct
        frequencies_recon_contour = zeros(size(wavevectors_contour,1),size(frequencies,2));
    end
    for eig_idx = 1:size(frequencies,2)
        frequencies_contour(:,eig_idx) = interp_true{eig_idx}(wavevectors_contour(:,1),wavevectors_contour(:,2));
        if can_reconstruct
            frequencies_recon_contour(:,eig_idx) = interp_recon{eig_idx}(wavevectors_contour(:,1),wavevectors_contour(:,2));
        end
    end

    % Save plot points if requested
    if isSavePlotPoints
        plot_points_data.(['struct_' num2str(struct_idx) '_wavevectors_contour']) = wavevectors_contour;
        plot_points_data.(['struct_' num2str(struct_idx) '_frequencies_contour']) = frequencies_contour;
        plot_points_data.(['struct_' num2str(struct_idx) '_contour_param']) = contour_info.wavevector_parameter;
        plot_points_data.(['struct_' num2str(struct_idx) '_use_interpolation']) = true;  % MATLAB uses interpolation
    end

    %% Plot the dispersion relation (as originally computed) on the IBZ contour

    fig = figure();
    ax = axes(fig);
    plot(ax,contour_info.wavevector_parameter,frequencies_contour)
    xlabel(ax,'wavevector contour parameter [-]')
    ylabel(ax, 'frequency [Hz]')
    title('original')

    for i = 0:contour_info.N_segment
        xline(ax,i)
    end

    if isExportPng
        png_path = fullfile(output_root, 'dispersion', [num2str(struct_idx) '.png']);
        if ~isfolder(fileparts(png_path))
            mkdir(fileparts(png_path))
        end
        exportgraphics(fig,png_path,'Resolution',png_resolution)
    end

    %% Plot the dispersion relation (reconstructed with eigenvector) on the IBZ contour (if available)

    if can_reconstruct
        fig = figure();
        ax = axes(fig);
        p_ = plot(ax,contour_info.wavevector_parameter,frequencies_contour,'LineStyle','-','Marker','o','Color',uint8([150 150 250]),'LineWidth',3); % first the original
        p(1) = p_(1);
        hold(ax,'on')
        p_ = plot(ax,contour_info.wavevector_parameter,frequencies_recon_contour,'LineStyle','-','Marker','.','Color',uint8([180 80 80]),'LineWidth',1); % first the original % then the reconstructed overlaying it
        p(2) = p_(1);
        xlabel(ax,'wavevector contour parameter [-]')
        ylabel(ax, 'frequency [Hz]')
        title('reconstructed from eigenvectors')

        p(1).DisplayName = 'true';
        p(2).DisplayName = 'reconstructed';

        for i = 0:contour_info.N_segment
            xline(ax,i)
        end

        legend(ax,p)

        if isExportPng
            png_path = fullfile(output_root, 'dispersion', [num2str(struct_idx) '_recon.png']);
            if ~isfolder(fileparts(png_path))
                mkdir(fileparts(png_path))
            end
            exportgraphics(fig,png_path,'Resolution',png_resolution)
        end
    end
end

% Save plot points if requested
if isSavePlotPoints && isfield(plot_points_data, 'struct_1_wavevectors_contour')
    plot_points_path = fullfile(output_root, 'plot_points.mat');
    save(plot_points_path, 'plot_points_data', '-v7.3');
    fprintf('\nPlot point locations saved to: %s\n', plot_points_path);
    
    % DEBUG: Also save plot points to debug folder
    debug_plot_points_path = fullfile(debug_output_root, 'plot_points_debug.mat');
    save(debug_plot_points_path, 'plot_points_data', '-v7.3');
    fprintf('DEBUG: Plot points also saved to: %s\n', debug_plot_points_path);
end

% DEBUG: Final summary
fprintf('\n========================================\n');
fprintf('DEBUG MODE COMPLETE\n');
fprintf('========================================\n');
fprintf('Debug output folder: %s\n', debug_output_root);
fprintf('All computed objects have been saved.\n');
fprintf('========================================\n\n');