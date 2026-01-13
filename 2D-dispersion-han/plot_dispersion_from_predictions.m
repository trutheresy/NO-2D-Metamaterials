clear; close all;

% plot_dispersion_from_predictions.m
%
% This script is specifically designed for prediction data that has no EIGENVALUE_DATA.
% It will reconstruct the EIGENVALUE_DATA by:
%   1. Computing K and M matrices from geometry using get_system_matrices()
%   2. Computing T matrices for each wavevector using get_transformation_matrix()
%   3. Reconstructing eigenvalues from eigenvectors using the method from plot_dispersion.m
%   4. Then plotting the dispersion curves
%
% This script relies on existing MATLAB library functions (get_system_matrices,
% get_transformation_matrix, etc.) as working, tested, ground truth sources of functionality.
% No library functions are modified.

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

% Save plot points if requested
if isSavePlotPoints && isfield(plot_points_data, 'struct_1_wavevectors_contour')
    plot_points_path = fullfile(output_root, 'plot_points.mat');
    save(plot_points_path, 'plot_points_data', '-v7.3');
    fprintf('\nPlot point locations saved to: %s\n', plot_points_path);
end

% flags
isExportPng = true;
png_resolution = 150;
isSavePlotPoints = true;  % Save plot point locations for comparison with Python

% Output root: save under plots/<dataset_name>_mat/... (in current directory)
output_root = fullfile(pwd, 'plots', [fn '_mat']);

% Make plots for one unit cell or multiple
struct_idxs = 1:10;

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

% Check if EIGENVALUE_DATA is missing and needs to be reconstructed
if ~isfield(data, 'EIGENVALUE_DATA') || isempty(data.EIGENVALUE_DATA)
    fprintf('\n========================================\n');
    fprintf('EIGENVALUE_DATA not found. Reconstructing from eigenvectors...\n');
    fprintf('========================================\n');
    
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
                % Handle column vector of uint16 characters
                if size(data.const.design_scale, 2) == 1
                    design_scale = char(data.const.design_scale(:,1)');
                else
                    design_scale = char(data.const.design_scale(1,:));
                end
            else
                design_scale = char(data.const.design_scale);
            end
            % Ensure it's a row vector string and trim whitespace
            design_scale = strtrim(design_scale(:)');
        else
            design_scale = 'linear';
        end
        if isfield(data.const, 'symmetry_type')
            symmetry_type = char(data.const.symmetry_type(:,1)');
        else
            symmetry_type = 'p4mm';
        end
        if isfield(data.const, 'wavevectors')
            wavevectors_const = data.const.wavevectors;
        else
            % Extract from WAVEVECTOR_DATA (use first structure's wavevectors)
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
            % Convert from uint16 character array to string
            if isnumeric(data.const.design_scale)
                % Handle column vector of uint16 characters
                if size(data.const.design_scale, 2) == 1
                    design_scale = char(data.const.design_scale(:,1)');
                else
                    design_scale = char(data.const.design_scale(1,:));
                end
            else
                design_scale = char(data.const.design_scale);
            end
            % Ensure it's a row vector string and trim whitespace
            design_scale = strtrim(design_scale(:)');
        else
            design_scale = 'linear';
        end
        if isfield(data.const, 'symmetry_type')
            symmetry_type = char(data.const.symmetry_type(:,1)');
        else
            symmetry_type = 'p4mm';
        end
        if isfield(data.const, 'wavevectors')
            wavevectors_const = data.const.wavevectors;
        else
            wavevectors_const = data.WAVEVECTOR_DATA(:,:,1)';
        end
    end
    
    % Determine number of structures and wavevectors
    N_struct_total = size(data.designs, ndims(data.designs));
    N_wv = size(wavevectors_const, 1);
    
    % Initialize EIGENVALUE_DATA array: (N_wv, N_eig, N_struct) in MATLAB order
    % But h5read transposes to (N_struct, N_eig, N_wv), so we'll create it in that order
    EIGENVALUE_DATA = zeros(N_struct_total, N_eig, N_wv);
    
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
    
    % Debug: Check design_scale value
    fprintf('  design_scale value: "%s" (length: %d)\n', design_scale, length(design_scale));
    if ~strcmp(design_scale, 'linear') && ~strcmp(design_scale, 'log')
        warning('design_scale is "%s", expected "linear" or "log". Forcing to "linear".', design_scale);
        design_scale = 'linear';
        const_template.design_scale = design_scale;
    end
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
    for struct_idx = 1:N_struct_total
        fprintf('Reconstructing EIGENVALUE_DATA for structure %d/%d...\n', struct_idx, N_struct_total);
        
        % Extract geometry for this structure
        if size(data.designs, 1) == size(data.designs, 2) && size(data.designs, 3) == 3
            % MATLAB order: (N_pix, N_pix, 3, N_struct)
            design_3d = data.designs(:,:,:,struct_idx);
        else
            % HDF5 order: (N_struct, 3, N_pix, N_pix)
            design_3d = permute(data.designs(struct_idx, :, :, :), [3, 4, 2, 1]); % (N_pix, N_pix, 3)
        end
        
        % Set up const structure for this structure
        const = const_template;
        const.design = design_3d;
        
        % Compute K and M matrices using library function
        fprintf('  Computing K and M matrices...\n');
        [K, M] = get_system_matrices(const);
        
        % Reconstruct frequencies for each wavevector
        frequencies_recon = zeros(N_wv, N_eig);
        for wv_idx = 1:N_wv
            % Ensure wavevector is a row vector of doubles
            wavevector = double(wavevectors_const(wv_idx, :));
            if size(wavevector, 1) > size(wavevector, 2)
                wavevector = wavevector';
            end
            
            % Compute T matrix using library function
            T = get_transformation_matrix(wavevector, const);
            
            % Transform to reduced space
            Kr = T'*K*T;
            Mr = T'*M*T;
            
            % Reconstruct eigenvalues from eigenvectors
            % EIGENVECTOR_DATA is in MATLAB order: (dof, wv, band, struct) after h5read transpose
            % So we access it as: data.EIGENVECTOR_DATA(:, wv_idx, band_idx, struct_idx)
            for band_idx = 1:N_eig
                eigvec = data.EIGENVECTOR_DATA(:, wv_idx, band_idx, struct_idx);
                eigvec = double(eigvec); % Cast to double for computation
                
                % Reconstruct eigenvalue: eigval = norm(Kr*eigvec)/norm(Mr*eigvec)
                % This is equivalent to: Kr*eigvec = eigval*Mr*eigvec
                eigval = norm(Kr*eigvec)/norm(Mr*eigvec);
                
                % Convert to frequency
                frequencies_recon(wv_idx, band_idx) = sqrt(eigval)/(2*pi);
            end
        end
        
        % Store in EIGENVALUE_DATA (transpose to match expected format)
        % MATLAB expects (N_wv, N_eig, N_struct), but we're storing as (N_struct, N_eig, N_wv)
        EIGENVALUE_DATA(struct_idx, :, :) = frequencies_recon';
    end
    
    % Transpose EIGENVALUE_DATA to MATLAB order: (N_wv, N_eig, N_struct)
    % But since h5read will transpose it back, we'll keep it as (N_struct, N_eig, N_wv)
    % and access it accordingly
    data.EIGENVALUE_DATA = permute(EIGENVALUE_DATA, [3, 2, 1]); % (N_wv, N_eig, N_struct)
    
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
    % Fix: WAVEVECTOR_DATA shape is (N_struct, 2, N_wv), so use (struct_idx, :, :) then transpose
    % Old (wrong): wavevectors = double(data.WAVEVECTOR_DATA(:,:,struct_idx));  % Gives (N_struct, 2)
    wavevectors = double(data.WAVEVECTOR_DATA(struct_idx, :, :));  % Gives (2, N_wv)
    wavevectors = wavevectors';  % Transpose to (N_wv, 2) for scatteredInterpolant

    % EIGENVALUE_DATA should now be available (either from file or reconstructed)
    if isfield(data, 'EIGENVALUE_DATA') && ~isempty(data.EIGENVALUE_DATA)
        disp('size(data.EIGENVALUE_DATA)')
        disp(size(data.EIGENVALUE_DATA))
        % Fix: EIGENVALUE_DATA shape is (N_struct, N_eig, N_wv), so use (struct_idx, :, :) then transpose
        % Old (wrong): frequencies = double(data.EIGENVALUE_DATA(:,:,struct_idx));  % Gives (N_struct, N_eig)
        frequencies = double(data.EIGENVALUE_DATA(struct_idx, :, :));  % Gives (N_eig, N_wv)
        frequencies = frequencies';  % Transpose to (N_wv, N_eig)
    else
        error('EIGENVALUE_DATA is still missing after reconstruction attempt');
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
    for eig_idx = 1:N_eig
        interp_true{eig_idx} = scatteredInterpolant(wavevectors,frequencies(:,eig_idx),interp_method,extrap_method);
    end

    % Get the IBZ contour wave vectors. NOTE: These only exist for some symmetry
    % groups. symmetry = 'p4mm' has a valid IBZ contour.
    N_k = 10; % Number of interpolation query points per contour segment
    % Get symmetry_type and a from const (handle both struct and array formats)
    if isstruct(data.const)
        a_val = data.const.a(1,1);
        if isfield(data.const, 'symmetry_type')
            % symmetry_type is stored as uint16 character array, convert to string
            if isnumeric(data.const.symmetry_type)
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
            sym_type = char(data.const.symmetry_type(:,1)');
        else
            sym_type = 'p4mm';  % Default
        end
    end
    % Debug: Check symmetry_type value
    fprintf('  symmetry_type value: "%s" (length: %d)\n', sym_type, length(sym_type));
    if ~strcmp(sym_type, 'p4mm') && ~strcmp(sym_type, 'none')
        warning('symmetry_type is "%s", expected "p4mm" or "none". Forcing to "p4mm".', sym_type);
        sym_type = 'p4mm';
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
    for eig_idx = 1:size(frequencies,2)
        frequencies_contour(:,eig_idx) = interp_true{eig_idx}(wavevectors_contour(:,1),wavevectors_contour(:,2));
    end

    % Save plot points if requested
    if isSavePlotPoints
        plot_points_data.(['struct_' num2str(struct_idx) '_wavevectors_contour']) = wavevectors_contour;
        plot_points_data.(['struct_' num2str(struct_idx) '_frequencies_contour']) = frequencies_contour;
        plot_points_data.(['struct_' num2str(struct_idx) '_contour_param']) = contour_info.wavevector_parameter;
        plot_points_data.(['struct_' num2str(struct_idx) '_use_interpolation']) = true;  % MATLAB uses interpolation
    end

    %% Plot the dispersion relation (reconstructed from eigenvectors) on the IBZ contour

    fig = figure();
    ax = axes(fig);
    plot(ax,contour_info.wavevector_parameter,frequencies_contour)
    xlabel(ax,'wavevector contour parameter [-]')
    ylabel(ax, 'frequency [Hz]')
    title('reconstructed from eigenvectors (predictions)')

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
end

% Save plot points if requested
if isSavePlotPoints && isfield(plot_points_data, 'struct_1_wavevectors_contour')
    plot_points_path = fullfile(output_root, 'plot_points.mat');
    save(plot_points_path, 'plot_points_data', '-v7.3');
    fprintf('\nPlot point locations saved to: %s\n', plot_points_path);
end

