clear; close all;

% demonstrates how eigenvectors are indexed (dof ordering)
%
% As a validation check, after you have written your Python script to convert
% your eigenvector format to my eigenvector format, run the data through
% this script and make sure the images look exactly the same as when you
% plot them in your Python plotting scripts.

% dispersion_library_path = '../../';
% addpath(dispersion_library_path)

data_fn = "D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1\out_binarized_1_predictions.mat";
% data_fn = "D:\Research\NO-2D-Metamaterials\test_matlab\eigenvectors.mat";
% data_fn = "generate_dispersion_dataset_Han\OUTPUT\output 23-Sep-2025 12-31-16\binarized 23-Sep-2025 12-31-16.mat";
% data_fn = "generate_dispersion_dataset_Han\OUTPUT\output 15-Sep-2025 15-36-03\continuous 15-Sep-2025 15-36-03.mat";
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

% pick an eigenvector
struct_idx = 1;
wv_idx = 5;
band_idx = 4;

% get the eigenvector
%
% eigvec is one single eigenvector,
% that means it corresponds to one specific design
% at one specific wavevector
% on one specific band (eigenvalue index)
%
% Note: h5read automatically transposes HDF5 dimensions to MATLAB column-major order
% HDF5 order (struct, band, wv, dof) becomes MATLAB order (dof, wv, band, struct)
eigvec = data.EIGENVECTOR_DATA(:,wv_idx,band_idx,struct_idx);

% get each displacement component from the eigenvector
% u = horizontal displacement
% v = vertical displacement
% starting at first entry, eigvec alternates u-v-u-v-u-v-...
u = eigvec(1:2:end);
v = eigvec(2:2:end);

% reshape
% note that matlab is column major. 
% I don't know what convention numpy/python follow.
%
% Column-major vs row-major is *extremely* important to pay attention to
% here, because it directly affects how arrays are reshaped.
design_size = size(data.designs); % When you do your sanity check, you may have to replace this line
u = reshape(u,design_size(1:2)); % only need first two entries of design size because the third entry is for E,rho,nu
v = reshape(v,design_size(1:2)); % apply same to v

% plot
fig = figure();
ax = axes(fig);
imagesc(ax,real(u))
title('real(u)')
colorbar
daspect([1 1 1])

fig = figure();
ax = axes(fig);
imagesc(ax,imag(u))
title('imag(u)')
colorbar
daspect([1 1 1])

fig = figure();
ax = axes(fig);
imagesc(ax,real(v))
title('real(v)')
colorbar
daspect([1 1 1])

fig = figure();
ax = axes(fig);
imagesc(ax,imag(v))
title('imag(v)')
colorbar
daspect([1 1 1])