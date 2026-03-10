clear; clc;

repo_root = fileparts(mfilename('fullpath'));
manifest_path = fullfile(repo_root, 'OUTPUT', 'fixed_geometry_samples', 'fixed_geometry_manifest.mat');
n_plot = 5;

if ~exist(manifest_path, 'file')
    error('Manifest not found: %s', manifest_path);
end

M = load(manifest_path);
n_total = numel(M.matlab_output_path);
n_use = min(n_plot, n_total);

fprintf('Generating MATLAB dispersion plots for %d samples...\n', n_use);

for i = 1:n_use
    data_fn = char(M.matlab_output_path{i});
    if ~exist(data_fn, 'file')
        warning('Missing MATLAB output for plotting: %s', data_fn);
        continue;
    end
    fprintf('  [%d/%d] plot_dispersion for %s\n', i, n_use, data_fn);
    fid = fopen(fullfile(repo_root, 'temp_data_fn.txt'), 'w');
    fprintf(fid, '%s', data_fn);
    fclose(fid);
    run(fullfile(repo_root, '2D-dispersion-mat', 'plot_dispersion.m'));
end

if exist(fullfile(repo_root, 'temp_data_fn.txt'), 'file')
    delete(fullfile(repo_root, 'temp_data_fn.txt'));
end

fprintf('MATLAB_PLOT_DONE_COUNT=%d\n', n_use);
