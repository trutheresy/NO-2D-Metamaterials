clear; clc;

repo_root = fileparts(mfilename('fullpath'));
manifest_path = fullfile(repo_root, 'OUTPUT', 'fixed_geometry_samples', 'fixed_geometry_manifest.mat');
max_to_run = inf; % set finite value for quick smoke runs

if ~exist(manifest_path, 'file')
    error('Manifest not found: %s', manifest_path);
end

manifestData = load(manifest_path);
n_total = numel(manifestData.fixed_geometry_path);
n_run = min(n_total, max_to_run);

fprintf('Running fixed-geometry MATLAB generation for %d/%d samples...\n', n_run, n_total);
success = false(n_run,1);
error_msg = strings(n_run,1);

for i = 1:n_run
    fixed_geometry_path = char(manifestData.fixed_geometry_path{i});
    matlab_output_path = char(manifestData.matlab_output_path{i});
    fprintf('  [%d/%d] %s -> %s\n', i, n_run, fixed_geometry_path, matlab_output_path);
    try
        run(fullfile(repo_root, 'generate_dispersion_dataset_fixed_geometry.m'));
        success(i) = true;
    catch ME
        success(i) = false;
        error_msg(i) = string(ME.message);
        warning('Failed sample %d: %s', i, ME.message);
    end
end

summary_path = fullfile(repo_root, 'OUTPUT', 'fixed_geometry_samples', 'fixed_geometry_run_summary.mat');
save(summary_path, 'manifest_path', 'n_total', 'n_run', 'success', 'error_msg');

fprintf('FIXED_GEOM_RUN_SUMMARY=%s\n', summary_path);
fprintf('FIXED_GEOM_SUCCESS_COUNT=%d\n', sum(success));
