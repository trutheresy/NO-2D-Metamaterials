% MATLAB script to run plot_dispersion.m for Case 2 (with eigenfrequencies)
% Dataset: 2D-dispersion-han/OUTPUT/out_test_10/out_binarized_1.mat

% Get current directory
current_dir = pwd;

% Change to MATLAB library directory
matlab_lib_dir = fullfile(current_dir, '2D-dispersion-han');
if ~exist(matlab_lib_dir, 'dir')
    error('MATLAB library directory not found: %s', matlab_lib_dir);
end
cd(matlab_lib_dir);

% Set data file path (absolute path)
data_file = fullfile(current_dir, '2D-dispersion-han', 'OUTPUT', 'out_test_10', 'out_binarized_1.mat');

% Check if file exists
if ~exist(data_file, 'file')
    error('Data file not found: %s', data_file);
end

fprintf('Running plot_dispersion.m for Case 2 (with eigenfrequencies)\n');
fprintf('Data file: %s\n', data_file);
fprintf('========================================\n\n');

% Create a temporary script that sets the data path and runs plot_dispersion
temp_script = fullfile(tempdir, 'run_plot_dispersion_temp.m');
fid = fopen(temp_script, 'w');
if fid == -1
    error('Could not create temporary script file');
end

% Write script content
fprintf(fid, '%% Temporary script to run plot_dispersion.m\n');
fprintf(fid, 'data_fn = "%s";\n', strrep(data_file, '\', '\\'));
fprintf(fid, 'plot_dispersion;\n');
fclose(fid);

% Run the temporary script
try
    run(temp_script);
    fprintf('\n========================================\n');
    fprintf('MATLAB script completed successfully!\n');
catch ME
    fprintf('\n========================================\n');
    fprintf('ERROR: MATLAB script failed:\n');
    fprintf('%s\n', ME.message);
    rethrow(ME);
end

% Clean up temporary file
if exist(temp_script, 'file')
    delete(temp_script);
end

% Return to original directory
cd(current_dir);

fprintf('\nDone!\n');

