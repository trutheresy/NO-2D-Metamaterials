% Script to run MATLAB dispersion plotting scripts and generate plot_points.mat files
% 
% This script runs both plot_dispersion.m and plot_dispersion_from_predictions.m
% to generate plot_points.mat files for comparison with Python results.
%
% Usage: In MATLAB, navigate to this directory and run:
%   run_matlab_scripts

fprintf('======================================================================\n');
fprintf('Running MATLAB Dispersion Plotting Scripts\n');
fprintf('======================================================================\n\n');

% Get current directory
current_dir = pwd;
matlab_scripts_dir = fullfile(current_dir, '2D-dispersion-han');

% Change to MATLAB scripts directory
cd(matlab_scripts_dir);

%% Case 1: Without eigenfrequencies (reconstructed from eigenvectors)
fprintf('\n');
fprintf('======================================================================\n');
fprintf('CASE 1: Without eigenfrequencies (plot_dispersion_from_predictions.m)\n');
fprintf('======================================================================\n');

% Set data file path for Case 1
% Try multiple possible locations
case1_data_file = fullfile(current_dir, 'data', 'out_test_10_verify', 'out_binarized_1', 'out_binarized_1_predictions.mat');
if ~exist(case1_data_file, 'file')
    % Try alternative location
    case1_data_file = fullfile(current_dir, 'data', 'out_test_10', 'out_binarized_1', 'out_binarized_1_predictions.mat');
end

% Check if file exists
if ~exist(case1_data_file, 'file')
    fprintf('WARNING: Case 1 data file not found: %s\n', case1_data_file);
    fprintf('  Skipping Case 1...\n');
else
    % Create temp file with data path
    temp_file = fullfile(matlab_scripts_dir, 'temp_data_fn.txt');
    fid = fopen(temp_file, 'w');
    if fid ~= -1
        fprintf(fid, '%s', case1_data_file);
        fclose(fid);
        
        fprintf('Running plot_dispersion_from_predictions.m...\n');
        fprintf('  Data file: %s\n', case1_data_file);
        
        try
            plot_dispersion_from_predictions;
            fprintf('  ✓ Case 1 completed successfully!\n');
        catch ME
            fprintf('  ✗ Case 1 failed: %s\n', ME.message);
        end
        
        % Clean up temp file
        temp_file_check = fullfile(matlab_scripts_dir, 'temp_data_fn.txt');
        if exist(temp_file_check, 'file')
            delete(temp_file_check);
        end
    else
        fprintf('  ERROR: Could not create temp file\n');
    end
end

%% Case 2: With eigenfrequencies
fprintf('\n');
fprintf('======================================================================\n');
fprintf('CASE 2: With eigenfrequencies (plot_dispersion.m)\n');
fprintf('======================================================================\n');

% Set data file path for Case 2
case2_data_file = fullfile(current_dir, 'data', 'out_test_10_matlab', 'out_binarized_1.mat');

% Check if file exists
if ~exist(case2_data_file, 'file')
    fprintf('WARNING: Case 2 data file not found: %s\n', case2_data_file);
    fprintf('  Skipping Case 2...\n');
else
    % Create temp file with data path
    temp_file = fullfile(matlab_scripts_dir, 'temp_data_fn.txt');
    fid = fopen(temp_file, 'w');
    if fid ~= -1
        fprintf(fid, '%s', case2_data_file);
        fclose(fid);
        
        fprintf('Running plot_dispersion.m...\n');
        fprintf('  Data file: %s\n', case2_data_file);
        
        try
            plot_dispersion;
            fprintf('  ✓ Case 2 completed successfully!\n');
        catch ME
            fprintf('  ✗ Case 2 failed: %s\n', ME.message);
        end
        
        % Clean up temp file
        temp_file_check = fullfile(pwd, 'temp_data_fn.txt');
        if exist(temp_file_check, 'file')
            delete(temp_file_check);
        end
    else
        fprintf('  ERROR: Could not create temp file\n');
    end
end

%% Summary
fprintf('\n');
fprintf('======================================================================\n');
fprintf('Summary\n');
fprintf('======================================================================\n');

% Check for generated plot_points.mat files
plots_dir = fullfile(current_dir, 'plots');
case1_output = fullfile(plots_dir, 'out_binarized_1_mat', 'plot_points.mat');
case2_output = fullfile(plots_dir, 'out_binarized_1_mat', 'plot_points.mat');

fprintf('\nGenerated files:\n');
if exist(case1_output, 'file')
    fprintf('  ✓ Case 1: %s\n', case1_output);
else
    fprintf('  ✗ Case 1: NOT FOUND\n');
end

if exist(case2_output, 'file')
    fprintf('  ✓ Case 2: %s\n', case2_output);
else
    fprintf('  ✗ Case 2: NOT FOUND\n');
    fprintf('    (Note: Case 2 may overwrite Case 1 if same output directory)\n');
end

fprintf('\nNext step: Run "python compare_plot_points.py" to compare results.\n');
fprintf('======================================================================\n');

% Return to original directory
cd(current_dir);

