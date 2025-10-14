function convert_mat_to_float16(input_file, output_file)
% CONVERT_MAT_TO_FLOAT16 Converts all higher precision numeric data in a .mat file to single/float32
%
% Usage:
%   convert_mat_to_float16(input_file, output_file)
%
% Parameters:
%   input_file  - Path to input .mat file
%   output_file - Path to output .mat file (will be created/overwritten)
%
% The function will:
%   - Load all variables from the input .mat file
%   - Convert any numeric data that is float64 or integer types to single (float32)
%   - Leave single/float32 data unchanged
%   - Preserve non-numeric data (strings, structs, cells, etc.) as-is
%   - Save to the output file
%
% Note: Uses single precision (float32) instead of half (float16) because
%       half precision requires special toolboxes not available in base MATLAB
%
% Example:
%   convert_mat_to_float16('data.mat', 'data_float32.mat')

    % Input validation
    if nargin < 2
        error('Must provide both input and output file paths');
    end
    
    if ~exist(input_file, 'file')
        error('Input file does not exist: %s', input_file);
    end
    
    fprintf('Loading data from: %s\n', input_file);
    
    % Load all variables from the input file
    data = load(input_file);
    var_names = fieldnames(data);
    
    fprintf('Found %d variables to process\n', length(var_names));
    
    % Variables to skip (never convert these)
    skip_vars = {'CONSTITUTIVE_DATA', 'rng_seed_offset', 'imag_tol'};
    
    % Process each variable
    for i = 1:length(var_names)
        var_name = var_names{i};
        var_data = data.(var_name);
        
        % Skip specific variables
        if ismember(var_name, skip_vars)
            fprintf('  %-30s: %s (skipped)\n', var_name, class(var_data));
            continue;
        end
        
        % Convert to float16 if appropriate
        [converted_data, was_converted] = convert_to_float16(var_data, var_name);
        
        if was_converted
            fprintf('  %-30s: %s -> single\n', var_name, class(var_data));
        else
            fprintf('  %-30s: %s (unchanged)\n', var_name, class(var_data));
        end
        
        % Update the data structure
        data.(var_name) = converted_data;
    end
    
    fprintf('\nSaving converted data to: %s\n', output_file);
    
    % Save all variables to output file
    save(output_file, '-struct', 'data', '-v7.3');
    
    fprintf('Conversion complete!\n');
    
    % Display file size comparison
    input_info = dir(input_file);
    output_info = dir(output_file);
    fprintf('\nFile size comparison:\n');
    fprintf('  Input:  %.2f MB\n', input_info.bytes / 1024^2);
    fprintf('  Output: %.2f MB\n', output_info.bytes / 1024^2);
    fprintf('  Reduction: %.1f%%\n', 100 * (1 - output_info.bytes / input_info.bytes));
end

function [converted_data, was_converted] = convert_to_float16(data, var_name)
    % Recursively convert data to single precision (float32)
    was_converted = false;
    
    if issparse(data) && isnumeric(data)
        % Handle sparse matrices FIRST (before other numeric checks)
        % Note: MATLAB doesn't support sparse single precision, so we keep as sparse double
        % This is a trade-off: sparse storage vs. precision
        % fprintf('  Warning: %s is sparse - keeping as sparse double (no single sparse support)\n', var_name);
        converted_data = data;
        was_converted = false;
    elseif isnumeric(data) && ~isa(data, 'single')
        % Convert numeric data (except if already single precision)
        % Note: Using 'single' (float32) because 'half' requires special toolboxes
        if isfloat(data) || isinteger(data)
            converted_data = single(data);
            was_converted = true;
        else
            converted_data = data;
        end
    elseif isstruct(data)
        % Leave structs unchanged (don't convert fields)
        converted_data = data;
        was_converted = false;
    elseif iscell(data)
        % Recursively process cell arrays
        converted_data = data;
        for i = 1:numel(data)
            [converted_cell, cell_converted] = convert_to_float16(data{i}, ...
                sprintf('%s{%d}', var_name, i));
            converted_data{i} = converted_cell;
            was_converted = was_converted || cell_converted;
        end
    elseif isa(data, 'containers.Map')
        % Handle containers.Map (like CONSTITUTIVE_DATA)
        converted_data = containers.Map;
        keys_list = keys(data);
        for i = 1:length(keys_list)
            key = keys_list{i};
            [converted_value, value_converted] = convert_to_float16(data(key), ...
                sprintf('%s(''%s'')', var_name, key));
            converted_data(key) = converted_value;
            was_converted = was_converted || value_converted;
        end
    else
        % Leave other data types unchanged (strings, etc.)
        converted_data = data;
    end
end

