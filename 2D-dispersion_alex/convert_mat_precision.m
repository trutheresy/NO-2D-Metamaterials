function output_file = convert_mat_precision(input_file, precision, output_file)
% CONVERT_MAT_PRECISION Convert numeric arrays in a .mat file to float16 or float32.
%
% Usage:
%   % Default to float16, auto-named output alongside input
%   convert_mat_precision('input.mat')
%
%   % Explicit precision ('half' or 'single') and auto-named output
%   convert_mat_precision('input.mat', 'half')
%   convert_mat_precision('input.mat', 'single')
%
%   % Explicit output path
%   convert_mat_precision('input.mat', 'half', 'output_f16.mat')
%
% Notes:
% - Structs are left unchanged
% - Sparse matrices are preserved as sparse double (no float16/float32 sparse)
% - Skips variables: CONSTITUTIVE_DATA, rng_seed_offset, imag_tol
% - Saves with -v7.3

    if nargin < 1
        error('Must provide input_file');
    end
    if ~exist('precision','var') || isempty(precision)
        precision = 'half'; % default
    end
    if ~ischar(precision) && ~isstring(precision)
        error('precision must be ''half'' or ''single''');
    end
    precision = char(lower(string(precision)));
    if ~ismember(precision, {'half','single'})
        error('precision must be ''half'' or ''single''');
    end

    if ~exist(input_file, 'file')
        error('Input file does not exist: %s', input_file);
    end

    % If half is requested but unavailable, fall back to single
    if strcmp(precision,'half')
        has_half = (exist('half','file') == 2) || (exist('half','class') == 8);
        if ~has_half
            fprintf('half() not available in this MATLAB. Falling back to single (float32).\n');
            precision = 'single';
        end
    end

    % Derive output file if not provided
    if ~exist('output_file','var') || isempty(output_file)
        [in_dir, in_name, ~] = fileparts(input_file);
        switch precision
            case 'half',   suffix = '_f16';
            case 'single', suffix = '_f32';
        end
        output_file = fullfile(in_dir, [in_name suffix '.mat']);
    end

    fprintf('Loading data from: %s\n', input_file);
    data = load(input_file);
    var_names = fieldnames(data);
    fprintf('Found %d variables to process\n', length(var_names));

    % Variables to skip (never convert these)
    skip_vars = {'CONSTITUTIVE_DATA', 'rng_seed_offset', 'imag_tol'};

    % Process variables
    for i = 1:length(var_names)
        var_name = var_names{i};
        var_data = data.(var_name);

        if ismember(var_name, skip_vars)
            fprintf('  %-30s: %s (skipped)\n', var_name, class(var_data));
            continue
        end

        [converted_data, was_converted] = convert_value(var_data, var_name, precision);
        if was_converted
            tgt = ternary(strcmp(precision,'half'),'half','single');
            fprintf('  %-30s: %s -> %s\n', var_name, class(var_data), tgt);
            data.(var_name) = converted_data;
        else
            fprintf('  %-30s: %s (unchanged)\n', var_name, class(var_data));
        end
    end

    fprintf('\nSaving converted data to: %s\n', output_file);
    save(output_file, '-struct', 'data', '-v7.3');
    fprintf('Conversion complete!\n');

    % Size report
    try
        input_info = dir(input_file);
        output_info = dir(output_file);
        fprintf('\nFile size comparison:\n');
        fprintf('  Input:  %.2f MB\n', input_info.bytes / 1024^2);
        fprintf('  Output: %.2f MB\n', output_info.bytes / 1024^2);
        fprintf('  Reduction: %.1f%%\n', 100 * (1 - output_info.bytes / input_info.bytes));
    catch
        % ignore
    end
end

function [converted_data, was_converted] = convert_value(data, var_name, precision)
    % Convert recursively using desired precision
    was_converted = false;

    % Sparse numerics: keep as-is (MATLAB lacks sparse half/single)
    if issparse(data) && isnumeric(data)
        converted_data = data;
        return
    end

    if isnumeric(data)
        switch precision
            case 'half'
                if ~isa(data, 'half')
                    % half() requires MATLAB Coder, GPU Coder, or Fixed-Point Designer
                    try
                        converted_data = half(data);
                        was_converted = true;
                    catch ME
                        error(['Converting %s to half failed: %s. Ensure required toolboxes are installed.'], var_name, ME.message);
                    end
                else
                    converted_data = data;
                end
            case 'single'
                if ~isa(data, 'single')
                    converted_data = single(data);
                    was_converted = true;
                else
                    converted_data = data;
                end
        end
        return
    end

    if isstruct(data)
        % Leave whole struct unchanged
        converted_data = data;
        return
    end

    if iscell(data)
        converted_data = data;
        for i = 1:numel(data)
            [cell_val, cell_conv] = convert_value(data{i}, sprintf('%s{%d}', var_name, i), precision);
            converted_data{i} = cell_val;
            was_converted = was_converted || cell_conv;
        end
        return
    end

    if isa(data, 'containers.Map')
        % Preserve keys, convert values recursively
        converted_data = containers.Map;
        k = keys(data);
        for i = 1:numel(k)
            key = k{i};
            [val2, conv2] = convert_value(data(key), sprintf('%s(''%s'')', var_name, key), precision);
            converted_data(key) = val2;
            was_converted = was_converted || conv2;
        end
        return
    end

    % Other types unchanged
    converted_data = data;
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
