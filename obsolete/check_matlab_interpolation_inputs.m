% Check what inputs MATLAB uses for scatteredInterpolant
% This script should be run in MATLAB to debug the interpolation issue

% Load the data
data_fn = 'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat';
data = load(data_fn);

% Extract wavevectors and frequencies for struct_idx = 1
struct_idx = 1;
wavevectors = double(data.WAVEVECTOR_DATA(:,:,struct_idx));  % Convert to double
frequencies = data.EIGENVALUE_DATA(:,:,struct_idx);

fprintf('Input data shapes:\n');
fprintf('  WAVEVECTOR_DATA shape: %s\n', mat2str(size(data.WAVEVECTOR_DATA)));
fprintf('  EIGENVALUE_DATA shape: %s\n', mat2str(size(data.EIGENVALUE_DATA)));
fprintf('  wavevectors shape (after extraction): %s\n', mat2str(size(wavevectors)));
fprintf('  frequencies shape (after extraction): %s\n', mat2str(size(frequencies)));

% Check if wavevectors needs transposing
% scatteredInterpolant expects (N, 2) for 2D points
if size(wavevectors, 1) == 2 && size(wavevectors, 2) > 2
    fprintf('  ⚠ wavevectors is (2, N) - needs transpose to (N, 2)\n');
    wavevectors = wavevectors';
    fprintf('  ✓ Transposed to: %s\n', mat2str(size(wavevectors)));
elseif size(wavevectors, 2) == 2 && size(wavevectors, 1) > 2
    fprintf('  ✓ wavevectors is already (N, 2)\n');
else
    fprintf('  ⚠ Unexpected wavevectors shape\n');
end

% Check frequencies shape
if size(frequencies, 1) == size(wavevectors, 1) && size(frequencies, 2) > 1
    fprintf('  ✓ frequencies is (N_wv, N_eig)\n');
elseif size(frequencies, 2) == size(wavevectors, 1) && size(frequencies, 1) > 1
    fprintf('  ⚠ frequencies is (N_eig, N_wv) - needs transpose\n');
    frequencies = frequencies';
    fprintf('  ✓ Transposed to: %s\n', mat2str(size(frequencies)));
else
    fprintf('  ⚠ Unexpected frequencies shape\n');
end

% Create interpolant
interp_method = 'linear';
extrap_method = 'linear';
N_eig = size(frequencies, 2);
fprintf('\nCreating interpolants for %d bands...\n', N_eig);

interp_true = cell(N_eig,1);
for eig_idx = 1:N_eig
    try
        interp_true{eig_idx} = scatteredInterpolant(wavevectors, frequencies(:,eig_idx), interp_method, extrap_method);
        fprintf('  ✓ Interpolant %d created successfully\n', eig_idx);
    catch ME
        fprintf('  ✗ Interpolant %d failed: %s\n', eig_idx, ME.message);
    end
end

% Test interpolation on a few points
fprintf('\nTesting interpolation...\n');
test_points = [0, 0; 1, 0; 0, 1];
fprintf('  Test points shape: %s\n', mat2str(size(test_points)));

for eig_idx = 1:min(3, N_eig)
    try
        result = interp_true{eig_idx}(test_points(:,1), test_points(:,2));
        fprintf('  Band %d: result shape = %s, NaN count = %d/%d\n', ...
            eig_idx, mat2str(size(result)), sum(isnan(result)), numel(result));
    catch ME
        fprintf('  Band %d: interpolation failed: %s\n', eig_idx, ME.message);
    end
end

