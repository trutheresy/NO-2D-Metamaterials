% Check what get_element_stiffness_VEC actually returns
% This is critical for understanding the transpose behavior

E_test = [100e9, 200e9];  % 2 elements
nu_test = [0.3, 0.3];
t_test = 1.0;

fprintf('Testing get_element_stiffness output format\n');
fprintf('E = [%.2e, %.2e] (2 elements)\n', E_test(1), E_test(2));

% Call the function (file is get_element_stiffness_VEC.m but function is get_element_stiffness)
k_ele = get_element_stiffness(E_test(:), nu_test(:), t_test);

fprintf('\nOutput shape: %s\n', mat2str(size(k_ele)));
fprintf('Total elements: %d\n', numel(k_ele));

% Check if it's (N_ele, 64) - each row is a flattened 8x8 matrix
if size(k_ele, 1) == length(E_test) && size(k_ele, 2) == 64
    fprintf('Format: (N_ele=2, 64) - each ROW is a flattened element matrix\n');
    fprintf('Element 0 (first row), first 8 values: ');
    fprintf('%.6e ', k_ele(1, 1:8));
    fprintf('\n');
    fprintf('Element 1 (second row), first 8 values: ');
    fprintf('%.6e ', k_ele(2, 1:8));
    fprintf('\n');
elseif size(k_ele, 1) == 64 && size(k_ele, 2) == length(E_test)
    fprintf('Format: (64, N_ele=2) - each COLUMN is a flattened element matrix\n');
elseif size(k_ele, 1) == 8 && size(k_ele, 2) == 8
    fprintf('Format: (8, 8) - only first element (not vectorized!)\n');
elseif ndims(k_ele) == 3
    fprintf('Format: 3D array %s\n', mat2str(size(k_ele)));
end

% Now transpose
k_ele_transposed = k_ele';

fprintf('\nAfter transpose:\n');
fprintf('  Shape: %s\n', mat2str(size(k_ele_transposed)));

% Flatten
k_ele_flat = k_ele_transposed(:);

fprintf('\nAfter flattening (k_ele_transposed(:)):\n');
fprintf('  Length: %d\n', length(k_ele_flat));
fprintf('  First 10 values: ');
fprintf('%.6e ', k_ele_flat(1:10));
fprintf('\n');
fprintf('  Values at indices 1, 65 (first value of each element if interleaved):\n');
fprintf('    Index 1: %.6e\n', k_ele_flat(1));
if length(k_ele_flat) >= 65
    fprintf('    Index 65: %.6e\n', k_ele_flat(65));
end

fprintf('\nIf interleaved, value[1] and value[65] should be first value of element 0 and 1\n');
fprintf('If sequential, value[1] and value[65] should be first and 65th value of element 0\n');

