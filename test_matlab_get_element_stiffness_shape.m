% Test what shape get_element_stiffness_VEC returns for vectorized inputs
% This will help understand the transpose behavior

E_test = [100e9, 200e9, 300e9];  % 3 elements
nu_test = [0.3, 0.3, 0.3];
t_test = 1.0;

fprintf('Testing get_element_stiffness_VEC with vectorized inputs\n');
fprintf('E = [%.2e, %.2e, %.2e] (3 elements)\n', E_test(1), E_test(2), E_test(3));

% Call the function
k_ele = get_element_stiffness(E_test(:), nu_test(:), t_test);

fprintf('\nResult BEFORE transpose:\n');
fprintf('  Shape: %s\n', mat2str(size(k_ele)));
fprintf('  Total elements: %d\n', numel(k_ele));

% Check if it's (N_ele, 64) - each row is a flattened 8x8 matrix
if size(k_ele, 2) == 64 && size(k_ele, 1) == length(E_test)
    fprintf('  Format: (N_ele=3, 64) - each row is a flattened 8x8 element matrix\n');
    fprintf('  First element (first 10 values): ');
    fprintf('%.6e ', k_ele(1, 1:10));
    fprintf('\n');
elseif size(k_ele, 1) == 64 && size(k_ele, 2) == length(E_test)
    fprintf('  Format: (64, N_ele=3) - each column is a flattened 8x8 element matrix\n');
elseif ndims(k_ele) == 3
    fprintf('  Format: 3D array, shape %s\n', mat2str(size(k_ele)));
end

% Now transpose
k_ele_transposed = k_ele';

fprintf('\nResult AFTER transpose:\n');
fprintf('  Shape: %s\n', mat2str(size(k_ele_transposed)));

% Flatten
k_ele_flat = k_ele_transposed(:);

fprintf('\nResult AFTER flattening (k_ele_transposed(:)):\n');
fprintf('  Length: %d (should be 3*64 = 192)\n', length(k_ele_flat));
fprintf('  First 20 values (element 0, positions 0-19):\n');
fprintf('  ');
for i = 1:min(20, length(k_ele_flat))
    fprintf('%.6e ', k_ele_flat(i));
end
fprintf('\n');

fprintf('\n  Values at indices 1, 65, 129 (should be first value of each element):\n');
if length(k_ele_flat) >= 129
    fprintf('    Index 1: %.6e\n', k_ele_flat(1));
    fprintf('    Index 65: %.6e\n', k_ele_flat(65));
    fprintf('    Index 129: %.6e\n', k_ele_flat(129));
    fprintf('  This tells us how elements are interleaved\n');
end

