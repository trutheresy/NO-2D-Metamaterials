% Test how MATLAB's get_element_stiffness handles vectorized inputs
% Understanding what get_element_stiffness_VEC(E(:),nu(:),t) returns

E_test = [100e9, 200e9];  % Two elements
nu_test = [0.3, 0.4];
t_test = 1.0;

fprintf('Testing MATLAB get_element_stiffness with vectorized inputs\n');
fprintf('E = [%.2e, %.2e]\n', E_test(1), E_test(2));
fprintf('nu = [%.2f, %.2f]\n', nu_test(1), nu_test(2));

% Call the function (assuming it's in path or same directory)
try
    k_ele = get_element_stiffness(E_test(:), nu_test(:), t_test);
    fprintf('\nResult:\n');
    fprintf('  Shape: %s\n', mat2str(size(k_ele)));
    
    if numel(k_ele) == 64
        fprintf('  Single flattened matrix (64 elements)\n');
        fprintf('  This means it only handles the first element!\n');
    elseif numel(k_ele) == 128
        fprintf('  Two flattened matrices (128 elements)\n');
    elseif all(size(k_ele) == [8, 8])
        fprintf('  Single 8x8 matrix\n');
        fprintf('  This means it only handles the first element!\n');
    elseif size(k_ele, 3) == 2
        fprintf('  Two 8x8 matrices (3D array)\n');
    end
    
    % Try with transpose
    k_ele_transposed = k_ele';
    fprintf('\nAfter transpose:\n');
    fprintf('  Shape: %s\n', mat2str(size(k_ele_transposed)));
    
    % Flatten
    k_ele_flat = k_ele_transposed(:);
    fprintf('\nAfter flattening (k_ele_transposed(:)):\n');
    fprintf('  Length: %d\n', length(k_ele_flat));
    fprintf('  First 20 values:\n');
    fprintf('  ');
    fprintf('%.6e ', k_ele_flat(1:min(20, length(k_ele_flat))));
    fprintf('\n');
    
catch ME
    fprintf('Error: %s\n', ME.message);
    fprintf('Make sure get_element_stiffness.m is in the MATLAB path\n');
end

