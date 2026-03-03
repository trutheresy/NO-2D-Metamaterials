% Test MATLAB transpose behavior on 3D arrays
% Understanding how AllLEle is transposed

% Simulate what get_element_stiffness_VEC might return
% If it returns (N_ele, 64) - each row is a flattened 8x8 matrix
N_ele = 4;
AllLEle_test = reshape(1:(N_ele*64), N_ele, 64); % Each row is one element's flattened matrix

fprintf('Test: Understanding MATLAB transpose on element matrices\n');
fprintf('AllLEle_test shape (before transpose): %s\n', mat2str(size(AllLEle_test)));

% MATLAB does: AllLEle = get_element_stiffness_VEC(E(:),nu(:),t)'
AllLEle_transposed = AllLEle_test';

fprintf('AllLEle_transposed shape (after transpose): %s\n', mat2str(size(AllLEle_transposed)));
fprintf('Each column now represents one element\n');

% Then: value_K = AllLEle(:)
value_K = AllLEle_transposed(:);

fprintf('value_K length: %d\n', length(value_K));
fprintf('First 20 values: ');
fprintf('%d ', value_K(1:20));
fprintf('\n');

% Compare with direct flatten (row-major, Python way)
value_K_rowmajor = AllLEle_test(:);
fprintf('value_K_rowmajor (direct flatten) length: %d\n', length(value_K_rowmajor));
fprintf('First 20 values: ');
fprintf('%d ', value_K_rowmajor(1:20));
fprintf('\n');

fprintf('\nDifference: MATLAB flattens column-wise (elements interleaved), Python flattens row-wise (elements sequential)\n');

