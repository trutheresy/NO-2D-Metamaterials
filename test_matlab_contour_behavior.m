% Test MATLAB's get_IBZ_contour_wavevectors behavior
% This script tests how MATLAB generates contour points

N_k = 10;
a = 1.0;
symmetry_type = 'p4mm';

% Test the get_contour_from_vertices function behavior
vertices = [0 0; pi/a 0; pi/a pi/a; 0 0];

fprintf('Testing MATLAB contour generation with N_k=%d\n', N_k);
fprintf('Vertices: [0 0; pi 0; pi pi; 0 0]\n');
fprintf('Expected segments: Gamma->X, X->M, M->Gamma\n\n');

% Simulate MATLAB's behavior
wavevectors = [];
for vertex_idx = 1:(size(vertices,1)-1)
    start_pt = vertices(vertex_idx,:);
    end_pt = vertices(vertex_idx+1,:);
    n_dims = length(start_pt);
    pts = zeros(N_k, n_dims);
    for dim = 1:n_dims
        pts(:, dim) = linspace(start_pt(dim), end_pt(dim), N_k)';
    end
    
    fprintf('Segment %d: %s -> %s\n', vertex_idx, mat2str(start_pt), mat2str(end_pt));
    fprintf('  Generated %d points\n', size(pts, 1));
    
    if isempty(wavevectors)
        fprintf('  wavevectors is empty, wavevectors(1:(end-1),:) = []\n');
        wavevectors_before = [];
    else
        fprintf('  wavevectors before: %d points\n', size(wavevectors, 1));
        wavevectors_before = wavevectors(1:(end-1),:);
        fprintf('  After removing last point: %d points\n', size(wavevectors_before, 1));
    end
    
    wavevectors = [wavevectors_before; pts];
    fprintf('  Total after adding segment: %d points\n\n', size(wavevectors, 1));
end

fprintf('Final total: %d points\n', size(wavevectors, 1));

% Now test the actual function
[wavevectors_actual, contour_info] = get_IBZ_contour_wavevectors(N_k, a, symmetry_type);
fprintf('\nActual function result: %d points\n', size(wavevectors_actual, 1));
fprintf('Expected from MATLAB: 55 points\n');

