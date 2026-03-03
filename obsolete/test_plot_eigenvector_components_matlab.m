% MATLAB script to generate and save plot_eigenvector_components data for comparison
% This matches the Python test_plot_eigenvector_components test

clear; close all;

% Add path to dispersion library
dispersion_library_path = '../2D-dispersion-han';
addpath(dispersion_library_path);

% Create test constants (matching Python create_test_const)
N_pix = 5;
design = get_design('homogeneous', N_pix);

const = struct();
const.N_pix = N_pix;
const.N_ele = 4;
const.a = 1.0;
const.design = design;
const.design_scale = 'linear';
const.E_min = 2e9;
const.E_max = 200e9;
const.rho_min = 1000;
const.rho_max = 8000;
const.poisson_min = 0.0;
const.poisson_max = 0.5;
const.t = 0.01;
const.N_eig = 5;
const.sigma_eig = 'SM';
const.isSaveEigenvectors = true;
const.isSaveMesh = false;
const.isUseGPU = false;
const.isUseParallel = false;
const.isUseImprovement = false;
const.isUseSecondImprovement = false;

% Get wavevector at Gamma point (k=0) for testing
wv_gamma = [0.0, 0.0];

% Calculate actual dispersion
[wv, fr, ev] = dispersion(const, wv_gamma);

% Extract eigenvector for first wavevector (k=0) and first mode
k_idx = 1;  % MATLAB uses 1-based indexing
eig_idx = 1;  % MATLAB uses 1-based indexing
u_reduced = squeeze(ev(:, k_idx, eig_idx));  % Reduced space eigenvector

% Transform to full space (matching MATLAB: u = T*u_reduced)
T = get_transformation_matrix(wv_gamma, const);
u_full = T * u_reduced;  % Full space eigenvector

% Extract u and v components
u = u_full(1:2:end);  % Horizontal displacement (x)
v = u_full(2:2:end);  % Vertical displacement (y)

% Reshape to spatial grid (matching MATLAB)
N_node = const.N_ele * const.N_pix + 1;
u_reshaped = reshape(u(1:N_node*N_node), N_node, N_node);  % Column-major order
v_reshaped = reshape(v(1:N_node*N_node), N_node, N_node);  % Column-major order

% Save plot data for comparison
save('test_plots/plot_eigenvector_components_data_matlab.mat', ...
     'wv_gamma', 'fr', 'ev', 'u_reduced', 'u_full', 'u', 'v', ...
     'u_reshaped', 'v_reshaped', 'N_node', 'k_idx', 'eig_idx', 'const', ...
     '-v7');

fprintf('\n✅ Saved MATLAB plot data: test_plots/plot_eigenvector_components_data_matlab.mat\n');
fprintf('   Contains: eigenvectors, u, v, u_reshaped, v_reshaped matrices\n');
fprintf('   Compare with: test_plots/plot_eigenvector_components_data.mat (Python)\n');

% Generate the plot (if plot_eigenvector function exists)
if exist('test_plots', 'dir') == 0
    mkdir('test_plots');
end

% Create plot manually (since we don't have plot_eigenvector function)
try
    fig = figure();
    
    % Create subplots with tighter spacing
    axes(1, 1) = subplot(2, 2, 1);
    imagesc(real(u_reshaped));
    set(gca, 'YDir', 'normal');  % Fix Y-axis orientation (0 at bottom)
    colorbar;
    title('real(u)');
    axis equal;
    axis tight;  % Remove empty rows/columns
    
    axes(1, 2) = subplot(2, 2, 2);
    imagesc(imag(u_reshaped));
    set(gca, 'YDir', 'normal');  % Fix Y-axis orientation
    colorbar;
    title('imag(u)');
    axis equal;
    axis tight;
    
    axes(2, 1) = subplot(2, 2, 3);
    imagesc(real(v_reshaped));
    set(gca, 'YDir', 'normal');  % Fix Y-axis orientation
    colorbar;
    title('real(v)');
    axis equal;
    axis tight;
    
    axes(2, 2) = subplot(2, 2, 4);
    imagesc(imag(v_reshaped));
    set(gca, 'YDir', 'normal');  % Fix Y-axis orientation
    colorbar;
    title('imag(v)');
    axis equal;
    axis tight;
    
    % Reduce spacing between subplots
    set(fig, 'Units', 'inches');
    set(fig, 'Position', [1 1 12 12]);
    
    % Manual spacing adjustment to reduce white space
    % Adjust subplot positions to reduce white space
    % Reduce gaps between subplots
    gap = 0.05;  % Gap between subplots
    subplot_width = (1 - 3*gap) / 2;
    subplot_height = (1 - 3*gap) / 2;
    
    set(axes(1, 1), 'Position', [gap, 0.5 + gap, subplot_width, subplot_height]);
    set(axes(1, 2), 'Position', [0.5 + 2*gap, 0.5 + gap, subplot_width, subplot_height]);
    set(axes(2, 1), 'Position', [gap, gap, subplot_width, subplot_height]);
    set(axes(2, 2), 'Position', [0.5 + 2*gap, gap, subplot_width, subplot_height]);
    
    saveas(fig, 'test_plots/plot_eigenvector_components_matlab.png');
    close(fig);
    fprintf('✅ Generated MATLAB plot: test_plots/plot_eigenvector_components_matlab.png\n');
catch ME
    fprintf('⚠️  Could not generate plot: %s\n', ME.message);
    fprintf('   Data saved successfully, plot generation skipped\n');
end

