% MATLAB script to generate and save plot_mode data for comparison
% This matches the Python test_plot_mode_basic test

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
u = T * u_reduced;  % Full space eigenvector

% Normalize (matching MATLAB: u = u/max(abs(u))*(1/10)*const.a)
scale_factor = (1/10) * const.a;
u = u / max(abs(u)) * scale_factor;

% Extract displacement components (matching MATLAB plot_mode.m)
U_vec = u(1:2:end);  % x-displacements
V_vec = u(2:2:end);  % y-displacements

% Create coordinate grids (matching MATLAB)
N_nodes = const.N_ele * const.N_pix + 1;
original_nodal_locations = linspace(0, const.a, N_nodes);
[X, Y] = meshgrid(original_nodal_locations, flip(original_nodal_locations));

% Reshape to spatial grid (matching MATLAB: reshape with transpose)
U_mat = reshape(U_vec, N_nodes, N_nodes)';  % Transpose to match MATLAB
V_mat = reshape(V_vec, N_nodes, N_nodes)';  % Transpose to match MATLAB

% Save plot data for comparison (use -v7 format for compatibility with scipy.io.loadmat)
save('test_plots/plot_mode_basic_data_matlab.mat', ...
     'wv_gamma', 'fr', 'ev', 'u_reduced', 'u', 'U_vec', 'V_vec', ...
     'U_mat', 'V_mat', 'X', 'Y', 'N_nodes', 'k_idx', 'eig_idx', 'const', ...
     '-v7');

fprintf('\n✅ Saved MATLAB plot data: test_plots/plot_mode_basic_data_matlab.mat\n');
fprintf('   Contains: eigenvectors, U_mat, V_mat, X, Y matrices\n');
fprintf('   Compare with: test_plots/plot_mode_basic_data.mat (Python)\n');

% Generate the plot (optional - comment out if figure2() is not available)
if exist('test_plots', 'dir') == 0
    mkdir('test_plots');
end
% Try to generate plot if plot_mode function is available
try
    if exist('plot_mode', 'file') == 2
        [fig, ax] = plot_mode(wv, fr, ev, eig_idx, k_idx, 'still', 0.1, const);
        saveas(fig, 'test_plots/plot_mode_basic_matlab.png');
        close(fig);
        fprintf('✅ Generated MATLAB plot: test_plots/plot_mode_basic_matlab.png\n');
    else
        fprintf('⚠️  plot_mode function not found, skipping plot generation\n');
    end
catch ME
    fprintf('⚠️  Could not generate plot: %s\n', ME.message);
    fprintf('   Data saved successfully, plot generation skipped\n');
end

