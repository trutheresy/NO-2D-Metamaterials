% MATLAB script to generate and save plot_dispersion_contour data for comparison
% This matches the Python test_plot_dispersion_contour test

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

% Create square grid of wavevectors (matching Python test)
N_k = 20;
a = const.a;
kx = linspace(-pi/a, pi/a, N_k);
ky = linspace(-pi/a, pi/a, N_k);
[KX, KY] = meshgrid(kx, ky);
% Flatten in column-major order (MATLAB default)
wv_grid = [KX(:), KY(:)];

% Calculate actual dispersion
[wv, fr, ev, mesh] = dispersion(const, wv_grid);

% Reshape for contour plot (matching MATLAB plot_dispersion_contour.m)
N_k_y = N_k;
N_k_x = N_k;
X = reshape(wv_grid(:,1), N_k_y, N_k_x);
Y = reshape(wv_grid(:,2), N_k_y, N_k_x);
Z = reshape(fr(:,1), N_k_y, N_k_x);  % First frequency band

% Save plot data for comparison
save('test_plots/plot_dispersion_contour_data_matlab.mat', ...
     'wv_grid', 'fr', 'X', 'Y', 'Z', 'N_k_x', 'N_k_y', 'const', ...
     '-v7.3');

fprintf('\n✅ Saved MATLAB plot data: test_plots/plot_dispersion_contour_data_matlab.mat\n');
fprintf('   Contains: wv_grid, fr, X, Y, Z matrices\n');
fprintf('   Compare with: test_plots/plot_dispersion_contour_data.mat (Python)\n');

% Generate the plot
[fig, ax] = plot_dispersion_contour(wv_grid, fr(:,1), N_k_x, N_k_y);
if exist('test_plots', 'dir') == 0
    mkdir('test_plots');
end
saveas(fig, 'test_plots/plot_dispersion_contour_matlab.png');
close(fig);

fprintf('✅ Generated MATLAB plot: test_plots/plot_dispersion_contour_matlab.png\n');

