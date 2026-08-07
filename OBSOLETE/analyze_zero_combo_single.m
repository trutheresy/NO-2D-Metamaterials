function result = analyze_zero_combo_single(combo_mat_path, combo_index, combo_mode, output_root, generate_plot)
% Analyze one (geometry, wavevector, band) combo from catalog.
% Saves conventional MATLAB-style outputs + a 2x4 diagnostic figure.
%
% combo_mode: 'representative' (default) or 'all'

if nargin < 1 || isempty(combo_mat_path)
    combo_mat_path = fullfile(fileparts(mfilename('fullpath')), 'OUTPUT', 'zero_output_combinations_catalog.mat');
end
if nargin < 2 || isempty(combo_index)
    combo_index = 1;
end
if nargin < 3 || isempty(combo_mode)
    combo_mode = 'representative';
end
if nargin < 4 || isempty(output_root)
    output_root = fullfile(fileparts(mfilename('fullpath')), 'OUTPUT', 'zero_combo_single_analysis');
end
if nargin < 5 || isempty(generate_plot)
    generate_plot = false;
end
if ~exist(output_root, 'dir')
    mkdir(output_root);
end

repo_root = fileparts(mfilename('fullpath'));
addpath(fullfile(repo_root, '2D-dispersion-mat'));

C = load(combo_mat_path);
if strcmpi(combo_mode, 'all')
    dataset_name = string(C.combo_all_dataset_name{combo_index});
    pt_dir = string(C.combo_all_pt_dir{combo_index});
    design_number = double(C.combo_all_design_number(combo_index));
    geometry_idx_0based = double(C.combo_all_geometry_idx_0based(combo_index));
    wavevector_idx_1based = double(C.combo_all_wavevector_idx_1based(combo_index));
    band_idx_1based = double(C.combo_all_band_idx_1based(combo_index));
    wavevector_xy = double(C.combo_all_wavevector_xy(combo_index, :));
else
    dataset_name = string(C.combo_rep_dataset_name{combo_index});
    pt_dir = string(C.combo_rep_pt_dir{combo_index});
    design_number = double(C.combo_rep_design_number(combo_index));
    geometry_idx_0based = double(C.combo_rep_geometry_idx_0based(combo_index));
    wavevector_idx_1based = double(C.combo_rep_wavevector_idx_1based(combo_index));
    band_idx_1based = double(C.combo_rep_band_idx_1based(combo_index));
    wavevector_xy = double(C.combo_rep_wavevector_xy(combo_index, :));
end

% Match generation constants.
const = struct();
const.N_ele = 1;
const.N_pix = 32;
const.N_wv = [25 NaN]; const.N_wv(2) = ceil(const.N_wv(1)/2);
const.N_eig = 6;
const.sigma_eig = 1e-2;
const.a = 1;
const.design_scale = 'linear';
const.E_min = 200e6;
const.E_max = 200e9;
const.rho_min = 8e2;
const.rho_max = 8e3;
const.poisson_min = 0;
const.poisson_max = 0.5;
const.t = 1;
const.isUseGPU = false;
const.isUseImprovement = true;
const.isUseSecondImprovement = false;
const.isUseParallel = false;
const.isSaveEigenvectors = true;
const.isSaveKandM = true;
const.isSaveMesh = false;
const.eigenvector_dtype = 'double';

% These combinations came from zero-filled dataset entries, so use the same
% placeholder geometry content.
const.design = zeros(const.N_pix, const.N_pix, 3);

wavevector = reshape(wavevector_xy, 1, 2);

% Compute matrices and one-wavevector eigensolve.
[K, M] = get_system_matrices_VEC(const);
T = get_transformation_matrix(wavevector, const);
Kr = T' * K * T;
Mr = T' * M * T;
[eig_vecs, eig_vals_mat] = eigs(Kr, Mr, const.N_eig, const.sigma_eig);
[eig_vals_sorted, idxs] = sort(diag(eig_vals_mat));
eig_vecs_sorted = eig_vecs(:, idxs);
fr_all = sqrt(real(eig_vals_sorted)) / (2 * pi);

if band_idx_1based > numel(fr_all)
    error('Requested band index %d exceeds solved bands %d', band_idx_1based, numel(fr_all));
end

eigval_sel = fr_all(band_idx_1based);
eigvec_sel = eig_vecs_sorted(:, band_idx_1based);

N_dof_grid = const.N_pix * const.N_ele;
ux = reshape(eigvec_sel(1:2:end), N_dof_grid, N_dof_grid);
uy = reshape(eigvec_sel(2:2:end), N_dof_grid, N_dof_grid);
u_mag = sqrt(abs(ux).^2 + abs(uy).^2);

diag_K = local_matrix_diag(K, 'K');
diag_M = local_matrix_diag(M, 'M');
diag_T = local_matrix_diag(T, 'T');
diag_Kr = local_matrix_diag(Kr, 'Kr');
diag_Mr = local_matrix_diag(Mr, 'Mr');

combo_id = sprintf('%s_g%04d_w%03d_b%02d', char(dataset_name), geometry_idx_0based, wavevector_idx_1based, band_idx_1based);
combo_folder = fullfile(output_root, combo_id);
if ~exist(combo_folder, 'dir')
    mkdir(combo_folder);
end

% Conventional output variables (single combo).
WAVEVECTOR_DATA = reshape(wavevector, [1, 2, 1]);
EIGENVALUE_DATA = reshape(real(eigval_sel), [1, 1, 1]);
EIGENVECTOR_DATA = reshape(eigvec_sel, [numel(eigvec_sel), 1, 1, 1]);
CONSTITUTIVE_DATA = containers.Map({'modulus', 'density', 'poisson'}, ...
    {const.E_min + (const.E_max - const.E_min) * const.design(:,:,1), ...
     const.rho_min + (const.rho_max - const.rho_min) * const.design(:,:,2), ...
     const.poisson_min + (const.poisson_max - const.poisson_min) * const.design(:,:,3)});
K_DATA = {K};
M_DATA = {M};
T_DATA = {T};
KR_DATA = {Kr};
MR_DATA = {Mr};

combo_metadata = struct();
combo_metadata.dataset_name = char(dataset_name);
combo_metadata.pt_dir = char(pt_dir);
combo_metadata.design_number = design_number;
combo_metadata.geometry_idx_0based = geometry_idx_0based;
combo_metadata.wavevector_idx_1based = wavevector_idx_1based;
combo_metadata.band_idx_1based = band_idx_1based;
combo_metadata.wavevector_xy = wavevector_xy;

matrix_diagnostics = struct('K', diag_K, 'M', diag_M, 'T', diag_T, 'Kr', diag_Kr, 'Mr', diag_Mr);

mat_output_path = fullfile(combo_folder, [combo_id '_single_combo_output.mat']);
save(mat_output_path, ...
    'WAVEVECTOR_DATA', 'EIGENVALUE_DATA', 'EIGENVECTOR_DATA', ...
    'CONSTITUTIVE_DATA', 'K_DATA', 'M_DATA', 'T_DATA', 'KR_DATA', 'MR_DATA', ...
    'const', 'combo_metadata', 'matrix_diagnostics', '-v7.3');

fig_path = '';
if generate_plot
    % 2x4 diagnostic plot.
    fig = figure('Visible', 'off', 'Position', [100 100 1800 900]);

    subplot(2,4,1);
    imagesc(const.design(:,:,1)); axis image; colorbar;
    title('Input 1: Geometry (E pane)');

    subplot(2,4,2);
    plot(wavevector(1), wavevector(2), 'ro', 'MarkerFaceColor', 'r'); grid on; axis equal;
    xlabel('k_x'); ylabel('k_y');
    title(sprintf('Input 2: Wavevector [%0.6g, %0.6g]', wavevector(1), wavevector(2)));

    subplot(2,4,3);
    axis off;
    text(0.05, 0.80, sprintf('Input 3: Band = %d', band_idx_1based), 'FontSize', 12);
    text(0.05, 0.65, sprintf('Design # = %d', round(design_number)), 'FontSize', 12);
    text(0.05, 0.50, sprintf('Geom idx (0b) = %d', round(geometry_idx_0based)), 'FontSize', 12);
    text(0.05, 0.35, sprintf('Dataset = %s', char(dataset_name)), 'FontSize', 12, 'Interpreter', 'none');
    title('Input metadata');

    subplot(2,4,4);
    axis off;
    text(0.05, 0.80, sprintf('Eigenvalue (freq) = %.9g', real(eigval_sel)), 'FontSize', 12);
    text(0.05, 0.65, sprintf('Imag(freq) = %.3e', imag(eigval_sel)), 'FontSize', 12);
    text(0.05, 0.50, sprintf('Wavevector idx = %d', wavevector_idx_1based), 'FontSize', 12);
    title('Output eigenvalue');

    subplot(2,4,5);
    imagesc(real(ux)); axis image; colorbar;
    title('Eigenvector field u_x (real)');

    subplot(2,4,6);
    imagesc(real(uy)); axis image; colorbar;
    title('Eigenvector field u_y (real)');

    subplot(2,4,7);
    axis off;
    text(0.05, 0.90, sprintf('K near-singular: %d', diag_K.near_singular), 'FontSize', 11);
    text(0.05, 0.80, sprintf('M near-singular: %d', diag_M.near_singular), 'FontSize', 11);
    text(0.05, 0.70, sprintf('T near-singular: %d', diag_T.near_singular), 'FontSize', 11);
    text(0.05, 0.60, sprintf('Kr near-singular: %d', diag_Kr.near_singular), 'FontSize', 11);
    text(0.05, 0.50, sprintf('Mr near-singular: %d', diag_Mr.near_singular), 'FontSize', 11);
    text(0.05, 0.35, sprintf('smin(Kr)=%.3e, smin(Mr)=%.3e', diag_Kr.smin, diag_Mr.smin), 'FontSize', 10);
    title('Matrix conditioning');

    subplot(2,4,8);
    imagesc(u_mag); axis image; colorbar;
    title('|u| = sqrt(|u_x|^2+|u_y|^2)');

    sgtitle(sprintf('Single-combo diagnostic: %s', combo_id), 'Interpreter', 'none');
    fig_path = fullfile(combo_folder, [combo_id '_diagnostic_2x4.png']);
    saveas(fig, fig_path);
    close(fig);
end

result = struct();
result.combo_id = combo_id;
result.output_folder = combo_folder;
result.mat_output_path = mat_output_path;
result.figure_path = fig_path;
result.eigenvalue = eigval_sel;
result.matrix_diagnostics = matrix_diagnostics;
end


function out = local_matrix_diag(A, name)
out = struct();
out.name = name;
out.m = size(A, 1);
out.n = size(A, 2);
out.nnz = nnz(A);
out.smin = NaN;
out.smax = NaN;
out.cond_est = NaN;
out.near_singular = false;

try
    out.smin = svds(A, 1, 'smallest');
catch
    try
        out.smin = min(svd(full(A)));
    catch
        out.smin = NaN;
    end
end

try
    out.smax = svds(A, 1, 'largest');
catch
    try
        out.smax = max(svd(full(A)));
    catch
        out.smax = NaN;
    end
end

if ~isnan(out.smin) && ~isnan(out.smax)
    out.cond_est = out.smax / max(out.smin, eps);
end

if ~isnan(out.smin) && ~isnan(out.cond_est)
    out.near_singular = (out.smin < 1e-10) || (out.cond_est > 1e12);
end
end
