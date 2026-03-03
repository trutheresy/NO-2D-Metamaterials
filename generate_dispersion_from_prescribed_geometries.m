close all;

% Generate MATLAB dispersion outputs from prescribed geometries loaded from
% a .mat converted from geometries_full.pt.
%
% Required caller-provided variables:
%   geometry_mat_path : path to .mat file containing 'geometries_full'
%   output_mat_path   : path to output .mat file

if ~exist('geometry_mat_path', 'var')
    error('geometry_mat_path must be provided by caller');
end
if ~exist('output_mat_path', 'var')
    error('output_mat_path must be provided by caller');
end

[repo_root, ~, ~] = fileparts(mfilename('fullpath'));
addpath(fullfile(repo_root, '2D-dispersion-mat'));

if ~exist(geometry_mat_path, 'file')
    error('geometry_mat_path does not exist: %s', geometry_mat_path);
end

data = load(geometry_mat_path);
if ~isfield(data, 'geometries_full')
    error('Expected variable geometries_full in %s', geometry_mat_path);
end

geometries_full = double(data.geometries_full);
if ndims(geometries_full) ~= 3
    error('Expected geometries_full to be 3D (N_struct x N_pix x N_pix), got ndims=%d', ndims(geometries_full));
end

N_struct = size(geometries_full, 1);
N_pix = size(geometries_full, 2);

% Match Python generator constants.
const = struct();
const.N_ele = 1;
const.N_pix = N_pix;
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
const.isSaveEigenvectors = false;
const.isSaveKandM = false;
const.isSaveMesh = false;
const.eigenvector_dtype = 'double';
const.symmetry_type = 'none';
const.wavevectors = get_IBZ_wavevectors(const.N_wv, const.a, 'none', 1);

imag_tol = 1e-3;
rng_seed_offset = 0;

WAVEVECTOR_DATA = zeros(prod(const.N_wv), 2, N_struct);
EIGENVALUE_DATA = zeros(prod(const.N_wv), const.N_eig, N_struct);
designs = zeros(N_pix, N_pix, 3, N_struct);

for struct_idx = 1:N_struct
    g = squeeze(geometries_full(struct_idx, :, :));  % single channel from PT
    % Start from one prescribed channel then apply MATLAB steel/rubber mapping
    % to produce physically consistent E/rho/nu panes.
    design3 = cat(3, g, g, g);
    design3 = apply_steel_rubber_paradigm(design3, const);
    const.design = design3;
    designs(:,:,:,struct_idx) = design3;

    [wv, fr, ~, ~, ~, ~, ~] = dispersion_with_matrix_save_opt(const, const.wavevectors);
    WAVEVECTOR_DATA(:,:,struct_idx) = wv;
    EIGENVALUE_DATA(:,:,struct_idx) = real(fr);

    if max(max(abs(imag(fr)))) > imag_tol
        warning('Large imaginary frequency component for structure %d', struct_idx);
    end
end

save(output_mat_path, 'WAVEVECTOR_DATA', 'EIGENVALUE_DATA', 'designs', 'const', 'N_struct', 'imag_tol', 'rng_seed_offset');
disp(['Saved MATLAB prescribed-geometry results to: ' output_mat_path]);
