% DEBUG ONLY: delete after debugging is complete.
close all;

% Required variables from caller:
%   geometry_mat_path : path to .mat containing FIXED_DESIGN or geometries_full
%   output_mat_path   : output .mat path
% Optional:
%   debug_save_dir    : folder to store intermediate debug artifacts

if ~exist('geometry_mat_path', 'var')
    error('geometry_mat_path must be provided by caller');
end
if ~exist('output_mat_path', 'var')
    error('output_mat_path must be provided by caller');
end

[repo_root, ~, ~] = fileparts(mfilename('fullpath'));
addpath(fullfile(repo_root, '2D-dispersion-mat'));

data = load(geometry_mat_path);
if isfield(data, 'FIXED_DESIGN')
    fixed_design = double(data.FIXED_DESIGN);
    if ndims(fixed_design) ~= 3 || size(fixed_design,3) ~= 3
        error('FIXED_DESIGN must be (N_pix, N_pix, 3)');
    end
    N_struct = 1;
    N_pix = size(fixed_design, 1);
    use_fixed_design = true;
elseif isfield(data, 'geometries_full')
    geometries_full = double(data.geometries_full);
    N_struct = size(geometries_full, 1);
    N_pix = size(geometries_full, 2);
    use_fixed_design = false;
else
    error('Expected FIXED_DESIGN or geometries_full in %s', geometry_mat_path);
end

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
const.isSaveEigenvectors = true;
const.isSaveKandM = true;
const.isSaveMesh = false;
const.eigenvector_dtype = 'double';
const.symmetry_type = 'none';
const.wavevectors = get_IBZ_wavevectors(const.N_wv, const.a, 'none', 1);

imag_tol = 1e-3;
rng_seed_offset = 0;

WAVEVECTOR_DATA = zeros(prod(const.N_wv), 2, N_struct);
EIGENVALUE_DATA = zeros(prod(const.N_wv), const.N_eig, N_struct);
N_dof = 2*(const.N_pix*const.N_ele)^2;
EIGENVECTOR_DATA = zeros(N_dof, prod(const.N_wv), const.N_eig, N_struct);
designs = zeros(N_pix, N_pix, 3, N_struct);

if exist('debug_save_dir', 'var') && ~isempty(debug_save_dir)
    if ~exist(debug_save_dir, 'dir')
        mkdir(debug_save_dir);
    end
end

for struct_idx = 1:N_struct
    if use_fixed_design
        design_input = fixed_design;
    else
        g = squeeze(geometries_full(struct_idx, :, :));
        design_input = cat(3, g, g, g);
    end
    design3 = apply_steel_rubber_paradigm(design_input, const);
    const.design = design3;
    designs(:,:,:,struct_idx) = design3;

    [wv, fr, ev, ~, K, M, T] = dispersion_with_matrix_save_opt(const, const.wavevectors);
    WAVEVECTOR_DATA(:,:,struct_idx) = wv;
    EIGENVALUE_DATA(:,:,struct_idx) = real(fr);
    EIGENVECTOR_DATA(:,:,:,struct_idx) = ev;

    if exist('debug_save_dir', 'var') && ~isempty(debug_save_dir)
        [Ki, Kj, Kv] = find(K);
        [Mi, Mj, Mv] = find(M);
        K_triplet = [Ki, Kj, real(Kv), imag(Kv)];
        M_triplet = [Mi, Mj, real(Mv), imag(Mv)];
        T_diag = zeros(length(T), 2);
        for tidx = 1:length(T)
            T_diag(tidx,1) = size(T{tidx},1);
            T_diag(tidx,2) = size(T{tidx},2);
        end
        save(fullfile(debug_save_dir, sprintf('matlab_stage_struct_%d.mat', struct_idx)), ...
            'design_input', 'design3', 'wv', 'fr', 'ev', 'K_triplet', 'M_triplet', 'T_diag');
    end
end

save(output_mat_path, 'WAVEVECTOR_DATA', 'EIGENVALUE_DATA', 'EIGENVECTOR_DATA', 'designs', 'const', 'N_struct', 'imag_tol', 'rng_seed_offset');
disp(['Saved MATLAB debug prescribed-geometry results to: ' output_mat_path]);
