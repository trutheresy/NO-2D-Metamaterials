close all;

% Deterministic MATLAB generation flow for fixed-geometry equivalence checks.
% Required variables from orchestrator:
%   fixed_geometry_path : .mat file containing FIXED_DESIGN (32x32x3)
%   matlab_output_path  : output .mat file path

if ~exist('fixed_geometry_path','var')
    error('fixed_geometry_path must be provided before running this script.');
end
if ~exist('matlab_output_path','var')
    error('matlab_output_path must be provided before running this script.');
end

[repo_root, ~, ~] = fileparts(mfilename('fullpath'));
addpath(fullfile(repo_root,'2D-dispersion-han'));

if ~exist(fixed_geometry_path,'file')
    error('Fixed geometry file does not exist: %s', fixed_geometry_path);
end

geom_struct = load(fixed_geometry_path);
if ~isfield(geom_struct,'FIXED_DESIGN')
    error('FIXED_DESIGN variable not found in %s', fixed_geometry_path);
end

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

design = double(geom_struct.FIXED_DESIGN);
if ~isequal(size(design), [32 32 3])
    error('Expected FIXED_DESIGN size [32 32 3], got [%s]', num2str(size(design)));
end

const.design = design;
const.symmetry_type = 'p4mm';
const.wavevectors = get_IBZ_wavevectors(const.N_wv,const.a,'none',1);
imag_tol = 1e-3;

[wv,fr,ev,~,K,M,T] = dispersion_with_matrix_save_opt(const,const.wavevectors);

ELASTIC_MODULUS_DATA = const.E_min + (const.E_max - const.E_min) * const.design(:,:,1);
DENSITY_DATA = const.rho_min + (const.rho_max - const.rho_min) * const.design(:,:,2);
POISSON_DATA = const.poisson_min + (const.poisson_max - const.poisson_min) * const.design(:,:,3);

WAVEVECTOR_DATA = reshape(wv, [size(wv,1), size(wv,2), 1]);
EIGENVALUE_DATA = reshape(real(fr), [size(fr,1), size(fr,2), 1]);
EIGENVECTOR_DATA = reshape(ev, [size(ev,1), size(ev,2), size(ev,3), 1]);
designs = reshape(const.design, [size(const.design,1), size(const.design,2), size(const.design,3), 1]);
K_DATA = {K};
M_DATA = {M};
T_DATA = T;
N_struct = 1;
rng_seed_offset = -1;
CONSTITUTIVE_DATA = containers.Map({'modulus','density','poisson'},...
    {reshape(ELASTIC_MODULUS_DATA,[size(ELASTIC_MODULUS_DATA,1), size(ELASTIC_MODULUS_DATA,2), 1]), ...
     reshape(DENSITY_DATA,[size(DENSITY_DATA,1), size(DENSITY_DATA,2), 1]), ...
     reshape(POISSON_DATA,[size(POISSON_DATA,1), size(POISSON_DATA,2), 1])});

save(matlab_output_path, ...
    'WAVEVECTOR_DATA','EIGENVALUE_DATA','EIGENVECTOR_DATA', ...
    'CONSTITUTIVE_DATA','K_DATA','M_DATA','T_DATA','const','designs','N_struct','imag_tol','rng_seed_offset');

disp(['MATLAB_FIXED_OUTPUT=' matlab_output_path]);
