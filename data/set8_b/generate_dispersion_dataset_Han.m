clear; close all;

dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex';
addpath(dispersion_library_path)

datetime_var = datetime;
mfilename_fullpath_var = mfilename('fullpath');
mfilename_var = mfilename;

% Output flags
isSaveOutput = true;
isSaveEigenvectors = true; % Flag for whether to save eigenvector information
isProfile = false; % Flag for whether to use MPI parallel profiler

% Discretization parameters
const.N_ele = 1; % Number of elements along one pixel side length. This is a FEM discretization parameter.
const.N_pix = 32; % Number of pixels along one unit cell side length
const.N_wv = [25 NaN]; const.N_wv(2) = ceil(const.N_wv(1)/2); % Define the number of wavevectors along each side of the IBZ. This is an IBZ discretization parameter.

% Flags for computational improvements (parallelization, GPU usage, etc)
const.isUseGPU = false; % Leave as false
const.isUseImprovement = true; % Leave as true
const.isUseSecondImprovement = false; % Leave as false
const.isUseParallel = true; % Flag for parallelization in dispersion loop, not structure loop
const.isSaveEigenvectors = isSaveEigenvectors;

% Define design parameters, including design_params, which controls how random designs will be generated
N_struct = 600; % Determines how many designs will be generated
%rng_seed_offset = 0; % Determines rng seed at which random designs will start to be generated. The rng seed used for each design is rng_seed_offset + struct_idx.
rng_seed_offset = 4200;
const.a = 1; % [m], the side length of the square unit cell
binarize = true; % Set to false for continuous designs

design_params = design_parameters;
design_params.design_number = []; % leave empty
design_params.design_style = 'kernel';
design_params.design_options = struct('kernel','periodic','sigma_f',1,'sigma_l',1,'symmetry_type','p4mm','N_value',inf);
% N_Value is the number of intervals between 0-1 it will round to. N_value
% = 11 will give 0, 0.1, ... 1
design_params.N_pix = [const.N_pix const.N_pix];
design_params = design_params.prepare();

const.design_scale = 'linear';
const.design = nan(const.N_pix,const.N_pix,3); % Initialize

% Set material parameters
const.E_min = 200e6; % 2e9
const.E_max = 200e9; % 200e9
const.rho_min = 8e2; % 1e3
const.rho_max = 8e3; % 8e3
const.poisson_min = 0; % 0
const.poisson_max = .5; % .5
const.t = 1;

% Set eigenvalue solution parameters
const.N_eig = 6; % Number of eigenvalue bands to compute
% const.sigma_eig = 'smallestabs';
const.sigma_eig = 1e-2;
imag_tol = 1e-3; % Tolerance for imaginary values in eigenvalues

% Define wavevector array
const.symmetry_type = design_params.design_options{1}.symmetry_type; IBZ_shape = 'rectangle'; num_tesselations = 1;
% const.wavevectors = get_IBZ_wavevectors(const.N_wv,const.a,const.symmetry_type,num_tesselations);

const.wavevectors = get_IBZ_wavevectors(const.N_wv,const.a,'none',num_tesselations);
if ~strcmp(const.symmetry_type,'none')
    warning('Forcing the IBZ to be defined based on symmetry-less unit cell. Some computations are guaranteed to be redundant.')
end

% plot_wavevectors(const.wavevectors)
if isProfile
    mpiprofile on %#ok<UNRCH>
end

if ~isSaveOutput
    warning('isSaveOutput is set to false. Output will not be saved.') %#ok<UNRCH>
end

% Initialize storage arrays
designs = zeros(const.N_pix,const.N_pix,3,N_struct);
WAVEVECTOR_DATA = zeros(prod(const.N_wv),2,N_struct);
EIGENVALUE_DATA = zeros(prod(const.N_wv),const.N_eig,N_struct);
N_dof = 2*(const.N_pix*const.N_ele)^2;
EIGENVECTOR_DATA = zeros(N_dof,prod(const.N_wv),const.N_eig,N_struct);
ELASTIC_MODULUS_DATA = zeros(const.N_pix,const.N_pix,N_struct);
DENSITY_DATA = zeros(const.N_pix,const.N_pix,N_struct);
POISSON_DATA = zeros(const.N_pix,const.N_pix,N_struct);

%% Generate dataset
%pfwb = parfor_wait(N_struct,'Waitbar', true);
for struct_idx = 1:N_struct % THIS MUST NOT BE PARFOR
    design_params.design_number = struct_idx + rng_seed_offset;
    design_params = design_params.prepare();
    const.design = get_design2(design_params);
    const.design = convert_design(const.design,'linear',const.design_scale,const.E_min,const.E_max,const.rho_min,const.rho_max);
    if binarize
        const.design = round(const.design); % Binarize to 0 or 1
    end

    designs(:,:,:,struct_idx) = const.design;
    %designs = round(designs); % Disallow/allow gradient materials
    %disp(designs(:,:,1,1))

    % Solve the dispersion problem
    [wv,fr,ev] = dispersion(const,const.wavevectors);
    WAVEVECTOR_DATA(:,:,struct_idx) = wv;
    EIGENVALUE_DATA(:,:,struct_idx) = real(fr);
    if isSaveEigenvectors
        EIGENVECTOR_DATA(:,:,:,struct_idx) = ev;
    end

    if max(max(abs(imag(fr))))>imag_tol
        warning(['Large imaginary component in frequency for structure ' num2str(struct_idx)])
    end

    % Save the material properties
    ELASTIC_MODULUS_DATA(:,:,struct_idx) = const.E_min + (const.E_max - const.E_min)*const.design(:,:,1);
    DENSITY_DATA(:,:,struct_idx) = const.rho_min + (const.rho_max - const.rho_min)*const.design(:,:,2);
    POISSON_DATA(:,:,struct_idx) = const.poisson_min + (const.poisson_max - const.poisson_min)*const.design(:,:,3);
    %pfwb.Send;
end
%pfwb.Destroy;

% Collect constitutive data in a container
CONSTITUTIVE_DATA = containers.Map({'modulus','density','poisson'},...
    {ELASTIC_MODULUS_DATA, DENSITY_DATA, POISSON_DATA});

% View result of parallel profiler
if isProfile
    mpiprofile viewer %#ok<UNRCH>
end

%% Set up save locations
script_start_time = replace(char(datetime_var),':','-');
if isSaveOutput
    %output_folder = ['OUTPUT/output ' script_start_time];
    output_folder = fullfile(mfilename_fullpath_var, ['OUTPUT/output ' script_start_time]);
    mkdir(output_folder);
    copyfile([mfilename_fullpath_var '.m'],[output_folder '/' mfilename_var '.m']);
end

%% Save the results
vars_to_save = {'WAVEVECTOR_DATA','EIGENVALUE_DATA','CONSTITUTIVE_DATA','const','design_params','designs','N_struct','imag_tol','rng_seed_offset'};
if isSaveEigenvectors
    vars_to_save{end+1} = 'EIGENVECTOR_DATA';
end
if isSaveOutput
    if binarize
        design_type_label = 'binarized';
    else
        design_type_label = 'continuous';
    end
    % Set up path for output file
    output_file_path = [output_folder '/DATA' ...
        ' N_pix' num2str(const.N_pix) 'x' num2str(const.N_pix)...
        ' N_ele' num2str(const.N_ele) 'x' num2str(const.N_ele)...
        ' N_wv' num2str(const.N_wv(1)) 'x' num2str(const.N_wv(2))...
        ' N_disp' num2str(N_struct)...
        ' N_eig' num2str(const.N_eig)...
        ' offset' num2str(rng_seed_offset) ' ' design_type_label ...
        ' ' script_start_time '.mat'];

    % Save output
    save(output_file_path,vars_to_save{:},'-v7.3');
    disp(['Outputs saved successfully to: ' output_file_path]);
end