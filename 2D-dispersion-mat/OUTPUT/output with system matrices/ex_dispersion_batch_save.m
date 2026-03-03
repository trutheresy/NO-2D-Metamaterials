clear; close all;

% dispersion_library_path = 'D:\Research\NO-2D-Metamaterials\2D-dispersion_alex';
% dispersion_library_path = '../../';
% addpath(dispersion_library_path)

datetime_var = datetime;
mfilename_fullpath_var = mfilename('fullpath');
mfilename_var = mfilename;

% Output flags
isSaveOutput = true;
isSaveEigenvectors = true; % Flag for whether to save eigenvector information
eigenvector_dtype = 'single'; % NEW: Save eigenvectors as either 'single' or 'double' datatype (do *NOT* use 'half', it is not supported)
isProfile = false; % Flag for whether to use MPI parallel profiler
const.isSaveMesh = false; % NEW: Unrelated to what we're doing, but set this for compatibility reasons.
const.isSaveKandM = true; % NEW: Flag for whether to save stiffness and mass matrices

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
const.eigenvector_dtype = eigenvector_dtype;

% Define design parameters, including design_params, which controls how random designs will be generated
N_struct = 10; % Determines how many TOTAL designs will be generated, per parameter set.
N_struct_batch = 10; % Determines max number of designs that get saved per file
struct_idx_ranges = make_chunks(N_struct,N_struct_batch); % N_batch x 2 array with starting indices in first column and ending indices in second column

%rng_seed_offset = 0; % Determines rng seed at which random designs will start to be generated. The rng seed used for each design is rng_seed_offset + struct_idx.
rng_seed_offset = 25000; %24000;
const.a = 1; % [m], the side length of the square unit cell
N_values = [inf 2]; % First do binary, then do continuous

design_params = design_parameters;
design_params.design_number = []; % leave empty
design_params.design_style = 'kernel';
design_params.design_options = struct('kernel','periodic - not squared','sigma_f',1,'sigma_l',1,'symmetry_type','p4mm','N_value',NaN);
% N_Value is the number of intervals between 0-1 it will round to. N_value
% = 11 will give 0, 0.1, ... 1
design_params.N_pix = [const.N_pix const.N_pix];
design_params = design_params.prepare();

const.design_scale = 'linear';
const.design = nan(const.N_pix,const.N_pix,3); % Initialize

% Set material parameters
const.E_min = 20e6; % 2e9
const.E_max = 200e9; % 200e9
const.rho_min = 1200; % 1e3
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
const.wavevectors = get_IBZ_wavevectors(const.N_wv,const.a,const.symmetry_type,num_tesselations); % NEW: Leverage symmetry so that you are not solving/saving redundant solutions

% const.wavevectors = get_IBZ_wavevectors(const.N_wv,const.a,'none',num_tesselations);
% if ~strcmp(const.symmetry_type,'none')
%     warning('Forcing the IBZ to be defined based on symmetry-less unit cell. Some computations are guaranteed to be redundant.')
% end

% plot_wavevectors(const.wavevectors)
if isProfile
    mpiprofile on %#ok<UNRCH>
end

if ~isSaveOutput
    warning('isSaveOutput is set to false. Output will not be saved.') %#ok<UNRCH>
end

for N_value = N_values
    for property_idx = 1:3
        design_params.design_options{property_idx}.N_value = N_value;
    end
    
    N_batch = size(struct_idx_ranges,1);
    for batch_idx = 1:N_batch

        [designs,WAVEVECTOR_DATA,EIGENVALUE_DATA,N_dof,DESIGN_NUMBERS,EIGENVECTOR_DATA,ELASTIC_MODULUS_DATA,DENSITY_DATA,POISSON_DATA,K_DATA,M_DATA,T_DATA] = init_storage(const,N_struct_batch);

        %% Generate dataset
        struct_counter = 0;
        wb = waitbar(struct_counter/N_struct_batch,['Working on batch ' num2str(batch_idx) '/' num2str(N_batch) ' | unit cell ' num2str(struct_counter) '/' num2str(N_struct_batch)]);
        for struct_idx = struct_idx_ranges(batch_idx,1):struct_idx_ranges(batch_idx,2) % THIS MUST NOT BE PARFOR
            struct_counter = struct_counter + 1;
            design_params.design_number = struct_idx + rng_seed_offset;
            design_params = design_params.prepare();
            const.design = get_design2(design_params);
            const.design = convert_design(const.design,'linear',const.design_scale,const.E_min,const.E_max,const.rho_min,const.rho_max);
            const.design = apply_steel_rubber_paradigm(const.design,const);

            designs(:,:,:,struct_counter) = const.design;

            % NEW: Solve the dispersion problem, with the option of saving the matrices
            % Transformation matrix T is output here as a cell array
            % Each entry of T is a sparse matrix
            % length(T) = size(const.wavevectors,1)
            % K and M are output as sparse matrices.
            [wv,fr,ev,~,K,M,T] = dispersion_with_matrix_save_opt(const,const.wavevectors); % NEW: dispersion_with_matrix_save_opt()
            
            % NEW: Store mass and stiffness matrices to global variables
            % Now these are *nested* cell arrays. Outer loop over struct_idx, inner
            % loop over wavevector_idx
            K_DATA{struct_counter} = K;
            M_DATA{struct_counter} = M;
            if struct_counter == 1
                T_DATA = T;
            end

            WAVEVECTOR_DATA(:,:,struct_counter) = wv;
            EIGENVALUE_DATA(:,:,struct_counter) = real(fr);
            if isSaveEigenvectors
                EIGENVECTOR_DATA(:,:,:,struct_counter) = ev;
            end

            if max(max(abs(imag(fr))))>imag_tol
                warning(['Large imaginary component in frequency for structure ' num2str(struct_counter)])
            end

            % Save the material properties
            ELASTIC_MODULUS_DATA(:,:,struct_counter) = const.E_min + (const.E_max - const.E_min)*const.design(:,:,1);
            DENSITY_DATA(:,:,struct_counter) = const.rho_min + (const.rho_max - const.rho_min)*const.design(:,:,2);
            POISSON_DATA(:,:,struct_counter) = const.poisson_min + (const.poisson_max - const.poisson_min)*const.design(:,:,3);

            DESIGN_NUMBERS(struct_counter) = design_params.design_number(1);
            waitbar(struct_counter/N_struct_batch,wb,['Working on batch ' num2str(batch_idx) '/' num2str(N_batch) ' | unit cell ' num2str(struct_counter) '/' num2str(N_struct_batch)]);
        end
        close(wb)

        % Collect constitutive data in a container
        CONSTITUTIVE_DATA = containers.Map({'modulus','density','poisson'},...
            {ELASTIC_MODULUS_DATA, DENSITY_DATA, POISSON_DATA});

        % View result of parallel profiler
        if isProfile
            mpiprofile viewer %#ok<UNRCH>
        end

        %% Set up save locations
        if batch_idx == 1 && N_value == N_values(1)
            script_start_time = replace(char(datetime_var),':','-');
            if isSaveOutput
                %output_folder = ['OUTPUT/output ' script_start_time];
                output_folder = ['OUTPUT/output ' script_start_time];
                mkdir(output_folder);
                copyfile([mfilename_fullpath_var '.m'],[output_folder '/' mfilename_var '.m']);
            end
        end

        %% Save the results
        vars_to_save = {'WAVEVECTOR_DATA','EIGENVALUE_DATA','CONSTITUTIVE_DATA','DESIGN_NUMBERS','struct_idx_ranges','batch_idx','N_batch','N_value','K_DATA','M_DATA','T_DATA','const','design_params','designs','N_struct','N_struct_batch','imag_tol','rng_seed_offset'};
        if isSaveEigenvectors
            vars_to_save{end+1} = 'EIGENVECTOR_DATA';
        end
        if isSaveOutput
            if N_value == 2
                design_type_label = 'binarized';
            elseif N_value == inf
                design_type_label = 'continuous';
            else
                disp('design_type_label not recognized')
                design_type_label = 'misc';
            end
            % Set up path for output file
            % output_file_path = [output_folder '/DATA' ...
            %     ' N_pix' num2str(const.N_pix) 'x' num2str(const.N_pix)...
            %     ' N_ele' num2str(const.N_ele) 'x' num2str(const.N_ele)...
            %     ' N_wv' num2str(const.N_wv(1)) 'x' num2str(const.N_wv(2))...
            %     ' N_disp' num2str(N_struct)...
            %     ' N_eig' num2str(const.N_eig)...
            %     ' offset' num2str(rng_seed_offset) ' ' design_type_label ...
            %     ' ' script_start_time '.mat'];

            % NEW: I had to make this path shorter on my machine, it was getting
            % way too long. Feel free to change it back for yourself.
            output_file_path = [output_folder filesep ...
                'out_' ...
                design_type_label ...
                '_' num2str(batch_idx) '.mat'];

            % Save output
            save(output_file_path,vars_to_save{:},'-v7.3');
            disp(['Finished N_value = ' num2str(N_value) ' | batch_number = ' num2str(batch_idx)])
            disp(['Output saved to: ' output_file_path]);
            struct_counter = 0;
        end
    end
end

