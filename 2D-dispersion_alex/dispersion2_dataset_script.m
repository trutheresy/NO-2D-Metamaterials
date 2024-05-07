clear; close all;

isSaveOutput = false;
isSaveEigenvectors = false;
isIncludeHomogeneous = false;
isProfile = false;
N_struct = 100;
imag_tol = 1e-3;
rng_seed_offset = 0;

const.a = 1; % [m]
const.N_ele = 2;
const.N_pix = 4;
const.N_wv = [11 NaN]; const.N_wv(2) = ceil(const.N_wv(1)/2); % used for full IBZ calculations
const.N_k = []; % used for IBZ contour calculations
const.N_eig = 4;
const.isUseGPU = false;
const.isUseImprovement = true;
const.isUseParallel = true; % parallelize dispersion loop, not structure loop
const.isSaveEigenvectors = isSaveEigenvectors;
const.isComputeFrequencyDesignSensitivity = false;
const.isComputeGroupVelocity = false;
const.isComputeGroupVelocityDesignSensitivity = false;


const.E_min = 2e9;
const.E_max = 200e9;
const.rho_min = 1e3;
const.rho_max = 8e3;
const.poisson_min = 0;
const.poisson_max = .5;
const.t = 1;
const.sigma_eig = 1;

% const.wavevectors = create_wavevector_array(const.N_k,const.a);
symmetry_type = 'none'; IBZ_shape = 'rectangle'; num_tesselations = 1;
const.wavevectors = get_IBZ_wavevectors(const.N_wv,const.a,symmetry_type,num_tesselations);

% const.wavevectors = linspaceNDim([0;0],[pi/const.a;0],const.N_k);

const.design_scale = 'linear';
const.design = nan(const.N_pix,const.N_pix,3); % This is just a temporary value so that 'const' has the field 'design' used in the parfor loop


%% Set up save locations
if ~isSaveOutput
    warning('isSaveOutput is set to false. Output will not be saved.')
end
script_start_time = replace(char(datetime),':','-');
if isSaveOutput
    output_folder = ['OUTPUT/output ' script_start_time];
    mkdir(output_folder);
    copyfile([mfilename('fullpath') '.m'],[output_folder '/' mfilename '.m']);
end

% plot_wavevectors(const.wavevectors)

if isProfile
    mpiprofile on
end

%% Generate dataset
pfwb = parfor_wait(N_struct,'Waitbar', true);
for struct_idx = 1:N_struct
% for struct_idx = 1:N_struct
    pfc = const;
    if struct_idx == 1 && isIncludeHomogeneous
        pfc.design = get_design('homogeneous',pfc.N_pix);
        pfc.design = convert_design(pfc.design,'linear',pfc.design_scale,pfc.E_min,pfc.E_max,pfc.rho_min,pfc.rho_max);
    else
        pfc.design = get_design(struct_idx + rng_seed_offset,pfc.N_pix);
        pfc.design = convert_design(pfc.design,'linear',pfc.design_scale,pfc.E_min,pfc.E_max,pfc.rho_min,pfc.rho_max);
    end
    
    
    designs(struct_idx,:,:,:) = pfc.design;
    
    % Solve the dispersion problem
    [wv,fr,ev] = dispersion2(pfc,pfc.wavevectors);
    WAVEVECTOR_DATA(:,:,struct_idx) = wv;
    EIGENVALUE_DATA(:,:,struct_idx) = real(fr);
    if isSaveEigenvectors
        EIGENVECTOR_DATA(:,:,:,struct_idx) = ev;
    end
    
    if max(max(abs(imag(fr))))>imag_tol
        warning(['Large imaginary component in frequency for structure ' num2str(struct_idx)])
    end
    
    % Save the material properties
    ELASTIC_MODULUS_DATA(:,:,struct_idx) = pfc.E_min + (pfc.E_max - pfc.E_min)*pfc.design(:,:,1);
    DENSITY_DATA(:,:,struct_idx) = pfc.rho_min + (pfc.rho_max - pfc.rho_min)*pfc.design(:,:,2);
    POISSON_DATA(:,:,struct_idx) = pfc.poisson_min + (pfc.poisson_max - pfc.poisson_min)*pfc.design(:,:,3);
    pfwb.Send; %#ok<PFBNS>
end
pfwb.Destroy;

if isProfile
    mpiprofile viewer
end

% figure2();
% hold on
% for eig_idx = 1:const.N_eig
%     line(squeeze(WAVEVECTOR_DATA(:,1,:))',squeeze(EIGENVALUE_DATA(:,eig_idx,:))','CreateFcn',@(l,e) set(l,'Color',[0 0 0 .1]),'Color',[0 0 0 .1])
% end

%% Save the results
if isSaveOutput
    CONSTITUTIVE_DATA = containers.Map({'modulus','density','poisson'},...
        {ELASTIC_MODULUS_DATA, DENSITY_DATA, POISSON_DATA});
    output_file_path = [output_folder '/DATA' ...
        ' N_pix' num2str(const.N_pix) 'x' num2str(const.N_pix)...
        ' N_ele' num2str(const.N_ele) 'x' num2str(const.N_ele)...
        ' N_wv' num2str(const.N_wv(1)) 'x' num2str(const.N_wv(2))...
        ' N_disp' num2str(N_struct)...
        ' N_eig' num2str(const.N_eig)...
        ' offset' num2str(rng_seed_offset) ' ' script_start_time '.mat'];
    if isSaveEigenvectors
        save(output_file_path,'WAVEVECTOR_DATA','EIGENVALUE_DATA','EIGENVECTOR_DATA','CONSTITUTIVE_DATA','-v7.3');
    else
        save(output_file_path,'WAVEVECTOR_DATA','EIGENVALUE_DATA','CONSTITUTIVE_DATA','-v7.3');
    end
end

% delete(gcp('nocreate'))

%% Plot a subset of the data
% struct_idx_to_plot = 1;
% 
% fig = figure2();
% ax = axes(fig);
% plot_design(cat(3,squeeze(ELASTIC_MODULUS_DATA(:,:,struct_idx_to_plot)), squeeze(DENSITY_DATA(:,:,struct_idx_to_plot)), squeeze(POISSON_DATA(:,:,struct_idx_to_plot))))
% 
% fig = figure2();
% ax = axes(fig);
% hold on
% for eig_idx_to_plot = 1:const.N_eig
%     wv_plot = squeeze(WAVEVECTOR_DATA(:,:,struct_idx_to_plot));
%     fr_plot = squeeze(EIGENVALUE_DATA(:,eig_idx_to_plot,struct_idx_to_plot));
%     plot_dispersion_surface(wv_plot,fr_plot,[],[],ax);
% end
% view(3)
