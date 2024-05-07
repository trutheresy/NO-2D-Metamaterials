clear; close all;
datetime_var = datetime;
mfilename_fullpath_var = mfilename('fullpath');
mfilename_var = mfilename;

isSaveOutput = true;
isSaveEigenvectors = false;
isIncludeHomogeneous = false;
isProfile = false;
N_struct = 20;
imag_tol = 1e-3;
rng_seed_offset = 0;

const.a = 1; % [m]
const.N_ele = 1;
const.N_pix = 16;
const.N_wv = [51 NaN]; const.N_wv(2) = ceil(const.N_wv(1)/2); % used for full IBZ calculations
% const.N_wv = [16 31];
const.N_k = []; % used for IBZ contour calculations
const.N_eig = 5;
const.isUseGPU = false;
const.isUseImprovement = true;
const.isUseParallel = true; % parallelize dispersion loop, not structure loop
const.isSaveEigenvectors = isSaveEigenvectors;

design_params = design_parameters;
design_params.design_number = []; % leave empty
design_params.design_style = 'kernel';
design_params.design_options = struct('kernel','periodic','sigma_f',1,'sigma_l',0.5,'symmetry_type','none','N_value',2);
design_params.N_pix = [const.N_pix const.N_pix];
design_params = design_params.prepare();

const.E_min = 200e6; % 2e9
const.E_max = 200e9; % 200e9
const.rho_min = 8e2; % 1e3
const.rho_max = 8e3; % 8e3
const.poisson_min = 0; % 0
const.poisson_max = .5; % .5
const.t = 1;
const.sigma_eig = 1;

% const.wavevectors = create_wavevector_array(const.N_k,const.a);
const.symmetry_type = design_params.design_options{1}.symmetry_type; IBZ_shape = 'rectangle'; num_tesselations = 1;
const.wavevectors = get_IBZ_wavevectors(const.N_wv,const.a,const.symmetry_type,num_tesselations);

% const.wavevectors = linspaceNDim([0;0],[pi/const.a;0],const.N_k);

const.design_scale = 'linear';
const.design = nan(const.N_pix,const.N_pix,3); % This is just a temporary value so that 'const' has the field 'design' used in the parfor loop

% plot_wavevectors(const.wavevectors)

if isProfile
    mpiprofile on
end

%% Generate dataset
pfwb = parfor_wait(N_struct,'Waitbar', true);
for struct_idx = 1:N_struct
% for struct_idx = 1:N_struct
    pfc = const;
    pfdp = design_params;
    if struct_idx == 1 && isIncludeHomogeneous
        pfc.design = get_design('homogeneous',pfc.N_pix);
        pfc.design = convert_design(pfc.design,'linear',pfc.design_scale,pfc.E_min,pfc.E_max,pfc.rho_min,pfc.rho_max);
    else
        pfdp.design_number = struct_idx + rng_seed_offset;
        pfdp = pfdp.prepare();
        pfc.design = get_design2(pfdp);
%         pfc.design = get_design(struct_idx + rng_seed_offset,pfc.N_pix);
        pfc.design = convert_design(pfc.design,'linear',pfc.design_scale,pfc.E_min,pfc.E_max,pfc.rho_min,pfc.rho_max);
    end
    
    
    designs(struct_idx,:,:,:) = pfc.design;
    
    % Solve the dispersion problem
    [wv,fr,ev] = dispersion(pfc,pfc.wavevectors);
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

%% Set up save locations
if ~isSaveOutput
    warning('isSaveOutput is set to false. Output will not be saved.')
end
script_start_time = replace(char(datetime_var),':','-');
if isSaveOutput
    output_folder = ['OUTPUT/output ' script_start_time];
    mkdir(output_folder);
    copyfile([mfilename_fullpath_var '.m'],[output_folder '/' mfilename_var '.m']);
end

%% Save the results
vars_to_save = {'WAVEVECTOR_DATA','EIGENVALUE_DATA','CONSTITUTIVE_DATA','const','design_params','N_struct','imag_tol','rng_seed_offset'};
if isSaveEigenvectors
    vars_to_save{end+1} = 'EIGENVECTOR_DATA';
end
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
    
        save(output_file_path,vars_to_save{:},'-v7.3');
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
