clear; close all; %delete(findall(0));

isSaveOutput = false;

%% Save output setup ... 
script_start_time = replace(char(datetime),':','-');
output_folder = ['OUTPUT/output ' script_start_time];
% if isSaveOutput
%     mkdir(output_folder);
%     copyfile([mfilename('fullpath') '.m'],[output_folder '/' mfilename '.m']);
%     plot_folder = create_new_folder('plots',output_folder);
%     create_new_folder('pdf',plot_folder)
%     create_new_folder('fig',plot_folder)
%     create_new_folder('svg',plot_folder)
%     create_new_folder('eps',plot_folder)
% end

%%
const.a = .01; % [m]
const.N_ele = 1;
const.N_pix = 16;
const.N_wv = [16 31];
const.N_eig = 3;
const.isUseGPU = false;
const.isUseImprovement = true;
const.isUseParallel = true;
const.isSaveEigenvectors = false;

const.E_min = 200e6;
const.E_max = 200e9;
const.rho_min = 8e2;
const.rho_max = 8e3;
const.poisson_min = 0;
const.poisson_max = .5;
const.t = .01;
const.sigma_eig = 1;

design_params = design_parameters;
design_params.design_number = 15;
design_params.design_style = 'kernel';
design_params.design_options = struct('kernel','periodic','sigma_f',1,'sigma_l',0.5,'symmetry_type','none','N_value',2);
design_params.N_pix = [const.N_pix const.N_pix];
design_params = design_params.prepare();

[const.wavevectors,contour_info] = get_IBZ_contour_wavevectors(const.N_wv(1),const.a,design_params.design_options{1}.symmetry_type);

%% Random cell
const.design_scale = 'linear';
% const.design = get_design(struct_tag,const.N_pix);
% const.design = all_designs(:,:,:,zhi_design_idx);
const.design = get_design2(design_params);
% const.design = convert_design(const.design,'linear',const.design_scale,const.E_min,const.E_max,const.rho_min,const.rho_max);

%% Plot the design
fig = plot_design(const.design);
if isSaveOutput
    fix_pdf_border(fig)
    save_in_all_formats(fig,'design',plot_folder,false)
end

%% Solve the dispersion problem
[wv,fr,ev] = dispersion(const,const.wavevectors);
% fr(end+1,:) = fr(1,:);
% ev(:,end + 1,:) = ev(:,1,:);
wn = linspace(0,contour_info.N_segment+1,size(const.wavevectors,1))';
wn = repmat(wn,1,const.N_eig);

%% Plot the discretized Irreducible Brillouin Zone
fig = plot_wavevectors(wv);
if isSaveOutput
    fig = fix_pdf_border(fig);
    save_in_all_formats(fig,'wavevectors',plot_folder,false)
end

%% Plot the dispersion relation
fig = plot_dispersion(wn,fr,contour_info.N_segment);
if isSaveOutput
    fig = fix_pdf_border(fig);
    save_in_all_formats(fig,'dispersion',plot_folder,false)
end

%% Plot the modes
% k_idx = 2;
% eig_idx = 5;
% wavevector = wv(:,k_idx);
% plot_mode_ui(wv,fr,ev,const);
% plot_mode(wv,fr,ev,eig_idx,k_idx,'both',const)

