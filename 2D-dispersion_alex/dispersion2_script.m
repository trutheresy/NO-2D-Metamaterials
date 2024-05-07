clear; close all; %delete(findall(0));

isSaveOutput = false;
isPlotDesign = true;
design_parameters.design_number = [];
design_parameters.design_style = 'matern52';
design_parameters.design_options = struct('sigma_f',0.5,'sigma_l',0.1,'symmetry_type','c1m1','N_value',2);
design_parameters.N_pix = [const.N_pix const.N_pix];
const.symmetry_type = 'c1m1';

%% Save output setup ...
script_start_time = replace(char(datetime),':','-');
output_folder = ['OUTPUT/output ' script_start_time];
if isSaveOutput
    mkdir(output_folder);
    copyfile([mfilename('fullpath') '.m'],[output_folder '/' mfilename '.m']);
    plot_folder = create_new_folder('plots',output_folder);
    create_new_folder('pdf',plot_folder)
    create_new_folder('fig',plot_folder)
    create_new_folder('svg',plot_folder)
    create_new_folder('eps',plot_folder)
end

%%
const.a = 1; % [m]
const.N_ele = 1;
const.N_pix = design_parameters.N_pix(1);
const.N_wv = [51 NaN]; const.N_wv(2) = ceil(const.N_wv(1)/2); % used for full IBZ calculations
const.N_k = 50; % used for IBZ contour calculations
const.N_eig = 3;
const.isUseGPU = false;
const.isUseImprovement = false; % group velocity not supported by get_system_matrices_VEC()
const.isUseParallel = false;
const.isComputeGroupVelocity = false;
const.isComputeFrequencyDesignSensitivity = false;
const.isComputeGroupVelocityDesignSensitivity = false;
const.isSaveEigenvectors = false;

% symmetry_type = 'none'; IBZ_shape = 'rectangle';
% num_tesselations = 1;
% const.wavevectors = get_IBZ_wavevectors(const.N_wv,const.a,symmetry_type,num_tesselations);
const.wavevectors = get_IBZ_contour_wavevectors(const.N_k,const.a,const.symmetry_type);

const.E_min = 2e9; % 2e9
const.E_max = 200e9;
const.rho_min = 1e3; % 1e3
const.rho_max = 8e3;
const.poisson_min = 0;
const.poisson_max = .5;
const.t = 1;
const.sigma_eig = 1;

const.design_scale = 'linear';
% const.design = get_design(struct_tag,const.N_pix);
const.design = get_design2(design_parameters);
const.design = convert_design(const.design,'linear',const.design_scale,const.E_min,const.E_max,const.rho_min,const.rho_max);

%% Plot the design
if isPlotDesign
    fig = plot_design(const.design);
    if isSaveOutput
        fix_pdf_border(fig)
        save_in_all_formats(fig,'design',plot_folder,false)
    end
end

%% Solve the dispersion problem
% [wv,fr,ev,cg,dfrddesign,dcgddesign] = dispersion2(const,const.wavevectors);
[wv,fr,ev,cg] = dispersion2(const,const.wavevectors);
% save('compare_file1','wv','fr','ev','cg','dfrddesign','dcgddesign')
% data = load('compare_file1');
% assert(all(data.wv == wv,'all'))
% assert(all(data.fr == fr,'all'))
% assert(all(data.ev == ev,'all'))
% assert(all(data.cg == cg,'all'))
% assert(all(data.dfrddesign == dfrddesign,'all'))
% assert(all(data.dcgddesign == dcgddesign,'all'))
% disp('cleared assertions')

% wn = linspace(0,3,size(const.wavevectors,2) + 1);
% wn = repmat(wn,const.N_eig,1);

%% Plot the discretized Irreducible Brillouin Zone
% fig = plot_wavevectors(wv);
% if isSaveOutput
%     fix_pdf_border(fig)
%     save_in_all_formats(fig,'wavevectors',plot_folder,false)
% end

%% Plot the dispersion relation (surface)
% fig = figure2();
% ax = axes(fig);
% hold(ax,'on');
% view(ax,3);
% eig_idxs_to_plot = 1:const.N_eig;
% for eig_idx_to_plot = eig_idxs_to_plot
%     plot_dispersion_surface(wv,fr(:,eig_idx_to_plot),[],[],ax);
% end
% title(ax,'dispersion relation')

if isSaveOutput
    fix_pdf_border(fig)
    save_in_all_formats(fig,'dispersion',plot_folder,false)
end

%% Plot the dispersion relation (contour)
% fr(end+1,:) = fr(1,:);
% ev(:,end + 1,:) = ev(:,1,:);
% wn = linspace(0,3,size(const.wavevectors,1) + 1)';
% wn = linspace(0,5,size(const.wavevectors,1))';
% wn = repmat(wn,1,const.N_eig);
% fig = plot_dispersion(wn,fr);
% if isSaveOutput
%     fig = fix_pdf_border(fig);
%     save_in_all_formats(fig,'dispersion',plot_folder,false)
% end

% %% Plot the group velocity (x-component)
% fig = figure2();
% ax = axes(fig);
% hold(ax,'on');
% view(ax,3);
% for eig_idx_to_plot = eig_idxs_to_plot
%     plot_dispersion_surface(wv,cg(:,eig_idx_to_plot,1),IBZ_shape,const.N_k,const.N_k,ax);
% end
% title(ax,'group velocity x-component')
% if isSaveOutput
%     fix_pdf_border(fig)
%     save_in_all_formats(fig,'dispersion',plot_folder,false)
% end
% 
% %% Plot the group velocity (y-component)
% fig = figure2();
% ax = axes(fig);
% hold(ax,'on');
% view(ax,3);
% for eig_idx_to_plot = eig_idxs_to_plot
%     plot_dispersion_surface(wv,cg(:,eig_idx_to_plot,2),IBZ_shape,const.N_k,const.N_k,ax);
% end
% title(ax,'group velocity y-component')
% if isSaveOutput
%     fix_pdf_border(fig)
%     save_in_all_formats(fig,'dispersion',plot_folder,false)
% end

%% Plot the modes
% plot_mode_ui(wv,fr,ev,const);

