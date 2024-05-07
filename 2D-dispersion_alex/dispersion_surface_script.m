clear; close all; %delete(findall(0));

isSaveOutput = false;
struct_tag = '2';

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
const.N_ele = 4;
const.N_pix = 8;
const.N_wv = [51 NaN]; const.N_wv(2) = ceil(const.N_wv(1)/2); % used for full IBZ calculations
const.N_eig = 1;
const.isUseGPU = false;
const.isUseImprovement = true;
const.isUseParallel = true;
const.isSaveEigenvectors = false;

symmetry_type = 'none'; IBZ_shape = 'rectangle';
num_tesselations = 1;
const.wavevectors = get_IBZ_wavevectors(const.N_wv,const.a,symmetry_type,num_tesselations);

const.E_min = 2e9;
const.E_max = 200e9;
const.rho_min = 1e3;
const.rho_max = 8e3;
const.poisson_min = 0;
const.poisson_max = .5;
const.t = 1;
const.sigma_eig = 1;

const.design_scale = 'linear';
const.design = get_design(struct_tag,const.N_pix);
const.design = convert_design(const.design,'linear',const.design_scale,const.E_min,const.E_max,const.rho_min,const.rho_max);


%% Plot the design
fig = plot_design(const.design);
if isSaveOutput
    fix_pdf_border(fig)
    save_in_all_formats(fig,'design',plot_folder,false)
end

%% Solve the dispersion problem
tic
[wv,fr,ev] = dispersion(const,const.wavevectors);
toc
fr = real(fr);
% wn = linspace(0,3,size(const.wavevectors,2) + 1);
% wn = repmat(wn,const.N_eig,1);

%% Plot the discretized Irreducible Brillouin Zone
fig = plot_wavevectors(wv);
if isSaveOutput
    fix_pdf_border(fig)
    save_in_all_formats(fig,'wavevectors',plot_folder,false)
end

%% Plot the dispersion relation
fig = figure2();
ax = axes(fig);
hold(ax,'on');
for eig_idx_to_plot = 1:const.N_eig
    plot_dispersion_surface(wv,fr(:,eig_idx_to_plot),const.N_wv(1),const.N_wv(2),ax);
end
zlabel(ax,'\omega')
title(ax,'Dispersion Relation')
view(ax,2);
fix_pdf_border(fig)
if isSaveOutput
    save_in_all_formats(fig,'dispersion',plot_folder,false)
end

fig = figure;
ax = axes(fig);
for eig_idx_to_plot = 1:const.N_eig
    plot_dispersion_contour(wv,fr(:,eig_idx_to_plot),const.N_wv(1),const.N_wv(2),ax);
end
view(ax,2)
colorbar

%% Plot the dispersion surface intersections
% fig = figure2();
% ax = axes(fig);
% hold(ax,'on')
% view(ax,3)
% plot_dispersion_surface_intersections(wv,fr,[],[],ax)
% fig = fix_pdf_border(fig);
% title(ax,'Mode Intersection Points/Lines')
% ax.ZLim = [0 5000];
% zlabel(ax,'\omega')

%% Plot the modes
% plot_mode_ui(wv,fr,ev,const);

