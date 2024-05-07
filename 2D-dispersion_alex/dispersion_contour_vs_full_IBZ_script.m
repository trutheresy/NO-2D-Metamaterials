clear; close all; %delete(findall(0));

isSaveOutput = false;

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
const.N_pix = 16;
const.N_wv = [30 59];
const.N_k = 500;
const.N_eig = 6;
const.isUseGPU = false;
const.isUseImprovement = true;
const.isUseParallel = true;
const.isSaveEigenvectors = false;

const.E_min = 200e6; % 2e9
const.E_max = 200e9; % 200e9
const.rho_min = 8e2; % 1e3
const.rho_max = 8e3; % 8e3
const.poisson_min = 0; % 0
const.poisson_max = .5; % .5
const.t = 1;
const.sigma_eig = 1;

design_parameters.design_number = 272;
design_parameters.design_style = 'matern52';
design_parameters.design_options = struct('sigma_f',0.5,'sigma_l',0.05,'symmetry_type','c1m1','feature_size',8,'N_value',2);
design_parameters.N_pix = [const.N_pix const.N_pix];

[const.wavevectors,N_contour_segments,critical_point_labels] = get_IBZ_contour_wavevectors(const.N_k,const.a,design_parameters.design_options.symmetry_type);

%% Random cell
const.design_scale = 'linear';
const.design = get_design2(design_parameters);
const.design = convert_design(const.design,'linear',const.design_scale,const.E_min,const.E_max,const.rho_min,const.rho_max);

%% Plot the design
fig = plot_design(const.design);

%% Solve the dispersion problem along the contour
[wv_contour,fr_contour,~] = dispersion(const,const.wavevectors);
wv_parameter = linspace(0,N_contour_segments,size(const.wavevectors,1))';
wv_parameter = repmat(wv_parameter,1,const.N_eig);

%% Solve the dispersion problem over the full BZ
const.wavevectors = get_IBZ_wavevectors(const.N_wv,const.a,'omit',1);
[wv_full,fr_full,~] = dispersion(const,const.wavevectors);

%% Plot the discretized Irreducible Brillouin Zone
fig = plot_wavevectors(wv_contour);

%% Plot the dispersion relation on the contour
fig = plot_dispersion(wv_parameter,fr_contour);
hold on

% Evaluate contour bandgaps
for eig_idx = 1:const.N_eig-1
    contour_max = max(fr_contour(:,eig_idx),[],'all');
    contour_min = min(fr_contour(:,eig_idx+1),[],'all');
    if contour_min > contour_max
        yline(contour_min,'m')
        yline(contour_max,'b')
        vertices = [0 contour_max; N_contour_segments contour_max; N_contour_segments contour_min; 0 contour_min];
        faces = [1 2 3 4];
        colors = [.3 .3 .3];
        patch('Faces',faces,'Vertices',vertices,'FaceVertexCData',[.3 .3 .3])
    end
end

%% Plot the dispersion relation over the full BZ
fig = figure;
ax = axes(fig);
hold(ax,'on');
for eig_idx = 1:const.N_eig
    plot_dispersion_surface(wv_full,fr_full(:,eig_idx),const.N_wv(1),const.N_wv(2),ax);
end
view(ax,[1 0 0])
% Evaluate full bandgaps
for eig_idx = 1:const.N_eig-1
    full_max = max(fr_full(:,eig_idx),[],'all');
    full_min = min(fr_full(:,eig_idx+1),[],'all');
    if full_min > full_max
%         yline(full_min,'m')
%         yline(full_max,'b')
        vertices = [0 min(wv_full(:,2)) full_max; 0 max(wv_full(:,2)) full_max; 0 max(wv_full(:,2)) full_min; 0 min(wv_full(:,2)) full_min];
        faces = [1 2 3 4];
        colors = [.3 .3 .3];
        patch('Faces',faces,'Vertices',vertices,'FaceVertexCData',[.3 .3 .3])
    end
end
