clear; close all;

% Visualization options
isPlayAnimation = true;
isVisualizeDesigns = true;
N_designs_to_plot_fcn = @(designs) min(10,size(designs,4));

% Toy datasets
% data_fn = "C:\Users\alex\OneDrive - California Institute of Technology\Documents\Graduate\Research\Shared-Rayehe-Han-Alex\neural-operator\OUTPUT\output 01-Mar-2024 15-36-15\DATA N_pix32x32 N_ele1x1 N_wv25x13 N_disp3 N_eig6 offset0 01-Mar-2024 15-36-15.mat";
% data_fn = "C:\Users\alex\OneDrive - California Institute of Technology\Documents\Graduate\Research\Shared-Rayehe-Han-Alex\neural-operator\OUTPUT\output 01-Mar-2024 18-25-55\DATA N_pix32x32 N_ele1x1 N_wv25x13 N_disp3 N_eig6 offset0 01-Mar-2024 18-25-55.mat";
% data_fn = "C:\Users\alex\OneDrive - California Institute of Technology\Documents\Graduate\Research\Shared-Rayehe-Han-Alex\neural-operator\OUTPUT\output 01-Mar-2024 18-39-41\DATA N_pix32x32 N_ele1x1 N_wv25x13 N_disp10 N_eig6 offset0 01-Mar-2024 18-39-41.mat";

% Real datasets
data_fn = "C:\Users\alex\OneDrive - California Institute of Technology\Documents\Graduate\Research\SHDA46~1\NEURAL~1\OUTPUT\PRELIM~1\OUTPUT~1\DATAN_~1.MAT";

% Load data
data = load(data_fn);

unpack_struct(data)

wv = WAVEVECTOR_DATA(:,:,1);

% Plot the first structure
fig = figure();
ax = axes(fig);

struct_idx_to_plot = 1;
material_property_idx_to_plot = 1; % 1 for modulus, 2 for density, 3 for poisson
imagesc([0 const.a],[0 const.a],designs(:,:,material_property_idx_to_plot,struct_idx_to_plot)')
set(ax,'YDir','normal')
daspect([1 1 1])
xlabel('spatial coordinate x')
ylabel('spatial coordinate y')
title(['The first design array' newline 'Modulus (color) over space (axes)'])

% Plot the first band of the first structure
fig = figure();
ax = axes(fig);

eig_idx_to_plot = 1;
struct_idx_to_plot = 1;
imagesc(wv(:,1),wv(:,2),reshape(EIGENVALUE_DATA(:,eig_idx_to_plot,struct_idx_to_plot),flip(const.N_wv)))
set(ax,'YDir','normal')
daspect([1 1 1])
xlabel('wavevector x')
ylabel('wavevector y')
title(['One band of the dispersion relation' newline 'Frequency (color) as a function of wavevector (axes)'])

% Plot the mode corresponding with the zero wavevector, of the first band, of the first structure
fig = figure();
ax = axes(fig);

wavevector_to_plot = [0 0];

eig_idx_to_plot = 1;
struct_idx_to_plot = 1;
wavevector_x_idx = find(wv(:,1) == wavevector_to_plot(1)); wavevector_y_idx = find(wv(:,2) == wavevector_to_plot(2)); wavevector_idx = intersect(wavevector_x_idx,wavevector_y_idx);
wavevector_idx_to_plot = wavevector_idx;
displacement_component_idx_to_plot = 1;
dof_idxs_to_plot = displacement_component_idx_to_plot:2:size(EIGENVECTOR_DATA,1);
N_dof_in_each_direction = [const.N_pix*const.N_ele const.N_pix*const.N_ele];

imagesc([0 const.a],[0 const.a],reshape(real(EIGENVECTOR_DATA(dof_idxs_to_plot,wavevector_idx_to_plot,eig_idx_to_plot,struct_idx_to_plot)),N_dof_in_each_direction)')
set(ax,'YDir','normal')
daspect([1 1 1])
xlabel('spatial coord x')
ylabel('spatial coord y')
title(['One mode of one band of the dispersion relation of one structure' newline 'Horizontal displacement (color) as a function of position (axes)' newline 'wavevector = ' num2str(wavevector_to_plot)])

% Plot the mode corresponding with the [0 pi] wavevector, of the first band, of the first structure
fig = figure();
ax = axes(fig);

wavevector_to_plot = [0 pi];

eig_idx_to_plot = 1;
struct_idx_to_plot = 1;
wavevector_x_idx = find(wv(:,1) == wavevector_to_plot(1)); wavevector_y_idx = find(wv(:,2) == wavevector_to_plot(2)); wavevector_idx = intersect(wavevector_x_idx,wavevector_y_idx);
wavevector_idx_to_plot = wavevector_idx;
displacement_component_idx_to_plot = 1;
dof_idxs_to_plot = displacement_component_idx_to_plot:2:size(EIGENVECTOR_DATA,1);
N_dof_in_each_direction = [const.N_pix*const.N_ele const.N_pix*const.N_ele];

imagesc([0 const.a],[0 const.a],reshape(real(EIGENVECTOR_DATA(dof_idxs_to_plot,wavevector_idx_to_plot,eig_idx_to_plot,struct_idx_to_plot)),N_dof_in_each_direction)')
set(ax,'YDir','normal')
daspect([1 1 1])
xlabel('spatial coord x')
ylabel('spatial coord y')
title(['One mode of one band of the dispersion relation of one structure' newline 'Horizontal displacement (color) as a function of position (axes)' newline 'wavevector = ' num2str(wavevector_to_plot)])

% Animate the mode corresponding with the [0 pi] wavevector, of the first band, of the first structure
N_animate = 4;
phases = linspace(0,2*pi,100);
if isPlayAnimation
    fig = figure();
    ax = axes(fig);

    % For wavevector_to_plot = [1.57 1.57]
    wv_idx_to_plot = 241;
    wavevector_to_plot = wv(wv_idx_to_plot,:);

    % % For wavevector_to_plot = [0 pi]
    % wavevector_to_plot = [0 pi];

    eig_idx_to_plot = 1;
    struct_idx_to_plot = 1;
    wavevector_x_idx = find(wv(:,1) == wavevector_to_plot(1)); wavevector_y_idx = find(wv(:,2) == wavevector_to_plot(2)); wavevector_idx = intersect(wavevector_x_idx,wavevector_y_idx);
    wavevector_idx_to_plot = wavevector_idx;
    displacement_component_idx_to_plot = 1;
    dof_idxs_to_plot = displacement_component_idx_to_plot:2:size(EIGENVECTOR_DATA,1);
    N_dof_in_each_direction = [const.N_pix*const.N_ele const.N_pix*const.N_ele];
    mode_to_plot = EIGENVECTOR_DATA(dof_idxs_to_plot,wavevector_idx_to_plot,eig_idx_to_plot,struct_idx_to_plot);

    colorlimits = [-max(abs(mode_to_plot),[],'all') max(abs(mode_to_plot),[],'all')];

    for j = 1:N_animate
        for phase_idx = 1:length(phases)
            phase = phases(phase_idx);
            complex_scalar = exp(i*phase);
            imagesc([0 const.a],[0 const.a],reshape(real(complex_scalar*mode_to_plot),N_dof_in_each_direction)')
            set(ax,'YDir','normal')
            daspect([1 1 1])
            xlabel('spatial coord x')
            ylabel('spatial coord y')
            title(['Animation of one mode of one band of the dispersion relation of one structure' newline 'Horizontal displacement (color) as a function of position (axes)' newline 'wavevector = ' num2str(wavevector_to_plot)])
            set(ax,'clim',colorlimits)
            colorbar
            drawnow
            pause(0.01)
        end
    end
end

% Visualize all designs
if isVisualizeDesigns
    N_designs_to_plot = N_designs_to_plot_fcn(designs);
    fig = figure;
    tlo = tiledlayout(3,N_designs_to_plot,'TileIndexing','ColumnMajor','TileSpacing','Compact','Padding','Compact');
    title(tlo,'N designs')

    for struct_idx = 1:N_designs_to_plot
        for prop_idx = 1:3
            d = designs(:,:,prop_idx,struct_idx);
            tile_idx = sub2ind(tlo.GridSize,prop_idx,struct_idx);
            ax = nexttile(tile_idx);

            imagesc([0 const.a],[0 const.a],d)
            set(ax,'YDir','normal')
            daspect([1 1 1])

            ax.Visible = 'off';

            if struct_idx == 1
                switch prop_idx
                    case 1
                        ylabel('modulus')
                    case 2
                        ylabel('density')
                    case 3
                        ylabel('poisson')
                end
            end
        end
    end
end
