% Run your own clear, then manually load whatever mat file
close all;

% Plot the wavevectors
plot_wavevectors(WAVEVECTOR_DATA(:,:,1));

%% Plot first three designs and last three designs
struct_idxs = [1:3 N_struct_batch-(1:3)+1];

% Constitution
for struct_idx = struct_idxs
    plot_design(designs(:,:,:,struct_idx))
end

% Dispersion relations
for struct_idx = struct_idxs
    fig = figure();
    ax = axes(fig);

    for eig_idx = 1:const.N_eig
        x = WAVEVECTOR_DATA(:,1,struct_idx);
        y = WAVEVECTOR_DATA(:,2,struct_idx);
        z = EIGENVALUE_DATA(:,eig_idx,struct_idx);
        scatter3(ax,x,y,z)
        hold(ax,'on')
        dasp = daspect(ax);
        dasp = [1 1 dasp(3)];
        daspect(ax,dasp)
    end
end