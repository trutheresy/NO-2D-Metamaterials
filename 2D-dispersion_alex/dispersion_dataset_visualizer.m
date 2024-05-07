clear; close all;

cmap = 'gray';

dataset_filename = 'design_data_pixelated_OG.mat';
dataset_folder = ['C:\Users\alex\OneDrive - California Institute of Technology\Documents\Graduate\Research\'...
    'comsol-2D-dispersion\OUTPUT\designs\'];
dataset_path = [dataset_folder dataset_filename];

load(dataset_path)

% pick param idxs to plot
N_design = [3 6];

% f = figure('WindowState','Maximized');
% for param1_idx = 1:N_param{1}
%     for param2_idx = 1:N_param{2}
%         pos = [(param2_idx-1)*plot_width (N_param{2}-param1_idx-1)*plot_height (plot_height-plot_spacing_y) (plot_width-plot_spacing_x)];
% %         pos = [((param2_idx-1))*plot_height (N_param{2} - 1 - (param1_idx-1))*plot_width (plot_height-plot_spacing_y) (plot_width-plot_spacing_x)];
%         ax(param1_idx,param2_idx) = subplot('Position',pos);
%         daspect([1 1 1])
% %         set(gca,'Visible','off')
%         prev_pos = pos;
%     end
% end

f = figure('WindowState','Maximized');
t = tiledlayout(N_design(1),N_design(2));
t.TileSpacing = 'tight';
t.Padding = 'tight';
for param1_idx = 1:N_design(1)
    for param2_idx = 1:N_design(2)
        ax(param1_idx,param2_idx) = nexttile;
    end
end

for design_idx_row = 1:N_design(1)
    for design_idx_col = 1:N_design(2)
        design_idx = sub2ind(N_design,design_idx_row,design_idx_col);
        axes(ax(design_idx_row,design_idx_col));
        design = designs(:,:,design_idx);
        imagesc(design)
        daspect([1 1 1])
        axis tight
        set(gca,'Visible','off')
    end
end

colormap(cmap)

f = fix_pdf_border(f);

print(f,['pixelated_dataset_visualization_' dataset_tag],'-painters','-dpdf')