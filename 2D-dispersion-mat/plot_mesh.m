function [fig,ax,pat] = plot_mesh(mesh,ax)

    if ~exist('ax','var')
        fig = figure();
        ax = axes(fig);
    else
        fig = ax.Parent;
    end
    
    % inds = cell(1,mesh.dim);
    % for i = 1:mesh.dim
    %     inds{i} = {}
    % end

    left_coords_x = mesh.node_coords{1}(1:end-1,:); % x-coord of all nodes that are on the left edge of at least one element
    right_coords_x = mesh.node_coords{1}(2:end,:); % x-coord of all nodes that are on the right edge of at least one element
    lower_coords_y = mesh.node_coords{2}(:,1:end-1); % y-coord of all nodes that are on the lower edge of at least one element
    upper_coords_y = mesh.node_coords{1}(:,2:end); % y-coord of all nodes that are on the upper edge of at least one element

    lower_left_coords = [left_coords_x(:) lower_coords_y(:)];
    lower_right_coords = [right_coords_x(:) lower_coords_y(:)];
    upper_right_coords = [right_coords_x(:) upper_coords_y(:)];
    upper_left_coords = [left_coords_x(:) upper_coords_y(:)];

    comp_ind = 1;
    X = [lower_left_coords(:,comp_ind) lower_right_coords(:,comp_ind) upper_right_coords(:,comp_ind) upper_left_coords(:,comp_ind)];
    
    comp_ind = 2;
    Y = [lower_left_coords(:,comp_ind) lower_right_coords(:,comp_ind) upper_right_coords(:,comp_ind) upper_left_coords(:,comp_ind)];

    X = X'; Y = Y';

    C = ones(size(X,2),1,3);

    pat = patch(ax,X,Y,C);

    daspect(ax,[1 1 1])
end