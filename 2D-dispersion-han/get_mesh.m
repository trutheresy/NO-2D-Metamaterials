function mesh = get_mesh(const)
    % Dimensionality of physical space
    dim = 2; % Hard code 2-dimensional for 2D-dispersion

    % Compute number of nodes along each side length of the unit cell
    N_node = const.N_pix.*const.N_ele + 1; % Assume scalar N_pix and N_ele
    N_node = repelem(N_node,1,dim);

    % Compute size of unit cell
    unit_cell_size = const.a; % Assume scalar
    unit_cell_size = repelem(unit_cell_size,1,dim);

    % Compute node locations
    node_loc_grid_vec = cell(1,dim);
    for i = 1:dim
        node_loc_grid_vec{i} = linspace(0,unit_cell_size(i),N_node(i));
    end
    
    node_coords_grid = cell(1,dim);
    [node_coords_grid{:}] = ndgrid(node_loc_grid_vec{:});
    
    % node_coords_list = cell(1,dim);
    % for i = 1:dim
    %     node_coords_list{i} = node_coords_grid{i}(:);
    % end
    
    % define nodal coordinates, size [prod(N_node) dim]
    % node_coords = [node_coords_list{:}];
    node_coords = node_coords_grid;

    % Attach to mesh struct
    mesh.dim = dim;
    mesh.node_coords = node_coords;
end