classdef gridMesh 
    properties
        dim = []; % size [1], physical dimensionality of the mesh
        N_node = []; % size [1 dim], defines the number of nodes along each side length
        node_coordinates = {}; % cell size {dim}, each entry array size [prod(N_node)], defines the coordinates of each node in the mesh
    end
    methods
        function node_coordinates_grid = node_coordinates_to_list(self)
            
        end
    end
end