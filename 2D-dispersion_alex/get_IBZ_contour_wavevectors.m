function [wavevectors,contour_info] = get_IBZ_contour_wavevectors(N_k,a,symmetry_type)
    vertex_labels = {};
    if length(N_k) > 2
        N_k = N_k(1);
        warning('received N_k as a vector')
    end
    switch symmetry_type
        case 'p4mm'
            vertices = [0 0; pi/a 0; pi/a pi/a; 0 0];
            wavevectors = get_contour_from_vertices(vertices);
            vertex_labels = {'$\Gamma$','$X$','$M$','$\Gamma$'};
        case 'c1m1'
            vertices = [0 0; pi/a 0; pi/a pi/a; 0 0; pi/a -pi/a; pi/a 0]; % Gamma X M Gamma \bar{O} X
            wavevectors = get_contour_from_vertices(vertices);
            vertex_labels = {'$\Gamma$','$X$','$M$','$\Gamma$','$\bar{O}$','$X$'};
        case 'p6mm'
            vertices = [0 0; pi/a*cosd(30)*cosd(30) -pi/a*cosd(30)*sind(30); pi/a 0; 0 0];
            wavevectors = get_contour_from_vertices(vertices);
            vertex_labels = {'$\Gamma$','$X$','$M$','$\Gamma$'};
        case 'none'
            vertices = [0 0; pi/a 0; pi/a pi/a; 0 0; 0 pi/a; -pi/a pi/a; 0 0];
            wavevectors = get_contour_from_vertices(vertices);
            vertex_labels = {'$\Gamma$','$X$','$M$','$\Gamma$','$Y$','$O$','$\Gamma$'};
        case 'all contour segments'
            vertices = ...
                [0 0; pi/a 0;
                0 0; pi/a pi/a;
                0 0; 0 pi/a;
                0 0; -pi/a pi/a;
                0 0; -pi/a 0;
                0 0; -pi/a -pi/a;
                0 0; 0 -pi/a;
                0 0; pi/a -pi/a;
                pi/a 0; pi/a pi/a;
                pi/a pi/a; 0 pi/a;
                0 pi/a; -pi/a pi/a;
                -pi/a pi/a; -pi/a 0;
                -pi/a 0; -pi/a -pi/a;
                -pi/a -pi/a; 0 -pi/a;
                0 -pi/a; pi/a -pi/a;
                pi/a -pi/a; pi/a 0];
            wavevectors = [];
            for vertex_idx = 1:2:(size(vertices,1)-1)
                if vertex_idx <= 17
                    wavevectors = [wavevectors; linspaceNDim(vertices(vertex_idx,:),vertices(vertex_idx+1,:),N_k)'];
                else
                    wavevectors = [wavevectors(1:(end-1),:); linspaceNDim(vertices(vertex_idx,:),vertices(vertex_idx+1,:),N_k)'];
                end
            end
        otherwise
            error('symmetry_type not recognized')
    end
    N_segment = size(vertices,1) - 1;
    if isempty(vertex_labels)
        warning('critical_point_labels not yet defined for this symmetry_type')
    end
    contour_info.N_segment = N_segment;
    contour_info.vertex_labels = vertex_labels;
    contour_info.vertices = vertices;
    contour_info.wavevector_parameter = linspace(0,N_segment,size(wavevectors,1));
    
    function wavevectors = get_contour_from_vertices(vertices)
        wavevectors = [];
        for vertex_idx = 1:(size(vertices,1)-1)
            wavevectors = [wavevectors(1:(end-1),:); linspaceNDim(vertices(vertex_idx,:),vertices(vertex_idx+1,:),N_k)'];
        end
    end
end

