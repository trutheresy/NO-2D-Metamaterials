function [T,dTdwavevector] = get_transformation_matrix(wavevector,const)
    
    N_node = (const.N_ele*const.N_pix(1)) + 1;
    
    r = [const.a; 0];
    xphase = exp(1i*dot(wavevector,r));
    if nargout == 2
        dxphasedwavevector = 1i*r*xphase;
    end
    
    r = [0; -const.a]; % [0; -const.a] --> [0; const.a]?
    yphase = exp(1i*dot(wavevector,r));
    if nargout == 2
        dyphasedwavevector = 1i*r*yphase;
    end
    
    r = [const.a; -const.a]; % [const.a; -const.a] --> [const.a; const.a]?
    cornerphase = exp(1i*dot(wavevector,r));
    if nargout == 2
        dcornerphasedwavevector = 1i*r*cornerphase;
    end
    
    node_idx_x = [reshape(meshgrid(1:(N_node-1),1:(N_node - 1)),[],1)' N_node*ones(1,N_node - 1) 1:(N_node-1) N_node];
    node_idx_y = [reshape(meshgrid(1:(N_node-1),1:(N_node - 1))',[],1)' 1:(N_node-1) N_node*ones(1,N_node - 1) N_node];
    % should we change to ndgrid? they are the unch_idxs --> node_idx_x = [reshape(ndgrid(1:(N_node-1),1:(N_node - 1)),[],1)' N_node*ones(1,N_node - 1) 1:(N_node-1) N_node];
    % should we change to ndgrid? they are the unch_idxs --> node_idx_y = [reshape(ndgrid(1:(N_node-1),1:(N_node - 1))',[],1)' 1:(N_node-1) N_node*ones(1,N_node - 1) N_node];
    global_node_idx = (node_idx_y - 1)*N_node + node_idx_x;
    global_dof_idxs = [2*global_node_idx - 1 2*global_node_idx];
    unch_idxs = 1:((N_node-1)^2);
    x_idxs = (((N_node-1)^2) + 1):(((N_node-1)^2) + 1 + N_node - 2);
    y_idxs = (((N_node-1)^2) + N_node):(((N_node-1)^2) + N_node + N_node - 2);

    reduced_global_node_idx = [(node_idx_y(unch_idxs) - 1)*(N_node - 1) + node_idx_x(unch_idxs)...
        (node_idx_y(x_idxs) - 1)*(N_node - 1) + node_idx_x(x_idxs) - (N_node - 1) ...
        node_idx_x(y_idxs)...
        1];
    reduced_global_dof_idxs = [2*reduced_global_node_idx - 1 2*reduced_global_node_idx];
    
    row_idxs = global_dof_idxs';
    col_idxs = reduced_global_dof_idxs';
    value_T = repmat([ones((N_node-1)^2,1); xphase*ones(N_node - 1,1); yphase*ones(N_node - 1,1); cornerphase],2,1);
    
    T = sparse(row_idxs,col_idxs,value_T);
    
%     if nargout == 2
%        value_dTdwavevector = [repmat([zeros((N_node-1)^2,1); dxphasedwavevector(1)*ones(N_node - 1,1); dyphasedwavevector(1)*ones(N_node - 1,1); dcornerphasedwavevector(1)],2,1);...
%                               repmat([zeros((N_node-1)^2,1); dxphasedwavevector(2)*ones(N_node - 1,1); dyphasedwavevector(2)*ones(N_node - 1,1); dcornerphasedwavevector(2)],2,1)];
%        wv_component_idxs = [1*ones(size(row_idxs)); 2*ones(size(row_idxs))];
%        dTdwavevector_nds = ndSparse.build([ [row_idxs; row_idxs] [col_idxs; col_idxs] wv_component_idxs ], value_dTdwavevector);
%     end
    if nargout == 2
        dTdwavevector = cell(2,1);
        row_idxs2 = [row_idxs; row_idxs];
        col_idxs2 = [col_idxs; col_idxs];
        wv_component_idxs = [1*ones(size(row_idxs)); 2*ones(size(row_idxs))];
        value_dTdwavevector = [repmat([zeros((N_node-1)^2,1); dxphasedwavevector(1)*ones(N_node - 1,1); dyphasedwavevector(1)*ones(N_node - 1,1); dcornerphasedwavevector(1)],2,1);...
            repmat([zeros((N_node-1)^2,1); dxphasedwavevector(2)*ones(N_node - 1,1); dyphasedwavevector(2)*ones(N_node - 1,1); dcornerphasedwavevector(2)],2,1)];
        for wv_comp_idx = 1:2
            idx_idxs = find(wv_component_idxs == wv_comp_idx);
            dTdwavevector{wv_comp_idx} = sparse(row_idxs2(idx_idxs),col_idxs2(idx_idxs),value_dTdwavevector(idx_idxs));
        end
    end
end

