function global_idxs = get_global_idxs(ele_idx_x,ele_idx_y,const)
    global_idxs = zeros(8,1);
    N_node_y = (const.N_ele*const.N_pix) + 1;
    
    % local node 1
    node_idx_y = ele_idx_y + 1; node_idx_x = ele_idx_x;
    global_node_idx = (node_idx_y - 1)*N_node_y + node_idx_x;
    global_idxs(1:2) = (2*global_node_idx - 1):(2*global_node_idx);
    
    % local node 2
    node_idx_y = ele_idx_y + 1; node_idx_x = ele_idx_x + 1;
    global_node_idx = (node_idx_y - 1)*N_node_y + node_idx_x;
    global_idxs(3:4) = (2*global_node_idx - 1):(2*global_node_idx);
    
    % local node 3
    node_idx_y = ele_idx_y; node_idx_x = ele_idx_x + 1;
    global_node_idx = (node_idx_y - 1)*N_node_y + node_idx_x;
    global_idxs(5:6) = (2*global_node_idx - 1):(2*global_node_idx);
    
    % local node 4
    node_idx_y = ele_idx_y; node_idx_x = ele_idx_x;
    global_node_idx = (node_idx_y - 1)*N_node_y + node_idx_x;
    global_idxs(7:8) = (2*global_node_idx - 1):(2*global_node_idx);
end