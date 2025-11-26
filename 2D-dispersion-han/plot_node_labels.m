function plot_node_labels(mesh,ax)
    spacing1 = 0.05*max(abs([mesh.node_coords{1}(:); mesh.node_coords{1}(:)]));
    spacing2 = -spacing1;
    
    text(ax,mesh.node_coords{1}(:) + spacing1,mesh.node_coords{2}(:) + spacing2,num2cell(1:numel(mesh.node_coords{1})))
end