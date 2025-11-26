clear; close all;
warning('This script is not verified. Verify this is the actual mesh ordering before trusting this.')
mesh_resolution = [3 3]; % Number of elements in each direction
domain_size = [1 1];
element_size = domain_size./mesh_resolution;

for i = 1:2
    node_grid_vectors{i} = linspace(0,domain_size(i),mesh_resolution(i) + 1); % In each direction there is one more node than elements %#ok<SAGROW>
end

% Generate coordinates of nodes
[X,Y] = ndgrid(node_grid_vectors{:});
node_coords = [X(:) Y(:)];

% Generate global labels of nodes
number_of_nodes = numel(X);
node_labels = (1:number_of_nodes)';

% Generate labels of degrees of freedom
dof_labels = node_labels*2 - [1 0];

% Generate coordinates of element centers
for i = 1:2
    element_grid_vectors{i} = linspace(0,domain_size(i)-element_size(i),mesh_resolution(i)); % Compute only the location of the node with smallest x,y,z in each element %#ok<SAGROW>
end
[X,Y] = ndgrid(element_grid_vectors{:});
element_coords = [X(:) Y(:)];
element_coords = element_coords + element_size./2;

% Generate labels of elements
number_of_elements = numel(X);
element_labels = (1:number_of_elements)';

% Plot
fig = figure;
tlo = tiledlayout(2,2);
spacing1 = .1./mesh_resolution;
spacing2 = -spacing1;

% Plot global node labels
nexttile
scatter(node_coords(:,1),node_coords(:,2),'k','filled')
text(node_coords(:,1) + spacing1(1),node_coords(:,2) + spacing1(2),num2cell(node_labels))
xlabel('x'); ylabel('y');
xticklabels([]); yticklabels([]);
daspect([1 1 1])
title('nodes and their global labels')

% Plot global degree of freedom labels
nexttile
scatter(node_coords(:,1),node_coords(:,2),'k','filled')
text(node_coords(:,1) + spacing1(1),node_coords(:,2) + spacing1(2),array2cell(dof_labels))
xlabel('x'); ylabel('y');
xticklabels([]); yticklabels([]);
daspect([1 1 1])
title('nodes and their global degrees of freedom')

% Plot global degree of freedom and node labels on the same plot
nexttile
scatter(node_coords(:,1),node_coords(:,2),'k','filled')
text(node_coords(:,1) + spacing1(1),node_coords(:,2) + spacing1(2),array2cell(dof_labels))
xlabel('x'); ylabel('y');
xticklabels([]); yticklabels([]); zticklabels([]);
daspect([1 1 1])

hold on

text(node_coords(:,1) + spacing2(1),node_coords(:,2) + spacing2(2),num2cell(node_labels),'color','red')
xlabel('x'); ylabel('y');
xticklabels([]); yticklabels([]);
daspect([1 1 1])
title('nodes and their global degrees of freedom')

% Plot global degree of freedom and node labels on the same plot, with
% element numbers
nexttile
scatter(node_coords(:,1),node_coords(:,2),'k','filled')
text(node_coords(:,1) + spacing1(1),node_coords(:,2) + spacing1(2),array2cell(dof_labels))
xlabel('x'); ylabel('y');
xticklabels([]); yticklabels([]);
daspect([1 1 1])

hold on

text(node_coords(:,1) + spacing2(1),node_coords(:,2) + spacing2(2),num2cell(node_labels),'color','red')
xlabel('x'); ylabel('y');
xticklabels([]); yticklabels([]);
daspect([1 1 1])

text(element_coords(:,1),element_coords(:,2),num2cell(element_labels),'color','blue')
xlabel('x'); ylabel('y');
xticklabels([]); yticklabels([]);
daspect([1 1 1])
title('nodes, global degrees of freedom, and elements')

function out = array2cell(array)
    char_array = num2str(array);
    dim1distances = ones(1,size(char_array,1));
    dim2distances = size(char_array,2);
    row_wise_cell_array = mat2cell(char_array,dim1distances,dim2distances);
    row_wise_cell_array = cellfun(@(x) strtrim(x),row_wise_cell_array,'UniformOutput',false);
    row_wise_cell_array = cellfun(@(x) regexprep(x,'\s+',','),row_wise_cell_array,'UniformOutput',false); % Replace any number of repeated spaces with a comma
    out = row_wise_cell_array;
end