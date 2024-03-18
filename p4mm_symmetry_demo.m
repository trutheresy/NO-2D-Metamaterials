clear; close all;

dispersion_library_path = 'C:\Users\alex\OneDrive - California Institute of Technology\Documents\Graduate\Research\2D-dispersion';
addpath(dispersion_library_path)

N = 30;

unit_cell_size = [1 1]; % [m]
gridvec = linspace(0,unit_cell_size(1),N);
[X,Y] = ndgrid(gridvec,gridvec);
points = [X(:) Y(:)];
property_variance = 1;
lengthscale = 1;

sigma = periodic_kernel(points,points,property_variance,lengthscale,unit_cell_size); % Covariance matrix
mu = 0.5*ones(N^2,1);

A = mvnrnd(mu,sigma);
A = reshape(A,N,N);
A(A<0) = 0;
A(A>1) = 1;

A_sym = apply_p4mm_symmetry(A);

fig = figure;
tlo = tiledlayout('flow');

ax = nexttile;
imagesc(A)
set(ax,'YDir','normal')
daspect([1 1 1])
title('design before p4mm symmetrification')
colorbar

axs(1) = ax;

ax = nexttile;
imagesc(A_sym)
set(ax,'YDir','normal')
daspect([1 1 1])
title('design after p4mm symmetrification')
colorbar

axs(2) = ax;

set(axs,'CLim',[min([axs.CLim]) max([axs.CLim])])