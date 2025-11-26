clear; close all;

% demonstrates how eigenvectors are indexed (dof ordering)
%
% As a validation check, after you have written your Python script to convert
% your eigenvector format to my eigenvector format, run the data through
% this script and make sure the images look exactly the same as when you
% plot them in your Python plotting scripts.

% dispersion_library_path = '../../';
% addpath(dispersion_library_path)

data_fn = "generate_dispersion_dataset_Han\OUTPUT\output 23-Sep-2025 12-31-16\binarized 23-Sep-2025 12-31-16.mat";
% data_fn = "generate_dispersion_dataset_Han\OUTPUT\output 15-Sep-2025 15-36-03\continuous 15-Sep-2025 15-36-03.mat";
[~,fn,~] = fileparts(data_fn);
fn = char(fn);

data = load(data_fn);

% pick an eigenvector
struct_idx = 1;
wv_idx = 5;
band_idx = 4;

% get the eigenvector
%
% eigvec is one single eigenvector,
% that means it corresponds to one specific design
% at one specific wavevector
% on one specific band (eigenvalue index)
eigvec = data.EIGENVECTOR_DATA(:,wv_idx,band_idx,struct_idx);

% get each displacement component from the eigenvector
% u = horizontal displacement
% v = vertical displacement
% starting at first entry, eigvec alternates u-v-u-v-u-v-...
u = eigvec(1:2:end);
v = eigvec(2:2:end);

% reshape
% note that matlab is column major. 
% I don't know what convention numpy/python follow.
%
% Column-major vs row-major is *extremely* important to pay attention to
% here, because it directly affects how arrays are reshaped.
design_size = size(data.const.design); % When you do your sanity check, you may have to replace this line
u = reshape(u,design_size(1:2)); % only need first two entries of design size because the third entry is for E,rho,nu
v = reshape(v,design_size(1:2)); % apply same to v

% plot
fig = figure();
ax = axes(fig);
imagesc(ax,real(u))
title('real(u)')
colorbar
daspect([1 1 1])

fig = figure();
ax = axes(fig);
imagesc(ax,imag(u))
title('imag(u)')
colorbar
daspect([1 1 1])

fig = figure();
ax = axes(fig);
imagesc(ax,real(v))
title('real(v)')
colorbar
daspect([1 1 1])

fig = figure();
ax = axes(fig);
imagesc(ax,imag(v))
title('imag(v)')
colorbar
daspect([1 1 1])