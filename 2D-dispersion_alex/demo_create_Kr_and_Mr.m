clear; close all;

data_fn = "generate_dispersion_dataset_Han\OUTPUT\output 15-Sep-2025 15-33-28\binarized 15-Sep-2025 15-33-28.mat";
% data_fn = "generate_dispersion_dataset_Han\OUTPUT\output 15-Sep-2025 15-36-03\continuous 15-Sep-2025 15-36-03.mat";
[~,fn,~] = fileparts(data_fn);
fn = char(fn);

data = load(data_fn);

% pick a unit cell
struct_idx = 1;

% pick a wavevector
wv_idx = 1;

% K M and T are all sparse matrices
K = data.K_DATA{struct_idx};
M = data.M_DATA{struct_idx};
T = data.T_DATA{wv_idx};

% create the reduced mass and stiffness matrices
Kr = T'*K*T;
Mr = T'*M*T;

% examine their sparsity pattern as sanity check
figure
spy(Kr)
title('sparsity pattern for K_r (reduced stiffness matrix)')

figure
spy(Mr)
title('sparsity diagram for M_r (reduced mass matrix)')