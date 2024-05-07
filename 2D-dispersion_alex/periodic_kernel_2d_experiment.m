clear; close all;

N = 20;
sigma_f = 1;
sigma_l = 1;
p = [1 1];

gridvec = linspace(0,1,N);
[X,Y] = meshgrid(gridvec,gridvec);
points = [X(:) Y(:)];

points_i = points;
points_j = points;
points_i = permute(points_i,[1 3 2]);
points_j = permute(points_j,[1 3 2]);
displacements = points_i - permute(points_j,[2 1 3]);

sin_arg1 = pi*abs(displacements(:,:,1))/p(1);
sin_arg2 = pi*abs(displacements(:,:,2))/p(2);
% C = sigma_f^2*exp(-2*sin(sin_arg1 + sin_arg2).^2/sigma_l^2);
C1 = sigma_f^2*exp(-2*sin(sin_arg1).^2/sigma_l^2); % periodicity in one direction
C2 = sigma_f^2*exp(-2*sin(sin_arg2).^2/sigma_l^2); % periodicity in the other direction
C = C1.*C2;

figure
imagesc(C)
colorbar
min(eig(C))

z = mvnrnd(zeros(N^2,1),C);
Z = reshape(z,N,N);
imagesc([0 1],[0 1],Z);
set(gca,'YDir','normal')