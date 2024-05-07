clear; close all;

N = 20;
sigma_f_periodic = 1;
sigma_l_periodic = 1;
sigma_f_matern = 1;
sigma_l_matern = 10;
p = [1 1];

gridvec = linspace(0,1,N);
[X,Y] = meshgrid(gridvec,gridvec);
points = [X(:) Y(:)];

points_i = points;
points_j = points;
points_i = permute(points_i,[1 3 2]);
points_j = permute(points_j,[1 3 2]);
displacements = points_i - permute(points_j,[2 1 3]);
r = vecnorm(displacements,2,3);

sin_arg1 = pi*abs(displacements(:,:,1))/p(1);
sin_arg2 = pi*abs(displacements(:,:,2))/p(2);
% C = sigma_f^2*exp(-2*sin(sin_arg1 + sin_arg2).^2/sigma_l^2);
C_periodic1 = sigma_f_periodic^2*exp(-2*sin(sin_arg1).^2/sigma_l_periodic^2);
C_periodic2 = sigma_f_periodic^2*exp(-2*sin(sin_arg2).^2/sigma_l_periodic^2);
C_matern = sigma_f_matern^2*(1 + sqrt(5)*r/sigma_l_matern + 5*r.^2/(3*sigma_l_matern^2)).*exp(-sqrt(5)*r/sigma_l_matern);
C_periodic = C_periodic1.*C_periodic2;
C = C_periodic.*C_matern;

figure
imagesc(C_matern)
title('matern52')
colorbar
figure
imagesc(C_periodic)
title('bi-directional periodic')
colorbar
figure
imagesc(C)
title('global covariance')
colorbar
min(eig(C))

z = mvnrnd(zeros(N^2,1),C);
Z = reshape(z,N,N);
figure
imagesc([0 1],[0 1],Z);
set(gca,'YDir','normal')
title('random sample')

figure
imagesc([0 3],[0 3],repmat(Z,3,3))
set(gca,'YDir','normal')
title('random sample, 3x3 tesselation')