clear; close all;

N_struct = 100;
N_pix = [50 50];
a = 1;
sigma_f = .3;
sigma_l = .3;

save_folder = 'OUTPUT\designs\';

dataset_tag = 'pixelated_OG2';

[X,Y] = meshgrid(linspace(0,a,N_pix(1)),linspace(0,a,N_pix(2)));
coord_points = [X(:) Y(:)];

designs = zeros([N_pix N_struct]);

C = material_kfcn(coord_points,sigma_f,sigma_l);

Z = mvnrnd(zeros(prod(N_pix),1),C,N_struct);
Z = permute(Z,[2 1]);
Z = reshape(Z,[N_pix N_struct]);

designs = Z > 0;

figure
imagesc(Z(:,:,1))
colorbar

figure
imagesc(designs(:,:,1))
colorbar

save([save_folder 'design_data_' dataset_tag],'designs','Z','X','Y','sigma_f','sigma_l','N_pix','N_struct','a','dataset_tag');

function C = material_kfcn(x,sigma_f,sigma_l)
    r = pdist(x);
    r = squareform(r);
    C = sigma_f^2*(1 + sqrt(3)*r/sigma_l).*exp(-sqrt(3)*r/sigma_l);
end