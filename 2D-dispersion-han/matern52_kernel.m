function C = matern52(points_i,points_j,sigma_f,sigma_l)
    points_i = permute(points_i,[1 3 2]);
    points_j = permute(points_j,[1 3 2]);
    displacements = points_i - permute(points_j,[2 1 3]);
    r = vecnorm(displacements,2,3);
    
    C = sigma_f^2*(1 + sqrt(5)*r/sigma_l + 5*r.^2/(3*sigma_l^2)).*exp(-sqrt(5)*r/sigma_l);
end