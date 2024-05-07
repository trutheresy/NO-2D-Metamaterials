function C = periodic_kernel(points_i,points_j,sigma_f,sigma_l,period)
    points_i = permute(points_i,[1 3 2]);
    points_j = permute(points_j,[1 3 2]);
    displacements = points_i - permute(points_j,[2 1 3]);

    sin_arg1 = pi*abs(displacements(:,:,1)/period(1));
    C1 = sigma_f^2*exp(-2*sin(sin_arg1).^2/sigma_l^2);
    sin_arg2 = pi*abs(displacements(:,:,2)/period(2));
    C2 = sigma_f^2*exp(-2*sin(sin_arg2).^2/sigma_l^2);
    C = C1.*C2;
end