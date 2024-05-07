function design = convert_vector_to_design(design_vec,N_pix)
    % this function is to convert Zhi's vectors into const.design format
    subdesign = zeros(N_pix/2);
    subdesign(triu(true(N_pix/2))) = vec;
    subdesign_diag = diag(diag(subdesign));
    subdesign = subdesign + subdesign' - subdesign_diag;
    design(:,:,1) = [subdesign, fliplr(subdesign); flipud(subdesign), flipud(fliplr(subdesign))]; % Young's Modulus
    design(:,:,2) = design(:,:,1); % Density
    design(:,:,3) = .3; % Poisson's Ratio
end