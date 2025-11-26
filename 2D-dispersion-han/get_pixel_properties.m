function [E,nu,t,rho] = get_pixel_properties(pix_idx_x,pix_idx_y,const)
    if strcmp(const.design_scale,'linear')
        E = const.E_min + const.design(pix_idx_y,pix_idx_x,1)*(const.E_max - const.E_min);
        nu = const.poisson_min + const.design(pix_idx_y,pix_idx_x,3)*(const.poisson_max - const.poisson_min);
        t = const.t;
        rho = const.rho_min + const.design(pix_idx_y,pix_idx_x,2)*(const.rho_max - const.rho_min);
    elseif strcmp(const.design_scale,'log')
        E = exp(const.design(pix_idx_y,pix_idx_x,1));
        nu = const.poisson_min + const.design(pix_idx_y,pix_idx_x,3)*(const.poisson_max - const.poisson_min);
        t = const.t;
        rho = exp(const.design(pix_idx_y,pix_idx_x,2));
    else
        error('const.design_scale not recognized as log or linear')
    end
end