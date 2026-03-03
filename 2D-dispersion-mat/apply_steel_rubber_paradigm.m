function design = apply_steel_rubber_paradigm(design,const)
    % design should be a N_pix matrix (with only one pane)
    design_in_polymer = 0;
    design_in_steel = 1;

    E_polymer = 100e6;
    E_steel = 200e9;

    rho_polymer = 1200;
    rho_steel = 8e3;

    nu_polymer = 0.45;
    nu_steel = 0.3;

    % 0 --> const.prop_min, 1 --> const.prop_max
    %
    % Invert this:
    % E = (const.E_min + const.design(:,:,1).*(const.E_max - const.E_min))';

    design_out_polymer_E = (E_polymer - const.E_min)/(const.E_max - const.E_min);
    design_out_polymer_rho = (rho_polymer - const.rho_min)/(const.rho_max - const.rho_min);
    design_out_polymer_nu = (nu_polymer - const.poisson_min)/(const.poisson_max - const.poisson_min);

    design_out_steel_E = (E_steel - const.E_min)/(const.E_max - const.E_min);
    design_out_steel_rho = (rho_steel - const.rho_min)/(const.rho_max - const.rho_min);
    design_out_steel_nu = (nu_steel - const.poisson_min)/(const.poisson_max - const.poisson_min);

    design_vals = [
        design_out_polymer_E design_out_steel_E;
        design_out_polymer_rho design_out_steel_rho;
        design_out_polymer_nu design_out_steel_nu
        ];

    for prop_idx = 1:3
        dvs = design_vals(prop_idx,:);
        design(:,:,prop_idx) = interp1([design_in_polymer design_in_steel],dvs,design(:,:,prop_idx));
    end
end