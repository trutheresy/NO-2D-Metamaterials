function converted_design = convert_design(design,initial_format,target_format,E_min,E_max,rho_min,rho_max,poisson_min,poisson_max)
    % initial_format and target_format can be 'linear','log','explicit'
    
    if strcmp(initial_format,target_format)
        converted_design = design;
        return
    end
    
    if strcmp(initial_format,'linear')
        explicit_design = cat(3,E_min + (E_max - E_min)*design(:,:,1), rho_min + (rho_max - rho_min)*design(:,:,2), poisson_min + (poisson_max - poisson_min)*design(:,:,3));
    elseif strcmp(initial_format,'log')
        explicit_design = cat(3,exp(design(:,:,1:2)),poisson_min + (poisson_max - poisson_min)*design(:,:,3));
    elseif strcmp(initial_format,'explicit')
        explicit_design = design;
    end
        
    if strcmp(target_format,'linear')
        converted_design = cat(3,(explicit_design(:,:,1) - E_min)/(E_max - E_min),(explicit_design(:,:,2) - rho_min)/(rho_max - rho_min),(explicit_design(:,:,3 - poisson_min))/(poisson_max - poisson_min));
    elseif strcmp(target_format,'log')
        converted_design = cat(3,log(explicit_design(:,:,1:2)),(explicit_design(:,:,3 - poisson_min))/(poisson_max - poisson_min));
    end
end