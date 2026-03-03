function design = get_design2(design_parameters)
    %     switch design_parameters.property_coupling
    %         case 'uncoupled'
    design = zeros(design_parameters.N_pix);
    for prop_idx = 1:3
        design(:,:,prop_idx) = get_prop(design_parameters,prop_idx);
    end
    %         case 'coupled'
    %             prop = get_prop(design_parameters);
    %             design = repmat(prop,1,1,2);
    %             design(:,:,3) = .6*ones(size(design(:,:,1)));
    %         otherwise
    %             error('code not written for this case')
end
