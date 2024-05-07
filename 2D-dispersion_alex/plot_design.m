function [fig_handle,subax_handle] = plot_design(design)
    fig = figure();
    
    N_prop = 3;
    titles = {'Modulus','Density','Poisson'};
    
    for prop_idx = 1:N_prop
        subax(prop_idx) = axes(fig);
        subplot(1,3,prop_idx,subax(prop_idx))
        subplot(subax(prop_idx))
        imagesc(squeeze(design(:,:,prop_idx)));
        colormap('gray');
        caxis([0 1]);
        daspect([1 1 1]);
        title(titles{prop_idx});
    end
    
    c = colorbar('southoutside');
    c.Position = [.2 .2 .6 .02];
    if nargout > 0
        fig_handle = fig;
        if nargout > 1
            subax_handle = subax;
        end
    end
end