function [fig_handle,ax_handle,plot_handle] = plot_dispersion(wn,fr,N_contour_segments,ax)
    if ~exist('ax','var')
        fig = figure();
        ax = axes(fig);
    end
    
    plot_handle = plot(ax,wn,fr,'k.-');
    ax.YMinorGrid = 'on';
    ax.XMinorGrid = 'on';
    hold(ax,'on')
    
    for i = 1:N_contour_segments-1
        xline(ax,i);
    end
    
    xlabel(ax,'wavevector parameter')
    ylabel(ax,'frequency [Hz]')
    
    if nargout > 0
        fig_handle = ax.Parent;
        ax_handle = ax;
    end
end