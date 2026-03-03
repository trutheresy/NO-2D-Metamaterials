function [fig_handle,ax_handle] = plot_dispersion_contour(wv,fr,N_k_x,N_k_y,ax)
    
    if ~exist('ax','var')
        fig = figure2();
        ax = axes(fig);
    else
        fig = ax.Parent;
    end
    
    if isempty(N_k_x)||isempty(N_k_y)||~exist('N_k_x','var')||~exist('N_k_y','var')
        N_k_y = sqrt(size(wv,1));
        N_k_x = sqrt(size(wv,1));
    end
    
    X = reshape(squeeze(wv(:,1)),N_k_y,N_k_x); % rect
    Y = reshape(squeeze(wv(:,2)),N_k_y,N_k_x); % rect
    Z = reshape(fr,N_k_y,N_k_x); % rect
    
    contour(ax,X,Y,Z)
    xlabel(ax,'\gamma_x')
    ylabel(ax,'\gamma_y')
%     zlabel(ax,'\omega')
    tighten_axes(ax,X,Y)
    daspect(ax,[pi pi max(max(Z))])
    view(ax,3)
    
    if nargout > 0
        fig_handle = fig;
        ax_handle = ax;
    end
end

function tighten_axes(ax,X,Y)
    set(ax,'XLim',[min(min(X)) max(max(X))])
    set(ax,'YLim',[min(min(Y)) max(max(Y))])
end