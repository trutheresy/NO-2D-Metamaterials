function [fig_handle,ax_handle] = plot_dispersion_surface_intersections(wv,fr,N_k_x,N_k_y,ax)
    
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
    Z = reshape(fr,N_k_y,N_k_x,size(fr,2)); % rect
    
    hold(ax,'on')
    for eig_idx = 2:size(fr,2)
%         [row_idxs,col_idxs] = find(abs(Z(:,:,eig_idx) - Z(:,:,eig_idx-1))<1e-3);
%         scatter3(ax,reshape(X(row_idxs,col_idxs),[],1),...
%                     reshape(Y(row_idxs,col_idxs),[],1),...
%                     reshape(Z(row_idxs,col_idxs,eig_idx),[],1))
        idxs = find(abs((Z(:,:,eig_idx) - Z(:,:,eig_idx-1))./Z(:,:,eig_idx))<0.5e-2);
        Z_temp = Z(:,:,eig_idx);
        scatter3(ax,X(idxs),Y(idxs),Z_temp(idxs),'.')
    end
    xlabel(ax,'\gamma_x')
    ylabel(ax,'\gamma_y')
%     zlabel(ax,'\omega')
    tighten_axes(ax,X,Y)
    daspect(ax,[pi pi max(Z,[],'all')])
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