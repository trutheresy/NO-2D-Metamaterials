function [fig_handle,ax_handle] = plot_dispersion_surface(wv,fr,cg,opts,ax)
    % Syntax:
    % plot_dispersion_surface(wv,fr,[],opts,ax)
    % plot_dispersion_surface(wv,[],cg,opts,ax)
    % plot_dispersion_surface(wv,[],cg,[],ax)
    % plot_dispersion_surface(wv,fr,[],[],ax)
    
    if ~exist('opts','var') || isempty(opts) || opts.isGetDefaultOpts
        opts.IBZ_shape = 'rectangle';
        [opts.N_k_x, opts.N_k_y] = set_N_ks(wv,opts.IBZ_shape);
        opts.Frequency.isPlot = ~isempty(fr);
        opts.GroupVelocityX.isPlot = ~isempty(cg);
        opts.GroupVelocityY.isPlot = ~isempty(cg);
        opts.GroupVelocityX.isPlotNumerical = ~isempty(fr);
        opts.GroupVelocityY.isPlotNumerical = ~isempty(fr);
        if isfield(opts,'isGetDefaultOpts') && opts.isGetDefaultOpts
            opts.isGetDefaultOpts = false;
            fig_handle = opts; % This is just to return the default opts structure for the given input. It's clearly not really a figure handle.
            return
        end
    end
    
    N_k_y = opts.N_k_y;
    N_k_x = opts.N_k_x;
    
    Z = nan(N_k_y,N_k_x);
    X = nan(N_k_y,N_k_x);
    Y = nan(N_k_y,N_k_x);
    
    if strcmp(opts.IBZ_shape,'triangle')
        % This section is outdated. Not taking the time to update it now.
        Z(triu(true(N_k))) = squeeze(fr); % tri
        X(triu(true(N_k))) = squeeze(wv(1,:)); % tri
        Y(triu(true(N_k))) = squeeze(wv(2,:)); %tri
        % end outdated section
    elseif strcmp(opts.IBZ_shape,'rectangle')
        X = reshape(squeeze(wv(:,1)),N_k_y,N_k_x); % rect
        Y = reshape(squeeze(wv(:,2)),N_k_y,N_k_x); % rect
        if opts.Frequency.isPlot || opts.GroupVelocityX.isPlotNumerical || opts.GroupVelocityY.isPlotNumerical
            Z = reshape(fr,N_k_y,N_k_x); % rect
        end
    end
    
    fig = figure2();
    %         ax = axes(fig);
    tiledlayout(fig,'flow')
    ax_counter = 0;
    
    if opts.Frequency.isPlot
        ax_counter = ax_counter + 1;
        ax(ax_counter) = nexttile;
        surf(ax,X,Y,Z)
        xlabel(ax,'\gamma_x')
        ylabel(ax,'\gamma_y')
        zlabel(ax,'\omega')
        tighten_axes(X,Y)
        daspect(ax,[pi pi max(max(Z))])
    end
    
    if opts.GroupVelocityX.isPlotNumerical
        ax_counter = ax_counter + 1;
        ax(ax_counter) = nexttile;
        [Z_x_num,~] = gradient(Z,X(1,2)-X(1,1));
        surf(ax,X,Y,Z_x_num)
        xlabel(ax,'\gamma_x')
        ylabel(ax,'\gamma_y')
        zlabel(ax,'cg_x numerical')
        tighten_axes(X,Y)
        daspect(ax,[pi pi max(max(Z_x_num))])
    end
    
    if opts.GroupVelocityY.isPlotNumerical
        ax_counter = ax_counter + 1;
        ax(ax_counter) = nexttile;
        [~,Z_y_num] = gradient(Z,Y(2,1)-Y(1,1));
        surf(ax,X,Y,Z_y_num)
        xlabel(ax,'\gamma_x')
        ylabel(ax,'\gamma_y')
        zlabel(ax,'cg_y numerical')
        tighten_axes(X,Y)
        daspect(ax,[pi pi max(max(Z_y_num))])
    end
    
    if opts.GroupVelocityX.isPlot
        nexttile;
        ax_counter = ax_counter + 1;
        ax(ax_counter) = nexttile;
        Z_x = reshape(cg(:,1),N_k_y,N_k_x);
        surf(ax,X,Y,Z_x)
        xlabel(ax,'\gamma_x')
        ylabel(ax,'\gamma_y')
        zlabel(ax,'cg_x')
        tighten_axes(X,Y)
        daspect(ax,[pi pi max(max(Z_x))])
    end
    
    if opts.GroupVelocityY.isPlot
        ax_counter = ax_counter + 1;
        ax(ax_counter) = nexttile;
        Z_y = reshape(cg(:,2),N_k_y,N_k_x);
        surf(ax,X,Y,Z_y)
        xlabel(ax,'\gamma_x')
        ylabel(ax,'\gamma_y')
        zlabel(ax,'cg_y')
        tighten_axes(X,Y)
        daspect(ax,[pi pi max(max(Z_y))])
    end
    
    % FOR DEBUGGING GROUP VELOCITY CALCULATIONS
    if opts.GroupVelocityX.isPlot && opts.GroupVelocityX.isPlotNumerical
        nexttile;
        ax_counter = ax_counter + 1;
        ax(ax_counter) = nexttile;
        surf(ax,X,Y,Z_x - Z_x_num)
        xlabel(ax,'\gamma_x')
        ylabel(ax,'\gamma_y')
        zlabel(ax,'cg_x - cg_x numerical')
        title('x difference')
        tighten_axes(X,Y)
        daspect(ax,[pi pi max(max(abs(Z_x-Z_x_num)))])
        view(2)
        colorbar
    end
    
    if opts.GroupVelocityY.isPlot && opts.GroupVelocityY.isPlotNumerical
        ax_counter = ax_counter + 1;
        ax(ax_counter) = nexttile;
        surf(ax,X,Y,Z_y - Z_y_num)
        xlabel(ax,'\gamma_x')
        ylabel(ax,'\gamma_y')
        zlabel(ax,'cg_y - cg_y numerical')
        title('y difference')
        tighten_axes(X,Y)
        daspect(ax,[pi pi max(max(abs(Z_y-Z_y_num)))])
        view(2)
        colorbar
    end
    
    if nargout > 0
        fig_handle = fig;
        ax_handle = ax;
    end
end

function [N_k_x,N_k_y] = set_N_ks(wv,IBZ_shape)
    [N_k_size,~] = size(wv);
    if strcmp(IBZ_shape,'triangle')
        N_k = -1/2 + 1/2*sqrt(1 + 8*N_k_size); % From the quadratic formula ... %tri
    elseif strcmp(IBZ_shape,'rectangle')
        N_k = sqrt(N_k_size); % rect
    end
    N_k_x = N_k;
    N_k_y = N_k;
end

function tighten_axes(X,Y)
    ax = gca();
    set(ax,'XLim',[min(min(X)) max(max(X))])
    set(ax,'YLim',[min(min(Y)) max(max(Y))])
end