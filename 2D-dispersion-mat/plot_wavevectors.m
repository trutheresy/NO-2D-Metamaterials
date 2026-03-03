function [fig_handle,ax_handle] = plot_wavevectors(wv,ax)
    if ~exist('ax','var')
        fig = figure();
        ax = axes(fig);
    end

    hold(ax,'on')
    % axis([-pi/const.a pi/const.a -pi/const.a pi/const.a])
    scatter(ax,wv(:,1),wv(:,2),[],[0 0 0],'MarkerFaceColor',[0 0 0],'MarkerFaceAlpha',.5)
    daspect(ax,[1 1 1])
    for i = 1:size(wv,1)
        text(ax,wv(i,1)+.05,wv(i,2)+.05,num2str(i));
    end
    hold(ax,'off')
    
    if nargout > 0
        fig_handle = ax.Parent;
        ax_handle = ax;
    end
end