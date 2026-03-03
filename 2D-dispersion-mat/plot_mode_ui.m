function plot_mode_ui(wv,fr,ev,const)
    k_idx = 1;
    eig_idx = 1;
    scale = .1;
    fig = uifigure('handlevisibility','on');
    ax_still = uiaxes('Parent', fig, 'Position', [10 10 200 200],'DataAspectRatio',[1 1 1],'handlevisibility','on'); % In future data aspect ratio can be defined by const.a (if unit cell is rectangle)
    ax_animation = uiaxes('Parent', fig, 'Position', [10 210 200 200],'DataAspectRatio',[1 1 1],'handlevisibility','on'); % In future data aspect ratio can be defined by const.a (if unit cell is rectangle)
%     axes(ax);
%     figure(fig);

    % mesh = get_mesh(const);
    
    plot_mode(wv,fr,ev,eig_idx,k_idx,'still',scale,const,ax_still);
    
    dd_wv = uidropdown(fig,...
        'Position',[430 150 100 22],...
        'Items', array2cell(wv'),...
        'ItemsData',1:size(wv,1),...
        'ValueChangedFcn', @(dd_wv, event) update_plot_wv(dd_wv));

    dd_wv_label = uilabel(fig,...
        'position',[340 150 100 22],...
        'text','wavevector');

    dd_eig =  uidropdown(fig,...
        'Position',[430 110 100 22],...
        'Items', array2cell(1:size(fr,2)),...
        'ItemsData',1:size(fr,2),...
        'ValueChangedFcn', @(dd_eig, event) update_plot_ev(dd_eig));

    dd_eig_label = uilabel(fig,...
        'position',[340 110 100 22],...
        'text','eigenvalue band');
    
    sld = uislider(fig,...
        'Position',[450 200 150 3],...
        'Orientation','vertical',...
        'Limits',[-1 1],...
        'Value',.1,...
        'MajorTicks',linspace(-1,1,11),...
        'ValueChangedFcn',@(sld, event) update_scale(sld));

    dd_sld_label = uilabel(fig,...
        'position',[360 200 100 22],...
        'text','scale');
    
    clkbut = uibutton(fig,...
        'Position',[430 50 100 22],...
        'Text','Play animation',...
        'ButtonPushedFcn',@(clkbut, event) play_animation());
        
        
    
    function update_plot_wv(dd_wv)
        k_idx = dd_wv.Value;
%         axes(ax);
%         figure(fig);
        plot_mode(wv,fr,ev,eig_idx,k_idx,'still',scale,const,ax_still);
    end
    
    function update_plot_ev(dd_ev)
        eig_idx = dd_ev.Value;
%         axes(ax);
%         figure(fig);
        plot_mode(wv,fr,ev,eig_idx,k_idx,'still',scale,const,ax_still);
    end
    
    function update_scale(sld)
        scale = sld.Value;
%         axes(ax);
%         figure(fig);
        plot_mode(wv,fr,ev,eig_idx,k_idx,'still',scale,const,ax_still);
    end
    
    function play_animation()
        plot_mode(wv,fr,ev,eig_idx,k_idx,'animation',scale,const,ax_animation);
    end
    
    function C = array2cell(A)
        % Each column becomes a character array in the cell
        dbl_cell = num2cell(A',2);
        C = cell(size(dbl_cell));
        for i = 1:length(C)
            C{i} = num2str(dbl_cell{i});
        end
    end
end