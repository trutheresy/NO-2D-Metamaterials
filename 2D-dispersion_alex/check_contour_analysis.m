clear; close all;

data_fn = "C:\Users\alex\OneDrive - California Institute of Technology\Documents\Graduate\Research\2D-dispersion\OUTPUT\output 13-Dec-2021 18-10-01\DATA N_pix16x16 N_ele1x1 N_wv16x31 N_disp1000 N_eig10 offset0 13-Dec-2021 18-10-01.mat";
full_bg_thresh = 100;
contour_bg_thresh = 100;
bg_width_error_thresh = 50;

data = load(data_fn);
unpack_struct(data);
unpack_struct(const);
symmetry_type = design_parameters.design_options.symmetry_type;

% Create list of wavevectors that define the IBZ contour
[contour_wavevectors,N_contour_segments,critical_point_labels] = get_IBZ_contour_wavevectors(N_wv(1),a,symmetry_type);

% Z = reshape(EIGENVALUE_DATA,[fliplr(N_wv) size(EIGENVALUE_DATA,[2 3])]);

N_struct = size(EIGENVALUE_DATA,3);

wv_x = WAVEVECTOR_DATA(:,1,1);
wv_y = WAVEVECTOR_DATA(:,2,1);
contour_bg = false(const.N_eig-1,N_struct);
full_bg = false(const.N_eig-1,N_struct);
for struct_idx = 1:N_struct
    for eig_idx = 1:const.N_eig - 1
        % Lower band
        full_frequencies = EIGENVALUE_DATA(:,eig_idx,struct_idx);
        
        F = scatteredInterpolant(wv_x,wv_y,full_frequencies);
        contour_frequencies = F(contour_wavevectors);
        
        % Eval max of lower band
        full_max = max(full_frequencies,[],'all');
        contour_max = max(contour_frequencies,[],'all');
        
        % Upper band
        full_frequencies = EIGENVALUE_DATA(:,eig_idx+1,struct_idx);
        
        F = scatteredInterpolant(wv_x,wv_y,full_frequencies);
        contour_frequencies = F(contour_wavevectors);
        
        % Eval min of upper band
        full_min = min(full_frequencies,[],'all');
        contour_min = min(contour_frequencies,[],'all');
        
        % Process mins and maxes
        if (contour_min - contour_max) > contour_bg_thresh
            contour_bg(eig_idx,struct_idx) = true;
        end
        contour_bg_width(eig_idx,struct_idx) = max(contour_min - contour_max,0);
        
        if full_min - full_max > full_bg_thresh
            full_bg(eig_idx,struct_idx) = true;
        end
        full_bg_width(eig_idx,struct_idx) = max(full_min - full_max,0);
        
        bg_width_error(eig_idx,struct_idx) = contour_bg_width(eig_idx,struct_idx) - full_bg_width(eig_idx,struct_idx);
        if bg_width_error(eig_idx,struct_idx)>bg_width_error_thresh
            disp(['Contour analysis fails for struct_idx = ' num2str(struct_idx) ', eig_idxs = [' num2str(eig_idx) ' ' num2str(eig_idx+1) ']'])
        end
        
        if contour_bg(eig_idx,struct_idx)
            if ~full_bg(eig_idx,struct_idx)
                %                 disp(['Contour analysis fails for struct_idx = ' num2str(struct_idx) ', eig_idxs = [' num2str(eig_idx) ' ' num2str(eig_idx+1) ']'])
            end
        end
    end
end

N_full_bg = nnz(full_bg)

N_contour_bg = nnz(contour_bg)

% for struct_idx = 1:N_struct
%     for eig_idx = 1:N_eig
%         if bg_width_error(eig_idx,struct_idx)>bg_width_error_thresh
%             fig = figure;
%             ax = axes(fig);
%             hold on
%             for
%         end
%     end
% end

