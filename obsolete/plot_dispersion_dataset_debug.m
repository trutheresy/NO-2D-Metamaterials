% DEBUG ONLY: save intermediate plotting arrays for parity comparison.
%
% Required caller vars:
%   data_fn_path   : .mat dataset with WAVEVECTOR_DATA / EIGENVALUE_DATA
%   debug_out_dir  : output folder for plot-debug artifacts
% Optional:
%   n_structs      : number of structures (default 1)

if ~exist('data_fn_path', 'var') || isempty(data_fn_path)
    error('data_fn_path must be provided.');
end
if ~exist('debug_out_dir', 'var') || isempty(debug_out_dir)
    error('debug_out_dir must be provided.');
end
if ~exist('n_structs', 'var') || isempty(n_structs)
    n_structs = 1;
end

if ~exist(debug_out_dir, 'dir')
    mkdir(debug_out_dir);
end

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, '2D-dispersion-mat'));

data = load(data_fn_path);
wv_all = double(data.WAVEVECTOR_DATA);
fr_all = double(data.EIGENVALUE_DATA);

if ndims(wv_all) == 2
    wv_all = reshape(wv_all, size(wv_all,1), size(wv_all,2), 1);
end
if ndims(fr_all) == 2
    fr_all = reshape(fr_all, size(fr_all,1), size(fr_all,2), 1);
end

n_total = size(wv_all, 3);
n_plot = min(n_structs, n_total);
[contour_wv, contour_info] = get_IBZ_contour_wavevectors(10, 1.0, 'none');

for struct_idx = 1:n_plot
    wv = double(wv_all(:, :, struct_idx));          % (N_wv, 2)
    fr = double(fr_all(:, :, struct_idx));          % (N_wv, N_eig)
    n_eig = size(fr, 2);

    interp_true = cell(n_eig,1);
    for eig_idx = 1:n_eig
        interp_true{eig_idx} = scatteredInterpolant(wv, fr(:, eig_idx), 'linear', 'linear');
    end

    frequencies_contour = zeros(size(contour_wv,1), n_eig);
    for eig_idx = 1:n_eig
        frequencies_contour(:, eig_idx) = interp_true{eig_idx}(contour_wv(:,1), contour_wv(:,2));
    end

    fig = figure('Visible', 'off');
    ax = axes(fig);
    plot(ax, contour_info.wavevector_parameter, frequencies_contour);
    xlabel(ax, 'Wavevector Contour Parameter');
    ylabel(ax, 'Frequency [Hz]');
    title(ax, 'Dispersion Relation (MATLAB debug)');
    for i = 0:contour_info.N_segment
        xline(ax, i, '--k');
    end
    exportgraphics(fig, fullfile(debug_out_dir, sprintf('struct_%d_dispersion.png', struct_idx-1)), 'Resolution', 150);
    close(fig);

    plot_debug = struct();
    plot_debug.wavevectors_raw = wv;
    plot_debug.frequencies_raw = fr;
    plot_debug.contour_wavevectors = contour_wv;
    plot_debug.contour_parameter = contour_info.wavevector_parameter(:);
    plot_debug.frequencies_contour = frequencies_contour;
    plot_debug.plot_x = contour_info.wavevector_parameter(:);
    plot_debug.plot_y = frequencies_contour;
    save(fullfile(debug_out_dir, sprintf('struct_%d_plot_debug.mat', struct_idx-1)), 'plot_debug');
end

manifest = struct();
manifest.n_structs = n_plot;
manifest.data_fn_path = data_fn_path;
save(fullfile(debug_out_dir, 'manifest.mat'), 'manifest');
fprintf('Saved MATLAB plot debug outputs to: %s\n', debug_out_dir);
