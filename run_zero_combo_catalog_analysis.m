clear; clc;

repo_root = fileparts(mfilename('fullpath'));
combo_mat_path = fullfile(repo_root, 'OUTPUT', 'zero_output_combinations_catalog.mat');
output_root = fullfile(repo_root, 'OUTPUT', ['zero_combo_analysis_' datestr(now,'yyyymmdd_HHMMSS')]);
mkdir(output_root);

if ~exist(combo_mat_path, 'file')
    error('Combo catalog not found: %s', combo_mat_path);
end

C = load(combo_mat_path);
n_rep = double(C.combo_representative_count);
fprintf('Running representative combo analysis for %d combinations...\n', n_rep);
generate_plot = false; % default off

summary = repmat(struct( ...
    'combo_index', [], ...
    'combo_id', '', ...
    'dataset_name', '', ...
    'geometry_idx_0based', [], ...
    'wavevector_idx_1based', [], ...
    'band_idx_1based', [], ...
    'eigenvalue_real', [], ...
    'eigenvalue_imag', [], ...
    'K_near_singular', [], ...
    'M_near_singular', [], ...
    'T_near_singular', [], ...
    'Kr_near_singular', [], ...
    'Mr_near_singular', [], ...
    'smin_Kr', [], ...
    'smin_Mr', [], ...
    'figure_path', '', ...
    'mat_output_path', ''), n_rep, 1);

for i = 1:n_rep
    fprintf('  [%d/%d] combo idx %d\n', i, n_rep, i);
    try
        r = analyze_zero_combo_single(combo_mat_path, i, 'representative', output_root, generate_plot);
        summary(i).combo_index = i;
        summary(i).combo_id = r.combo_id;
        summary(i).dataset_name = C.combo_rep_dataset_name{i};
        summary(i).geometry_idx_0based = double(C.combo_rep_geometry_idx_0based(i));
        summary(i).wavevector_idx_1based = double(C.combo_rep_wavevector_idx_1based(i));
        summary(i).band_idx_1based = double(C.combo_rep_band_idx_1based(i));
        summary(i).eigenvalue_real = real(r.eigenvalue);
        summary(i).eigenvalue_imag = imag(r.eigenvalue);
        summary(i).K_near_singular = r.matrix_diagnostics.K.near_singular;
        summary(i).M_near_singular = r.matrix_diagnostics.M.near_singular;
        summary(i).T_near_singular = r.matrix_diagnostics.T.near_singular;
        summary(i).Kr_near_singular = r.matrix_diagnostics.Kr.near_singular;
        summary(i).Mr_near_singular = r.matrix_diagnostics.Mr.near_singular;
        summary(i).smin_Kr = r.matrix_diagnostics.Kr.smin;
        summary(i).smin_Mr = r.matrix_diagnostics.Mr.smin;
        summary(i).figure_path = r.figure_path;
        summary(i).mat_output_path = r.mat_output_path;
    catch ME
        warning('Combo %d failed: %s', i, ME.message);
    end
end

summary_mat = fullfile(output_root, 'zero_combo_representative_summary.mat');
save(summary_mat, 'summary', 'combo_mat_path', 'output_root');

summary_txt = fullfile(output_root, 'zero_combo_representative_summary.txt');
fid = fopen(summary_txt, 'w');
fprintf(fid, 'combo_index\tdataset\tgeom_idx0\twv_idx1\tband_idx1\teig_real\teig_imag\tK_sing\tM_sing\tT_sing\tKr_sing\tMr_sing\tsmin_Kr\tsmin_Mr\n');
for i = 1:numel(summary)
    if isempty(summary(i).combo_index)
        continue;
    end
    fprintf(fid, '%d\t%s\t%d\t%d\t%d\t%.12g\t%.3e\t%d\t%d\t%d\t%d\t%d\t%.6e\t%.6e\n', ...
        summary(i).combo_index, summary(i).dataset_name, summary(i).geometry_idx_0based, ...
        summary(i).wavevector_idx_1based, summary(i).band_idx_1based, ...
        summary(i).eigenvalue_real, summary(i).eigenvalue_imag, ...
        summary(i).K_near_singular, summary(i).M_near_singular, summary(i).T_near_singular, ...
        summary(i).Kr_near_singular, summary(i).Mr_near_singular, ...
        summary(i).smin_Kr, summary(i).smin_Mr);
end
fclose(fid);

disp(['ZERO_COMBO_OUTPUT_ROOT=' output_root]);
disp(['ZERO_COMBO_SUMMARY_MAT=' summary_mat]);
disp(['ZERO_COMBO_SUMMARY_TXT=' summary_txt]);
