% Wrapper script to run plot_dispersion.m on continuous dataset
data_fn_cli = 'D:\Research\NO-2D-Metamaterials\OUTPUT\test dataset\out_continuous_1.mat';
% Write to temp file for plot_dispersion.m to read
fid = fopen('temp_data_fn.txt', 'w');
fprintf(fid, '%s', data_fn_cli);
fclose(fid);
run('plot_dispersion.m');
delete('temp_data_fn.txt');

