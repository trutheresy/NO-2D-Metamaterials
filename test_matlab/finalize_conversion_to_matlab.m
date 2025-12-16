clear; close all;

input_data_path = "D:\Research\NO-2D-Metamaterials\data\dispersion_binarized_1_predictions_mat\dispersion_binarized_1_predictions.mat";

var_name = "EIGENVECTOR_DATA";

ev_data = h5read(input_data_path,['/' char(var_name)]);
designs = h5read(input_data_path,['/designs']);

EIGENVECTOR_DATA = ev_data.real + 1i*ev_data.imag;

output_data_path = 'eigenvectors';
vars_to_save = {'EIGENVECTOR_DATA','designs'};

save(output_data_path,vars_to_save{:},'-v7.3')