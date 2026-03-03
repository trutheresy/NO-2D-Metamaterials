function out = cellofsparse_to_full(system_matrix_data)
    N = length(system_matrix_data);

    out = zeros([N, size(system_matrix_data{1})]);
    for i = 1:N
        out(i,:,:) = single(full(system_matrix_data{i}));
    end
end