function unpack_struct(input_struct)
    var_names = fieldnames(input_struct);
    for i = 1:length(var_names)
        fni = string(var_names(i));
        assignin('caller',var_names{i},input_struct.(fni));
    end
end