function mask = get_mask(symmetry_type,N_wv)

if strcmp(symmetry_type,'c1m1')
    mask = triu(true([N_wv(1) (N_wv(2)+1)/2]));
    mask = [flipud(mask(2:end,:)); mask];
end