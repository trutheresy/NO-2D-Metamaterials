function A_sym = apply_p4mm_symmetry(A)
    % Measure original data range of A
    orig_range = [min(A,[],'all') max(A,[],'all')];
    
    % Create A_sym, a symmetrified version of A
    A = 1/2*(A + fliplr(A));
    A = 1/2*(A + flipud(A));
    A = 1/2*(A + A');
    A = 1/2*(fliplr(flipud(A)) + fliplr(flipud(A))');
    A_sym = A;
    
    % Normalize so min(A_sym) is zero
    A_sym = A_sym - min(A_sym,[],'all');

    % Normalize so range(A_sym) == 1
    A_sym = A_sym./max(A_sym,[],'all');

    % Multiply so that range(A_sym) == range(A)
    A_sym = A_sym*(orig_range(2) - orig_range(1));

    % Add back the original min
    A_sym = A_sym + orig_range(1);
end