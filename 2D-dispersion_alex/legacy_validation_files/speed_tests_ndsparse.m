clear; close all;
rng(1)
breaker = repmat('=',1,50);
% ndSparse

% s = [32 32 2 2 2];
% s = [256 256 4 4 4];
% s = [512 512 8 8 8];
s = [2^14 2^13 2 1 1];

N_nz_frac = .05;
N_total = prod(s);
N_nz = round(N_total*N_nz_frac);

disp(['Array size = ' num2str(s)])
disp(['Array numel = ' num2str(prod(s))])
disp(['Array nonzero frac = ' num2str(N_nz_frac)])
disp(['Array num nonzero = ' num2str(N_nz)])

for i = 1:length(s)
    idxs(:,i) = randi(s(i),N_nz,1);
end
V = rand(N_nz,1);

disp(breaker)
disp('ndSparse')
disp(breaker)

disp('Creating with ndSparse.build')
tic
A_nds = ndSparse.build(idxs,V,s);
toc

disp('Accessing with A_nds(:,:,i,j,k)')
tic
for i = 1:s(3)
    for j = 1:s(4)
        for k = 1:s(5)
            temp = A_nds(:,:,i,j,k);
        end
    end
end
toc

% cell

disp(breaker)
disp('Cell')
disp(breaker)

disp('Creating with for loop')
tic
A_cell = cell(s(3:5));
for i = 1:s(3)
    for j = 1:s(4)
        for k = 1:s(5)
            A_cell{i,j,k} = sprand(s(1),s(2),N_nz_frac);
        end
    end
end
toc

disp('Accessing with A_cell{i,j,k}')
tic
for i = 1:s(3)
    for j = 1:s(4)
        for k = 1:s(5)
            temp = A_cell{i,j,k};
        end
    end
end
toc

disp(breaker)
disp('Full')
disp(breaker)
disp('Creating with full(ndSparse)')
tic
A_full = full(A_nds);
toc

disp('Accessing with A_full(:,:,i,j,k)')
tic
for i = 1:s(3)
    for j = 1:s(4)
        for k = 1:s(5)
            temp = A_full(:,:,i,j,k);
        end
    end
end
toc