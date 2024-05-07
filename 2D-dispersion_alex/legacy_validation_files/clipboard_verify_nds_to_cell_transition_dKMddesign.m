s = size(dKddesign_nds);

max_diffs_dK = zeros(s(3:4));

for i = 1:s(3)
    for j = 1:s(4)
        max_diffs_dK(i,j) = max(full(abs(dKddesign{i,j} - dKddesign_nds(:,:,i,j))),[],'all');
    end
end

for i = 1:s(3)
    for j = 1:s(4)
        max_diffs_dM(i,j) = max(full(abs(dMddesign{i,j} - dMddesign_nds(:,:,i,j))),[],'all');
    end
end

disp(['Max dKddesign difference is ' num2str(max(max_diffs_dK,[],'all'))])

disp(['Max dMddesign difference is ' num2str(max(max_diffs_dM,[],'all'))])