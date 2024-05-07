s = size(dTdwavevector_nds);

max_diffs_dT = zeros(s(3),1);

for i = 1:s(3)
        max_diffs_dT(i) = max(full(abs(dTdwavevector{i} - dTdwavevector_nds(:,:,i))),[],'all');
end

disp(['Max dTdwavevector difference is ' num2str(max(max_diffs_dT,[],'all'))])