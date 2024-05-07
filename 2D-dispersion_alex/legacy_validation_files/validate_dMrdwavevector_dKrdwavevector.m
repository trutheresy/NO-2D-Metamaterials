clear; close all;
%% Step 1
% For numerical gradients
Kr1 = Kr;
Mr1 = Mr;

% Compute analytical gradient
dTf = full(dTdwavevector);
Tf = full(T);
Kf = full(K);
Mf = full(M);

dMa = pagemtimes(dTf,'ctranspose',Mf*Tf,'none') + pagemtimes(Tf'*Mf,dTf);

dKa = pagemtimes(dTf,'ctranspose',Kf*Tf,'none') + pagemtimes(Tf'*Kf,dTf);

%% Step 2
% For numerical gradients
Kr2 = Kr;
Mr2 = Mr;

% Compute numerical gradients
dKn = (Kr2 - Kr1)/(1e-8);
dMn = (Mr2 - Mr1)/(1e-8);

%% Derivative with respect to SECOND wavevector component
plot_fcn = @(data) imagesc(data);

figure2()
plot_fcn(abs(dKn))
title('numerical')
colorbar
set(gca(),'ColorScale','log')
cax = caxis();

figure2()
plot_fcn(abs(dKa(:,:,2)))
title('analytical')
colorbar
caxis(cax);
set(gca(),'ColorScale','log')

figure2()
plot_fcn(abs(dKa(:,:,2) - dKn)./abs(dKn(:,:,2)))
title('analytical minus numerical, normalized')
colorbar
set(gca(),'ColorScale','log')

%% Derivative with respect to FIRST wavevector component
figure2()
imagesc(abs(dKn))
title('numerical')
colorbar

figure2()
imagesc(abs(dKa(:,:,1)))
title('analytical')
colorbar

figure2()
imagesc(abs(dKa(:,:,1) - dKn)./abs(Kr))
title('analytical minus numerical, normalized')
colorbar
    