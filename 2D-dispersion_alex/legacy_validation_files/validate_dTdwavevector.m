clear; close all;

%% Step 1
T0 = T;
dTa = dTdwavevector;

%% Step 2
dTn = ndSparse.build([size(T) 2]);

T_y = T;
dTn(:,:,2) = (T_y - T0)/(1e-8);

%% Step 3
T_x = T;
dTn(:,:,1) = (T_x - T0)/(1e-8);

%% Plot
figure2();
tiledlayout('flow')
for i = 1:2
    nexttile
    surf(abs(dTa(:,:,i) - dTn(:,:,i)))
    title(['analytical minus numerical, component ' num2str(i)])
end

figure2();
tiledlayout('flow')
for i = 1:2
    nexttile
    surf(abs(dTa(:,:,i)))
    title(['analytical comp ' num2str(i)])
    nexttile
    surf(abs(dTn(:,:,i)))
    title(['numerical comp ' num2str(i)])
end

