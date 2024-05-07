clear; close all;

N = 1000;

x = linspace(0,1,N);

sigma_f = 1;
sigma_l = .25;
p = .33;

r = abs(x - x');

sin_arg = pi*r/p;
C = sigma_f^2*exp(-2*sin(sin_arg).^2/sigma_l^2);

figure
imagesc(C)
colorbar
min(eig(C))

y = mvnrnd(zeros(N,1),C);
plot(x,y)