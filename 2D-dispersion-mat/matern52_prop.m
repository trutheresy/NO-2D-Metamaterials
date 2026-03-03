function prop = matern52_prop(rng_seed,N_pix,sigma_f,sigma_l)
    rng(rng_seed,'twister');
    xx = linspace(0,1,N_pix(1)); yy = linspace(0,1,N_pix(2));
    
    [X,Y] = meshgrid(xx,yy);
    
    points = [reshape(X,[],1),reshape(Y,[],1)];
    %     scatter(points(:,1),points(:,2))
    
    mu = 0.5*ones(1,size(points,1));
    C = matern52(points,points,sigma_f,sigma_l);
    prop = mvnrnd(mu,C);
    prop = reshape(prop,N_pix);
    prop(prop<0) = 0;
    prop(prop>1) = 1;
end