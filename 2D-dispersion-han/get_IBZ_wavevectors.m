function wavevectors = get_IBZ_wavevectors(N_wv,a,symmetry_type,N_tesselations)
    if ~exist('symmetry_type','var')
        symmetry_type = 'none';
    end
    if ~exist('num_tesselations','var')
        N_tesselations = 1;
    end
    if numel(N_wv) == 1
        N_wv = [N_wv N_wv];
    end
    
    switch symmetry_type
        case 'omit'
            [X,Y] = meshgrid(linspace(-pi/a,pi/a,N_wv(1)),linspace(-pi/a,pi/a,N_wv(2))); % a square centered at the origin of side length 2*pi/a
            gamma_x = X(true(size(X))); gamma_y = Y(true(size(Y))); % rect
            %         IBZ_shape = 'square';
        case 'none'
            [X,Y] = meshgrid(linspace(-pi/a,pi/a,N_wv(1)),linspace(0,pi/a,N_wv(2))); % true asymmetric IBZ - note that this IBZ can be rotated arbitrarily!
            gamma_x = X(true(size(X))); gamma_y = Y(true(size(Y))); % rect
            %         IBZ_shape = 'rectangle';
        case 'p4mm'
            % [X,Y] = meshgrid(linspace(0,pi/a,N_wv(1)),linspace(0,pi/a,N_wv(2))); % these are more points than we need, so we chop it with triu
            % gamma_x = X(triu(true(size(X)))); gamma_y = Y(triu(true(size(Y)))); % tri
            [X,Y] = meshgrid(linspace(-pi/a,pi/a,N_wv(1)),linspace(0,pi/a,N_wv(2))); % true asymmetric IBZ - note that this IBZ can be rotated arbitrarily!
            tol = 1e-6;
            mask = X >= 0-tol & Y >= 0-tol & (Y - X) <= tol;
            gamma_x = X(mask); gamma_y = Y(mask);
        case 'c1m1'
            if (floor(N_wv(2)/2) == N_wv(2)/2)
                error('For symmetry type c1m1, N_wv(2) must be an odd integer')
            end
            [X,Y] = meshgrid(linspace(0,pi/a,N_wv(1)),linspace(-pi/a,pi/a,N_wv(2))); % these are more points than we need, so we chop it with triu
            mask = triu(true([N_wv(1) (N_wv(2)+1)/2]));
            mask = [flipud(mask(2:end,:)); mask];
            gamma_x = X(mask); gamma_y = Y(mask);
        case 'p2mm'
            [X,Y] = meshgrid(linspace(0,pi/a,N_wv(1)),linspace(0,pi/a,N_wv(2)));
            gamma_x = X(:); gamma_y = Y(:);
        otherwise
            error('symmetry_type not recognized')
    end
    %         IBZ_shape = 'triangle';
    %     [X,Y] = meshgrid(linspace(0,pi/a,N_k),linspace(0,pi/a,N_k));
    %     [X,Y] = meshgrid(linspace(-pi/a,pi/a,N_k),linspace(-pi/a,pi/a,N_k)); % big rect
    %     [X,Y] = meshgrid(3*linspace(-pi/a,pi/a,N_k),3*linspace(-pi/a,pi/a,N_k)); % 3x big rect
    
    %     k_x = X(triu(true(size(X)))); k_y = Y(triu(true(size(Y)))); % tri
    %     k_x = X(true(size(X))); k_y = Y(true(size(Y))); % rect
    
    wavevectors = N_tesselations.*cat(2,gamma_x,gamma_y);
end