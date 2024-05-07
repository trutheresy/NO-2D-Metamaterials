function [wv,fr,ev] = dispersion(const,wavevectors)
    
    fr = zeros(size(wavevectors,2),const.N_eig);
    if const.isSaveEigenvectors
        ev = zeros(((const.N_ele*const.N_pix)^2)*2,size(wavevectors,2),const.N_eig);
    else
        ev = [];
    end
    if const.isUseImprovement
        [K,M] = get_system_matrices_VEC(const);
    else
        [K,M] = get_system_matrices(const);
    end
    if const.isUseParallel
        parforArg = Inf;
    else
        parforArg = 0;
    end
%     for k_idx = 1:size(wavevectors,1) % USE THIS TO DEBUG
    parfor (k_idx = 1:size(wavevectors,1), parforArg)
        wavevector = wavevectors(k_idx,:);
        T = get_transformation_matrix(wavevector,const);
        Kr = T'*K*T;
        Mr = T'*M*T;
        
        if ~const.isUseGPU
            % DONT USE THE GPU WITH EIGS           
            [eig_vecs,eig_vals] = eigs(Kr,Mr,const.N_eig,const.sigma_eig);
            [eig_vals,idxs] = sort(diag(eig_vals));
            eig_vecs = eig_vecs(:,idxs);
            
%             ev(:,k_idx,:) = (eig_vecs./(diag(eig_vecs'*Mr*eig_vecs)'))'; % normalize by mass matrix
            if const.isSaveEigenvectors
                ev(:,k_idx,:) = (eig_vecs./vecnorm(eig_vecs,2,1)).*exp(-1i*angle(eig_vecs(1,:))); % normalize by p-norm, align complex angle
            end
            %             ev(:,k_idx,:) = (eig_vecs./max(eig_vecs))'; % normalize by max
%             ev(:,k_idx,:) = eig_vecs'; % don't normalize
            
            fr(k_idx,:) = sqrt(real(eig_vals));
            fr(k_idx,:) = fr(k_idx,:)/(2*pi);
        elseif const.isUseGPU
            % USE THE GPU WITH EIG
            error('GPU use is not currently developed');
            MinvK_gpu = gpuArray(full(inv(Mr)*Kr));
            [eig_vecs,eig_vals] = eig(MinvK_gpu);
            [eig_vals,idxs] = sort(diag(eig_vals));
            eig_vecs = eig_vecs(:,idxs);
            fr(k_idx,:) = sqrt(real(eig_vals(1:const.N_eig)));
            fr(k_idx,:) = fr(k_idx,:)/(2*pi);
        end
    end
    wv = wavevectors;
end


